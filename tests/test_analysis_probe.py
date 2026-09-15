"""Tests for ``wingspan.analysis``: the architecture-probe package.

Covers:

1. ``ProbeAttention`` FULL mode vs. ``nn.MultiheadAttention`` (numerical fidelity).
2. ZERO / UNIFORM / HEAD_KNOCKOUT ablation semantics.
3. ``install`` / ``uninstall`` wiring and the empty-board finite/KL=0 guarantee.
4. ``measure()`` on a shared-attention net: report well-formedness.
5. ``measure()`` with a reference net, with board attention off, and that a
   mid-measurement exception still restores ``net``'s mode and wrappers.
6. ``summarize_for_loop()``.
7. ``evaluate_substitution()``.
8. The ``wingspan analysis probe`` CLI.

All nets here are tiny (32-wide trunk/choice, 8-dim card embedding) so the
whole file runs in a few seconds.
"""

from __future__ import annotations

import io
import pathlib
import random
import sys

import pytest
import torch
import torch.nn.functional as F

from wingspan import model
from wingspan.analysis import (
    attention_probe,
    cli,
    head_to_head,
    models,
    probe_set,
    representation,
)
from wingspan.training import collect, config

_SMALL_LAYERS = (32, 32)
_SMALL_HEAD_LAYERS = (16,)
_SMALL_CARD_EMBED_DIM = 8
_DEVICE = torch.device("cpu")


###### HELPERS #######


def _tiny_config(
    *, use_board_attention: bool = True, board_attention_shared: bool = True
) -> config.RunConfig:
    """A tiny ``RunConfig`` for fast probe tests, mirroring
    ``tests/test_loop_iteration_order.py``'s pattern."""
    return config.RunConfig(
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=_SMALL_LAYERS,
                choice_layers=_SMALL_LAYERS,
                head_layers=_SMALL_HEAD_LAYERS,
                card_embed_dim=_SMALL_CARD_EMBED_DIM,
                use_board_attention=use_board_attention,
                board_attention_heads=2,
                board_attention_positions=use_board_attention,
                board_attention_shared=board_attention_shared,
            ),
        ),
    )


def _tiny_net(cfg: config.RunConfig) -> model.PolicyValueNet:
    net = model.PolicyValueNet(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )
    net.eval()
    return net


def _save_checkpoint(
    net: model.PolicyValueNet, cfg: config.RunConfig, path: pathlib.Path
) -> None:
    """A minimal self-describing checkpoint that ``loaders.load_policy_net`` accepts."""
    torch.save(
        {
            "config": cfg.model_dump(),
            "model": net.state_dict(),
            "version": cfg.encoding_version,
        },
        path,
    )


def _run_probe_cli(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Drive ``cli.main_analysis`` with stdout swapped for a plain
    ``StringIO`` (no ``.buffer`` attribute, so ``cli._utf8_stdout`` falls back
    to it directly) — mirrors ``tests/test_inspect_era.py``'s ``_run_inspect``,
    avoiding a real ``io.TextIOWrapper`` over ``sys.stdout.buffer`` closing
    pytest's captured stdout out from under later tests."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    return cli.main_analysis(argv)


def _hand_computed_uniform(
    reference: torch.nn.MultiheadAttention,
    tokens: torch.Tensor,
    key_padding_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """A masked mean of the value projections through ``out_proj`` — the
    UNIFORM ablation computed independently of ``ProbeAttention``."""
    batch_size, seq_len, embed_dim = tokens.shape
    head_dim = embed_dim // num_heads
    projected = F.linear(tokens, reference.in_proj_weight, reference.in_proj_bias)
    _, _, value = projected.chunk(3, dim=-1)
    value_heads = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    keep = (
        (~key_padding_mask)
        .float()[:, None, None, :]
        .expand(batch_size, num_heads, seq_len, seq_len)
    )
    uniform_weights = keep / keep.sum(-1, keepdim=True).clamp(min=1.0)
    merged = (
        (uniform_weights @ value_heads)
        .transpose(1, 2)
        .reshape(batch_size, seq_len, embed_dim)
    )
    return F.linear(merged, reference.out_proj.weight, reference.out_proj.bias)


###### 1-2: ProbeAttention math #######


@pytest.mark.parametrize("num_heads,embed_dim", [(1, 16), (4, 24)])
def test_probe_attention_full_matches_torch(num_heads: int, embed_dim: int) -> None:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    reference = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, batch_first=True
    )
    reference.eval()
    probe = attention_probe.ProbeAttention(reference, true_width=embed_dim)

    tokens = torch.randn(4, 15, embed_dim)
    key_padding_mask = torch.zeros(4, 15, dtype=torch.bool)
    key_padding_mask[0, 5:] = True
    key_padding_mask[1, 0] = True

    with torch.no_grad():
        expected, _ = reference(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        actual, _ = probe(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

    assert torch.allclose(expected, actual, atol=1e-5)


def test_probe_attention_zero_returns_zeros() -> None:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    reference = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    probe = attention_probe.ProbeAttention(reference, true_width=16)
    probe.mode = models.AblationMode.ZERO

    tokens = torch.randn(2, 15, 16)
    mask = torch.zeros(2, 15, dtype=torch.bool)
    out, weights = probe(
        tokens, tokens, tokens, key_padding_mask=mask, need_weights=False
    )

    assert torch.equal(out, torch.zeros_like(tokens))
    assert weights is None


def test_probe_attention_uniform_matches_hand_computed_mean() -> None:
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    embed_dim, num_heads = 16, 4
    reference = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, batch_first=True
    )
    probe = attention_probe.ProbeAttention(reference, true_width=embed_dim)
    probe.mode = models.AblationMode.UNIFORM

    tokens = torch.randn(3, 15, embed_dim)
    mask = torch.zeros(3, 15, dtype=torch.bool)
    mask[0, 5:] = True
    mask[1, 10:] = True

    with torch.no_grad():
        actual, _ = probe(
            tokens, tokens, tokens, key_padding_mask=mask, need_weights=False
        )
        expected = _hand_computed_uniform(reference, tokens, mask, num_heads)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_probe_attention_head_knockout_changes_output() -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    embed_dim, num_heads = 16, 4
    reference = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, batch_first=True
    )
    probe = attention_probe.ProbeAttention(reference, true_width=embed_dim)

    tokens = torch.randn(2, 15, embed_dim)
    mask = torch.zeros(2, 15, dtype=torch.bool)

    with torch.no_grad():
        probe.mode = models.AblationMode.FULL
        full_out, _ = probe(
            tokens, tokens, tokens, key_padding_mask=mask, need_weights=False
        )
        probe.mode = models.AblationMode.HEAD_KNOCKOUT
        probe.knockout_head = 1
        knocked_out, _ = probe(
            tokens, tokens, tokens, key_padding_mask=mask, need_weights=False
        )

    assert not torch.allclose(full_out, knocked_out)


###### 3: install/uninstall + empty-board guarantee #######


def test_install_returns_expected_wrapper_counts_and_uninstall_restores() -> None:
    shared_net = _tiny_net(_tiny_config(board_attention_shared=True))
    shared_original = shared_net.board_attn
    shared_wrappers = attention_probe.install(shared_net)
    assert len(shared_wrappers) == 1
    attention_probe.uninstall(shared_net, shared_wrappers)
    assert shared_net.board_attn is shared_original

    unshared_net = _tiny_net(_tiny_config(board_attention_shared=False))
    pov_original = unshared_net.board_attn_me
    opp_original = unshared_net.board_attn_opp
    unshared_wrappers = attention_probe.install(unshared_net)
    assert len(unshared_wrappers) == 2
    attention_probe.uninstall(unshared_net, unshared_wrappers)
    assert unshared_net.board_attn_me is pov_original
    assert unshared_net.board_attn_opp is opp_original

    off_net = _tiny_net(_tiny_config(use_board_attention=False))
    assert attention_probe.install(off_net) == []


def test_forward_pass_finite_and_zero_equals_full_on_fresh_game() -> None:
    """The very first recorded decision of a fresh game has an empty board on
    every seat; every ablation mode should collapse to the same (finite,
    zero-contribution) trunk input there, so ZERO's policy exactly matches
    FULL's (KL 0)."""
    cfg = _tiny_config()
    net = _tiny_net(cfg)
    rng = random.Random(7)
    record = collect.play_game(
        net,
        _DEVICE,
        rng,
        seed=42,
        combine_gain_food=cfg.engine.combine_gain_food,
        num_players=cfg.num_players,
    )
    first_step = record.steps[0]
    state = torch.tensor(first_step.state, dtype=torch.float32).unsqueeze(0)
    choices = torch.tensor(first_step.choices, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, choices.shape[1])
    family_idx = torch.tensor([first_step.family_idx], dtype=torch.long)

    wrappers = attention_probe.install(net)
    outputs: dict[models.AblationMode, tuple[torch.Tensor, torch.Tensor]] = {}
    try:
        for mode in models.AblationMode:
            if mode == models.AblationMode.HEAD_KNOCKOUT:
                attention_probe.set_mode(wrappers, mode, head=0)
            else:
                attention_probe.set_mode(wrappers, mode)
            with torch.no_grad():
                logits, value = net(state, choices, mask, family_idx)
            assert torch.isfinite(logits).all()
            assert torch.isfinite(value).all()
            outputs[mode] = (logits, value)
    finally:
        attention_probe.uninstall(net, wrappers)

    full_logits, _ = outputs[models.AblationMode.FULL]
    zero_logits, _ = outputs[models.AblationMode.ZERO]
    full_log_probs = F.log_softmax(full_logits, dim=-1)
    zero_log_probs = F.log_softmax(zero_logits, dim=-1)
    kl = (full_log_probs.exp() * (full_log_probs - zero_log_probs)).sum(-1)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


###### 4-5: measure() #######


def test_measure_shared_attention_report_is_well_formed() -> None:
    cfg = _tiny_config(board_attention_shared=True)
    net = _tiny_net(cfg)
    probes = probe_set.from_self_play(net, cfg, n_games=2, seed=100, device=_DEVICE)

    report = representation.measure(
        net,
        probes,
        device=_DEVICE,
        score_norm=cfg.training.score_norm,
        checkpoint_label="test",
    )

    for layer in report.layers:
        assert 1 <= layer.rank95 <= layer.out_features
        assert 0.0 <= layer.linear_r2 <= 1.0
        assert 0.0 <= layer.dead_fraction <= 1.0
    for ablation in report.ablations:
        assert ablation.mean_kl >= 0.0
    assert sum(effect.n for effect in report.family_effects) == report.n_decisions
    assert report.attention is not None
    num_heads = len(report.attention.heads)
    assert num_heads == 2
    assert len(report.ablations) == 2 + num_heads
    total_share = sum(entry.share for entry in report.param_census)
    assert total_share == pytest.approx(1.0, abs=1e-6)


def test_measure_with_reference_net_reports_positive_kl() -> None:
    cfg = _tiny_config()
    torch.manual_seed(10)  # pyright: ignore[reportUnknownMemberType]
    net = _tiny_net(cfg)
    torch.manual_seed(20)  # pyright: ignore[reportUnknownMemberType]
    reference_net = _tiny_net(cfg)
    probes = probe_set.from_self_play(net, cfg, n_games=1, seed=200, device=_DEVICE)

    report = representation.measure(
        net,
        probes,
        device=_DEVICE,
        score_norm=cfg.training.score_norm,
        reference_net=reference_net,
        checkpoint_label="test",
    )

    assert report.reference is not None
    assert report.reference.mean_kl > 0.0


def test_measure_attention_off_has_no_attention_or_ablations() -> None:
    cfg = _tiny_config(use_board_attention=False)
    net = _tiny_net(cfg)
    probes = probe_set.from_self_play(net, cfg, n_games=1, seed=300, device=_DEVICE)

    report = representation.measure(
        net, probes, device=_DEVICE, score_norm=cfg.training.score_norm
    )

    assert report.attention is None
    assert report.ablations == []
    assert report.family_effects == []
    assert report.board_fill_effects == []
    assert report.layers  # trunk/choice layer stats are still produced


def test_measure_restores_net_mode_and_uninstalls_wrappers_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-measurement exception (a CUDA OOM, a linalg error in the ridge
    fit, ...) must not leave ``net`` stuck in eval mode or still wearing its
    ``ProbeAttention`` wrappers. Offline this was harmless (the CLI process
    just exits); it matters now that ``training.loop_probe`` calls ``measure``
    live, mid-run, on a net that collection/update keep using afterward."""
    cfg = _tiny_config(board_attention_shared=True)
    net = _tiny_net(cfg)
    net.train()  # the live loop's net is normally in train mode, not eval
    original_board_attn = net.board_attn
    probes = probe_set.from_self_play(net, cfg, n_games=1, seed=900, device=_DEVICE)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic measurement failure")

    # _run_ablations runs after _install_attention, so wrappers is non-empty
    # here — exercising the real uninstall path, not just its empty-list no-op.
    monkeypatch.setattr(representation, "_run_ablations", _boom)

    with pytest.raises(RuntimeError, match="synthetic measurement failure"):
        representation.measure(
            net, probes, device=_DEVICE, score_norm=cfg.training.score_norm
        )

    assert net.training is True
    assert net.board_attn is original_board_attn


###### 6: summarize_for_loop() #######


def test_summarize_for_loop_carries_layers_and_uniform_zero() -> None:
    cfg = _tiny_config()
    net = _tiny_net(cfg)
    probes = probe_set.from_self_play(net, cfg, n_games=1, seed=400, device=_DEVICE)
    report = representation.measure(
        net, probes, device=_DEVICE, score_norm=cfg.training.score_norm
    )

    loop_metrics = representation.summarize_for_loop(report)

    assert loop_metrics.probe_decisions == report.n_decisions
    assert [layer.name for layer in loop_metrics.layers] == [
        layer.name for layer in report.layers
    ]
    zero = report.effect(models.AblationMode.ZERO)
    uniform = report.effect(models.AblationMode.UNIFORM)
    assert zero is not None
    assert uniform is not None
    assert loop_metrics.zero_kl == pytest.approx(zero.mean_kl)
    assert loop_metrics.zero_flip_rate == pytest.approx(zero.flip_rate)
    assert loop_metrics.uniform_kl == pytest.approx(uniform.mean_kl)
    assert loop_metrics.uniform_flip_rate == pytest.approx(uniform.flip_rate)


###### 7: evaluate_substitution() #######


def test_evaluate_substitution_plays_expected_game_count(
    tmp_path: pathlib.Path,
) -> None:
    cfg = _tiny_config()
    net = _tiny_net(cfg)
    checkpoint_path = tmp_path / "tiny.pt"
    _save_checkpoint(net, cfg, checkpoint_path)

    result = head_to_head.evaluate_substitution(
        checkpoint_path,
        models.AblationMode.UNIFORM,
        n_pairs=1,
        seed=500,
        device=_DEVICE,
    )

    assert result.n_games == 2  # num_players (2) * n_pairs (1)
    assert 0.0 <= result.win_rate <= 1.0


###### 8: CLI #######


def test_cli_probe_writes_valid_json_report(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _tiny_config()
    net = _tiny_net(cfg)
    checkpoint_path = tmp_path / "tiny.pt"
    _save_checkpoint(net, cfg, checkpoint_path)
    out_path = tmp_path / "report.json"

    exit_code = _run_probe_cli(
        ["probe", str(checkpoint_path), "--games", "1", "--json", str(out_path)],
        monkeypatch,
    )

    assert exit_code == 0
    report = models.RepresentationReport.model_validate_json(
        out_path.read_text(encoding="utf-8")
    )
    assert report.n_games == 1


def test_cli_probe_rejects_random_target(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_probe_cli(["probe", "random"], monkeypatch) == 1
