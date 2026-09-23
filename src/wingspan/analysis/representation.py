"""The architecture-probe main entry: :func:`measure` and :func:`summarize_for_loop`.

:func:`measure` is the single place that runs a whole probe pass over a
:class:`probe_set.ProbeSet`: a parameter census, per-layer capacity
diagnostics on the trunk and choice encoder, board-attention statistics and
ablation effects (zero / uniform / per-head knockout), the same ablations
sliced by judgment family and by the deciding player's board-fill count, an
optional reference-checkpoint comparison, and — for the default
hand/tray-encoding regime — a trunk input-energy-share breakdown. Everything
here is read-only: no gradients, no checkpoint writes, and ``net`` is restored
to its original train/eval mode before returning.

**Caveats** (see ``docs/analysis/INDEX.md`` for the full list): the ablation
effects measure *dependence*, not *necessity* — a block scoring near-zero
effect may simply be redundant with information available elsewhere in the
state, not literally unused. A ``reference_net`` comparison assumes the same
artifact era as ``net``; this module does not check or reconcile eras. The
trunk input-share breakdown only supports the default (non-distinct-hand,
non-tray-set) encoding regime; other configurations report an empty list.
"""

from __future__ import annotations

import typing

import torch
import torch.nn.functional as F
import torch.utils.hooks as hooks
from torch import nn

from wingspan import decisions, encode, model, state
from wingspan.analysis import attention_probe, layer_probe, models
from wingspan.analysis import probe_set as probe_set_module

# The upper-tail percentile every "p90" policy-delta readout is computed at.
_HIGH_QUANTILE = 0.9

# Numerical floor so a degenerate (zero-variance) denominator never divides.
_VARIANCE_EPS = 1e-12

# State-layout stripe names bracketing the card-set multi-hot blocks the trunk
# embeds after the hand itself (see ``model.core._extract_hand_blocks``).
_HAND_MULTIHOT_STRIPE = "hand_multihot"
_DECISION_TYPE_STRIPE = "decision_type"


def measure(
    net: model.PolicyValueNet,
    probe_set: probe_set_module.ProbeSet,
    *,
    device: torch.device,
    score_norm: float,
    reference_net: model.PolicyValueNet | None = None,
    checkpoint_label: str = "",
) -> models.RepresentationReport:
    """Run every probe over ``probe_set`` and return one
    :class:`models.RepresentationReport`.

    ``net`` is switched to ``eval()`` for the duration (restored to its prior
    mode before returning) and every pass runs under ``torch.no_grad()`` — this
    function never updates ``net``'s weights. ``score_norm`` converts the raw
    value-head shift into score points (mirrors the training run's own
    ``TrainingConfig.score_norm``).

    ``probe_set``'s tensors may live on any device: every batch is moved to
    ``device`` for its forward pass, and the statistics run there too (the
    only cross-device copies are the batches themselves and the small
    per-decision selector vectors), so a cpu-built probe set measured on a
    cuda net — the training loop's cpu-collect / cuda-train case — works.

    The wrapper install/mode-restore is wrapped in ``try``/``finally`` so an
    exception anywhere in the probe (a CUDA OOM, a linalg error in the ridge
    fit, a bad ``probe_set``) can never leave ``net`` stuck in eval mode or
    still wearing its ``ProbeAttention`` wrappers. Offline this was harmless —
    the CLI process just exits — but ``training.loop_probe`` now calls this
    live, mid-run, on the same net that collection and the update step keep
    using afterward, so a failed probe must not corrupt it."""
    was_training = net.training
    net.eval()
    wrappers: list[attention_probe.ProbeAttention] = []
    try:
        with torch.no_grad():
            census, total_parameters = _param_census(net)
            wrappers, collector = _install_attention(net)
            full_pass, layer_stats, attention_stats, trunk_input = _run_full_pass(
                net, probe_set, device, wrappers, collector
            )
            ablations, family_effects, board_fill_effects = _run_ablations(
                net, probe_set, device, wrappers, full_pass, score_norm
            )
            reference = None
            if reference_net is not None:
                reference = _run_reference_pass(
                    reference_net,
                    probe_set,
                    device,
                    full_pass,
                    score_norm,
                    checkpoint_label,
                )
            trunk_shares = (
                _trunk_input_shares(net, trunk_input)
                if _trunk_shares_supported(net)
                else []
            )
    finally:
        attention_probe.uninstall(net, wrappers)
        if was_training:
            net.train()
    return models.RepresentationReport(
        checkpoint=checkpoint_label,
        n_games=probe_set.n_games,
        n_decisions=probe_set.n_decisions,
        param_census=census,
        total_parameters=total_parameters,
        layers=layer_stats,
        attention=attention_stats,
        ablations=ablations,
        reference=reference,
        family_effects=family_effects,
        board_fill_effects=board_fill_effects,
        trunk_input_shares=trunk_shares,
        head_to_head=[],
    )


def summarize_for_loop(
    report: models.RepresentationReport,
) -> models.RepresentationMetrics:
    """Project a full :class:`models.RepresentationReport` down to the
    lightweight :class:`models.RepresentationMetrics` shape a training-loop
    metrics row would carry (Stage 2 — this package does not write one)."""
    zero = report.effect(models.AblationMode.ZERO)
    uniform = report.effect(models.AblationMode.UNIFORM)
    attention_entropy_median = (
        [head.entropy_median for head in report.attention.heads]
        if report.attention is not None
        else []
    )
    return models.RepresentationMetrics(
        probe_decisions=report.n_decisions,
        layers=[layer.summary for layer in report.layers],
        attention_entropy_median=attention_entropy_median,
        uniform_kl=uniform.mean_kl if uniform is not None else None,
        uniform_flip_rate=uniform.flip_rate if uniform is not None else None,
        zero_kl=zero.mean_kl if zero is not None else None,
        zero_flip_rate=zero.flip_rate if zero is not None else None,
    )


###### PRIVATE #######

#### Forward-pass plumbing ####


class _ForwardPass(typing.NamedTuple):
    """Per-batch log-softmax policy plus concatenated value output of one
    full pass over a :class:`probe_set_module.ProbeSet`. Kept as a list per
    batch (not concatenated) since batches have different choice counts."""

    log_probs: list[torch.Tensor]
    values: torch.Tensor


class _PolicyDelta(typing.NamedTuple):
    """Per-decision KL(full‖other), greedy-flip indicator, and value shift in
    score points, concatenated across every batch of a probe pass pair."""

    kl: torch.Tensor
    flip: torch.Tensor
    value_shift: torch.Tensor


def _forward_pass(
    net: model.PolicyValueNet,
    probe_set: probe_set_module.ProbeSet,
    device: torch.device,
) -> _ForwardPass:
    """Run ``net`` over every batch of ``probe_set`` once, under whatever
    ablation mode its (already-installed) attention wrappers currently hold."""
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for batch in probe_set.batches:
        logits, value = net(
            batch.state.to(device),
            batch.choices.to(device),
            batch.mask.to(device),
            batch.family_idx.to(device),
        )
        log_probs.append(F.log_softmax(logits, dim=-1))
        values.append(value)
    return _ForwardPass(log_probs=log_probs, values=torch.cat(values))


def _pass_under_mode(
    net: model.PolicyValueNet,
    probe_set: probe_set_module.ProbeSet,
    device: torch.device,
    wrappers: list[attention_probe.ProbeAttention],
    mode: models.AblationMode,
    head: int | None = None,
) -> _ForwardPass:
    """Set every wrapper to ``mode`` (and ``head`` for HEAD_KNOCKOUT), then
    run and return one :func:`_forward_pass`."""
    if head is None:
        attention_probe.set_mode(wrappers, mode)
    else:
        attention_probe.set_mode(wrappers, mode, head=head)
    return _forward_pass(net, probe_set, device)


def _policy_delta(
    full_pass: _ForwardPass, other_pass: _ForwardPass, score_norm: float
) -> _PolicyDelta:
    """``KL(full‖other)`` and the greedy-flip indicator, batch-by-batch
    (choice counts differ per batch, so this cannot be one tensor op), plus
    the value shift in score points. KL is clamped at 0 — float32 rounding on
    two near-identical distributions (e.g. UNIFORM over an already-near-flat
    policy) can otherwise land a hair below zero, which is never a genuine
    divergence."""
    kl_parts: list[torch.Tensor] = []
    flip_parts: list[torch.Tensor] = []
    for full_lp, other_lp in zip(full_pass.log_probs, other_pass.log_probs):
        kl_parts.append((full_lp.exp() * (full_lp - other_lp)).sum(-1))
        flip_parts.append((full_lp.argmax(-1) != other_lp.argmax(-1)).float())
    value_shift = (full_pass.values - other_pass.values).abs() * score_norm
    return _PolicyDelta(
        kl=torch.cat(kl_parts).clamp(min=0.0),
        flip=torch.cat(flip_parts),
        value_shift=value_shift,
    )


def _policy_delta_stats(
    delta: _PolicyDelta, selector: torch.Tensor
) -> models.PolicyDeltaStats:
    """Roll up ``delta`` over the rows ``selector`` picks out; an empty
    selection reports an explicit all-zero, ``n=0`` row rather than NaN."""
    n = int(selector.sum().item())
    if n == 0:
        return models.PolicyDeltaStats(
            n=0, mean_kl=0.0, p90_kl=0.0, flip_rate=0.0, value_shift_points=0.0
        )
    return models.PolicyDeltaStats(
        n=n,
        mean_kl=float(delta.kl[selector].mean()),
        p90_kl=float(delta.kl[selector].quantile(_HIGH_QUANTILE)),
        flip_rate=float(delta.flip[selector].mean()),
        value_shift_points=float(delta.value_shift[selector].mean()),
    )


def _batch_context(
    probe_set: probe_set_module.ProbeSet, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate every batch's ``family_idx`` / ``own_bird_count``, in the
    same batch order :func:`_forward_pass` visits (so they align row-for-row
    with a ``_PolicyDelta``'s tensors), moved to ``device`` so they can select
    rows of the pass outputs — which live on ``device``, not wherever the
    probe set was built (cpu, during training)."""
    family_idx_all = torch.cat([batch.family_idx for batch in probe_set.batches])
    own_count_all = torch.cat([batch.own_bird_count for batch in probe_set.batches])
    return family_idx_all.to(device), own_count_all.to(device)


#### Parameter census ####


def _param_census(
    net: model.PolicyValueNet,
) -> tuple[list[models.ParamCensusEntry], int]:
    """Trainable-parameter count by top-level block name (the first dotted
    component of each parameter's qualified name), largest first."""
    counts: dict[str, int] = {}
    for name, parameter in net.named_parameters():
        block = name.split(".")[0]
        counts[block] = counts.get(block, 0) + parameter.numel()
    total = sum(counts.values())
    entries = [
        models.ParamCensusEntry(
            block=block, parameters=count, share=(count / total if total else 0.0)
        )
        for block, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
    return entries, total


#### Attention install + the one FULL pass that also captures layers/trunk input ####


def _install_attention(
    net: model.PolicyValueNet,
) -> tuple[
    list[attention_probe.ProbeAttention], attention_probe.AttentionStatsCollector | None
]:
    """Install the attention wrapper(s) (if any) and, when present, one
    shared :class:`attention_probe.AttentionStatsCollector` every wrapper
    reports into."""
    wrappers = attention_probe.install(net)
    if not wrappers:
        return wrappers, None
    collector = attention_probe.AttentionStatsCollector(
        wrappers[0].num_heads, wrappers[0].true_width
    )
    for wrapper in wrappers:
        wrapper.collector = collector
    return wrappers, collector


def _run_full_pass(
    net: model.PolicyValueNet,
    probe_set: probe_set_module.ProbeSet,
    device: torch.device,
    wrappers: list[attention_probe.ProbeAttention],
    collector: attention_probe.AttentionStatsCollector | None,
) -> tuple[
    _ForwardPass,
    list[models.LayerStats],
    models.AttentionBlockStats | None,
    torch.Tensor,
]:
    """The single FULL-mode pass that also captures trunk/choice layer
    activations, board-attention statistics, and the trunk's raw input (for
    the input-share breakdown) — the only pass that needs those hooks."""
    attention_probe.set_mode(wrappers, models.AblationMode.FULL)
    trunk_probe = layer_probe.LinearLayerProbe("trunk", net.state_trunk)
    choice_probe = layer_probe.LinearLayerProbe("choice", net.choice_encoder)
    trunk_input_handle, trunk_input_rows = _capture_trunk_input(net)

    full_pass = _forward_pass(net, probe_set, device)

    trunk_probe.remove()
    choice_probe.remove()
    trunk_input_handle.remove()
    layer_stats = trunk_probe.stats() + choice_probe.stats()
    attention_stats = collector.summarize() if collector is not None else None
    return full_pass, layer_stats, attention_stats, torch.cat(trunk_input_rows)


#### Ablations: zero / uniform / per-head knockout, and their slices ####


def _run_ablations(
    net: model.PolicyValueNet,
    probe_set: probe_set_module.ProbeSet,
    device: torch.device,
    wrappers: list[attention_probe.ProbeAttention],
    full_pass: _ForwardPass,
    score_norm: float,
) -> tuple[
    list[models.AblationEffect], list[models.FamilyEffect], list[models.BoardFillEffect]
]:
    """ZERO, UNIFORM, and one HEAD_KNOCKOUT per head, each scored against
    ``full_pass``, plus the zero/uniform effect sliced by judgment family and
    by board-fill band. Empty when ``net`` has no board attention to ablate."""
    if not wrappers:
        return [], [], []
    zero_pass = _pass_under_mode(
        net, probe_set, device, wrappers, models.AblationMode.ZERO
    )
    uniform_pass = _pass_under_mode(
        net, probe_set, device, wrappers, models.AblationMode.UNIFORM
    )
    zero_delta = _policy_delta(full_pass, zero_pass, score_norm)
    uniform_delta = _policy_delta(full_pass, uniform_pass, score_norm)
    ablations = [
        _ablation_effect_from_delta(models.AblationMode.ZERO, None, zero_delta),
        _ablation_effect_from_delta(models.AblationMode.UNIFORM, None, uniform_delta),
    ]
    for head in range(wrappers[0].num_heads):
        head_pass = _pass_under_mode(
            net,
            probe_set,
            device,
            wrappers,
            models.AblationMode.HEAD_KNOCKOUT,
            head=head,
        )
        head_delta = _policy_delta(full_pass, head_pass, score_norm)
        ablations.append(
            _ablation_effect_from_delta(
                models.AblationMode.HEAD_KNOCKOUT, head, head_delta
            )
        )

    family_idx_all, own_count_all = _batch_context(probe_set, device)
    families = tuple(
        family.value for family in decisions.active_decision_families(net.include_setup)
    )
    family_effects = _family_effects(
        families, family_idx_all, zero_delta, uniform_delta
    )
    board_fill_effects = _board_fill_effects(own_count_all, zero_delta, uniform_delta)
    return ablations, family_effects, board_fill_effects


def _ablation_effect_from_delta(
    mode: models.AblationMode, head: int | None, delta: _PolicyDelta
) -> models.AblationEffect:
    return models.AblationEffect(
        mode=mode,
        head=head,
        n=int(delta.kl.shape[0]),
        mean_kl=float(delta.kl.mean()),
        p90_kl=float(delta.kl.quantile(_HIGH_QUANTILE)),
        flip_rate=float(delta.flip.mean()),
        value_shift_points=float(delta.value_shift.mean()),
    )


def _family_effects(
    families: tuple[str, ...],
    family_idx_all: torch.Tensor,
    zero_delta: _PolicyDelta,
    uniform_delta: _PolicyDelta,
) -> list[models.FamilyEffect]:
    """Zero/uniform effect sliced to each judgment family present in the
    probe set (a family absent from this particular sample is skipped)."""
    effects: list[models.FamilyEffect] = []
    for index, family in enumerate(families):
        selector = family_idx_all == index
        n = int(selector.sum().item())
        if n == 0:
            continue
        effects.append(
            models.FamilyEffect(
                family=family,
                n=n,
                zero=_policy_delta_stats(zero_delta, selector),
                uniform=_policy_delta_stats(uniform_delta, selector),
            )
        )
    return effects


def _board_fill_effects(
    own_count_all: torch.Tensor, zero_delta: _PolicyDelta, uniform_delta: _PolicyDelta
) -> list[models.BoardFillEffect]:
    """Zero/uniform effect sliced to each :data:`models.BOARD_FILL_BANDS` band
    the probe set has at least one decision in."""
    effects: list[models.BoardFillEffect] = []
    for min_birds, max_birds in models.BOARD_FILL_BANDS:
        selector = (own_count_all >= min_birds) & (own_count_all <= max_birds)
        n = int(selector.sum().item())
        if n == 0:
            continue
        effects.append(
            models.BoardFillEffect(
                min_birds=min_birds,
                max_birds=max_birds,
                n=n,
                zero=_policy_delta_stats(zero_delta, selector),
                uniform=_policy_delta_stats(uniform_delta, selector),
            )
        )
    return effects


#### Reference-checkpoint comparison ####


def _run_reference_pass(
    reference_net: model.PolicyValueNet,
    probe_set: probe_set_module.ProbeSet,
    device: torch.device,
    full_pass: _ForwardPass,
    score_norm: float,
    checkpoint_label: str,
) -> models.ReferenceComparison:
    """Score ``reference_net`` over the same probe-set batches (its own
    natural, unwrapped forward pass — no ablation) and report its KL / flip /
    value-shift against ``full_pass``. Assumes the same artifact era as the
    net under test; see the module docstring."""
    reference_net.eval()
    reference_pass = _forward_pass(reference_net, probe_set, device)
    delta = _policy_delta(full_pass, reference_pass, score_norm)
    return models.ReferenceComparison(
        checkpoint=checkpoint_label,
        n=int(delta.kl.shape[0]),
        mean_kl=float(delta.kl.mean()),
        p90_kl=float(delta.kl.quantile(_HIGH_QUANTILE)),
        flip_rate=float(delta.flip.mean()),
        value_shift_points=float(delta.value_shift.mean()),
    )


#### Trunk input-energy shares ####


def _trunk_shares_supported(net: model.PolicyValueNet) -> bool:
    """The input-share breakdown only understands the default (pooled-hand,
    no-tray-set-embedding) trunk input layout."""
    return not net.arch.tray_set_embedding and not net.arch.use_distinct_hand_model


def _capture_trunk_input(
    net: model.PolicyValueNet,
) -> tuple[hooks.RemovableHandle, list[torch.Tensor]]:
    """Hook the trunk's first ``nn.Linear`` to capture every call's raw
    (embedded) input, for the input-share breakdown."""
    first_linear = _first_linear(net.state_trunk)
    captured: list[torch.Tensor] = []

    def hook(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor
    ) -> None:
        captured.append(inputs[0].detach())

    return first_linear.register_forward_hook(hook), captured


def _first_linear(sequential: nn.Sequential) -> nn.Linear:
    return next(module for module in sequential if isinstance(module, nn.Linear))


def _trunk_input_groups(net: model.PolicyValueNet) -> list[tuple[str, int]]:
    """Named trunk-input column groups, in the exact order
    ``model.core._embed_state`` / ``_embed_state_board_attention`` concatenate
    them after the leading continuous block: own board, opponent board(s),
    tray, hand pool, then one group per extra card-set multi-hot block (the
    playability stripes and, from v1.5, each opponent's known hand), labelled
    by the stripe each block embeds."""
    slots = encode.SLOTS_PER_BOARD
    card_embed_dim = net.arch.card_embed_dim
    token_width = card_embed_dim
    if net.arch.use_board_attention:
        token_width += encode.SLOT_SCALAR_DIM
        if net.arch.board_attention_positions_active:
            token_width += encode.BOARD_POSITION_DIM
    own_width = slots * token_width
    opp_width = (net.arch.num_players - 1) * slots * token_width
    tray_width = state.TRAY_SIZE * card_embed_dim
    hand_width = net.arch.pooled_hand_width
    groups = [
        ("own_board", own_width),
        ("opp_board", opp_width),
        ("tray", tray_width),
        ("hand_pool", hand_width),
    ]
    groups.extend((f"{name}_pool", hand_width) for name in _extra_hand_block_names(net))
    return groups


def _extra_hand_block_names(net: model.PolicyValueNet) -> list[str]:
    """Names of the card-set multi-hot stripes the trunk embeds after the hand
    itself, read from the net's own era-routed state layout (the stripes
    between ``hand_multihot`` and ``decision_type``) so each input group is
    labelled by what it encodes rather than by position. Falls back to
    positional names when the layout does not bracket exactly
    ``encode.n_extra_hand_multihots`` stripes there."""
    n_extra = encode.n_extra_hand_multihots(net.spec)
    names = [stripe.name for stripe in net.raw_state_stripe_layout().stripes]
    if _HAND_MULTIHOT_STRIPE in names and _DECISION_TYPE_STRIPE in names:
        start = names.index(_HAND_MULTIHOT_STRIPE) + 1
        end = names.index(_DECISION_TYPE_STRIPE)
        between = names[start:end]
        if len(between) == n_extra:
            return between
    return [f"extra_multihot_{index + 1}" for index in range(n_extra)]


def _trunk_input_shares(
    net: model.PolicyValueNet, trunk_input: torch.Tensor
) -> list[models.InputGroupShare]:
    """Share of the first trunk layer's pre-activation variance ("energy")
    each named input group carries — the ridge-free ``||W_g x_g||^2`` share
    computation ``trunk_inputs.py`` prototyped, generalized to attention-off
    nets and arbitrary seat counts."""
    first_linear = _first_linear(net.state_trunk)
    weight = first_linear.weight.detach()
    bias = first_linear.bias.detach()
    named_groups = _trunk_input_groups(net)
    total_width = trunk_input.shape[1]
    groups: list[tuple[str, int]] = [
        ("continuous", total_width - sum(width for _, width in named_groups)),
        *named_groups,
    ]

    full_pre_activation = trunk_input @ weight.T + bias
    centered_total = full_pre_activation - full_pre_activation.mean(0, keepdim=True)
    total_energy = centered_total.pow(2).sum().clamp(min=_VARIANCE_EPS)

    shares: list[models.InputGroupShare] = []
    cursor = 0
    for name, width in groups:
        columns = trunk_input[:, cursor : cursor + width]
        group_pre_activation = columns @ weight[:, cursor : cursor + width].T
        centered_group = group_pre_activation - group_pre_activation.mean(
            0, keepdim=True
        )
        share = float(centered_group.pow(2).sum() / total_energy)
        shares.append(
            models.InputGroupShare(group=name, dims=width, energy_share=share)
        )
        cursor += width
    return shares
