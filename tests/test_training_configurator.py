"""Tests for the FLIGHT PLAN configurator (``wingspan.training.configure``).

* Key decoding: the pure ``decode_*`` helpers (no console needed).
* Fields: read / format, validated commit + rejection, nudge clamping, choice
  cycling, changed-vs-saved detection.
* Runs: inspecting a checkpoint dir, architecture compatibility + status, and
  the archive / clear / list operations (with a fast hand-written checkpoint —
  no training run required).
* Screen: ``build`` renders empty / populated / edit / confirm states wide and
  narrow without error.
* Controller: the console-free ``build_initial_state`` + ``dispatch`` core —
  navigation, nudging, editing, the Start / New / Archive actions, and quit.
* App wiring: ``--config`` parsing and the centralized device fallback.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")
pytest.importorskip("rich")

import rich.console as rich_console
import torch

from wingspan import architecture, encode, model, setup_model, version
from wingspan.training import artifacts, config, loop, runstate, setup_net
from wingspan.training.configure import (
    arch_diagram,
    controller,
    fields,
    keys,
    runs,
    screen,
    state,
    user_defaults,
)

# --------------------------------------------------------------------------- #
# keys                                                                        #
# --------------------------------------------------------------------------- #


def test_decode_char_control_and_printable():
    assert keys.decode_char("\r").kind is keys.KeyKind.ENTER
    assert keys.decode_char("\n").kind is keys.KeyKind.ENTER
    assert keys.decode_char("\t").kind is keys.KeyKind.TAB
    assert keys.decode_char("\x08").kind is keys.KeyKind.BACKSPACE
    assert keys.decode_char("\x7f").kind is keys.KeyKind.BACKSPACE
    assert keys.decode_char("\x1b").kind is keys.KeyKind.ESCAPE
    assert keys.decode_char("\x03").kind is keys.KeyKind.INTERRUPT
    printable = keys.decode_char("7")
    assert printable.kind is keys.KeyKind.CHAR and printable.char == "7"


def test_decode_windows_special_arrows():
    assert keys.decode_windows_special("H").kind is keys.KeyKind.UP
    assert keys.decode_windows_special("P").kind is keys.KeyKind.DOWN
    assert keys.decode_windows_special("K").kind is keys.KeyKind.LEFT
    assert keys.decode_windows_special("M").kind is keys.KeyKind.RIGHT
    assert keys.decode_windows_special("?").kind is keys.KeyKind.OTHER


def test_decode_unix_escape_arrows():
    assert keys.decode_unix_escape("A").kind is keys.KeyKind.UP
    assert keys.decode_unix_escape("D").kind is keys.KeyKind.LEFT
    assert keys.decode_unix_escape("5~").kind is keys.KeyKind.PAGE_UP


# --------------------------------------------------------------------------- #
# fields                                                                      #
# --------------------------------------------------------------------------- #


def test_format_value_scientific_and_plain():
    cfg = config.TrainConfig(device="cpu")
    assert fields.format_value(cfg, fields.spec_for("lr")) == "3e-04"
    assert fields.format_value(cfg, fields.spec_for("value_coef")) == "0.5"
    assert fields.format_value(cfg, fields.spec_for("games_per_iter")) == "256"
    assert fields.format_value(cfg, fields.spec_for("device")) == "cpu"


def test_commit_validates_against_model_bounds():
    cfg = config.TrainConfig(device="cpu")
    lr = fields.spec_for("lr")
    updated, error = fields.commit(cfg, lr, "0.001")
    assert error is None and updated.lr == 0.001
    # lr must be strictly positive — the model rejects 0.
    rejected, error = fields.commit(cfg, lr, "0")
    assert error is not None and rejected.lr == cfg.lr
    # alphas are capped at 1.0.
    alpha = fields.spec_for("eval_ewma_alpha")
    _, alpha_error = fields.commit(cfg, alpha, "1.5")
    assert alpha_error is not None
    # non-numeric input is a parse error, not a crash.
    _, parse_error = fields.commit(cfg, lr, "abc")
    assert parse_error is not None


def test_nudge_steps_and_clamps():
    cfg = config.TrainConfig(device="cpu")
    games = fields.spec_for("games_per_iter")
    assert isinstance(games, fields.IntField)
    up, error = fields.nudge(cfg, games, 1)
    assert error is None and up.games_per_iter == cfg.games_per_iter + games.step
    # Nudging lr below its strictly-positive floor is rejected, value unchanged.
    low = config.TrainConfig(device="cpu", lr=1e-4)
    stepped, error = fields.nudge(low, fields.spec_for("lr"), -1)
    assert error is not None and stepped.lr == low.lr


def test_nudge_cycles_choice():
    cfg = config.TrainConfig(device="cpu")
    device = fields.spec_for("device")
    forward, _ = fields.nudge(cfg, device, 1)
    assert forward.device == "cuda"
    wrapped, _ = fields.nudge(forward, device, 1)
    assert wrapped.device == "cpu"


def test_is_changed_against_saved():
    saved = config.TrainConfig(device="cpu")
    working = saved.model_copy(update={"games_per_iter": 72})
    games = fields.spec_for("games_per_iter")
    lr = fields.spec_for("lr")
    assert fields.is_changed(working, saved, games)
    assert not fields.is_changed(working, saved, lr)
    assert not fields.is_changed(working, None, games)  # no saved run -> unchanged


def test_layers_field_format_and_commit():
    cfg = config.TrainConfig(device="cpu")
    trunk = fields.spec_for("trunk_layers")
    assert isinstance(trunk, fields.LayersField)
    assert fields.format_value(cfg, trunk) == "128, 128"
    # Typing comma/space separated widths sets the sizes (the final width stays
    # 128 so it still matches the choice encoder's last layer).
    updated, error = fields.commit(cfg, trunk, "256, 192, 128")
    assert error is None and updated.trunk_layers == (256, 192, 128)
    # An empty head list formats as "none" and parses back to ().
    head = fields.spec_for("head_layers")
    empty, error = fields.commit(cfg, head, "none")
    assert error is None and empty.head_layers == ()
    assert fields.format_value(empty, head) == "none"
    # Non-numeric tokens are a parse error, not a crash.
    _, parse_error = fields.commit(cfg, trunk, "256, wide")
    assert parse_error is not None


def test_layers_field_widths_are_independent():
    # Trunk and choice encoder widths are now fully independent; editing one
    # does not touch the other.
    cfg = config.TrainConfig(device="cpu")  # trunk=(128,128), choice=(128,128)
    choice = fields.spec_for("choice_layers")
    updated, error = fields.commit(cfg, choice, "128, 64")
    assert error is None
    assert updated.choice_layers == (128, 64)
    # trunk_layers is unchanged — no auto-sync.
    assert updated.trunk_layers == (128, 128)

    trunk = fields.spec_for("trunk_layers")
    updated2, error2 = fields.commit(cfg, trunk, "256, 32")
    assert error2 is None
    assert updated2.trunk_layers == (256, 32)
    # choice_layers is unchanged — no auto-sync.
    assert updated2.choice_layers == (128, 128)


def test_layers_field_nudge_changes_depth():
    cfg = config.TrainConfig(device="cpu")  # trunk defaults to (128, 128)
    trunk = fields.spec_for("trunk_layers")
    deeper, error = fields.nudge(cfg, trunk, 1)
    assert error is None and deeper.trunk_layers == (128, 128, 128)  # duplicates last
    shallower, error = fields.nudge(deeper, trunk, -1)
    assert error is None and shallower.trunk_layers == (128, 128)
    # A body block cannot drop below its single-layer floor.
    one_layer = cfg.model_copy(update={"trunk_layers": (128,), "choice_layers": (128,)})
    floored, error = fields.nudge(one_layer, trunk, -1)
    assert error is not None and floored.trunk_layers == (128,)


# --------------------------------------------------------------------------- #
# runs                                                                        #
# --------------------------------------------------------------------------- #


def _write_checkpoint(
    directory: pathlib.Path,
    cfg: config.TrainConfig,
    *,
    iteration: int = 4,
    total_games: int = 320,
    best_win_rate: float | None = 0.71,
    extras: bool = True,
) -> None:
    """Write a minimal but realistically-shaped ``last.pt`` (no training run)."""
    directory.mkdir(parents=True, exist_ok=True)
    progress = runstate.RunProgress(
        iteration=iteration,
        total_games=total_games,
        best_win_rate=best_win_rate,
        opponent_generation=0,
    )
    payload = {
        "config": cfg.model_dump(),
        "progress": progress.model_dump(),
        "git_sha": "abc1234",
    }
    torch.save(payload, directory / artifacts.LAST_CKPT)
    if extras:
        (directory / artifacts.BEST_CKPT).write_bytes(b"best")
        (directory / artifacts.METRICS_LOG).write_text("{}\n", encoding="utf-8")
        (directory / artifacts.GAMES_LOG).write_text("{}\n", encoding="utf-8")
        (directory / artifacts.MODEL_CONFIG_JSON).write_text("{}", encoding="utf-8")
        (directory / "process_20260530-120000.json").write_text("{}", encoding="utf-8")
        (directory / "dashboard.log").write_text("log", encoding="utf-8")
        (directory / "_test.pt").write_bytes(b"scratch")  # unrelated, must survive


def test_inspect_run_reads_metadata(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", run_name="alpha")
    _write_checkpoint(tmp_path, cfg)
    summary = runs.inspect_run(str(tmp_path))
    assert summary.exists and summary.readable
    assert summary.iteration == 4 and summary.total_games == 320
    assert summary.best_win_rate == 0.71
    assert summary.train_config is not None and summary.train_config.run_name == "alpha"
    assert summary.has_best and summary.has_metrics and summary.has_games


def test_inspect_run_empty_dir(tmp_path: pathlib.Path):
    summary = runs.inspect_run(str(tmp_path))
    assert not summary.exists and summary.train_config is None
    assert summary.archives == []


def test_architecture_compatible_and_status(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    # No readable embedded config -> never resumable (self-describing contract).
    assert not runs.architecture_compatible(None, cfg)
    assert runs.architecture_compatible(cfg, cfg)
    wider = cfg.model_copy(
        update={"trunk_layers": (256, 256), "choice_layers": (256, 256)}
    )
    assert not runs.architecture_compatible(cfg, wider)

    _write_checkpoint(tmp_path, cfg)
    summary = runs.inspect_run(str(tmp_path))
    assert runs.resolve_status(summary, cfg) is runs.RunStatus.RESUMABLE
    assert runs.resolve_status(summary, wider) is runs.RunStatus.INCOMPATIBLE
    empty = runs.inspect_run(str(tmp_path / "missing"))
    assert runs.resolve_status(empty, cfg) is runs.RunStatus.EMPTY


def test_archive_run_moves_artifacts_and_leaves_scratch(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    _write_checkpoint(tmp_path, cfg)
    result = runs.archive_run(str(tmp_path), "run_iter0004_T")
    assert result.ok
    archive_dir = tmp_path / artifacts.ARCHIVE_SUBDIR / "run_iter0004_T"
    assert (archive_dir / artifacts.LAST_CKPT).exists()
    assert (archive_dir / artifacts.BEST_CKPT).exists()
    assert (archive_dir / "dashboard.log").exists()
    # The new run artifacts ride along: the game log, model descriptor, and the
    # dated process record are all relocated, not left behind.
    assert (archive_dir / artifacts.GAMES_LOG).exists()
    assert (archive_dir / artifacts.MODEL_CONFIG_JSON).exists()
    assert (archive_dir / "process_20260530-120000.json").exists()
    # The live dir is now clean of run artifacts, but unrelated scratch survives.
    assert not (tmp_path / artifacts.LAST_CKPT).exists()
    assert not (tmp_path / artifacts.METRICS_LOG).exists()
    assert not (tmp_path / artifacts.GAMES_LOG).exists()
    assert not list(tmp_path.glob(artifacts.PROCESS_GLOB))
    assert (tmp_path / "_test.pt").exists()
    assert artifacts.LAST_CKPT in result.moved


def test_archive_run_unique_label(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    _write_checkpoint(tmp_path, cfg)
    runs.archive_run(str(tmp_path), "dup")
    _write_checkpoint(tmp_path, cfg)
    runs.archive_run(str(tmp_path), "dup")
    archive_root = tmp_path / artifacts.ARCHIVE_SUBDIR
    assert (archive_root / "dup").is_dir()
    assert (archive_root / "dup-1").is_dir()


def test_archive_missing_files_tolerated(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    _write_checkpoint(tmp_path, cfg, extras=False)  # only last.pt present
    result = runs.archive_run(str(tmp_path), "label")
    assert result.ok and result.moved == [artifacts.LAST_CKPT]


def test_clear_run_deletes_artifacts(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    _write_checkpoint(tmp_path, cfg)
    removed = runs.clear_run(str(tmp_path))
    assert artifacts.LAST_CKPT in removed
    assert artifacts.GAMES_LOG in removed and artifacts.MODEL_CONFIG_JSON in removed
    assert not (tmp_path / artifacts.LAST_CKPT).exists()
    assert not list(tmp_path.glob(artifacts.PROCESS_GLOB))
    assert (tmp_path / "_test.pt").exists()


def test_default_archive_label_sanitizes(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", run_name="my run!")
    _write_checkpoint(tmp_path, cfg, iteration=7)
    summary = runs.inspect_run(str(tmp_path))
    label = runs.default_archive_label(summary, "20260530-120000")
    assert label == "my_run_iter0007_20260530-120000"


def test_list_archives(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    _write_checkpoint(tmp_path, cfg)
    runs.archive_run(str(tmp_path), "first")
    entries = runs.list_archives(str(tmp_path))
    assert [entry.label for entry in entries] == ["first"]
    assert entries[0].has_checkpoint


# --------------------------------------------------------------------------- #
# screen                                                                      #
# --------------------------------------------------------------------------- #


# A frame on which the edit caret is in its "off" half (caret blinks every
# screen._CARET_BLINK_FRAMES frames: on for [0,8), off for [8,16)).
_BLINK_OFF_FRAME = 8


def _render(
    view: state.ConfiguratorState, width: int = 120, height: int = 44, frame: int = 0
) -> str:
    buffer = io.StringIO()
    term = rich_console.Console(
        file=buffer, width=width, height=height, force_terminal=True, color_system=None
    )
    term.print(screen.build(view, frame=frame))
    return buffer.getvalue()


def _empty_state() -> state.ConfiguratorState:
    cfg = config.TrainConfig(device="cpu", checkpoint_dir="checkpoints")
    summary = runs.RunSummary(checkpoint_dir="checkpoints")
    return state.ConfiguratorState(
        working=cfg, summary=summary, selected_attr=fields.editable_attrs()[0]
    )


@pytest.mark.parametrize("width", [128, 80])
def test_screen_renders_empty(width: int):
    out = _render(_empty_state(), width=width)
    assert "FLIGHT PLAN" in out
    assert "CONFIGURATION" in out and "RUN MANAGEMENT" in out and "FIELD" in out


def test_screen_renders_populated_and_edit():
    cfg = config.TrainConfig(device="cpu")
    summary = runs.RunSummary(
        checkpoint_dir="checkpoints",
        exists=True,
        train_config=cfg,
        iteration=12,
        total_games=768,
        best_win_rate=0.83,
        archives=[
            runs.ArchiveEntry(label="old_run", modified=1.0, has_checkpoint=True)
        ],
    )
    view = state.ConfiguratorState(
        working=cfg.model_copy(update={"lr": 1e-3}),
        saved=cfg,
        summary=summary,
        selected_attr="lr",
        seeded_from_saved=True,
    )
    assert "resume" in _render(view)
    view.mode = state.Mode.EDIT
    view.edit_buffer = "0.001"
    assert "EDITING" in _render(view)


def test_bootstrap_hint_fixed_values():
    """Selecting bootstrap_opponent with 'none' or 'random' shows descriptive hints."""
    summary = runs.RunSummary(checkpoint_dir="checkpoints")
    for value, expected_fragment in [
        ("none", "no bootstrap"),
        ("random", "random agent"),
    ]:
        cfg = config.TrainConfig(
            device="cpu", checkpoint_dir="checkpoints", bootstrap_opponent=value
        )
        view = state.ConfiguratorState(
            working=cfg, summary=summary, selected_attr="bootstrap_opponent"
        )
        # Render wide enough that the hint is not truncated.
        out = _render(view, width=200)
        assert expected_fragment in out, f"expected {expected_fragment!r} for {value!r}"


def test_dispatch_nudge_bootstrap_cycles_through_options(tmp_path: pathlib.Path):
    """Left/Right on the bootstrap_opponent field cycles none → random → archive."""
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    _write_checkpoint(tmp_path, cfg)
    runs.archive_run(str(tmp_path), "archived_run")
    view = controller.build_initial_state(cfg, cuda_available=False)
    view.selected_attr = "bootstrap_opponent"

    # Default is "random" (index 1 in [none, random, archive]).
    assert view.working.bootstrap_opponent == "random"

    # Right: random → archive path.
    controller.dispatch(view, _key(keys.KeyKind.RIGHT))
    archive_path = view.working.bootstrap_opponent
    assert archive_path not in ("none", "random")

    # Right again wraps: archive → none.
    controller.dispatch(view, _key(keys.KeyKind.RIGHT))
    assert view.working.bootstrap_opponent == "none"

    # Left from none wraps back to archive.
    controller.dispatch(view, _key(keys.KeyKind.LEFT))
    assert view.working.bootstrap_opponent == archive_path

    # Left: archive → random.
    controller.dispatch(view, _key(keys.KeyKind.LEFT))
    assert view.working.bootstrap_opponent == "random"


def test_bootstrap_hint_shows_archive_metadata():
    """Selecting bootstrap_opponent with a path matching a known archive entry
    shows the archive's version, game count, and session stamp in the detail hint."""
    checkpoint_dir = "checkpoints"
    archive_label = "my_run_iter0100"
    expected_path = str(
        pathlib.Path(checkpoint_dir)
        / artifacts.ARCHIVE_SUBDIR
        / archive_label
        / artifacts.LAST_CKPT
    )
    entry = runs.ArchiveEntry(
        label=archive_label,
        modified=1748649600.0,
        has_checkpoint=True,
        model_version="0.1",
        total_games=12345,
        first_session_stamp="20240611-142030",
    )
    cfg = config.TrainConfig(
        device="cpu", checkpoint_dir=checkpoint_dir, bootstrap_opponent=expected_path
    )
    summary = runs.RunSummary(checkpoint_dir=checkpoint_dir, archives=[entry])
    view = state.ConfiguratorState(
        working=cfg, summary=summary, selected_attr="bootstrap_opponent"
    )
    out = _render(view)
    assert "v0.1" in out
    assert "12,345" in out
    assert "20240611-142030" in out


def test_bootstrap_hint_archive_entry_no_stamp_uses_date():
    """When first_session_stamp is absent the hint falls back to the archived date."""
    checkpoint_dir = "checkpoints"
    archive_label = "old_run"
    expected_path = str(
        pathlib.Path(checkpoint_dir)
        / artifacts.ARCHIVE_SUBDIR
        / archive_label
        / artifacts.LAST_CKPT
    )
    entry = runs.ArchiveEntry(
        label=archive_label,
        modified=1748649600.0,  # 2025-05-30
        has_checkpoint=True,
        model_version=None,
        total_games=None,
        first_session_stamp=None,
    )
    cfg = config.TrainConfig(
        device="cpu", checkpoint_dir=checkpoint_dir, bootstrap_opponent=expected_path
    )
    summary = runs.RunSummary(checkpoint_dir=checkpoint_dir, archives=[entry])
    view = state.ConfiguratorState(
        working=cfg, summary=summary, selected_attr="bootstrap_opponent"
    )
    out = _render(view)
    assert "archived" in out


def test_bootstrap_hint_custom_path_not_in_archives():
    """A bootstrap_opponent path that does not match any archive shows 'custom'."""
    cfg = config.TrainConfig(
        device="cpu",
        checkpoint_dir="checkpoints",
        bootstrap_opponent="some/custom/path.pt",
    )
    summary = runs.RunSummary(checkpoint_dir="checkpoints")
    view = state.ConfiguratorState(
        working=cfg, summary=summary, selected_attr="bootstrap_opponent"
    )
    out = _render(view)
    assert "custom checkpoint" in out


def test_screen_renders_era_line_and_defaults_hints():
    # An era-pinned RESUMABLE run shows its frozen era; the footer always
    # offers [D] save defaults; a defaults-seeded editor names its source.
    pinned = config.with_encoding_version(
        config.TrainConfig(device="cpu", checkpoint_dir="checkpoints"), "0.2"
    )
    summary = runs.RunSummary(
        checkpoint_dir="checkpoints", exists=True, train_config=pinned, iteration=3
    )
    view = state.ConfiguratorState(
        working=pinned,
        saved=pinned,
        summary=summary,
        selected_attr=fields.editable_attrs()[0],
        seeded_from_saved=True,
    )
    out = _render(view)
    assert "era 0.2 (resume)" in out
    assert "save defaults" in out

    fresh = _empty_state()
    fresh.seeded_from_user_defaults = True
    fresh_out = _render(fresh)
    assert f"era {version.MODEL_VERSION} (new run)" in fresh_out
    assert "saved defaults" in fresh_out


def test_screen_renders_confirm_modal():
    view = _empty_state()
    view.mode = state.Mode.CONFIRM
    view.confirm = state.ConfirmPrompt(
        title="START A NEW RUN",
        lines=["A run already exists.", "Archive or overwrite?"],
        options=[
            state.ConfirmOption(
                key="a",
                label="archive & start",
                action=state.ConfirmAction.ARCHIVE_THEN_FRESH,
            ),
            state.ConfirmOption(
                key="c", label="cancel", action=state.ConfirmAction.CANCEL
            ),
        ],
        default_key="a",
    )
    out = _render(view)
    assert "START A NEW RUN" in out and "archive & start" in out


# --------------------------------------------------------------------------- #
# architecture diagram                                                        #
# --------------------------------------------------------------------------- #


# A direct render wide enough to engage the two-column box mode (not the narrow
# fallback) and tall enough that the whole diagram is visible without the viewport
# clipping the bottom — the per-size test below covers clipping.
_BOX_W = 64
_BOX_H = 120


def _arch_state(
    selected_attr: str = "trunk_layers", **overrides: object
) -> state.ConfiguratorState:
    base = config.TrainConfig(device="cpu", checkpoint_dir="checkpoints")
    cfg = base.model_copy(update=dict(overrides))
    summary = runs.RunSummary(checkpoint_dir="checkpoints")
    return state.ConfiguratorState(
        working=cfg, summary=summary, selected_attr=selected_attr
    )


def _render_diagram(view: state.ConfiguratorState, width: int, height: int) -> str:
    buffer = io.StringIO()
    term = rich_console.Console(
        file=buffer, width=width, height=height, force_terminal=True, color_system=None
    )
    term.print(arch_diagram.ArchitectureDiagram(view))
    return buffer.getvalue()


def _box_diagram(view: state.ConfiguratorState) -> str:
    """The diagram rendered wide + tall enough that box mode is active and the
    whole flow is visible."""
    return _render_diagram(view, width=_BOX_W, height=_BOX_H)


def _param_report_for(cfg: config.TrainConfig) -> architecture.ParamReport:
    return architecture.count_parameters(
        cfg.arch,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            cfg.state_dim,
            cfg.card_embed_dim,
            use_distinct_hand_model=cfg.use_distinct_hand_model,
            hand_embed_dim=cfg.hand_embed_dim,
            tray_set_embedding=cfg.tray_set_embedding,
        ),
        choice_in=encode.choice_input_dim(
            cfg.choice_dim,
            cfg.card_embed_dim,
            include_setup=cfg.encoding_spec.include_setup,
        ),
        num_families=len(cfg.family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )


@pytest.mark.parametrize("width,height", [(128, 44), (128, 18), (80, 44), (80, 18)])
def test_arch_diagram_renders_all_sizes(width: int, height: int):
    out = _render(_arch_state(), width=width, height=height)
    assert "ARCHITECTURE" in out and "TRUNK" in out  # top of the flow is always shown


def test_arch_diagram_all_blocks_present():
    out = _box_diagram(_arch_state())
    for block in (
        "SETUP MODEL",
        "CARD ENCODER",
        "STATE TRUNK",
        "CHOICE ENC",
        "VALUE",
        "DECISION",
    ):
        assert block in out


def test_arch_diagram_setup_box_toggles():
    # The separate setup net is its own box at the top when enabled, and a single
    # "off" line otherwise (it is trained independently of the in-game policy).
    on = _box_diagram(_arch_state())  # use_setup_model defaults True
    assert "SETUP MODEL" in on
    off = _box_diagram(_arch_state(use_setup_model=False))
    assert "SETUP MODEL" not in off
    assert "setup model · off" in off


def test_arch_diagram_extra_input_boxes():
    # Each trunk carries a small box counting the additional, non-card features it
    # consumes alongside the fanned-out card embeddings.
    out = _box_diagram(_arch_state())
    assert "feats" in out


def test_arch_diagram_card_encoder_is_mlp():
    # The card encoder renders as an MLP body block, so its hidden widths show up
    # as Linear layer boxes (here a distinctive 256-wide hidden layer).
    out = _box_diagram(_arch_state("card_encoder_layers", card_encoder_layers=(256,)))
    assert "CARD ENCODER" in out and "256" in out


def test_arch_diagram_dropout_appears_and_hides():
    assert "Dropout" not in _box_diagram(_arch_state())  # default 0 -> no card
    assert "Dropout" in _box_diagram(_arch_state("dropout", dropout=0.15))


def test_arch_diagram_layernorm_appears():
    assert "LayerNorm" not in _box_diagram(_arch_state())  # default off
    assert "LayerNorm" in _box_diagram(_arch_state("layernorm", layernorm=True))


def test_arch_diagram_activation_label():
    assert "relu" in _box_diagram(_arch_state())
    assert "gelu" in _box_diagram(
        _arch_state("activation", activation=architecture.ActivationName.GELU)
    )


def test_arch_diagram_readout_never_layernorms():
    # Even with LayerNorm enabled and a hidden scorer layer, the readout heads
    # must not draw a LayerNorm card (mirrors model._build_readout). LayerNorm
    # shows up in the body blocks (above the heads) but never from DECISION onward.
    view = _arch_state(layernorm=True, head_layers=(128,))
    out = _box_diagram(view)
    assert "LayerNorm" in out  # the trunk / choice bodies do carry it
    assert "LayerNorm" not in out.split("DECISION", 1)[1]


def test_arch_diagram_collapse_tag():
    view = _arch_state(
        trunk_layers=(128, 128, 128, 128), choice_layers=(128, 128, 128, 128)
    )
    assert "×4" in _box_diagram(view)  # four identical trunk layers fold to ×4


def test_arch_diagram_narrow_fallback():
    # Below the two-column floor the renderable drops to the compact text list;
    # it must still name the blocks and not crash.
    out = _render_diagram(_arch_state(), width=16, height=20)
    assert "TRUNK" in out


def test_arch_diagram_focus_highlight_smoke():
    out = _box_diagram(_arch_state("dropout", dropout=0.15))
    assert "Dropout" in out  # focusing the dropout handle still renders cleanly


def test_arch_diagram_param_count_matches_model():
    # The analytic per-block accounting equals sum(p.numel()) of the real net,
    # exercising LayerNorm params, a per-family scorer multiplier, both heads, and
    # asymmetric trunk/choice widths (M=16, N=24) so the scorer's M+N input is
    # distinct from 2M and 2N — a regression to a "2H" concat would fail here.
    cfg = config.TrainConfig(
        device="cpu",
        trunk_layers=(32, 16),
        choice_layers=(64, 24),
        head_layers=(8,),
        value_layers=(8,),
        card_embed_dim=8,
        layernorm=True,
        dropout=0.1,
    )
    net = model.PolicyValueNet(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
    )
    report = _param_report_for(cfg)
    assert report.total == sum(param.numel() for param in net.parameters())


def test_arch_diagram_param_count_scales_with_embed_dim():
    small = _param_report_for(config.TrainConfig(device="cpu", card_embed_dim=16))
    large = _param_report_for(config.TrainConfig(device="cpu", card_embed_dim=64))
    assert large.total > small.total


def test_arch_diagram_setup_param_count_matches_net():
    # The separate setup net's analytic accounting equals sum(p.numel()) of the
    # real SetupNet — the diagram's per-op / Σ source for the unconnected box.
    # Frozen embedder copies count in numel, so they must count analytically too.
    cfg = config.TrainConfig(device="cpu", setup_hidden_layers=(32, 16))
    block = setup_model.count_setup_parameters(
        cfg.setup_arch,
        feature_dim=setup_model.SETUP_FEATURE_DIM,
        main_arch=cfg.arch,
    )
    net = setup_net.SetupNet(
        encoding=setup_model.SetupEncoding(),
        arch=cfg.setup_arch,
        main_arch=cfg.arch,
    )
    assert block.total == sum(param.numel() for param in net.parameters())


def test_arch_diagram_param_display():
    out = _box_diagram(_arch_state())
    assert "TOTAL" in out and "params" in out


# --------------------------------------------------------------------------- #
# controller                                                                  #
# --------------------------------------------------------------------------- #


def _key(kind: keys.KeyKind, char: str = "") -> keys.KeyEvent:
    return keys.KeyEvent(kind=kind, char=char)


def test_initial_state_empty_uses_defaults(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    view = controller.build_initial_state(cfg, cuda_available=False)
    assert not view.seeded_from_saved
    assert view.saved is None
    assert view.status() is runs.RunStatus.EMPTY


def test_initial_state_seeds_from_compatible_run(tmp_path: pathlib.Path):
    saved = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), run_name="saved", lr=7e-4
    )
    _write_checkpoint(tmp_path, saved)
    launched = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    view = controller.build_initial_state(launched, cuda_available=False)
    assert view.seeded_from_saved
    assert view.working.lr == 7e-4  # tuned from the saved run, not the launch defaults
    assert view.status() is runs.RunStatus.RESUMABLE


def test_initial_state_seeds_from_run_with_other_architecture(tmp_path: pathlib.Path):
    # Reopening --config on a run whose architecture differs from the argparse
    # defaults must still load *its* saved settings, so the screen opens on the
    # actual run (RESUMABLE, nothing marked changed) rather than reverting to the
    # defaults and reporting a spurious "architecture changed / needs fresh run".
    saved = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), trunk_layers=(256, 256)
    )
    assert saved.trunk_layers != config.TrainConfig().trunk_layers  # not the default
    _write_checkpoint(tmp_path, saved)
    launched = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    view = controller.build_initial_state(launched, cuda_available=False)
    assert view.seeded_from_saved
    assert view.working.trunk_layers == (256, 256)
    assert view.status() is runs.RunStatus.RESUMABLE


def test_dispatch_navigation_and_nudge():
    view = _empty_state()
    attrs = fields.editable_attrs()
    assert controller.dispatch(view, _key(keys.KeyKind.DOWN)) is state.Outcome.CONTINUE
    assert view.selected_attr == attrs[1]
    controller.dispatch(view, _key(keys.KeyKind.UP))
    assert view.selected_attr == attrs[0]
    before = view.working.games_per_iter
    controller.dispatch(view, _key(keys.KeyKind.RIGHT))
    assert view.working.games_per_iter > before


def test_dispatch_edit_flow():
    view = _empty_state()
    view.selected_attr = "lr"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    assert view.mode is state.Mode.EDIT
    view.edit_buffer = ""  # retype from scratch
    for char in "0.002":
        controller.dispatch(view, _key(keys.KeyKind.CHAR, char))
    assert view.edit_buffer == "0.002"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    assert view.mode is state.Mode.NAVIGATE and view.working.lr == 0.002


def test_dispatch_edit_rejects_invalid():
    view = _empty_state()
    view.selected_attr = "lr"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    view.edit_buffer = "0"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    # Invalid commit keeps EDIT mode and surfaces an error message.
    assert view.mode is state.Mode.EDIT
    assert view.message is not None and view.message.kind is state.MessageKind.ERROR


def test_dispatch_start_empty_launches_fresh(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    view = controller.build_initial_state(cfg, cuda_available=False)
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "s"))
    assert outcome is state.Outcome.LAUNCH and view.working.resume is False


def test_dispatch_start_resumable_resumes(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    _write_checkpoint(tmp_path, cfg)
    view = controller.build_initial_state(cfg, cuda_available=False)
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "s"))
    assert outcome is state.Outcome.LAUNCH and view.working.resume is True


def test_dispatch_new_run_prompts_then_archives(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    _write_checkpoint(tmp_path, cfg)
    view = controller.build_initial_state(cfg, cuda_available=False)
    assert (
        controller.dispatch(view, _key(keys.KeyKind.CHAR, "n"))
        is state.Outcome.CONTINUE
    )
    assert view.mode is state.Mode.CONFIRM and view.confirm is not None
    # Choosing "archive & start" archives the existing run and launches fresh.
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "a"))
    assert outcome is state.Outcome.LAUNCH and view.working.resume is False
    assert (tmp_path / artifacts.ARCHIVE_SUBDIR).is_dir()
    assert not (tmp_path / artifacts.LAST_CKPT).exists()


def test_dispatch_archive_only_keeps_screen(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    _write_checkpoint(tmp_path, cfg)
    view = controller.build_initial_state(cfg, cuda_available=False)
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "a"))  # archive action
    assert view.mode is state.Mode.CONFIRM
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "a"))  # confirm
    assert outcome is state.Outcome.CONTINUE  # housekeeping — no launch
    assert view.mode is state.Mode.NAVIGATE
    assert view.status() is runs.RunStatus.EMPTY  # re-inspected: now clean


def test_dispatch_confirm_cancel(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    _write_checkpoint(tmp_path, cfg)
    view = controller.build_initial_state(cfg, cuda_available=False)
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "n"))
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "c"))  # cancel
    assert view.mode is state.Mode.NAVIGATE
    assert (tmp_path / artifacts.LAST_CKPT).exists()  # nothing was touched


def test_dispatch_quit():
    view = _empty_state()
    assert controller.dispatch(view, _key(keys.KeyKind.CHAR, "q")) is state.Outcome.QUIT
    assert controller.dispatch(view, _key(keys.KeyKind.INTERRUPT)) is state.Outcome.QUIT


def test_dispatch_edit_checkpoint_dir_reinspects(tmp_path: pathlib.Path):
    run_dir = tmp_path / "run"
    empty_dir = tmp_path / "elsewhere"
    saved = config.TrainConfig(device="cpu", checkpoint_dir=str(run_dir), lr=7e-4)
    _write_checkpoint(run_dir, saved)
    view = controller.build_initial_state(
        config.TrainConfig(device="cpu", checkpoint_dir=str(run_dir)),
        cuda_available=False,
    )
    assert view.seeded_from_saved and view.working.lr == 7e-4
    # Point checkpoint_dir at an empty directory; the re-inspect must drop the
    # "resumed run" framing and report EMPTY, keeping the user's edited values.
    view.selected_attr = "checkpoint_dir"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    view.edit_buffer = str(empty_dir)
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    assert view.working.checkpoint_dir == str(empty_dir)
    assert not view.seeded_from_saved
    assert view.status() is runs.RunStatus.EMPTY
    assert view.working.lr == 7e-4  # edits preserved across the re-inspect


# --------------------------------------------------------------------------- #
# era alignment (the working config's encoding_version follows the saved run) #
# --------------------------------------------------------------------------- #


def _pinned_config(directory: pathlib.Path, **overrides: object) -> config.TrainConfig:
    """A factory-architecture config pinned at era 0.2, as a run started before
    the 0.3 encoding change would have saved it."""
    base: dict[str, object] = {"device": "cpu", "checkpoint_dir": str(directory)}
    base.update(overrides)
    return config.with_encoding_version(config.TrainConfig.model_validate(base), "0.2")


def _pinned_view(tmp_path: pathlib.Path) -> state.ConfiguratorState:
    """A configurator opened on a directory holding an era-0.2 saved run."""
    _write_checkpoint(tmp_path, _pinned_config(tmp_path))
    return controller.build_initial_state(
        config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path)),
        cuda_available=False,
    )


def test_align_era_pins_and_unpins(tmp_path: pathlib.Path):
    pinned = _pinned_config(tmp_path)
    _write_checkpoint(tmp_path, pinned)
    summary = runs.inspect_run(str(tmp_path))
    live = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))

    # Same architecture at the saved era: pinned to it.
    assert runs.align_era(summary, live).encoding_version == "0.2"
    # A genuinely different architecture: live era (a fresh run would start).
    wider = live.model_copy(update={"trunk_layers": (256, 256)})
    assert runs.align_era(summary, wider).encoding_version == version.MODEL_VERSION
    # No run at all: live era, even for an already-pinned working config.
    empty = runs.inspect_run(str(tmp_path / "missing"))
    assert runs.align_era(empty, pinned).encoding_version == version.MODEL_VERSION
    # An unreadable checkpoint: live era too.
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / artifacts.LAST_CKPT).write_bytes(b"not a checkpoint")
    unreadable = runs.inspect_run(str(bad_dir))
    assert runs.align_era(unreadable, pinned).encoding_version == version.MODEL_VERSION


def test_fresh_edit_bumps_era_and_revert_repins(tmp_path: pathlib.Path):
    view = _pinned_view(tmp_path)
    assert view.working.encoding_version == "0.2"
    assert view.status() is runs.RunStatus.RESUMABLE
    default_widths = ",".join(str(width) for width in view.working.trunk_layers)

    # A FRESH-impact edit (trunk widths) breaks compatibility: the era bumps to
    # the live version, the derived dims re-sync, and the footer says so.
    view.selected_attr = "trunk_layers"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    view.edit_buffer = "256,128"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    assert view.working.encoding_version == version.MODEL_VERSION
    assert view.working.state_dim == encode.state_size(view.working.encoding_spec)
    assert view.status() is runs.RunStatus.INCOMPATIBLE
    assert view.message is not None and "era" in view.message.text

    # Reverting the edit restores compatibility: re-pinned to the saved era.
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    view.edit_buffer = default_widths
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    assert view.working.encoding_version == "0.2"
    assert view.status() is runs.RunStatus.RESUMABLE
    assert view.message is not None and "resume" in view.message.text


def test_regime_edit_keeps_saved_era(tmp_path: pathlib.Path):
    view = _pinned_view(tmp_path)
    view.selected_attr = "lr"
    controller.dispatch(view, _key(keys.KeyKind.RIGHT))  # nudge: REGIME impact
    assert view.working.encoding_version == "0.2"
    assert view.status() is runs.RunStatus.RESUMABLE


def test_new_run_over_old_era_launches_live(tmp_path: pathlib.Path):
    view = _pinned_view(tmp_path)
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "n"))
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "a"))  # archive & start
    assert outcome is state.Outcome.LAUNCH and view.working.resume is False
    assert view.working.encoding_version == version.MODEL_VERSION


def test_overwrite_over_old_era_launches_live(tmp_path: pathlib.Path):
    view = _pinned_view(tmp_path)
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "n"))
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "o"))  # overwrite
    assert outcome is state.Outcome.LAUNCH and view.working.resume is False
    assert view.working.encoding_version == version.MODEL_VERSION


def test_start_empty_launches_live_era(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    view = controller.build_initial_state(cfg, cuda_available=False)
    view.working = config.with_encoding_version(view.working, "0.2")  # stale pin
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "s"))
    assert outcome is state.Outcome.LAUNCH and view.working.resume is False
    assert view.working.encoding_version == version.MODEL_VERSION


def test_reinspect_to_empty_dir_unpins_era(tmp_path: pathlib.Path):
    run_dir = tmp_path / "run"
    empty_dir = tmp_path / "elsewhere"
    _write_checkpoint(run_dir, _pinned_config(run_dir, lr=7e-4))
    view = controller.build_initial_state(
        config.TrainConfig(device="cpu", checkpoint_dir=str(run_dir)),
        cuda_available=False,
    )
    assert view.working.encoding_version == "0.2"
    view.selected_attr = "checkpoint_dir"
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    view.edit_buffer = str(empty_dir)
    controller.dispatch(view, _key(keys.KeyKind.ENTER))
    # The era is a property of the inspected directory: an empty target means a
    # fresh run at the live version, while the user's other edits survive.
    assert view.working.encoding_version == version.MODEL_VERSION
    assert view.working.state_dim == encode.state_size(view.working.encoding_spec)
    assert view.working.lr == 7e-4
    assert view.status() is runs.RunStatus.EMPTY


# --------------------------------------------------------------------------- #
# user defaults ([D] save / seeding / reset chooser)                           #
# --------------------------------------------------------------------------- #


def test_save_defaults_roundtrip_and_exclusions(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="tuned",
        lr=7e-4,
        trunk_layers=(256, 128),
    )
    path = user_defaults.save_defaults(cfg, directory=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["saved_with_version"] == version.MODEL_VERSION
    for excluded in user_defaults.EXCLUDED_FIELDS:
        assert excluded not in raw["settings"]
    assert raw["settings"]["lr"] == 7e-4

    current = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path / "other"), run_name="fresh"
    )
    loaded = user_defaults.load_defaults(current, directory=tmp_path)
    assert loaded.warning is None and loaded.train_config is not None
    assert loaded.train_config.lr == 7e-4
    assert loaded.train_config.trunk_layers == (256, 128)
    # Run-identity fields stay the caller's, not the file's (nor factory).
    assert loaded.train_config.checkpoint_dir == str(tmp_path / "other")
    assert loaded.train_config.run_name == "fresh"
    assert loaded.train_config.device == "cpu"
    # The era is never persisted: a loaded config is always at the live version.
    assert loaded.train_config.encoding_version == version.MODEL_VERSION


def test_load_defaults_missing_file_is_empty(tmp_path: pathlib.Path):
    current = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    loaded = user_defaults.load_defaults(current, directory=tmp_path)
    assert loaded.train_config is None and loaded.warning is None


def test_corrupt_or_invalid_defaults_fall_back_with_warning(tmp_path: pathlib.Path):
    current = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    defaults_path = tmp_path / user_defaults.DEFAULTS_FILENAME

    # Garbage bytes: warned, no config.
    defaults_path.write_text("not json{", encoding="utf-8")
    garbage = user_defaults.load_defaults(current, directory=tmp_path)
    assert garbage.train_config is None and garbage.warning is not None

    # A valid envelope whose settings no longer validate: warned too.
    envelope = user_defaults.DefaultsFile(
        saved_with_version="0.2",
        saved_at="2026-01-01T00:00:00",
        settings={"lr": "zero"},
    )
    defaults_path.write_text(envelope.model_dump_json(), encoding="utf-8")
    invalid = user_defaults.load_defaults(current, directory=tmp_path)
    assert invalid.train_config is None
    assert invalid.warning is not None and "0.2" in invalid.warning

    # Renamed / removed fields from another era are simply ignored.
    renamed = user_defaults.DefaultsFile(
        saved_with_version="0.2",
        saved_at="2026-01-01T00:00:00",
        settings={"lr": 7e-4, "some_retired_field": 3},
    )
    defaults_path.write_text(renamed.model_dump_json(), encoding="utf-8")
    tolerant = user_defaults.load_defaults(current, directory=tmp_path)
    assert tolerant.train_config is not None and tolerant.train_config.lr == 7e-4


def test_dispatch_save_defaults_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    view = _empty_state()
    outcome = controller.dispatch(view, _key(keys.KeyKind.CHAR, "d"))
    assert outcome is state.Outcome.CONTINUE
    assert (tmp_path / user_defaults.DEFAULTS_FILENAME).exists()
    assert view.message is not None
    assert view.message.kind is state.MessageKind.SUCCESS


def test_initial_state_seeds_user_defaults_on_empty_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    tuned = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path), lr=7e-4)
    user_defaults.save_defaults(tuned)

    # An empty target seeds from the saved defaults at the live era.
    empty_dir = tmp_path / "empty"
    view = controller.build_initial_state(
        config.TrainConfig(device="cpu", checkpoint_dir=str(empty_dir)),
        cuda_available=False,
    )
    assert view.seeded_from_user_defaults and not view.seeded_from_saved
    assert view.working.lr == 7e-4
    assert view.working.encoding_version == version.MODEL_VERSION

    # A directory holding a readable run still wins over the defaults file.
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, config.TrainConfig(device="cpu", lr=5e-4))
    seeded_view = controller.build_initial_state(
        config.TrainConfig(device="cpu", checkpoint_dir=str(run_dir)),
        cuda_available=False,
    )
    assert seeded_view.seeded_from_saved
    assert not seeded_view.seeded_from_user_defaults
    assert seeded_view.working.lr == 5e-4


def test_initial_state_warns_on_unreadable_defaults(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / user_defaults.DEFAULTS_FILENAME).write_text("nope", encoding="utf-8")
    view = controller.build_initial_state(
        config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path / "empty")),
        cuda_available=False,
    )
    assert not view.seeded_from_user_defaults
    assert view.message is not None
    assert view.message.kind is state.MessageKind.WARN


def test_reset_prompt_offers_user_and_factory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    # Without a defaults file the chooser offers only factory / cancel.
    view = _empty_state()
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "r"))
    assert view.confirm is not None
    assert [option.key for option in view.confirm.options] == ["f", "c"]
    controller.dispatch(view, _key(keys.KeyKind.ESCAPE))

    # Save tuned defaults, drift the working config, then reset to each.
    view.working = view.working.model_copy(update={"lr": 7e-4})
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "d"))  # save as defaults
    view.working = view.working.model_copy(update={"lr": 9e-4})
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "r"))
    assert view.confirm is not None
    assert [option.key for option in view.confirm.options] == ["u", "f", "c"]
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "u"))
    assert view.working.lr == 7e-4
    assert view.seeded_from_user_defaults

    controller.dispatch(view, _key(keys.KeyKind.CHAR, "r"))
    controller.dispatch(view, _key(keys.KeyKind.CHAR, "f"))
    assert view.working.lr == config.TrainConfig().lr
    assert not view.seeded_from_user_defaults


# --------------------------------------------------------------------------- #
# review-confirmed regressions                                                #
# --------------------------------------------------------------------------- #


def test_inspect_run_invalid_config_is_unreadable(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    payload = {
        "config": {**cfg.model_dump(), "lr": 0.0},  # lr>0 — now out of bounds
        "progress": runstate.RunProgress(iteration=3, total_games=12).model_dump(),
    }
    torch.save(payload, tmp_path / artifacts.LAST_CKPT)
    summary = runs.inspect_run(str(tmp_path))
    assert summary.exists and summary.config_invalid
    assert summary.train_config is None
    # Start must route through the fresh-run prompt, never offer resume.
    assert runs.resolve_status(summary, cfg) is runs.RunStatus.UNREADABLE


def test_inspect_run_missing_config_is_unreadable(tmp_path: pathlib.Path):
    """A checkpoint with no embedded config at all is not self-describing and
    must never be offered for resume (the post-cutoff refusal contract)."""
    cfg = config.TrainConfig(device="cpu")
    payload = {
        "progress": runstate.RunProgress(iteration=3, total_games=12).model_dump(),
    }
    torch.save(payload, tmp_path / artifacts.LAST_CKPT)
    summary = runs.inspect_run(str(tmp_path))
    assert summary.exists and summary.config_invalid
    assert summary.train_config is None
    assert runs.resolve_status(summary, cfg) is runs.RunStatus.UNREADABLE


def test_loop_starts_fresh_on_missing_saved_config(tmp_path: pathlib.Path):
    """The resume gate refuses a config-less checkpoint (starts fresh with an
    alarm) rather than assuming compatibility."""
    cfg = config.TrainConfig(
        device="cpu",
        trunk_layers=(32, 32),
        choice_layers=(32, 32),
        checkpoint_dir=str(tmp_path),
    )
    payload = {
        "progress": runstate.RunProgress(iteration=4, total_games=8).model_dump(),
    }
    torch.save(payload, tmp_path / artifacts.LAST_CKPT)
    training = loop.TrainingLoop(cfg)  # must not raise on the config-less payload
    assert training.state.total_games == 0  # started fresh rather than resuming


def test_loop_starts_fresh_on_invalid_saved_config(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(
        device="cpu",
        trunk_layers=(32, 32),
        choice_layers=(32, 32),
        checkpoint_dir=str(tmp_path),
    )
    payload = {
        "config": {**cfg.model_dump(), "eval_ewma_alpha": 0.0},  # now out of bounds
        "progress": runstate.RunProgress(iteration=4, total_games=8).model_dump(),
    }
    torch.save(payload, tmp_path / artifacts.LAST_CKPT)
    training = loop.TrainingLoop(cfg)  # must not raise on the invalid saved config
    assert training.state.total_games == 0  # started fresh rather than resuming


def test_loop_truncates_history_on_fresh_start(tmp_path: pathlib.Path):
    (tmp_path / artifacts.METRICS_LOG).write_text("stale-row\n", encoding="utf-8")
    (tmp_path / artifacts.GAMES_LOG).write_text("stale-game\n", encoding="utf-8")
    (tmp_path / "process_20000101-000000.json").write_text("{}", encoding="utf-8")
    cfg = config.TrainConfig(
        device="cpu",
        trunk_layers=(32, 32),
        choice_layers=(32, 32),
        checkpoint_dir=str(tmp_path),
        resume=False,
    )
    loop.TrainingLoop(cfg)  # a non-resumed run clears a previous run's history
    assert (tmp_path / artifacts.METRICS_LOG).read_text(encoding="utf-8") == ""
    assert (tmp_path / artifacts.GAMES_LOG).read_text(encoding="utf-8") == ""
    # The prior run's dated session record is dropped; only this startup's
    # freshly-written one remains.
    assert not (tmp_path / "process_20000101-000000.json").exists()
    assert len(list(tmp_path.glob(artifacts.PROCESS_GLOB))) == 1


def test_edit_caret_blinks_across_frames():
    view = _empty_state()
    view.selected_attr = "lr"
    view.mode = state.Mode.EDIT
    view.edit_buffer = "0.001"
    assert "▏" in _render(view, frame=0)  # caret on
    assert "▏" not in _render(view, frame=_BLINK_OFF_FRAME)  # caret off


def test_modal_keeps_options_on_short_terminal():
    view = _empty_state()
    view.mode = state.Mode.CONFIRM
    view.confirm = state.ConfirmPrompt(
        title="START A NEW RUN",
        lines=[f"explanatory line {i}" for i in range(6)],
        options=[
            state.ConfirmOption(
                key="a",
                label="archive & start",
                action=state.ConfirmAction.ARCHIVE_THEN_FRESH,
            ),
            state.ConfirmOption(
                key="o",
                label="overwrite & start",
                action=state.ConfirmAction.OVERWRITE_THEN_FRESH,
                danger=True,
            ),
            state.ConfirmOption(
                key="c", label="cancel", action=state.ConfirmAction.CANCEL
            ),
        ],
        default_key="a",
    )
    out = _render(view, width=100, height=18)  # a deliberately short terminal
    assert "[A]" in out and "[O]" in out and "[C]" in out
