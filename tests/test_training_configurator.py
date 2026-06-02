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
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")
pytest.importorskip("rich")

import rich.console as rich_console
import torch

from wingspan import architecture, encode, model
from wingspan.training import artifacts, config, loop, runstate
from wingspan.training.configure import (
    arch_diagram,
    controller,
    fields,
    keys,
    runs,
    screen,
    state,
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
        "iteration": iteration,
        "total_games": total_games,
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
    assert runs.architecture_compatible(None, cfg)  # pre-descriptor -> compatible
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


# Tall enough that the whole flow (down to the VALUE box + TOTAL) fits without
# the viewport clipping the bottom — the per-size test below covers clipping.
_FULL_DIAGRAM_HEIGHT = 60


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


def _param_report_for(cfg: config.TrainConfig) -> architecture.ParamReport:
    return architecture.count_parameters(
        cfg.arch,
        trunk_in=encode.trunk_input_dim(cfg.state_dim, cfg.card_embed_dim),
        choice_in=encode.choice_input_dim(cfg.choice_dim, cfg.card_embed_dim),
        embed_rows=encode.HAND_MULTIHOT_DIM + 1,
        num_families=len(cfg.family_order),
    )


@pytest.mark.parametrize("width,height", [(128, 44), (128, 18), (80, 44), (80, 18)])
def test_arch_diagram_renders_all_sizes(width: int, height: int):
    out = _render(_arch_state(), width=width, height=height)
    assert "ARCHITECTURE" in out and "TRUNK" in out  # top of the flow is always shown


def test_arch_diagram_all_blocks_present():
    out = _render(_arch_state(), height=_FULL_DIAGRAM_HEIGHT)
    for block in ("EMBED", "TRUNK", "CHOICE", "CONCAT", "SCORER", "VALUE"):
        assert block in out


def test_arch_diagram_dropout_appears_and_hides():
    assert "Dropout" not in _render(_arch_state())  # default dropout 0 -> no row
    assert "Dropout" in _render(_arch_state("dropout", dropout=0.15))


def test_arch_diagram_layernorm_appears():
    assert "LayerNorm" not in _render(_arch_state())  # default off -> no row
    assert "LayerNorm" in _render(_arch_state("layernorm", layernorm=True))


def test_arch_diagram_activation_label():
    assert "relu" in _render(_arch_state())
    assert "gelu" in _render(
        _arch_state("activation", activation=architecture.ActivationName.GELU)
    )


def test_arch_diagram_readout_never_layernorms():
    # Even with LayerNorm enabled and a hidden scorer layer, the readout heads
    # must not draw a LayerNorm row (mirrors model._build_readout). LayerNorm
    # shows up in the body blocks (above SCORER) but never from SCORER onward.
    view = _arch_state(layernorm=True, head_layers=(128,))
    out = _render(view, width=128, height=_FULL_DIAGRAM_HEIGHT)
    assert "LayerNorm" in out  # the trunk / choice bodies do carry it
    readouts = out.split("SCORER", 1)[1]
    assert "LayerNorm" not in readouts


def test_arch_diagram_collapse_tag():
    view = _arch_state(
        trunk_layers=(128, 128, 128, 128), choice_layers=(128, 128, 128, 128)
    )
    assert "×4" in _render(view)  # four identical trunk layers fold to one ×4 group


def test_arch_diagram_narrow_fallback():
    # Below the box-width floor the renderable drops to the compact text list;
    # it must still name the blocks and not crash.
    out = _render_diagram(_arch_state(), width=16, height=20)
    assert "TRUNK" in out


def test_arch_diagram_focus_highlight_smoke():
    out = _render(_arch_state("dropout", dropout=0.15))
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


def test_arch_diagram_param_display():
    out = _render(_arch_state(), width=128, height=_FULL_DIAGRAM_HEIGHT)
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
