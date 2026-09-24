"""The dashboard application: parse args, run training on a worker thread, and
repaint the live ``top``-style display on the main thread.

``main`` wires a :class:`loop.TrainingLoop` to the :mod:`dashboard` through a
``rich.live.Live`` running on the alternate screen buffer — a fixed full-screen
window that repaints in place and never scrolls, restoring the terminal on exit.
Training runs on a background thread and mutates the shared
:class:`runstate.RunState`; the main thread only reads it (under the loop's
lock) to render, so the display never blocks on training work and the two
wall-clocks tick smoothly every frame.

``Ctrl+C`` requests a fast stop: the worker pool is killed immediately, the
current iteration's partial data is discarded, and the last completed iteration's
checkpoint is preserved. The dashboard shows the shutdown before the screen is
restored and a plain-text summary is printed.

When the training loop reaches its ``target_iterations`` milestone it pauses and
displays an acknowledgment overlay in the events panel; the main loop handles
``[C]``ontinue and ``[E]``nd keypresses, optionally setting a new target before
unblocking the worker thread. On ``[E]``nd the run's checkpoints are archived
and the interactive FLIGHT PLAN configurator is reopened.

``--config FILE`` supplies the whole run config from a file (a
``run_config_<stamp>.json`` artifact, a ``configurator_defaults.json`` envelope,
a cloud run-file's ``train:`` block, or a bare ``RunConfig`` dump — see
:mod:`wingspan.training.config_file`) instead of argparse defaults. Without
``--start`` the config screen still opens, seeded from the file for review;
with ``--start`` the screen is skipped and the run launches immediately via
:func:`wingspan.training.configure.controller.prepare_headless_launch`, which
never archives or overwrites an existing run on its own — an incompatible or
resume-disabled directory is refused instead. Only the five run-identity flags
(``--checkpoint-dir``, ``--run-name``, ``--collect-device``, ``--train-device``,
``--resume``/``--no-resume``) may be combined with ``--config``; any other flag
is a usage error, since the file already speaks for every other field.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import threading
import time
import typing

import torch
from rich import console, live

from wingspan import architecture
from wingspan.training import (
    artifacts,
    config,
    config_file,
    configure,
    dashboard,
    loop,
    runstate,
)
from wingspan.training.configure import controller, keys
from wingspan.training.configure import runs as config_runs

_REFRESH_HZ = 8.0
_STOP_GRACE_SECONDS = 30.0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``wingspan dashboard`` / ``python -m wingspan.training``.

    Without ``--config``, the FLIGHT PLAN configurator always opens first —
    tune any hyperparameters, then start or resume a run, which transitions
    into the live training display. Quitting the configurator without
    launching exits the process cleanly.

    With ``--config FILE``, the file supplies the whole run config (see the
    module docstring); ``--start`` skips the screen and launches immediately,
    refusing rather than archiving when the target directory is not safely
    launchable. Only the five run-identity flags may be combined with
    ``--config`` — any other explicit flag is a usage error (exit 2).

    When the training loop ends with the user choosing ``[E]nd run`` at a target
    milestone, the run is archived and the configurator is reopened so the user
    can adjust settings and start another run without leaving the application.
    """
    args = _parse_args(argv)
    term = console.Console()

    if args.config is None:
        if args.start:
            print("error: --start requires --config", file=sys.stderr)
            return 2
        return _configure_and_train(_config_from_namespace(args), term, seed_file=None)

    explicit = _explicit_dests(argv)
    conflicting_flag = _first_disallowed_flag(explicit)
    if conflicting_flag is not None:
        print(
            f"error: --config cannot be combined with {conflicting_flag}; edit "
            "the file or use the config screen",
            file=sys.stderr,
        )
        return 2

    try:
        cfg = config_file.load_run_config(args.config)
    except config_file.ConfigFileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    cfg = _apply_identity_overrides(cfg, args, explicit)

    if not args.start:
        return _configure_and_train(cfg, term, seed_file=args.config.name)

    try:
        cfg = controller.prepare_headless_launch(cfg)
    except controller.LaunchRefused as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _run_training(cfg, term)
    return 0


###### PRIVATE #######


def _configure_and_train(
    cfg: config.RunConfig, term: console.Console, seed_file: str | None
) -> int:
    """The FLIGHT PLAN loop: open the configurator (seeded from ``seed_file``
    only on this first entry — a ``--config FILE`` display name, or ``None``
    for the normal saved-run / user-defaults precedence), launch into
    training, and reopen the configurator after an ``[E]nd run`` archive.
    Quitting the configurator without launching exits the process cleanly."""
    show_config = True
    while True:
        if show_config:
            result = configure.run_configurator(
                cfg, term, torch.cuda.is_available(), seed_file=seed_file
            )
            if result is None:
                return 0  # user quit the configurator without launching a run
            cfg = result
        return_to_config = _run_training(cfg, term)
        if not return_to_config:
            return 0
        show_config = True  # always show configurator on re-entry after "end run"
        seed_file = None  # re-entry after "end run" uses the normal precedence


def _run_training(cfg: config.RunConfig, term: console.Console) -> bool:
    """Train + live-monitor one run.

    Returns ``True`` iff the user chose ``[E]nd run`` at a target milestone and
    the caller should re-open the configurator; ``False`` on a normal exit.
    """
    cfg = _resolve_device(cfg)
    _configure_file_logging(cfg)
    training = loop.TrainingLoop(cfg)
    worker = threading.Thread(target=training.run, name="wingspan-trainer", daemon=True)
    worker.start()

    _run_dashboard(term, training, worker)

    worker.join(timeout=_STOP_GRACE_SECONDS)
    _print_summary(term, training.state)

    # If the user chose "end run", archive the checkpoints and signal the caller
    # to re-open the configurator.
    if training.state.user_target_choice == "end":
        term.print(
            f"\n  Archiving run to {cfg.run.checkpoint_dir}/{artifacts.ARCHIVE_SUBDIR}/…"
        )
        config_runs.archive_run(cfg.run.checkpoint_dir, cfg.run.run_name)
        return True
    return False


def _resolve_device(cfg: config.RunConfig) -> config.RunConfig:
    """Downgrade a ``cuda`` request to ``cpu`` when CUDA is unavailable, so a
    configurator- or flag-chosen ``cuda`` on a CPU-only host still runs instead
    of crashing the loop at model construction."""
    return config.resolve_devices(cfg, torch.cuda.is_available())


def _configure_file_logging(cfg: config.RunConfig) -> None:
    """Route all logging to ``{checkpoint_dir}/{run_name}.log`` so the engine's
    soft warnings (e.g. the 504-wide setup decision) never bleed onto the
    alternate-screen dashboard. Installing a root handler also suppresses the
    default last-resort stderr handler."""
    log_dir = pathlib.Path(cfg.run.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"{cfg.run.run_name}.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _run_dashboard(
    term: console.Console,
    training: loop.TrainingLoop,
    worker: threading.Thread,
) -> None:
    """Drive the Live display until the worker finishes (or a second Ctrl+C).

    During ``PAUSED_AT_TARGET`` the KeyReader receives ``[C]``ontinue and
    ``[E]``nd keypresses and forwards them to the training loop via
    :meth:`loop.TrainingLoop.signal_target_response`.
    """
    root = dashboard.build_layout()
    interval = 1.0 / _REFRESH_HZ
    frame = 0
    stop_requested = False
    pause_buffer = ""

    with live.Live(
        root,
        console=term,
        screen=True,
        auto_refresh=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as display:
        with keys.KeyReader() as reader:
            while True:
                try:
                    # Render AND refresh under the lock: the chart/histogram
                    # renderables read live state (history, family counts) lazily at
                    # refresh time, so the worker must not mutate mid-frame.
                    with training.lock:
                        dashboard.render(root, training.state, frame, pause_buffer)
                        current_phase = training.state.phase
                        terminal = current_phase.is_terminal
                        display.refresh()
                    frame += 1
                    if terminal and not worker.is_alive():
                        break
                    event = reader.poll()
                    if event is not None:
                        if current_phase is runstate.Phase.PAUSED_AT_TARGET:
                            pause_buffer = _handle_pause_key(
                                training, event, pause_buffer
                            )
                    time.sleep(interval)
                except KeyboardInterrupt:
                    if stop_requested:
                        break  # second Ctrl+C — drop out immediately
                    training.request_stop()
                    stop_requested = True


def _handle_pause_key(
    training: loop.TrainingLoop,
    event: keys.KeyEvent,
    pause_buffer: str,
) -> str:
    """Route a keypress received while the dashboard is PAUSED_AT_TARGET.

    Returns the new pause buffer (possibly unchanged). Signals the training loop
    via :meth:`loop.TrainingLoop.signal_target_response` on ``[C]``, ``[E]``,
    or ``ENTER``; digit keys accumulate in the buffer for a new-target number.
    """
    if event.kind is keys.KeyKind.BACKSPACE:
        return pause_buffer[:-1]
    if event.kind is keys.KeyKind.ENTER:
        new_target = int(pause_buffer) if pause_buffer.isdigit() else 0
        training.signal_target_response("continue", new_target)
        return ""
    if event.char in ("e", "E"):
        training.signal_target_response("end", 0)
        return ""
    if event.char in ("c", "C"):
        # Immediate continue with no new target (clears the milestone).
        training.signal_target_response("continue", 0)
        return ""
    if event.char and event.char.isdigit():
        return pause_buffer + event.char
    return pause_buffer


def _print_summary(term: console.Console, state: runstate.RunState) -> None:
    """Plain-text recap printed to the restored terminal after the Live exits."""
    avg = state.avg_breakdown()
    term.rule("[bold]WINGSPAN // FLIGHT PLAN — run summary[/bold]")
    term.print(
        f"  phase            : {state.phase.value}\n"
        f"  iterations       : {state.iteration + 1 if state.last_iter else 0}\n"
        f"  total games      : {state.total_games:,}\n"
        f"  total decisions  : {state.total_decisions:,}\n"
        f"  elapsed          : {_summary_clock(state.elapsed())}\n"
        f"  avg score        : {avg.total:.1f} pts/game "
        f"(birds {avg.birds:.1f}, eggs {avg.eggs:.1f}, cached {avg.cached:.1f}, "
        f"tucked {avg.tucked:.1f}, goals {avg.goals:.1f}, bonus {avg.bonus:.1f})\n"
        f"  avg game length  : {state.avg_decisions():.0f} decisions"
    )
    if state.best_win_rate is not None:
        opponent = (
            "random"
            if state.opponent_generation == 0
            else f"self·gen{state.opponent_generation}"
        )
        term.print(
            f"  best win rate    : {state.best_win_rate * 100:.1f}% vs {opponent}"
        )
    artifact_list = (
        f"{artifacts.LAST_CKPT}, {artifacts.BEST_CKPT}, {artifacts.METRICS_LOG}, "
        f"{artifacts.GAMES_LOG}, {artifacts.MODEL_CONFIG_JSON}, {artifacts.PROCESS_GLOB}"
    )
    if state.opponent_generation > 0:
        artifact_list += f", {artifacts.OPPONENT_CKPT}"
    term.print(
        f"  checkpoints      : {state.config.run.checkpoint_dir}/  ({artifact_list})"
    )
    if state.error:
        term.rule("[bold red]error[/bold red]")
        term.print(state.error)


# The five run-identity flags that may still be combined with --config,
# overriding the file's corresponding field; every other flag is a usage error
# alongside --config, since the file already speaks for it.
_IDENTITY_OVERRIDE_DESTS = frozenset(
    {"checkpoint_dir", "run_name", "collect_device", "train_device", "resume"}
)
# Dests that name the --config machinery itself, never a conflict with it.
_CONFIG_MACHINERY_DESTS = frozenset({"config", "start"})


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser, _ = _build_parser(live_defaults=True)
    return parser.parse_args(argv)


def _explicit_dests(argv: list[str] | None) -> frozenset[str]:
    """The argparse dests the user actually typed on the command line, as
    opposed to argparse's own defaults.

    Parses ``argv`` against a twin parser whose every flag defaults to
    ``argparse.SUPPRESS`` (built by :func:`_build_parser`), so a flag the user
    did not pass is simply absent from the resulting namespace — this needs no
    access to argparse's private ``_actions`` / ``seen_actions`` bookkeeping.
    """
    parser, _ = _build_parser(live_defaults=False)
    namespace = parser.parse_args(argv)
    return frozenset(vars(namespace))


def _first_disallowed_flag(explicit: frozenset[str]) -> str | None:
    """The first flag (in registration order) explicitly passed alongside
    ``--config`` that is neither a run-identity override nor ``--config`` /
    ``--start`` themselves, or ``None`` when the combination is allowed."""
    _, dest_to_flag = _build_parser(live_defaults=True)
    for dest, flag in dest_to_flag.items():
        if dest in _CONFIG_MACHINERY_DESTS or dest in _IDENTITY_OVERRIDE_DESTS:
            continue
        if dest in explicit:
            return flag
    return None


def _apply_identity_overrides(
    cfg: config.RunConfig, args: argparse.Namespace, explicit: frozenset[str]
) -> config.RunConfig:
    """Override ``cfg``'s run-identity fields with whichever of the five
    identity flags the user explicitly passed alongside ``--config``; every
    other field comes from the file untouched."""
    run_updates: dict[str, object] = {
        field: getattr(args, field)
        for field in ("checkpoint_dir", "run_name", "resume")
        if field in explicit
    }
    misc_updates: dict[str, object] = {
        field: getattr(args, field)
        for field in ("collect_device", "train_device")
        if field in explicit
    }
    if not run_updates and not misc_updates:
        return cfg
    updated = cfg
    if run_updates:
        updated = updated.model_copy(
            update={"run": updated.run.model_copy(update=run_updates)}
        )
    if misc_updates:
        updated = updated.model_copy(
            update={"misc": updated.misc.model_copy(update=misc_updates)}
        )
    return updated


def _build_parser(
    *, live_defaults: bool
) -> tuple[argparse.ArgumentParser, dict[str, str]]:
    """Construct the ``wingspan dashboard`` argument parser.

    With ``live_defaults`` every flag gets its real default — the parser
    ``wingspan dashboard`` actually runs with, so a bare invocation behaves
    exactly as before this module gained ``--config``. Otherwise every flag
    defaults to ``argparse.SUPPRESS``, so parsing ``argv`` against the result
    (see :func:`_explicit_dests`) yields a namespace containing only the dests
    the user actually typed. Returns the parser and the dest→flag-string
    mapping (used to name the offending flag in a ``--config`` conflict error).
    """
    parser = argparse.ArgumentParser(
        prog="wingspan dashboard",
        description="Run and live-monitor Wingspan self-play training (TRAINING.md Phase 1).",
    )
    default_train_device = "cuda" if torch.cuda.is_available() else "cpu"
    dest_to_flag: dict[str, str] = {}

    def add(
        flag: str, *, default: object, dest: str | None = None, **kwargs: typing.Any
    ) -> None:
        resolved_dest = dest if dest is not None else flag.lstrip("-").replace("-", "_")
        dest_to_flag[resolved_dest] = flag
        parser.add_argument(
            flag,
            dest=dest,
            default=default if live_defaults else argparse.SUPPRESS,
            **kwargs,
        )

    add(
        "--collect-device",
        default="cpu",
        help="where self-play collection runs: cpu (worker pool, fastest) or cuda "
        "(in-process batched collector; requires --train-device cuda)",
    )
    add(
        "--train-device",
        default=default_train_device,
        help="where the learner (net, optimizer, update step) runs: cpu or cuda",
    )
    add("--games-per-iter", type=int, default=256)
    add("--iterations", type=int, default=0, help="max iterations (0 = until Ctrl+C)")
    add("--lr", type=float, default=3e-4)
    add("--entropy-coef", type=float, default=0.01)
    add("--value-coef", type=float, default=0.5)
    add(
        "--eval-every",
        type=int,
        default=5,
        help="run an eval block every N training iterations (0 disables eval)",
    )
    add(
        "--eval-games",
        type=int,
        default=128,
        help="held-out games per eval block (played as mirrored pairs)",
    )
    add(
        "--trunk-layers",
        default="128,128",
        help="state-trunk hidden widths, comma-separated (e.g. 256,128)",
    )
    add(
        "--choice-layers",
        default="128,128",
        help="per-choice encoder widths (independent of the trunk; ends at N)",
    )
    add(
        "--head-layers",
        default="128",
        help="per-family scorer hidden widths (empty string = direct (M+N)->1)",
    )
    add(
        "--value-layers",
        default="",
        help="value-head hidden widths (empty string = direct M->1)",
    )
    add(
        "--activation",
        dest="between_activation",
        default=architecture.ActivationName.RELU.value,
        choices=[name.value for name in architecture.ActivationName],
        help="between-layers activation function for every MLP block",
    )
    add(
        "--final-activation",
        dest="final_activation",
        default=architecture.ActivationName.NONE.value,
        choices=[name.value for name in architecture.ActivationName],
        help="final-layer activation for every MLP block (default: none = no final act)",
    )
    add("--dropout", type=float, default=0.0)
    add(
        "--layernorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply LayerNorm in the trunk / choice-encoder body blocks",
    )
    add("--card-embed-dim", dest="card_embed_dim", type=int, default=64)
    add("--seed", type=int, default=0)
    add("--checkpoint-dir", default="checkpoints")
    add("--run-name", default="dashboard")
    add(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from last.pt in --checkpoint-dir if present (--no-resume starts fresh)",
    )
    add(
        "--config",
        type=pathlib.Path,
        default=None,
        help="load the full run config from FILE (JSON, or YAML for .yaml/.yml) "
        "instead of the flags above; see wingspan.training.config_file",
    )
    add(
        "--start",
        action="store_true",
        default=False,
        help="with --config, skip the config screen and launch immediately",
    )
    return parser, dest_to_flag


def _config_from_namespace(args: argparse.Namespace) -> config.RunConfig:
    """Build the run config from parsed flags. The ``cuda``->``cpu`` fallback is
    deferred to :func:`_resolve_device` so both this and the configurator path
    funnel device safety through one place."""
    return config.RunConfig(
        run=config.RunSettings(
            games_per_iter=args.games_per_iter,
            max_iterations=args.iterations,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            resume=args.resume,
        ),
        training=config.TrainingConfig(
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=_parse_layers(args.trunk_layers),
                choice_layers=_parse_layers(args.choice_layers),
                head_layers=_parse_layers(args.head_layers),
                value_layers=_parse_layers(args.value_layers),
                between_activation=architecture.ActivationName(args.between_activation),
                final_activation=architecture.ActivationName(args.final_activation),
                dropout=args.dropout,
                layernorm=args.layernorm,
                card_embed_dim=args.card_embed_dim,
            ),
        ),
        misc=config.MiscConfig(
            seed=args.seed,
            collect_device=args.collect_device,
            train_device=args.train_device,
        ),
    )


def _parse_layers(layer_text: str) -> tuple[int, ...]:
    """Parse a comma-separated layer-width flag into a tuple (empty string → the
    empty tuple, for a head with no hidden layers)."""
    return tuple(int(part) for part in layer_text.replace(" ", "").split(",") if part)


def _summary_clock(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s"


if __name__ == "__main__":
    raise SystemExit(main())
