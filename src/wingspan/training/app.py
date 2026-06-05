"""The dashboard application: parse args, run training on a worker thread, and
repaint the live ``top``-style display on the main thread.

``main`` wires a :class:`loop.TrainingLoop` to the :mod:`dashboard` through a
``rich.live.Live`` running on the alternate screen buffer — a fixed full-screen
window that repaints in place and never scrolls, restoring the terminal on exit.
Training runs on a background thread and mutates the shared
:class:`runstate.RunState`; the main thread only reads it (under the loop's
lock) to render, so the display never blocks on training work and the two
wall-clocks tick smoothly every frame.

``Ctrl+C`` requests a graceful stop: the loop finishes the current game, writes
a final checkpoint, and the dashboard shows the shutdown before the screen is
restored and a plain-text summary is printed.

When the training loop reaches its ``target_iterations`` milestone it pauses and
displays an acknowledgment overlay in the events panel; the main loop handles
``[C]``ontinue and ``[E]``nd keypresses, optionally setting a new target before
unblocking the worker thread. On ``[E]``nd the run's checkpoints are archived
and the interactive FLIGHT PLAN configurator is reopened.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import threading
import time

import torch
from rich import console, live

from wingspan import architecture
from wingspan.training import artifacts, config, configure, dashboard, loop, runstate
from wingspan.training.configure import keys
from wingspan.training.configure import runs as config_runs

_REFRESH_HZ = 8.0
_STOP_GRACE_SECONDS = 30.0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``wingspan dashboard`` / ``python -m wingspan.training``.

    The FLIGHT PLAN configurator always opens first — tune any hyperparameters,
    then start or resume a run, which transitions into the live training display.
    Quitting the configurator without launching exits the process cleanly.

    When the training loop ends with the user choosing ``[E]nd run`` at a target
    milestone, the run is archived and the configurator is reopened so the user
    can adjust settings and start another run without leaving the application.
    """
    args = _parse_args(argv)
    cfg = _config_from_namespace(args)
    term = console.Console()
    show_config = True
    while True:
        if show_config:
            result = configure.run_configurator(cfg, term, torch.cuda.is_available())
            if result is None:
                return 0  # user quit the configurator without launching a run
            cfg = result
        return_to_config = _run_training(cfg, term)
        if not return_to_config:
            return 0
        show_config = True  # always show configurator on re-entry after "end run"


###### PRIVATE #######


def _run_training(cfg: config.TrainConfig, term: console.Console) -> bool:
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
            f"\n  Archiving run to {cfg.checkpoint_dir}/{artifacts.ARCHIVE_SUBDIR}/…"
        )
        config_runs.archive_run(cfg.checkpoint_dir, cfg.run_name)
        return True
    return False


def _resolve_device(cfg: config.TrainConfig) -> config.TrainConfig:
    """Downgrade a ``cuda`` request to ``cpu`` when CUDA is unavailable, so a
    configurator- or flag-chosen ``cuda`` on a CPU-only host still runs instead
    of crashing the loop at model construction."""
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        return cfg.model_copy(update={"device": "cpu"})
    return cfg


def _configure_file_logging(cfg: config.TrainConfig) -> None:
    """Route all logging to ``{checkpoint_dir}/{run_name}.log`` so the engine's
    soft warnings (e.g. the 504-wide setup decision) never bleed onto the
    alternate-screen dashboard. Installing a root handler also suppresses the
    default last-resort stderr handler."""
    log_dir = pathlib.Path(cfg.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"{cfg.run_name}.log", encoding="utf-8")
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
        f"(birds {avg.birds:.1f}, eggs {avg.eggs:.1f}, food {avg.food:.1f}, "
        f"tucked {avg.tucked:.1f}, rounds {avg.rounds:.1f}, bonus {avg.bonus:.1f})\n"
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
        f"  checkpoints      : {state.config.checkpoint_dir}/  ({artifact_list})"
    )
    if state.error:
        term.rule("[bold red]error[/bold red]")
        term.print(state.error)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        prog="wingspan dashboard",
        description="Run and live-monitor Wingspan self-play training (TRAINING.md Phase 1).",
    )
    parser.add_argument("--device", default=default_device, help="cpu or cuda")
    parser.add_argument("--games-per-iter", type=int, default=256)
    parser.add_argument(
        "--iterations", type=int, default=0, help="max iterations (0 = until Ctrl+C)"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="run an eval block every N training iterations (0 disables eval)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=128,
        help="held-out games per eval block (played as mirrored pairs)",
    )
    parser.add_argument(
        "--trunk-layers",
        default="128,128",
        help="state-trunk hidden widths, comma-separated (e.g. 256,128)",
    )
    parser.add_argument(
        "--choice-layers",
        default="128,128",
        help="per-choice encoder widths (independent of the trunk; ends at N)",
    )
    parser.add_argument(
        "--head-layers",
        default="128",
        help="per-family scorer hidden widths (empty string = direct (M+N)->1)",
    )
    parser.add_argument(
        "--value-layers",
        default="",
        help="value-head hidden widths (empty string = direct M->1)",
    )
    parser.add_argument(
        "--activation",
        default=architecture.ActivationName.RELU.value,
        choices=[name.value for name in architecture.ActivationName],
        help="activation function for every MLP block",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--layernorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply LayerNorm in the trunk / choice-encoder body blocks",
    )
    parser.add_argument("--card-embed-dim", type=int, default=64, dest="card_embed_dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--run-name", default="dashboard")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from last.pt in --checkpoint-dir if present (--no-resume starts fresh)",
    )
    return parser.parse_args(argv)


def _config_from_namespace(args: argparse.Namespace) -> config.TrainConfig:
    """Build the run config from parsed flags. The ``cuda``->``cpu`` fallback is
    deferred to :func:`_resolve_device` so both this and the configurator path
    funnel device safety through one place."""
    return config.TrainConfig(
        games_per_iter=args.games_per_iter,
        max_iterations=args.iterations,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        trunk_layers=_parse_layers(args.trunk_layers),
        choice_layers=_parse_layers(args.choice_layers),
        head_layers=_parse_layers(args.head_layers),
        value_layers=_parse_layers(args.value_layers),
        activation=architecture.ActivationName(args.activation),
        dropout=args.dropout,
        layernorm=args.layernorm,
        card_embed_dim=args.card_embed_dim,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        resume=args.resume,
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
