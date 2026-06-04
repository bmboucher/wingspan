"""The tournament application: pick competitors, play the round-robin live, and
write the JSON report.

``main`` resolves the competitors (the interactive :mod:`picker`, or ``--ai``
flags with ``--no-picker``), then runs the round-robin. By default the games run
on a background thread while the main thread repaints the :mod:`dashboard` on a
``rich.live.Live`` screen (mirroring the training app); ``--quiet`` skips the
live UI for scripted runs. Either way the final :class:`results.TournamentReport`
is written to ``--out`` and a plain-text standings recap is printed.

``q`` / ``Ctrl+C`` requests a graceful stop: in-flight games finish, the pending
ones are cancelled, and the report covers exactly the games that completed.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
import typing

import pydantic
import torch
from rich import box, console, live, table

from wingspan.tournament import config, dashboard, participants, picker, results, runner
from wingspan.tournament import state as state_module
from wingspan.training import runstate
from wingspan.training.configure import keys

_REFRESH_HZ = 8.0
_STOP_GRACE_SECONDS = 30.0


class _Outcome(pydantic.BaseModel):
    """The worker thread's result hand-off to the main thread."""

    report: results.TournamentReport | None = None
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``wingspan-tournament`` / ``wingspan.cli tournament``."""
    args = _parse_args(argv)
    term = console.Console()
    specs = _resolve_competitors(args, term)
    if specs is None:
        return 0  # picker cancelled
    if len(specs) < 2:
        term.print("[red]A tournament needs at least 2 competitors.[/red]")
        return 1

    cfg = config.TournamentConfig(
        participants=specs,
        games_per_pair=args.games_per_pair,
        elo_k=args.elo_k,
        elo_init=args.elo_init,
        base_seed=args.seed,
        out_path=args.out,
        device=_resolve_device(args.device),
    )
    unloadable = _unloadable_competitors(cfg)
    if unloadable:
        term.print(
            "[red]Could not load these competitors "
            "(incompatible or corrupt checkpoint):[/red]"
        )
        for name, reason in unloadable:
            term.print(f"  - {name}: {reason}")
        return 1
    report = _run_quiet(cfg, term) if args.quiet else _run_live(cfg, term)
    if report is None:
        return 1
    _write_report(report, term)
    _print_summary(report, term)
    return 0


###### PRIVATE #######


def _resolve_competitors(
    args: argparse.Namespace, term: console.Console
) -> list[participants.ParticipantSpec] | None:
    """The competitor specs, from the picker or from ``--ai`` flags."""
    if args.no_picker:
        ai_dirs = typing.cast("list[str]", args.ai or [])
        specs = [participants.spec_from_dir(directory) for directory in ai_dirs]
        if args.include_random:
            specs.append(participants.random_spec())
        return participants.with_unique_ids(specs)
    selected = picker.run_picker(
        args.base_dir,
        term,
        include_random_option=True,
        games_per_pair=args.games_per_pair,
    )
    if selected is None:
        return None
    return participants.with_unique_ids(selected)


def _unloadable_competitors(cfg: config.TournamentConfig) -> list[tuple[str, str]]:
    """Preflight every model competitor's checkpoint, returning (name, reason)
    for any that fail to load — e.g. a checkpoint from an older incompatible
    network — so the tournament aborts with a clear message instead of a worker
    traceback mid-run."""
    device = torch.device("cpu")
    failures: list[tuple[str, str]] = []
    for spec in cfg.participants:
        if spec.kind is not participants.ParticipantKind.MODEL:
            continue
        try:
            participants.load_player(spec, device, random.Random(0))
        except Exception as error:  # noqa: BLE001 — any load failure is reported
            reason = " ".join(str(error).split()) or type(error).__name__
            failures.append((spec.display_name, reason[:160]))
    return failures


def _run_live(
    cfg: config.TournamentConfig, term: console.Console
) -> results.TournamentReport | None:
    """Play the tournament on a worker thread while the main thread renders."""
    live_state = state_module.new_tournament_state(cfg)
    lock = threading.RLock()
    stop_flag = threading.Event()
    outcome = _Outcome()

    def on_result(result: results.GameResult) -> None:
        with lock:
            live_state.record_game(result)
            live_state.push_event(runstate.EventKind.INFO, _event_text(result))

    def work() -> None:
        try:
            report = runner.run_tournament(
                cfg, on_result=on_result, should_stop=stop_flag.is_set
            )
        except (
            Exception
        ) as error:  # noqa: BLE001 — surfaced on the dashboard, not raised
            with lock:
                live_state.error = str(error)
                live_state.finish(state_module.TournamentPhase.ERROR)
            outcome.error = str(error)
            return
        outcome.report = report
        with lock:
            live_state.finish(
                state_module.TournamentPhase.STOPPED
                if stop_flag.is_set()
                else state_module.TournamentPhase.DONE
            )

    worker = threading.Thread(target=work, name="wingspan-tournament", daemon=True)
    worker.start()
    _drive_dashboard(term, live_state, lock, worker, stop_flag)
    worker.join(timeout=_STOP_GRACE_SECONDS)
    if outcome.error is not None:
        term.print(f"[red]Tournament failed:[/red] {outcome.error}")
    return outcome.report


def _drive_dashboard(
    term: console.Console,
    live_state: state_module.TournamentState,
    lock: threading.RLock,
    worker: threading.Thread,
    stop_flag: threading.Event,
) -> None:
    """Repaint the Live display until the worker finishes; ``q`` / Ctrl+C stops."""
    root = dashboard.build_layout()
    interval = 1.0 / _REFRESH_HZ
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
                    with lock:
                        dashboard.render(root, live_state)
                        terminal = live_state.phase.is_terminal
                        display.refresh()
                    if terminal and not worker.is_alive():
                        break
                    event = reader.poll()
                    if event is not None and (
                        event.kind is keys.KeyKind.INTERRUPT or event.char in ("q", "Q")
                    ):
                        stop_flag.set()
                    time.sleep(interval)
                except KeyboardInterrupt:
                    stop_flag.set()


def _run_quiet(
    cfg: config.TournamentConfig, term: console.Console
) -> results.TournamentReport | None:
    """Play the tournament with no live UI, printing occasional progress."""
    term.print(
        f"Running tournament: {len(cfg.participants)} competitors, "
        f"{cfg.games_per_pair} games/pair, {cfg.total_games} total games…"
    )
    step = max(1, cfg.total_games // 20)
    done = [0]

    def on_result(_: results.GameResult) -> None:
        done[0] += 1
        if done[0] % step == 0 or done[0] == cfg.total_games:
            term.print(f"  {done[0]}/{cfg.total_games} games")

    return runner.run_tournament(cfg, on_result=on_result)


def _write_report(report: results.TournamentReport, term: console.Console) -> None:
    """Write the report JSON to its configured output path."""
    path = report.config.out_path
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report.model_dump_json(indent=2))
    term.print(f"\nWrote results to [bold]{path}[/bold]")


def _print_summary(report: results.TournamentReport, term: console.Console) -> None:
    """A plain-text final standings table on the restored terminal."""
    term.rule("[bold]WINGSPAN // TOURNAMENT — final standings[/bold]")
    grid = table.Table(box=box.SIMPLE_HEAD)
    grid.add_column("#", justify="right")
    grid.add_column("competitor")
    grid.add_column("elo", justify="right")
    grid.add_column("W-L-T", justify="right")
    grid.add_column("win%", justify="right")
    grid.add_column("margin", justify="right")
    for rank, participant in enumerate(report.participants, start=1):
        grid.add_row(
            str(rank),
            participant.display_name,
            f"{participant.final_elo:.0f}",
            f"{participant.wins}-{participant.losses}-{participant.ties}",
            f"{participant.win_rate * 100:.1f}%",
            f"{participant.avg_margin:+.1f}",
        )
    term.print(grid)


def _event_text(result: results.GameResult) -> str:
    """A one-line recent-events description of a finished game."""
    if result.a_score == result.b_score:
        return f"{result.player_a_id} ⇄ {result.player_b_id} tie ({result.a_score})"
    if result.a_score > result.b_score:
        winner, loser, margin = (
            result.player_a_id,
            result.player_b_id,
            result.a_score - result.b_score,
        )
    else:
        winner, loser, margin = (
            result.player_b_id,
            result.player_a_id,
            result.b_score - result.a_score,
        )
    return f"{winner} beat {loser} by {margin}"


def _resolve_device(device: str) -> str:
    """Fall back to CPU when a CUDA device is requested but unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wingspan tournament",
        description="Run a round-robin tournament between trained Wingspan AIs.",
    )
    parser.add_argument(
        "--ai",
        action="append",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="a model competitor's checkpoint dir (repeatable; used with --no-picker)",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="add the random agent as a competitor (with --no-picker)",
    )
    parser.add_argument(
        "--no-picker",
        action="store_true",
        help="skip the interactive picker and use --ai / --include-random",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=config.DEFAULT_GAMES_PER_PAIR,
        help="games each pair plays (must be even; played as mirrored deals)",
    )
    parser.add_argument("--elo-k", type=float, default=config.DEFAULT_ELO_K)
    parser.add_argument("--elo-init", type=float, default=config.DEFAULT_ELO_INIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="tournament_report.json")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--base-dir",
        default="checkpoints",
        help="base checkpoint dir the picker scans for runs",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no live UI; print periodic progress"
    )
    return parser.parse_args(argv)
