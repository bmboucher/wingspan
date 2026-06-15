"""The unified ``wingspan play`` entry point: any seats, 1..N games, optional logs.

Each seat is set with ``--p0`` / ``--p1`` using the shared player-spec grammar
(``human`` / ``random`` / ``last`` / ``best`` / ``opponent`` / a ``.pt`` path /
a run directory — see ``wingspan.players``), so interactive play, quick
random-vs-random games, and trained-AI matchups all run through one command.
The default matchup is ``last`` vs ``last``: the most recent trained model
playing itself.

When a seat is AI-driven, every genuine decision is annotated in the game log
with the policy's ranked probability distribution (see ``players.factory``),
and the opening-bonus regime is auto-derived from each checkpoint's stored
``TrainConfig`` so games mirror how the nets were trained.

The unified ``wingspan`` dispatcher lives in ``__main__.py``.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import typing

import torch
import yaml

from wingspan import engine, players, state
from wingspan.agents import display
from wingspan.instrumentation import config as instrumentation_config
from wingspan.instrumentation import dispatcher
from wingspan.instrumentation import events as instrumentation_events
from wingspan.players import decision_probe

if typing.TYPE_CHECKING:
    from wingspan.training import config as train_config


def main_play(argv: list[str] | None = None) -> int:
    """Run one or more games between any mix of seats, optionally writing logs.

    Returns a process exit code: 0 on success, 1 if a seat spec cannot be
    resolved (missing checkpoint, encoding-incompatible network, or a regime
    mismatch between two checkpoints)."""
    args = _build_parser().parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    rng = random.Random(seed)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    device = torch.device(args.device)

    # Resolve both seats up front so a bad checkpoint (or a regime mismatch
    # between two checkpoints) fails before any game runs, with a clean message
    # rather than a mid-game traceback.
    probe_0 = decision_probe.DecisionProbe()
    probe_1 = decision_probe.DecisionProbe()
    try:
        spec_a = players.parse_player_spec(args.p0, checkpoint_dir)
        spec_b = players.parse_player_spec(args.p1, checkpoint_dir)
        agent_a, config_a = players.build_agent(
            spec_a, device, rng, args.greedy, value_probe=probe_0
        )
        agent_b, config_b = players.build_agent(
            spec_b, device, rng, args.greedy, value_probe=probe_1
        )
        split_setup_bonus = players.resolve_split_setup_bonus((config_a, config_b))
        split_setup_food = players.resolve_split_setup_food((config_a, config_b))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading agent: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        regime_parts: list[str] = []
        if split_setup_bonus:
            regime_parts.append("bonus: split (CHOOSE_BONUS)")
        if split_setup_food:
            regime_parts.append("food: split (GAIN/SPEND_FOOD)")
        regime = "  |  opening " + ", ".join(regime_parts) if regime_parts else ""
        print(f"Seed: {seed}  |  P0: {args.p0}  vs  P1: {args.p1}{regime}")

    instrumentation = _open_instrumentation(
        args, seed, (config_a, config_b), (probe_0, probe_1)
    )
    try:
        for game_idx in range(args.games):
            eng, _, _, _ = engine.Engine.create(seed=seed + game_idx)
            engine.Engine.play_one_game(
                eng.state,
                (agent_a, agent_b),
                instrumentation=instrumentation,
                split_setup_bonus=split_setup_bonus,
                split_setup_food=split_setup_food,
            )
            scores = [player.final_score for player in eng.state.players]
            if not args.quiet:
                print(
                    f"Game {game_idx + 1}: scores={scores}, "
                    f"log lines={len(eng.state.log)}"
                )
            if args.log:
                log_path = args.log if args.games == 1 else f"{args.log}.{game_idx}"
                if args.collate:
                    _write_log(log_path, eng.state.log)
                    if not args.quiet:
                        print(f"  log -> {log_path}")
                else:
                    _write_split_logs(log_path, eng.state.log_entries)
                    if not args.quiet:
                        print(f"  log -> {log_path}_p0.log, {log_path}_p1.log")
            if args.html and not args.quiet:
                # The HTML file itself is written by the instrumentation
                # handler on game end; report where it landed.
                print(f"  html -> {_html_game_path(args.html, game_idx, args.games)}")
    finally:
        instrumentation.close()
    return 0


###### PRIVATE #######


#### Argument parsing ####


def _build_parser() -> argparse.ArgumentParser:
    """The ``play`` argument parser. ``--p0`` / ``--p1`` each take a player
    spec: ``human``, ``random``, a named checkpoint (``last`` / ``best`` /
    ``opponent``), a path to a ``.pt`` file, or a run directory."""
    parser = argparse.ArgumentParser(
        prog="wingspan play",
        description="Play Wingspan games between any mix of human, random, "
        "and trained-AI seats.",
    )
    spec_help = (
        "Player %s: 'human', 'random', a named checkpoint "
        "('last'/'best'/'opponent'), a path to a .pt file, or a run directory "
        "(default: last)."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument(
        "--log", type=str, default=None, help="Path to write detailed game log(s)."
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Path to write a navigable HTML game-log viewer (one phase/turn at "
        "a time, P0/P1/both toggle, 3x5 board grids). For a --games series the "
        "game index is inserted before the extension (out.html -> out.0.html).",
    )
    parser.add_argument(
        "--collate",
        action="store_true",
        help="Write a single interleaved log file instead of per-player files "
        "(FILE_p0.log / FILE_p1.log). Use when a single unified view is preferred.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--p0", type=str, default="last", help=spec_help % "0")
    parser.add_argument("--p1", type=str, default="last", help=spec_help % "1")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        dest="checkpoint_dir",
        help="Directory to resolve named checkpoint specs against.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device for AI inference."
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="AI seats pick the argmax option instead of sampling "
        "(ignored for human/random seats).",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Path to an instrumentation config (YAML/JSON): event handlers to "
        "attach to every game.",
    )
    parser.add_argument(
        "--instrument-out",
        type=str,
        default=None,
        dest="instrument_out",
        help="Directory the instrumentation handlers write their output under "
        "(default: current directory).",
    )
    return parser


#### Instrumentation ####


# The handler name and the events the HTML game-log recorder subscribes to.
# Phase-boundary events fire once per ``=== ... ===`` header in the log so the
# handler's per-phase snapshots align one-to-one with the log segments.
# MADE_DECISION is included so the handler can record per-decision timeline data.
_HTML_HANDLER_NAME = "__game_log_html__"
_HTML_HANDLER_EVENTS = (
    instrumentation_events.EventName.GAME_START,
    instrumentation_events.EventName.SETUP_START,
    instrumentation_events.EventName.ROUND_START,
    instrumentation_events.EventName.TURN_START,
    instrumentation_events.EventName.MADE_DECISION,
    instrumentation_events.EventName.GAME_END,
)


def _open_instrumentation(
    args: argparse.Namespace,
    seed: int,
    seat_configs: (
        tuple[train_config.TrainConfig | None, train_config.TrainConfig | None] | None
    ) = None,
    probes: (
        tuple[decision_probe.DecisionProbe | None, decision_probe.DecisionProbe | None]
        | None
    ) = None,
) -> dispatcher.Instrumentation:
    """Build and open the event-callback router for this run.

    Combines the standalone ``--instrument`` config (same shape as
    ``TrainConfig.instrumentation``) with the built-in HTML game-log recorder
    attached by ``--html``. Returns the no-op ``EMPTY`` router when neither flag
    is given. The caller must ``close`` whatever this returns when the run ends.

    When ``seat_configs`` and ``probes`` are both supplied (the normal ``play``
    path), they are injected into the HTML handler via
    :meth:`~wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler.configure_timeline`
    so the timeline chart can render value/target lines."""
    if args.instrument is None and args.html is None:
        return dispatcher.EMPTY

    handlers, events_map = _instrument_file_spec(args)
    if args.html is not None:
        handlers[_HTML_HANDLER_NAME] = {
            "class": "GameLogHtml",
            "output_path": args.html,
            "index_suffix": args.games > 1,
        }
        for event in _HTML_HANDLER_EVENTS:
            events_map.setdefault(event, []).append(_HTML_HANDLER_NAME)

    cfg = instrumentation_config.InstrumentationConfig.model_validate(
        {"handlers": handlers, "events": events_map}
    )

    # Inject timeline probes into the HTML handler before building the router.
    if args.html is not None and seat_configs is not None and probes is not None:
        from wingspan.instrumentation.handlers import (
            game_log_html as html_handler_module,
        )

        html_handler = cfg.handlers[_HTML_HANDLER_NAME]
        assert isinstance(html_handler, html_handler_module.GameLogHtmlHandler)
        html_handler.configure_timeline(seat_configs, probes)

    out_dir = (
        pathlib.Path(args.instrument_out)
        if args.instrument_out is not None
        else pathlib.Path(".")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumentation = cfg.build()
    instrumentation.open(
        instrumentation_config.RunContext(
            output_dir=out_dir,
            run_name="play",
            seed=seed,
            matchup=(str(args.p0), str(args.p1)),
        )
    )
    return instrumentation


def _instrument_file_spec(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[instrumentation_events.EventName, list[str]]]:
    """The handler-instance map and event assignment from ``--instrument``.

    Returns already-resolved handler instances (which the config validator
    passes through untouched) plus a mutable copy of the event assignment, so
    the ``--html`` recorder can be merged on top. Empty when ``--instrument`` is
    absent."""
    if args.instrument is None:
        return {}, {}
    text = pathlib.Path(args.instrument).read_text(encoding="utf-8")
    cfg = instrumentation_config.InstrumentationConfig.model_validate(
        yaml.safe_load(text)
    )
    handlers: dict[str, object] = dict(cfg.handlers)
    events_map = {event: list(names) for event, names in cfg.events.items()}
    return handlers, events_map


#### Log file output ####


def _html_game_path(base: str, game_idx: int, games: int) -> str:
    """The HTML file path for one game, matching the handler's naming: ``base``
    for a single game, else the game index inserted before the extension
    (``out.html`` -> ``out.0.html``)."""
    if games == 1:
        return base
    path = pathlib.Path(base)
    suffix = path.suffix or ".html"
    return str(path.with_name(f"{path.stem}.{game_idx}{suffix}"))


def _write_log(path: str, lines: list[str]) -> None:
    """Write the game log line-by-line to ``path`` (UTF-8, newline-terminated)."""
    with open(path, "w", encoding="utf-8") as log_file:
        for line in lines:
            log_file.write(display.strip_ansi(line) + "\n")


def _write_split_logs(base_path: str, entries: list[state.LogEntry]) -> None:
    """Write per-player log files from structured log entries.

    Produces ``<base_path>_p0.log`` and ``<base_path>_p1.log``.  Each file
    contains all entries attributed to that player (``player_id == N``) plus
    all global entries (``player_id is None``), preserving the original
    interleaved order.  This gives each player a coherent perspective on the
    game without the other player's private decision annotations."""
    for player_idx in (0, 1):
        player_path = f"{base_path}_p{player_idx}.log"
        with open(player_path, "w", encoding="utf-8") as log_file:
            for entry in entries:
                if entry.player_id is None or entry.player_id == player_idx:
                    log_file.write(display.strip_ansi(entry.text) + "\n")
