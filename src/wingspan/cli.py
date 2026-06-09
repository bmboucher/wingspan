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

import torch
import yaml

from wingspan import engine, players
from wingspan.agents import display
from wingspan.instrumentation import config as instrumentation_config
from wingspan.instrumentation import dispatcher


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
    try:
        spec_a = players.parse_player_spec(args.p0, checkpoint_dir)
        spec_b = players.parse_player_spec(args.p1, checkpoint_dir)
        agent_a, config_a = players.build_agent(spec_a, device, rng, args.greedy)
        agent_b, config_b = players.build_agent(spec_b, device, rng, args.greedy)
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

    instrumentation = _open_instrumentation(args, seed)
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
                _write_log(log_path, eng.state.log)
                if not args.quiet:
                    print(f"  log -> {log_path}")
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


def _open_instrumentation(
    args: argparse.Namespace, seed: int
) -> dispatcher.Instrumentation:
    """Build and open the event-callback router from ``--instrument`` — the
    standalone instrumentation config (same shape as ``TrainConfig.instrumentation``).
    Returns the no-op ``EMPTY`` router when the flag is absent. The caller must
    ``close`` whatever this returns when the run ends."""
    if args.instrument is None:
        return dispatcher.EMPTY
    text = pathlib.Path(args.instrument).read_text(encoding="utf-8")
    cfg = instrumentation_config.InstrumentationConfig.model_validate(
        yaml.safe_load(text)
    )
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


#### Log file output ####


def _write_log(path: str, lines: list[str]) -> None:
    """Write the game log line-by-line to ``path`` (UTF-8, newline-terminated)."""
    with open(path, "w", encoding="utf-8") as log_file:
        for line in lines:
            log_file.write(display.strip_ansi(line) + "\n")
