"""Entry points: manual CLI play, random self-play with log, and a training entry."""
from __future__ import annotations

import argparse
import logging
import random
import sys

from .agents import cli_agent, random_agent
from .cards import power_coverage, load_all
from .game import Engine, make_engine


def _print_board(state) -> None:
    for p in state.players:
        print(f"\n=== {p.name} ===  food={dict(p.food)}  hand={len(p.hand)}  cubes={p.action_cubes_left}")
        for h in (b for b in p.board.keys()):
            row = p.board[h]
            cells = [
                f"{pb.bird.name}({pb.eggs}e/{pb.cached_food}c/{pb.tucked_cards}t)"
                for pb in row
            ]
            print(f"  {h.value:10s}: " + (", ".join(cells) if cells else "(empty)"))


def main_manual(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play a Wingspan game manually against a random opponent.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--you", type=int, default=0, choices=[0, 1], help="Which player you control (default: 0).")
    parser.add_argument("--both-human", action="store_true", help="Two human players (hotseat).")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    print(f"Seed: {seed}")
    eng, birds, _, _ = make_engine(seed=seed)
    impl, total = power_coverage(birds)
    print(f"Bird power coverage: {impl}/{total} ({impl*100//total}%)")

    rng = random.Random(seed)
    if args.both_human:
        a = cli_agent(); b = cli_agent()
    else:
        a = cli_agent() if args.you == 0 else random_agent(rng)
        b = cli_agent() if args.you == 1 else random_agent(rng)

    eng.play_one_game((a, b))
    print("\n=== GAME LOG (tail) ===")
    for line in eng.state.log[-20:]:
        print(line)
    print()
    for p in eng.state.players:
        print(f"{p.name}: final_score={getattr(p, 'final_score', '?')}")
    return 0


def main_random(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a random-vs-random Wingspan game.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", type=str, default=None, help="Path to write detailed game log.")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    rng = random.Random(seed)
    for g in range(args.games):
        eng, birds, _, _ = make_engine(seed=seed + g)
        a = random_agent(rng); b = random_agent(rng)
        eng.play_one_game((a, b))
        scores = [getattr(p, "final_score", None) for p in eng.state.players]
        if not args.quiet:
            print(f"Game {g+1}: scores={scores}, log lines={len(eng.state.log)}")
        if args.log:
            path = args.log if args.games == 1 else f"{args.log}.{g}"
            with open(path, "w", encoding="utf-8") as f:
                for line in eng.state.log:
                    f.write(line + "\n")
            if not args.quiet:
                print(f"  log -> {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        sys.exit(main_manual(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "random":
        sys.exit(main_random(sys.argv[2:]))
    else:
        sys.exit(main_random(sys.argv[1:]))
