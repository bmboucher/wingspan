"""Entry points: manual CLI play, random self-play with log, and a training entry."""

from __future__ import annotations

import argparse
import random
import sys

from wingspan import agents, cards, engine


def main_manual(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Play a Wingspan game manually against a random opponent."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--you",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which player you control (default: 0).",
    )
    parser.add_argument(
        "--both-human", action="store_true", help="Two human players (hotseat)."
    )
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    print(f"Seed: {seed}")
    eng, birds, _, _ = engine.Engine.create(seed=seed)
    impl, total = cards.power_coverage(birds)
    print(f"Bird power coverage: {impl}/{total} ({impl*100//total}%)")

    rng = random.Random(seed)
    if args.both_human:
        a = agents.cli_agent()
        b = agents.cli_agent()
    else:
        a = agents.cli_agent() if args.you == 0 else agents.random_agent(rng)
        b = agents.cli_agent() if args.you == 1 else agents.random_agent(rng)

    engine.Engine.play_one_game(eng.state, (a, b))
    print("\n=== GAME LOG (tail) ===")
    for line in eng.state.log[-20:]:
        print(line)
    print()
    for p in eng.state.players:
        score = p.final_score if p.final_score is not None else "?"
        print(f"{p.name}: final_score={score}")
    return 0


def main_random(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a random-vs-random Wingspan game."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--log", type=str, default=None, help="Path to write detailed game log."
    )
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    rng = random.Random(seed)
    for g in range(args.games):
        eng, _, _, _ = engine.Engine.create(seed=seed + g)
        a = agents.random_agent(rng)
        b = agents.random_agent(rng)
        engine.Engine.play_one_game(eng.state, (a, b))
        scores = [p.final_score for p in eng.state.players]
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
