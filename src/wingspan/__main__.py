"""Unified ``wingspan`` CLI dispatcher.

Usage: wingspan <command> [args …]

Commands:
  play        Play a game manually against a random opponent.
  random      Run one or more random-vs-random games.
  selfplay    Run games with configurable per-seat agent matchups.
  dashboard   FLIGHT PLAN: configure a run, then train and monitor it live.
  tournament  Round-robin tournament between trained AIs.
  cloud       Run a training job headless with S3 persistence.
  monitor     FLOCK WATCH: watch all cloud runs from their S3 status snapshots.
  inspect     Print the model introspection report.
"""

from __future__ import annotations

import sys

# One-line description per verb — printed by the top-level help.
_VERBS: dict[str, str] = {
    "play": "Play a game manually against a random opponent.",
    "random": "Run one or more random-vs-random games.",
    "selfplay": "Run games with configurable per-seat agent matchups.",
    "dashboard": "FLIGHT PLAN: configure a run, then train and monitor it live.",
    "tournament": "Round-robin tournament between trained AIs.",
    "cloud": "Run a training job headless with S3 persistence.",
    "monitor": "FLOCK WATCH: watch all cloud runs from their S3 status snapshots.",
    "inspect": "Print the model introspection report.",
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the unified ``wingspan`` command."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0
    verb, rest = args[0], args[1:]
    match verb:
        case "play":
            import wingspan.cli as cli

            return cli.main_manual(rest)
        case "random":
            import wingspan.cli as cli

            return cli.main_random(rest)
        case "selfplay":
            import wingspan.selfplay as selfplay_mod

            return selfplay_mod.main_selfplay(rest)
        case "dashboard":
            import wingspan.training.app as training_app

            return training_app.main(rest)
        case "tournament":
            import wingspan.tournament.app as tournament_app

            return tournament_app.main(rest)
        case "cloud":
            import wingspan.cloud.__main__ as cloud_entry

            return cloud_entry.main(rest)
        case "monitor":
            import wingspan.cloud.monitor as cloud_monitor

            return cloud_monitor.main(rest)
        case "inspect":
            import wingspan.introspect as introspect

            return introspect.main_inspect(rest)
        case _:
            print(
                f"wingspan: unknown command '{verb}'  (try `wingspan --help`)",
                file=sys.stderr,
            )
            return 1


###### PRIVATE #######


def _print_help() -> None:
    """Print the top-level usage table."""
    width = max(len(verb) for verb in _VERBS)
    lines = [
        "usage: wingspan <command> [args …]",
        "",
        "Commands:",
        *(f"  {verb:<{width}}  {desc}" for verb, desc in _VERBS.items()),
        "",
        "Run `wingspan <command> --help` for per-command usage.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
