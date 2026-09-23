"""``wingspan summary`` — write the training-progression HTML report on demand.

Thin CLI wrapper around :func:`wingspan.reporting.training_summary
.write_training_summary`, the same writer the training loop calls
automatically at the target milestone (``training.loop_target
.handle_target_reached``). Useful for regenerating the report for an
in-progress or archived run without waiting for (or re-triggering) a target
milestone.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from wingspan.reporting import training_summary


def main_summary(argv: list[str] | None = None) -> int:
    """CLI entry point: write ``training_summary.html`` for a run directory."""
    parser = argparse.ArgumentParser(
        prog="wingspan summary",
        description="Training-progression HTML report for a run directory.",
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="checkpoints",
        help="Run directory containing metrics.jsonl (default: checkpoints).",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Output path (default: <run_dir>/training_summary.html).",
    )
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"wingspan summary: {run_dir} is not a directory", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.out) if args.out is not None else None
    written = training_summary.write_training_summary(str(run_dir), out_path)
    _print(f"HTML report written → {written}")
    return 0


def _print(message: str) -> None:
    """Print ``message``, tolerating a non-UTF-8 stdout encoding (cp1252 on a
    default Windows console can't encode the success message's arrow)."""
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(message.encode("utf-8") + b"\n")


if __name__ == "__main__":
    raise SystemExit(main_summary())
