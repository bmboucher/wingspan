"""``wingspan research`` — the CLI over the offline research studies.

Usage::

    wingspan research setup-keep [--setups N] [--out FILE] [--checkpoint-dir DIR]
                                 [--seed S] [--temperature T] [--workers W]
                                 [--device cpu] [--num-players N]

One sub-command per study. ``setup-keep`` writes the setup keep-rate experience
table (:mod:`wingspan.research.setup_keep`): one CSV row per dealt bird of every
sampled setup, flagged 0/1 by whether the run's setup model kept it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

from wingspan.research import constants, models, setup_keep
from wingspan.training import runmeta

_SETUP_KEEP = "setup-keep"


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``wingspan research``."""
    args = _parse_args(argv)
    if args.study == _SETUP_KEEP:
        return _run_setup_keep(args)
    raise AssertionError(f"unreachable: argparse admitted study {args.study!r}")


###### PRIVATE #######


def _run_setup_keep(args: argparse.Namespace) -> int:
    """Resolve the spec (the seat count defaults to the run's trained count),
    run the study, and print a one-line summary."""
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    out_path = pathlib.Path(args.out)
    try:
        num_players = (
            args.num_players
            if args.num_players is not None
            else _trained_num_players(checkpoint_dir)
        )
        spec = models.SetupKeepStudySpec(
            setups=args.setups,
            seed=args.seed,
            num_players=num_players,
            temperature=args.temperature,
            workers=args.workers,
        )
        print(
            f"setup-keep: {spec.setups} setups over {spec.games} "
            f"{spec.num_players}-seat games from {checkpoint_dir} "
            f"(workers={spec.workers}) -> {out_path}"
        )
        summary = setup_keep.run_setup_keep_study(
            spec, checkpoint_dir, torch.device(args.device), out_path
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"wingspan research: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {summary.rows} rows ({summary.setups} setups) to "
        f"{summary.out_path} in {summary.elapsed_seconds:.1f}s"
    )
    return 0


def _trained_num_players(checkpoint_dir: pathlib.Path) -> int:
    """The seat count the run trained at — its setup model only ever saw deals
    from tables of that size."""
    return runmeta.read_run_config(str(checkpoint_dir)).config.architecture.num_players


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wingspan research",
        description="Offline research studies over a trained checkpoint "
        "(docs/RESEARCH.md).",
    )
    studies = parser.add_subparsers(dest="study", required=True, metavar="<study>")
    keep = studies.add_parser(
        _SETUP_KEEP,
        help="Setup keep-rate experience table: one CSV row per dealt bird of "
        "every sampled setup, with a 0/1 flag for whether the setup model kept it.",
    )
    keep.add_argument(
        "--setups",
        type=int,
        default=constants.DEFAULT_SETUPS,
        help=f"Setups (seats) to sample (default: {constants.DEFAULT_SETUPS}).",
    )
    keep.add_argument(
        "--out",
        default=constants.DEFAULT_SETUP_KEEP_OUT,
        help=f"CSV path to write (default: {constants.DEFAULT_SETUP_KEEP_OUT}).",
    )
    keep.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=constants.DEFAULT_CHECKPOINT_DIR,
        help="Run directory whose setup.pt scores the keeps "
        f"(default: {constants.DEFAULT_CHECKPOINT_DIR}, the active run).",
    )
    keep.add_argument(
        "--seed",
        type=int,
        default=constants.DEFAULT_SEED,
        help="Seed of the first game; games use consecutive seeds "
        f"(default: {constants.DEFAULT_SEED}).",
    )
    keep.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sample the keep from the softmax at this temperature instead of "
        "taking the argmax (default: argmax).",
    )
    keep.add_argument(
        "--workers",
        type=int,
        default=constants.DEFAULT_WORKERS,
        help="Worker processes; 1 runs in-process "
        f"(default: {constants.DEFAULT_WORKERS}).",
    )
    keep.add_argument(
        "--device", default="cpu", help="Torch device for scoring (default: cpu)."
    )
    keep.add_argument(
        "--num-players",
        dest="num_players",
        type=int,
        default=None,
        help="Seats per dealt game (default: the run's trained seat count).",
    )
    return parser.parse_args(argv)
