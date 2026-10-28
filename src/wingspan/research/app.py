"""``wingspan research`` — the CLI over the offline research studies.

Usage::

    wingspan research setup-keep [--setups N] [--out FILE] [--checkpoint-dir DIR]
                                 [--seed S] [--temperature T] [--workers W]
                                 [--device cpu] [--num-players N]
    wingspan research game-stats [--games N] --out DIR [--checkpoint-dir DIR]
                                 [--seed S] [--workers W] [--device cpu]
                                 [--aggregate-only games_raw.csv] [--blog-snippets]
    wingspan research card-stats [--games N] --out DIR
                                 [--checkpoint-dir DIR | --release]
                                 [--seed S] [--workers W] [--device cpu]
                                 [--aggregate-only card_games_raw.csv]

One sub-command per study. ``setup-keep`` writes the setup keep-rate experience
table (:mod:`wingspan.research.setup_keep`): one CSV row per dealt bird of every
sampled setup, flagged 0/1 by whether the run's setup model kept it.
``game-stats`` plays greedy self-play games and writes ``games_raw.csv`` plus
chart-ready aggregates and an HTML preview into ``--out``
(:mod:`wingspan.research.game_stats`, :mod:`wingspan.research.game_stats_agg`).
``card-stats`` plays the same games and writes ``card_games_raw.csv`` plus
per-card ``bird_stats.csv`` / ``bonus_stats.csv`` "games in hand" win rates
(:mod:`wingspan.research.card_stats`, :mod:`wingspan.research.card_stats_agg`);
``--release`` plays the bundled release model.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

from wingspan import release
from wingspan.research import (
    card_stats,
    card_stats_agg,
    constants,
    game_stats,
    game_stats_agg,
    models,
    setup_keep,
)
from wingspan.training import runmeta

_SETUP_KEEP = "setup-keep"
_GAME_STATS = "game-stats"
_CARD_STATS = "card-stats"


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``wingspan research``."""
    args = _parse_args(argv)
    if args.study == _SETUP_KEEP:
        return _run_setup_keep(args)
    if args.study == _GAME_STATS:
        return _run_game_stats(args)
    if args.study == _CARD_STATS:
        return _run_card_stats(args)
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


def _run_game_stats(args: argparse.Namespace) -> int:
    """Play the study (unless ``--aggregate-only``), then aggregate the raw table
    into ``--out`` and print where everything landed."""
    out_dir = pathlib.Path(args.out)
    try:
        if args.aggregate_only is not None:
            raw_path = pathlib.Path(args.aggregate_only)
        else:
            raw_path = _play_game_stats(args, out_dir)
        written = game_stats_agg.aggregate(
            raw_path, out_dir, with_blog_snippets=args.blog_snippets
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"wingspan research: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}")
    return 0


def _play_game_stats(args: argparse.Namespace, out_dir: pathlib.Path) -> pathlib.Path:
    """Run the self-play half of ``game-stats`` and return the raw CSV path."""
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / constants.GAMES_RAW_FILENAME
    spec = models.GameStatsSpec(games=args.games, seed=args.seed, workers=args.workers)
    print(
        f"game-stats: {spec.games} greedy self-play games from {checkpoint_dir} "
        f"(workers={spec.workers}) -> {raw_path}"
    )
    summary = game_stats.run_game_stats_study(
        spec,
        checkpoint_dir,
        torch.device(args.device),
        raw_path,
        progress_stream=sys.stderr,
    )
    print(
        f"played {summary.games} games ({summary.rows} seat rows) in "
        f"{summary.elapsed_seconds:.1f}s "
        f"({summary.games / summary.elapsed_seconds:.2f} games/s)"
    )
    return raw_path


def _run_card_stats(args: argparse.Namespace) -> int:
    """Play the study (unless ``--aggregate-only``), then aggregate the raw table
    into ``--out`` and print where everything landed."""
    out_dir = pathlib.Path(args.out)
    try:
        if args.aggregate_only is not None:
            raw_path = pathlib.Path(args.aggregate_only)
        else:
            raw_path = _play_card_stats(args, out_dir)
        written = card_stats_agg.aggregate(raw_path, out_dir)
        base = card_stats_agg.baseline(card_stats_agg.read_raw_cards(raw_path))
    except (FileNotFoundError, ValueError) as error:
        print(f"wingspan research: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}")
    print(
        f"baseline win rate {base.wins}/{base.seat_games} seat-games "
        f"({base.ties} in tied games, counted as non-wins)"
    )
    return 0


def _play_card_stats(args: argparse.Namespace, out_dir: pathlib.Path) -> pathlib.Path:
    """Run the self-play half of ``card-stats`` and return the raw CSV path."""
    checkpoint_dir = _card_stats_run_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / constants.CARD_GAMES_RAW_FILENAME
    spec = models.GameStatsSpec(games=args.games, seed=args.seed, workers=args.workers)
    print(
        f"card-stats: {spec.games} greedy self-play games from {checkpoint_dir} "
        f"(workers={spec.workers}) -> {raw_path}"
    )
    summary = card_stats.run_card_stats_study(
        spec,
        checkpoint_dir,
        torch.device(args.device),
        raw_path,
        progress_stream=sys.stderr,
    )
    print(
        f"played {summary.games} games ({summary.rows} seat rows) in "
        f"{summary.elapsed_seconds:.1f}s "
        f"({summary.games / summary.elapsed_seconds:.2f} games/s)"
    )
    return raw_path


def _card_stats_run_dir(args: argparse.Namespace) -> pathlib.Path:
    """The run directory to play: the bundled release model for ``--release``
    (``FileNotFoundError`` when the package was built without one), otherwise
    ``--checkpoint-dir``."""
    if not args.release:
        return pathlib.Path(args.checkpoint_dir)
    if not release.has_release():
        raise FileNotFoundError(
            "no bundled release model in this install; use --checkpoint-dir"
        )
    return release.release_dir()


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
    stats = studies.add_parser(
        _GAME_STATS,
        help="Model performance stats: greedy self-play games with per-turn "
        "scores and actions, aggregated into chart-ready CSVs and an HTML preview.",
    )
    stats.add_argument(
        "--games",
        type=int,
        default=constants.DEFAULT_GAMES,
        help=f"Games to play (default: {constants.DEFAULT_GAMES}).",
    )
    stats.add_argument(
        "--out",
        required=True,
        help="Output directory (games_raw.csv, aggregates, preview.html).",
    )
    stats.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=constants.DEFAULT_CHECKPOINT_DIR,
        help="Run directory whose last.pt and setup.pt play "
        f"(default: {constants.DEFAULT_CHECKPOINT_DIR}).",
    )
    stats.add_argument(
        "--seed",
        type=int,
        default=constants.DEFAULT_SEED,
        help="Seed of the first game; games use consecutive seeds "
        f"(default: {constants.DEFAULT_SEED}).",
    )
    stats.add_argument(
        "--workers",
        type=int,
        default=constants.DEFAULT_WORKERS,
        help="Worker processes; 1 runs in-process "
        f"(default: {constants.DEFAULT_WORKERS}).",
    )
    stats.add_argument(
        "--device", default="cpu", help="Torch device for inference (default: cpu)."
    )
    stats.add_argument(
        "--aggregate-only",
        dest="aggregate_only",
        default=None,
        metavar="GAMES_RAW_CSV",
        help="Skip play: aggregate this games_raw.csv into --out.",
    )
    stats.add_argument(
        "--blog-snippets",
        dest="blog_snippets",
        action="store_true",
        help="Also write blog_snippets/ into --out: one ready-to-paste Plotly "
        "embed per figure (scattergram subsampled) plus the CDN script tag.",
    )
    card = studies.add_parser(
        _CARD_STATS,
        help="General card stats: greedy self-play games aggregated into "
        "per-card games-in-hand win rates and opener keep rates for birds "
        "and bonus cards.",
    )
    _add_card_stats_arguments(card)
    return parser.parse_args(argv)


def _add_card_stats_arguments(card: argparse.ArgumentParser) -> None:
    """Register the ``card-stats`` sub-command's arguments."""
    card.add_argument(
        "--games",
        type=int,
        default=constants.DEFAULT_GAMES,
        help=f"Games to play (default: {constants.DEFAULT_GAMES}).",
    )
    card.add_argument(
        "--out",
        required=True,
        help="Output directory (card_games_raw.csv, bird_stats.csv, "
        "bonus_stats.csv and their .md renderings).",
    )
    run_source = card.add_mutually_exclusive_group()
    run_source.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=constants.DEFAULT_CHECKPOINT_DIR,
        help="Run directory whose last.pt and setup.pt play "
        f"(default: {constants.DEFAULT_CHECKPOINT_DIR}).",
    )
    run_source.add_argument(
        "--release",
        action="store_true",
        help="Play the bundled release model instead of a checkpoint directory.",
    )
    card.add_argument(
        "--seed",
        type=int,
        default=constants.DEFAULT_SEED,
        help="Seed of the first game; games use consecutive seeds "
        f"(default: {constants.DEFAULT_SEED}).",
    )
    card.add_argument(
        "--workers",
        type=int,
        default=constants.DEFAULT_WORKERS,
        help="Worker processes; 1 runs in-process "
        f"(default: {constants.DEFAULT_WORKERS}).",
    )
    card.add_argument(
        "--device", default="cpu", help="Torch device for inference (default: cpu)."
    )
    card.add_argument(
        "--aggregate-only",
        dest="aggregate_only",
        default=None,
        metavar="CARD_GAMES_RAW_CSV",
        help="Skip play: aggregate this card_games_raw.csv into --out.",
    )
