"""Generate YAML run-file configs for RESEARCH.md Project 1: general architecture exploration.

Each "sweep" run has every submodel pinned to its "lite" version (one hidden layer sized
to the submodel's natural output width) except for one submodel, which uses the "heavy"
version (two hidden layers of the same width). One additional baseline run sets every
submodel to lite simultaneously.

Hidden-layer widths by submodel:

* Card encoding (``card_encoder_layers``, ``hand_encoder_layers``): 64 (= ``card_embed_dim``)
* Trunk (``trunk_layers``): 128
* Choice encoding (``choice_layers``): 128
* Scorer heads (``head_layers``): 128 (input = trunk + choice concat = 256)
* Value head (``value_layers``): 64
* Setup net (``setup_hidden_layers``): 64

Usage::

    python scripts/gen_arch_experiment.py [options]

    --out-dir DIR           Write runfiles here (default: runs/arch_experiment)
    --target-games N        Total games per run (default: 1,000,000)
    --games-per-iter N      Games per training iteration (default: 256)
    --prefix STR            Run-name prefix (default: arch_exp)
    --s3-bucket STR         If given, add an ``s3:`` block to each runfile for cloud use
    --s3-prefix STR         S3 object prefix (default: runs)
    --s3-region STR         AWS region (default: us-east-1)
"""

from __future__ import annotations

import argparse
import math
import pathlib
import typing

import yaml

# ---- Layer-width constants ----
# Each value is the natural output width of the corresponding submodel; hidden
# layers in both lite (depth=1) and heavy (depth=2) versions use this width.

_CARD_ENC_WIDTH: int = 64    # projects into card_embed_dim embedding space
_TRUNK_WIDTH: int = 128       # trunk state embedding; feeds scorer heads
_CHOICE_WIDTH: int = 128      # choice embedding; concatenated with trunk for scoring
_HEAD_WIDTH: int = 128        # scorer hidden width (input = trunk+choice = 256)
_VALUE_WIDTH: int = 64        # value head; projects to a scalar
_SETUP_WIDTH: int = 64        # setup net; projects to a scalar

# ---- Training-run constants ----

_DEFAULT_TARGET_GAMES: int = 1_000_000
_DEFAULT_GAMES_PER_ITER: int = 256
_DEFAULT_OUT_DIR: str = "runs/arch_experiment"
_DEFAULT_PREFIX: str = "arch_exp"
_DEFAULT_S3_PREFIX: str = "runs"
_DEFAULT_S3_REGION: str = "us-east-1"

# ---- Submodel sweep table ----
# Each entry: (label used in run names, [TrainConfig field names], hidden-layer width).
# Submodels that control multiple fields (card encoding) list all of them together so
# the lite/heavy transition is applied atomically.

_SUBMODELS: list[tuple[str, list[str], int]] = [
    ("card_enc", ["card_encoder_layers", "hand_encoder_layers"], _CARD_ENC_WIDTH),
    ("trunk",    ["trunk_layers"],                                _TRUNK_WIDTH),
    ("choice",   ["choice_layers"],                               _CHOICE_WIDTH),
    ("heads",    ["head_layers"],                                  _HEAD_WIDTH),
    ("value",    ["value_layers"],                                 _VALUE_WIDTH),
    ("setup",    ["setup_hidden_layers"],                          _SETUP_WIDTH),
]

# ---- Custom YAML dumper ----
# Renders flat lists of integers in flow style ([64, 64]) so layer configs read
# naturally; all other nodes use the default block style.


def _represent_list(dumper: yaml.Dumper, data: typing.Any) -> yaml.SequenceNode:
    """Emit flat integer lists as flow-style ``[a, b]``; all others block-style."""
    is_flat_int_list = isinstance(data, list) and bool(data) and all(
        isinstance(item, int) for item in data
    )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=is_flat_int_list
    )


class _PrettyDumper(yaml.Dumper):
    """YAML dumper that writes flat integer lists in flow style."""


_PrettyDumper.add_representer(list, _represent_list)


# ---- Public entry point ----


def main() -> None:
    """Generate runfiles for the architecture-exploration sweep and print a summary."""
    args = _parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-run iteration budget from total-game target.
    max_iters = math.ceil(args.target_games / args.games_per_iter)
    s3 = _s3_block(args)
    experiments = _build_experiments(args.prefix)

    # Write one YAML runfile per experiment.
    print(f"Generating {len(experiments)} runfiles -> {out_dir}")
    print(
        f"  max_iterations={max_iters}"
        f"  ({args.target_games:,} games / {args.games_per_iter} per iter)"
    )
    print()
    for run_name, overrides in experiments:
        runfile = _runfile_dict(run_name, overrides, max_iters, args.games_per_iter, s3)
        out_path = out_dir / f"{run_name}.yaml"
        with out_path.open("w", encoding="utf-8") as yaml_file:
            yaml.dump(
                runfile, yaml_file,
                Dumper=_PrettyDumper, default_flow_style=False, sort_keys=True,
            )
        print(f"  {out_path}")

    _print_tournament_command(experiments)


# ---- Private helpers ----


def _parse_args() -> argparse.Namespace:
    """Build and parse the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate arch-experiment runfiles (RESEARCH.md Project 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR,
                        help="Directory to write runfiles into")
    parser.add_argument("--target-games", type=int, default=_DEFAULT_TARGET_GAMES,
                        help="Total games per run (sets max_iterations)")
    parser.add_argument("--games-per-iter", type=int, default=_DEFAULT_GAMES_PER_ITER,
                        help="Games collected per training iteration")
    parser.add_argument("--prefix", default=_DEFAULT_PREFIX,
                        help="Prefix prepended to every run name")
    parser.add_argument("--s3-bucket", default="",
                        help="S3 bucket; if set, adds an s3: block to every runfile")
    parser.add_argument("--s3-prefix", default=_DEFAULT_S3_PREFIX,
                        help="S3 object prefix")
    parser.add_argument("--s3-region", default=_DEFAULT_S3_REGION,
                        help="AWS region")
    return parser.parse_args()


def _build_experiments(prefix: str) -> list[tuple[str, dict[str, list[int]]]]:
    """Return (run_name, train_overrides) pairs for the full sweep.

    The first entry is the all-lite baseline; subsequent entries each promote one
    submodel to its heavy version while all others remain lite.
    """
    # Build the shared lite baseline: one hidden layer per submodel.
    lite_base: dict[str, list[int]] = {}
    for _, fields, width in _SUBMODELS:
        for field in fields:
            lite_base[field] = [width]

    # Baseline: every submodel lite.
    experiments: list[tuple[str, dict[str, list[int]]]] = [
        (f"{prefix}_all_lite", dict(lite_base)),
    ]

    # One sweep per submodel: that submodel heavy, all others lite.
    for label, swept_fields, width in _SUBMODELS:
        overrides = dict(lite_base)
        for field in swept_fields:
            overrides[field] = [width, width]
        experiments.append((f"{prefix}_heavy_{label}", overrides))

    return experiments


def _runfile_dict(
    run_name: str,
    overrides: dict[str, list[int]],
    max_iterations: int,
    games_per_iter: int,
    s3: dict[str, str | None] | None,
) -> dict[str, object]:
    """Assemble one runfile as a plain dict ready for ``yaml.dump``."""
    train: dict[str, object] = {
        "checkpoint_dir": f"checkpoints/{run_name}",
        "games_per_iter": games_per_iter,
        "max_iterations": max_iterations,
        "run_name": run_name,
    }
    train.update(overrides)

    runfile: dict[str, object] = {"run_name": run_name, "train": train}
    if s3 is not None:
        runfile["s3"] = s3
    return runfile


def _s3_block(args: argparse.Namespace) -> dict[str, str | None] | None:
    """Return the ``s3:`` dict if ``--s3-bucket`` was provided, else ``None``."""
    if not args.s3_bucket:
        return None
    return {
        "bucket": args.s3_bucket,
        "endpoint_url": None,
        "prefix": args.s3_prefix,
        "region": args.s3_region,
    }


def _print_tournament_command(experiments: list[tuple[str, dict[str, list[int]]]]) -> None:
    """Print a ready-to-use ``wingspan tournament`` command for all runs."""
    run_names = [run_name for run_name, _ in experiments]
    print("\nTournament command (run after all training completes):")
    print("  wingspan tournament --no-picker \\")
    for run_name in run_names:
        print(f"    --ai checkpoints/{run_name} \\")
    print("    --games-per-pair 64 --out arch_exp_results.json --quiet")


if __name__ == "__main__":
    main()
