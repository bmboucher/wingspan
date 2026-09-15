"""``wingspan analysis`` — the CLI over the architecture-probe package.

Usage::

    wingspan analysis probe TARGET [--checkpoint-dir checkpoints] [--games 20]
                                    [--reference TARGET] [--head-to-head 0]
                                    [--seed 1234] [--device cpu] [--json PATH]
                                    [--width N]

``TARGET`` (and ``--reference``) use the player-spec grammar
(``players.spec.parse_player_spec``): a named checkpoint (``last`` / ``best`` /
``opponent``), a run directory, or a direct ``.pt`` path. ``human`` and
``random`` are rejected — there is no network to probe.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys

import rich.console as rich_console
import rich.panel as rich_panel
import rich.table as rich_table
import torch

from wingspan import model
from wingspan.analysis import head_to_head, models
from wingspan.analysis import probe_set as probe_set_module
from wingspan.analysis import representation
from wingspan.players import loaders
from wingspan.players import spec as player_spec

_PROBE_COMMAND = "probe"

_DEFAULT_CHECKPOINT_DIR = "checkpoints"
_DEFAULT_GAMES = 20
_DEFAULT_SEED = 1234
_DEFAULT_HEAD_TO_HEAD_PAIRS = 0

# TARGET / --reference specs that name no trainable network.
_REJECTED_KINDS = (player_spec.PlayerKind.HUMAN, player_spec.PlayerKind.RANDOM)

# Substitution modes a nonzero --head-to-head plays, in report order.
_HEAD_TO_HEAD_MODES = (models.AblationMode.UNIFORM, models.AblationMode.ZERO)


def main_analysis(argv: list[str] | None = None) -> int:
    """CLI entry point for ``wingspan analysis``."""
    args = _parse_args(argv)
    if args.command == _PROBE_COMMAND:
        return _run_probe(args)
    raise AssertionError(f"unreachable: argparse admitted command {args.command!r}")


###### PRIVATE #######

#### Argument parsing ####


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wingspan analysis",
        description="Offline architecture-importance probes over a trained checkpoint.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    probe = commands.add_parser(
        _PROBE_COMMAND,
        help="Measure how much each part of the policy network's architecture matters.",
    )
    probe.add_argument(
        "target",
        metavar="TARGET",
        help="Player spec naming the checkpoint to probe (last/best/opponent, a run "
        "directory, or a direct .pt path).",
    )
    probe.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=_DEFAULT_CHECKPOINT_DIR,
        help=f"Directory named specs (last/best/opponent) resolve against "
        f"(default: {_DEFAULT_CHECKPOINT_DIR}).",
    )
    probe.add_argument(
        "--games",
        type=int,
        default=_DEFAULT_GAMES,
        help=f"Self-play games to build the probe set from (default: {_DEFAULT_GAMES}).",
    )
    probe.add_argument(
        "--reference",
        default=None,
        metavar="TARGET",
        help="A second checkpoint spec to compare policy/value output against.",
    )
    probe.add_argument(
        "--head-to-head",
        dest="head_to_head",
        type=int,
        default=_DEFAULT_HEAD_TO_HEAD_PAIRS,
        help="Mirrored-deal pairs to play for a UNIFORM/ZERO substitution win rate "
        "(default: 0, skipped).",
    )
    probe.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Seed for self-play collection and head-to-head evaluation "
        f"(default: {_DEFAULT_SEED}).",
    )
    probe.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    probe.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write the full report as JSON to PATH.",
    )
    probe.add_argument(
        "--width", type=int, default=None, help="Override terminal column width."
    )
    return parser.parse_args(argv)


#### `probe` sub-command ####


def _run_probe(args: argparse.Namespace) -> int:
    """Resolve TARGET, build the probe set, measure, and print/write the report."""
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    target = player_spec.parse_player_spec(args.target, checkpoint_dir)
    if target.kind in _REJECTED_KINDS or target.checkpoint_path is None:
        print(
            f"wingspan analysis probe: TARGET must name a trained checkpoint, "
            f"not {args.target!r}",
            file=sys.stderr,
        )
        return 1

    try:
        report = _measure_target(args, target.checkpoint_path, checkpoint_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"wingspan analysis probe: {error}", file=sys.stderr)
        return 1

    console = rich_console.Console(
        file=_utf8_stdout(), width=args.width, legacy_windows=False
    )
    _print_report(console, report)
    if args.json:
        pathlib.Path(args.json).write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        console.print(f"[bold green]JSON report written ->[/bold green] {args.json}")
    return 0


def _measure_target(
    args: argparse.Namespace,
    checkpoint_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
) -> models.RepresentationReport:
    """Load the target (and optional reference) net, build the probe set, run
    :func:`representation.measure`, and attach a head-to-head evaluation when
    ``--head-to-head`` is nonzero."""
    device = torch.device(args.device)
    net, run_config = loaders.load_policy_net(checkpoint_path, device)
    probe_set = probe_set_module.from_self_play(
        net, run_config, args.games, args.seed, device
    )
    reference_net = _load_reference(args, checkpoint_dir, device)

    report = representation.measure(
        net,
        probe_set,
        device=device,
        score_norm=run_config.training.score_norm,
        reference_net=reference_net,
        checkpoint_label=str(checkpoint_path),
    )
    if args.head_to_head > 0:
        report.head_to_head = [
            head_to_head.evaluate_substitution(
                checkpoint_path, mode, args.head_to_head, args.seed, device
            )
            for mode in _HEAD_TO_HEAD_MODES
        ]
    return report


def _load_reference(
    args: argparse.Namespace, checkpoint_dir: pathlib.Path, device: torch.device
) -> model.PolicyValueNet | None:
    """Load ``--reference``'s net, or ``None`` when not given. Raises
    ``ValueError`` when the spec names no trainable network."""
    if args.reference is None:
        return None
    reference_spec = player_spec.parse_player_spec(args.reference, checkpoint_dir)
    if reference_spec.kind in _REJECTED_KINDS or reference_spec.checkpoint_path is None:
        raise ValueError(
            f"--reference must name a trained checkpoint, not {args.reference!r}"
        )
    net, _ = loaders.load_policy_net(reference_spec.checkpoint_path, device)
    return net


#### Console setup (mirrors reporting.inspect_cli) ####


def _utf8_stdout() -> io.TextIOWrapper:
    """A UTF-8 text stream over stdout — the default Windows cp1252 stdout
    cannot encode the box-drawing / Unicode glyphs rich tables use."""
    if hasattr(sys.stdout, "buffer"):
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    return sys.stdout  # type: ignore[return-value]


#### Report printing — one small helper per section ####


def _print_report(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    """Print every populated section of ``report`` as a titled rich panel."""
    _print_summary(console, report)
    _print_param_census(console, report)
    _print_layers(console, report)
    _print_attention(console, report)
    _print_ablations(console, report)
    _print_family_effects(console, report)
    _print_board_fill_effects(console, report)
    _print_trunk_shares(console, report)
    _print_reference(console, report)
    _print_head_to_head(console, report)


def _new_table(*columns: str) -> rich_table.Table:
    table = rich_table.Table(
        show_header=True, header_style="bold cyan", border_style="dim", pad_edge=False
    )
    for column in columns:
        table.add_column(column)
    return table


def _print_panel(
    console: rich_console.Console, renderable: rich_table.Table, title: str
) -> None:
    console.print()
    console.print(
        rich_panel.Panel(
            renderable, title=f"[bold]{title}[/bold]", border_style="bright_blue"
        )
    )


def _print_summary(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    console.print()
    console.print(
        f"[bold]{report.checkpoint}[/bold]  {report.n_games} games, "
        f"{report.n_decisions} decisions, {report.total_parameters:,} parameters"
    )


def _print_param_census(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    table = _new_table("Block", "Parameters", "Share")
    for entry in report.param_census:
        table.add_row(entry.block, f"{entry.parameters:,}", f"{100 * entry.share:.1f}%")
    _print_panel(console, table, "PARAMETER CENSUS")


def _print_layers(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    table = _new_table(
        "Layer", "In", "Out", "rank95", "rank95/width", "linR2", "dead%", "rows"
    )
    for layer in report.layers:
        table.add_row(
            layer.name,
            str(layer.in_features),
            str(layer.out_features),
            str(layer.rank95),
            f"{layer.rank95_over_width:.2f}",
            f"{layer.linear_r2:.3f}",
            f"{100 * layer.dead_fraction:.1f}%",
            str(layer.rows),
        )
    _print_panel(console, table, "LAYER CAPACITY")


def _print_attention(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    block = report.attention
    if block is None:
        return
    summary = _new_table(
        "Contribution ratio (med / p10 / p90)", "Cosine (med / p10 / p90)"
    )
    summary.add_row(
        f"{block.contribution_ratio_median:.3f} / {block.contribution_ratio_p10:.3f} / "
        f"{block.contribution_ratio_p90:.3f}",
        f"{block.cosine_median:.3f} / {block.cosine_p10:.3f} / {block.cosine_p90:.3f}",
    )
    _print_panel(console, summary, "BOARD ATTENTION")
    heads = _new_table("Head", "Entropy median", "Entropy p10", "Self-weight median")
    for head in block.heads:
        heads.add_row(
            str(head.head),
            f"{head.entropy_median:.3f}",
            f"{head.entropy_p10:.3f}",
            f"{head.self_weight_median:.3f}",
        )
    _print_panel(console, heads, "BOARD ATTENTION HEADS")


def _print_ablations(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    if not report.ablations:
        return
    table = _new_table("Mode", "Head", "n", "mean KL", "p90 KL", "flip%", "|dV| pts")
    for ablation in report.ablations:
        table.add_row(
            ablation.mode.value,
            "-" if ablation.head is None else str(ablation.head),
            str(ablation.n),
            f"{ablation.mean_kl:.4f}",
            f"{ablation.p90_kl:.4f}",
            f"{100 * ablation.flip_rate:.1f}%",
            f"{ablation.value_shift_points:.2f}",
        )
    _print_panel(console, table, "ABLATION EFFECTS")


def _print_family_effects(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    if not report.family_effects:
        return
    table = _new_table(
        "Family", "n", "zero meanKL", "zero flip%", "uniform meanKL", "uniform flip%"
    )
    for effect in report.family_effects:
        table.add_row(
            effect.family,
            str(effect.n),
            f"{effect.zero.mean_kl:.4f}",
            f"{100 * effect.zero.flip_rate:.1f}%",
            f"{effect.uniform.mean_kl:.4f}",
            f"{100 * effect.uniform.flip_rate:.1f}%",
        )
    _print_panel(console, table, "ZERO/UNIFORM EFFECT BY DECISION FAMILY")


def _print_board_fill_effects(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    if not report.board_fill_effects:
        return
    table = _new_table("Birds", "n", "zero meanKL", "uniform meanKL")
    for effect in report.board_fill_effects:
        table.add_row(
            f"{effect.min_birds}-{effect.max_birds}",
            str(effect.n),
            f"{effect.zero.mean_kl:.4f}",
            f"{effect.uniform.mean_kl:.4f}",
        )
    _print_panel(console, table, "ZERO/UNIFORM EFFECT BY BOARD FILL")


def _print_trunk_shares(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    if not report.trunk_input_shares:
        return
    table = _new_table("Group", "Dims", "Energy share")
    for share in report.trunk_input_shares:
        table.add_row(share.group, str(share.dims), f"{100 * share.energy_share:.1f}%")
    _print_panel(console, table, "TRUNK INPUT ENERGY SHARE")


def _print_reference(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    reference = report.reference
    if reference is None:
        return
    table = _new_table("Checkpoint", "n", "mean KL", "p90 KL", "flip%", "|dV| pts")
    table.add_row(
        reference.checkpoint,
        str(reference.n),
        f"{reference.mean_kl:.4f}",
        f"{reference.p90_kl:.4f}",
        f"{100 * reference.flip_rate:.1f}%",
        f"{reference.value_shift_points:.2f}",
    )
    _print_panel(console, table, "REFERENCE CHECKPOINT COMPARISON")


def _print_head_to_head(
    console: rich_console.Console, report: models.RepresentationReport
) -> None:
    if not report.head_to_head:
        return
    table = _new_table("Mode", "Games", "Win rate", "95% CI", "Mean margin")
    for result in report.head_to_head:
        table.add_row(
            result.mode.value,
            str(result.n_games),
            f"{100 * result.win_rate:.1f}%",
            f"+/- {100 * result.ci95:.1f}%",
            f"{result.mean_margin:.2f}",
        )
    _print_panel(console, table, "HEAD-TO-HEAD (FULL vs SUBSTITUTED)")


if __name__ == "__main__":
    sys.exit(main_analysis())
