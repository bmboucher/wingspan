"""Model introspection CLI: state/choice vector layout, architecture, and parameters.

Entry point: ``wingspan inspect`` (or ``python -m wingspan.reporting.inspect_cli``).
Prints four sections:

1. **STATE VECTOR** — every stripe in the ``encode_state`` output, named,
   described, with its offset, size, encoding kind, and value range.
2. **CHOICE VECTOR** — same breakdown for the per-candidate feature vector.
3. **ARCHITECTURE** — the same box-and-arrow flow shown by FLIGHT PLAN.
4. **PARAMETERS** — per-layer and per-block trainable-weight counts.

Without ``--checkpoint-dir`` the tool uses the default
:class:`~wingspan.architecture.ModelArchitecture` (the out-of-the-box network
shape). Pass a run's checkpoint directory to read its ``model_config.json`` and
show the exact topology that checkpoint was trained with: every table, width,
and count derives from the run's own descriptor through the era-routed
``runmeta`` seam, so a pre-0.1 run shows its frozen v0.0 choice geometry — not
the live encoder's.
"""

from __future__ import annotations

import argparse
import io
import sys

import rich.console as rich_console
import rich.panel as rich_panel
import rich.table as rich_table
from rich import text as rich_text

from wingspan import architecture, decisions, encode, setup_model, version
from wingspan.encode import stripes as encode_stripes
from wingspan.training import artifacts, runmeta, setup_runmeta
from wingspan.training.charts import text_helpers
from wingspan.training.configure import arch_diagram


def main_inspect(argv: list[str] | None = None) -> int:
    """CLI entry point: print the model introspection report."""
    parser = argparse.ArgumentParser(
        prog="wingspan inspect",
        description=(
            "Inspect the Wingspan model: state/choice vector layout, "
            "architecture, and parameter breakdown."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        metavar="PATH",
        default=None,
        help=(
            "Load architecture from model_config.json in this directory. "
            "Defaults to the standard ModelArchitecture() baseline."
        ),
    )
    parser.add_argument(
        "--section",
        choices=["state", "choice", "arch", "params", "all"],
        default="all",
        help="Which section(s) to print (default: all).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override terminal column width.",
    )
    parser.add_argument(
        "--html",
        metavar="FILE",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write a standalone HTML report instead of (or in addition to) "
            "terminal output.  FILE is the output path; omit to use "
            "<checkpoint-dir>/model_summary.html or ./model_summary.html."
        ),
    )
    args = parser.parse_args(argv)

    # Ensure UTF-8 output on Windows — box-drawing glyphs and Unicode symbols
    # in the arch diagram are not encodable in cp1252 (the Windows default).
    stdout = _utf8_stdout()
    console = rich_console.Console(file=stdout, width=args.width, legacy_windows=False)
    try:
        info = _load_arch_info(args.checkpoint_dir)
    except FileNotFoundError:
        console.print(
            f"[red]Error:[/red] no {artifacts.MODEL_CONFIG_JSON} in "
            f"{args.checkpoint_dir!r} — not a run directory."
        )
        return 1

    # --html: write the HTML report, then optionally also print terminal output.
    if args.html is not None:
        _write_html_report(args, info, console)
        if args.section == "all":
            return 0

    show_all = args.section == "all"
    if show_all or args.section == "state":
        _print_state_section(console, info)
    if show_all or args.section == "choice":
        _print_choice_section(console, info)
    if show_all or args.section == "arch":
        _print_arch_section(console, info)
    if show_all or args.section == "params":
        _print_params_section(console, info)

    return 0


###### PRIVATE #######

#### HTML report ####


def _write_html_report(
    args: argparse.Namespace,
    info: _ArchInfo,
    console: rich_console.Console,
) -> None:
    """Generate and write the HTML report; print the output path to the console.

    Built by the same :func:`runmeta.build_model_summary_html` the run-start
    writer uses, so regenerating into a run directory reproduces (for a
    current-era run) or era-corrects (for an older run) its original
    ``model_summary.html`` instead of clobbering it with live-encoder data."""
    import pathlib

    # Resolve the output path: explicit FILE, then <checkpoint-dir>/model_summary.html,
    # then ./model_summary.html as last resort.
    if args.html:
        out_path = pathlib.Path(args.html)
    elif args.checkpoint_dir:
        out_path = pathlib.Path(args.checkpoint_dir) / artifacts.MODEL_SUMMARY_HTML
    else:
        out_path = pathlib.Path(artifacts.MODEL_SUMMARY_HTML)

    html_content = runmeta.build_model_summary_html(info.descriptor, info.setup_arch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    console.print(f"[bold green]HTML report written →[/bold green] {out_path}")


#### Console setup ####


def _utf8_stdout() -> io.TextIOWrapper:
    """Return a UTF-8 text stream over stdout.

    On Windows the default stdout encoding is cp1252, which cannot encode the
    box-drawing characters and Greek letters the architecture diagram uses.
    Wrapping ``sys.stdout.buffer`` in a UTF-8 TextIOWrapper fixes that without
    affecting the encoding of other streams. Falls back to the original stdout
    when no ``buffer`` attribute is available (already a text stream without a
    backing binary buffer, e.g. in some test harnesses)."""
    if hasattr(sys.stdout, "buffer"):
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    return sys.stdout  # type: ignore[return-value]


#### Data loading ####


class _ArchInfo:
    """The resolved run descriptor for one introspection run, plus the setup-net
    topology — the one datum not on the descriptor (it lives in
    ``setup_config.json``). Every table, width, and count is derived from
    ``descriptor`` through the era-routed ``runmeta`` seam."""

    def __init__(
        self,
        descriptor: runmeta.ModelConfig,
        setup_arch: setup_model.SetupArchitecture | None = None,
    ):
        self.descriptor = descriptor
        self.setup_arch = setup_arch or setup_model.SetupArchitecture()

    @property
    def run_name(self) -> str:
        """The run's display name (``"(baseline)"`` for the no-dir default)."""
        return self.descriptor.run_name

    @property
    def use_setup_model(self) -> bool:
        """Whether the separate setup model is active — the inverse of the main
        net carrying setup (``include_setup``)."""
        return not self.descriptor.include_setup


def _default_family_order() -> tuple[str, ...]:
    """The baseline family order — matches the default spec (setup excluded)."""
    return tuple(
        family.value
        for family in decisions.active_decision_families(
            encode.DEFAULT_SPEC.include_setup
        )
    )


def _load_arch_info(checkpoint_dir: str | None) -> _ArchInfo:
    """Load the run descriptor from ``model_config.json``; with no checkpoint
    dir at all, describe the baseline (default-config) network instead.

    The baseline is a synthetic current-era descriptor, so every code path —
    tables, diagram, HTML — is descriptor-first. A checkpoint dir without a
    ``model_config.json`` raises ``FileNotFoundError`` — every run directory
    carries the descriptor, so its absence means the path is not a run
    directory. The setup net's topology comes from the run's
    ``setup_config.json`` when present; absent, the default
    :class:`~wingspan.setup_model.SetupArchitecture` is used."""
    if checkpoint_dir is None:
        return _ArchInfo(
            descriptor=runmeta.ModelConfig(
                run_name="(baseline)",
                state_dim=encode.state_size(),
                choice_dim=encode.CHOICE_FEATURE_DIM,
                family_order=_default_family_order(),
                architecture=architecture.ModelArchitecture(),
                include_setup=encode.DEFAULT_SPEC.include_setup,
                version=version.MODEL_VERSION,
            )
        )
    return _ArchInfo(
        descriptor=runmeta.read_model_config(checkpoint_dir),
        setup_arch=_load_setup_arch(checkpoint_dir),
    )


def _load_setup_arch(checkpoint_dir: str) -> setup_model.SetupArchitecture:
    """The run's setup-net topology from ``setup_config.json``, or the default."""
    try:
        return setup_runmeta.read_setup_config(checkpoint_dir).setup_arch
    except FileNotFoundError:
        return setup_model.SetupArchitecture()


#### Vector layout sections ####


def _print_state_section(console: rich_console.Console, info: _ArchInfo) -> None:
    """Print the STATE VECTOR breakdown table."""
    layout = runmeta.state_layout_for(info.descriptor)
    table = _make_stripe_table(layout, "STATE VECTOR")
    console.print()
    console.print(
        rich_panel.Panel(
            table,
            title=f"[bold]STATE VECTOR[/bold]  ({layout.total_size} elements)",
            subtitle=f"run: {info.run_name}",
            border_style="bright_blue",
        )
    )


def _print_choice_section(console: rich_console.Console, info: _ArchInfo) -> None:
    """Print the CHOICE VECTOR breakdown table — era-routed, so a pre-0.1 run
    shows its frozen v0.0 stripes (habitat one-hot, 180-wide bird identity)."""
    layout = runmeta.choice_layout_for(info.descriptor)
    table = _make_stripe_table(layout, "CHOICE VECTOR")
    console.print()
    console.print(
        rich_panel.Panel(
            table,
            title=f"[bold]CHOICE VECTOR[/bold]  ({layout.total_size} elements)",
            subtitle=f"run: {info.run_name}",
            border_style="bright_blue",
        )
    )


def _make_stripe_table(
    layout: encode_stripes.VectorLayout, section_name: str
) -> rich_table.Table:
    """Build a Rich Table for a :class:`VectorLayout`."""
    table = rich_table.Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("Name", style="bold", no_wrap=True)
    table.add_column("Offset", justify="right", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Encoding", style="green", no_wrap=True)
    table.add_column("Range", style="yellow", no_wrap=True)
    table.add_column("Description / Notes")

    for stripe in layout.stripes:
        desc = stripe.description
        if stripe.notes:
            desc = f"{stripe.description}\n[dim]{stripe.notes}[/dim]"
        table.add_row(
            stripe.name,
            str(stripe.offset),
            str(stripe.size),
            stripe.encoding,
            stripe.value_range,
            desc,
        )

    return table


#### Architecture section ####


def _print_arch_section(console: rich_console.Console, info: _ArchInfo) -> None:
    """Print the ARCHITECTURE block diagram (identical to FLIGHT PLAN), the
    choice-encoder widths era-routed through the descriptor seam."""
    # Use ~40 columns for the diagram box so it fits even in narrow terminals.
    box_width = min(48, (console.width or 80) - 4)
    descriptor = info.descriptor
    rows = arch_diagram.render_static(
        descriptor.architecture,
        state_dim=descriptor.state_dim,
        choice_dim=descriptor.choice_dim,
        family_order=descriptor.family_order,
        choice_in=runmeta.choice_input_dim_for(descriptor),
        choice_extra=runmeta.choice_extra_for(descriptor),
        use_setup_model=info.use_setup_model,
        setup_arch=info.setup_arch,
        width=box_width,
    )
    # Render each row separated by newlines inside a panel.
    lines = rich_text.Text()
    for index, row in enumerate(rows):
        if index:
            lines.append("\n")
        lines.append_text(row)
    console.print()
    console.print(
        rich_panel.Panel(
            lines,
            title="[bold]ARCHITECTURE[/bold]",
            subtitle=f"run: {info.run_name}",
            border_style="bright_blue",
        )
    )


#### Parameters section ####


def _print_params_section(console: rich_console.Console, info: _ArchInfo) -> None:
    """Print the per-layer / per-block parameter breakdown — era-routed, so the
    totals match ``sum(p.numel())`` of the run's actual checkpoint."""
    report = runmeta.param_report_for(info.descriptor)
    total = report.total
    table = _build_params_table(report, total)
    console.print()
    console.print(
        rich_panel.Panel(
            table,
            title=f"[bold]PARAMETERS[/bold]  ({text_helpers.human_count(total)} total)",
            subtitle=f"run: {info.run_name}",
            border_style="bright_blue",
        )
    )


def _build_params_table(
    report: architecture.ParamReport, total: int
) -> rich_table.Table:
    """Build the per-block/per-layer parameter Rich Table."""
    table = rich_table.Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("Block", style="bold", no_wrap=True)
    table.add_column("Layer", no_wrap=True)
    table.add_column("Params", justify="right")
    table.add_column("% Total", justify="right", style="dim")
    table.add_column("Cumulative", justify="right", style="dim")

    running = 0
    for block in report.blocks:
        # Per-layer rows for this block (multiplied by block.multiplier for scorer).
        block_label = block.label
        if block.multiplier > 1:
            block_label = f"{block.label} x{block.multiplier}"

        for index, layer in enumerate(block.layers):
            layer_label = f"Linear  {layer.in_features} -> {layer.out_features}"
            layer_params = layer.linear * block.multiplier
            running += layer_params
            table.add_row(
                block_label if index == 0 else "",
                layer_label,
                text_helpers.human_count(layer_params),
                f"{100.0 * layer_params / max(total, 1):.1f}%",
                text_helpers.human_count(running),
            )
            if layer.norm > 0:
                norm_params = layer.norm * block.multiplier
                running += norm_params
                table.add_row(
                    "",
                    f"LayerNorm  {layer.out_features}",
                    text_helpers.human_count(norm_params),
                    f"{100.0 * norm_params / max(total, 1):.1f}%",
                    text_helpers.human_count(running),
                )

        # Block subtotal row.
        table.add_row(
            "",
            f"[bold]Subtotal {block_label}[/bold]",
            f"[bold]{text_helpers.human_count(block.total)}[/bold]",
            f"[bold]{100.0 * block.total / max(total, 1):.1f}%[/bold]",
            "",
            style="on grey7",
        )

    # Grand total footer.
    table.add_row(
        "[bold bright_white]TOTAL[/bold bright_white]",
        "",
        f"[bold bright_white]{text_helpers.human_count(total)}[/bold bright_white]",
        "[bold bright_white]100%[/bold bright_white]",
        "",
        style="on grey11",
    )

    return table


if __name__ == "__main__":
    sys.exit(main_inspect())
