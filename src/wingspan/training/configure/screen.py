"""Assembles the FLIGHT PLAN configurator screen from a
:class:`state.ConfiguratorState` snapshot.

:func:`build` returns a fresh renderable each frame (rebuilt rather than region-
patched, so the modal confirmation can replace the whole body — a rich
``Layout`` tiles, it cannot z-stack an overlay). The bands mirror the dashboard:
a gradient wordmark header, a body split into the editable form and a
detail / run-management column, and a footer of key hints + a status line. The
form is a width/height-aware renderable that scrolls to keep the focused field
visible, reusing the ``options.max_height`` pattern from :mod:`charts`.
"""

from __future__ import annotations

import pathlib
import time

import rich.console as rich_console
from rich import align, box, layout, panel, table, text

from wingspan import version
from wingspan.training import artifacts, charts, theme
from wingspan.training.configure import arch_diagram, fields, runs, state

_WORDMARK = "🪶 WINGSPAN  FLIGHT PLAN"
_HEADER_H = 4
_FOOTER_H = 5
# Dynamically sized so even the longest label always has
# at least 2 spaces before its value column.
_LABEL_W = max(len(spec.label) for spec in fields.FIELD_SPECS) + 2
_VALUE_W = 12  # field-value column width (minimum width via ljust)
# Frames the edit caret stays on / off. Idle heartbeat renders land on frame
# multiples of the controller's heartbeat (4), so dividing by 8 toggles the
# caret every two heartbeats (~0.5 s) regardless of which frames get painted.
_CARET_BLINK_FRAMES = 8
_MODAL_WIDTH = 66  # confirmation modal width (clamped to the terminal)
_MODAL_MIN_WIDTH = 40

# Focus / change markers in the form's left gutter.
_MARKER_FOCUS = "▸"
_MARKER_CHANGED = "•"
_SCROLL_UP = "  ▲ more above"
_SCROLL_DOWN = "  ▼ more below"

# Per-mode accent for the header pill.
_MODE_COLOR: dict[state.Mode, str] = {
    state.Mode.NAVIGATE: theme.GAUGE_UTIL,  # wetland teal
    state.Mode.EDIT: theme.BORDER_HEADLINE,  # grassland gold
    state.Mode.CONFIRM: "#B08CD9",  # soft violet
}
_MODE_LABEL: dict[state.Mode, str] = {
    state.Mode.NAVIGATE: "CONFIGURE",
    state.Mode.EDIT: "EDITING",
    state.Mode.CONFIRM: "CONFIRM",
}

# Change-impact glyph + color for the value column and the detail panel.
_IMPACT_GLYPH: dict[fields.ChangeImpact, str] = {
    fields.ChangeImpact.NONE: "",
    fields.ChangeImpact.REGIME: "≈",
    fields.ChangeImpact.FRESH: "✶",
}
_IMPACT_COLOR: dict[fields.ChangeImpact, str] = {
    fields.ChangeImpact.NONE: theme.GOOD,
    fields.ChangeImpact.REGIME: theme.CAUTION,
    fields.ChangeImpact.FRESH: theme.BAD,
}

# Run-status verdict glyph + color for the run-management panel.
_STATUS_GLYPH: dict[runs.RunStatus, str] = {
    runs.RunStatus.EMPTY: "○",
    runs.RunStatus.RESUMABLE: "●",
    runs.RunStatus.INCOMPATIBLE: "▲",
    runs.RunStatus.UNREADABLE: "⚠",
}
_STATUS_COLOR: dict[runs.RunStatus, str] = {
    runs.RunStatus.EMPTY: theme.TEXT_DIM2,
    runs.RunStatus.RESUMABLE: theme.GOOD,
    runs.RunStatus.INCOMPATIBLE: theme.CAUTION,
    runs.RunStatus.UNREADABLE: theme.BAD,
}
_MESSAGE_COLOR: dict[state.MessageKind, str] = {
    state.MessageKind.INFO: theme.TEXT_DIM2,
    state.MessageKind.SUCCESS: theme.GOOD,
    state.MessageKind.WARN: theme.CAUTION,
    state.MessageKind.ERROR: theme.BAD,
}

_ARCHIVES_SHOWN = 4  # most-recent archives listed in the run-management panel


def build(view: state.ConfiguratorState, frame: int) -> layout.Layout:
    """The full configurator renderable for one frame."""
    root = layout.Layout(name="root")
    root.split_column(
        layout.Layout(_header(view), name="header", size=_HEADER_H),
        layout.Layout(name="body", ratio=1),
        layout.Layout(_footer(view), name="footer", size=_FOOTER_H),
    )
    if view.mode is state.Mode.CONFIRM and view.confirm is not None:
        root["body"].update(_modal(view.confirm))
    else:
        body = root["body"]
        body.split_row(
            layout.Layout(_form_panel(view, frame), name="form", ratio=50),
            layout.Layout(_arch_panel(view), name="arch", ratio=34, minimum_size=30),
            layout.Layout(name="side", ratio=16, minimum_size=26),
        )
        body["side"].split_column(
            layout.Layout(_detail(view), name="detail", ratio=46),
            layout.Layout(_runinfo(view), name="runinfo", ratio=54),
        )
    return root


###### PRIVATE #######

#### Header band ####


def _header(view: state.ConfiguratorState) -> panel.Panel:
    top = table.Table.grid(expand=True)
    top.add_column(justify="left")
    top.add_column(justify="right")
    top.add_row(theme.gradient_text(_WORDMARK), _mode_pill(view))
    return panel.Panel(
        rich_console.Group(top, _context_row(view)),
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _mode_pill(view: state.ConfiguratorState) -> text.Text:
    color = _MODE_COLOR[view.mode]
    pill = text.Text(no_wrap=True, end="")
    pill.append(f" {_MODE_LABEL[view.mode]} ", style=f"bold {theme.CANVAS} on {color}")
    pill.append(f"  {view.working.misc.device}", style=theme.TEXT_DIM2)
    return pill


def _context_row(view: state.ConfiguratorState) -> table.Table:
    grid = table.Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    if view.seeded_from_saved and view.summary.iteration is not None:
        source = f"resumed run @ iter {view.summary.iteration:04d}"
    elif view.seeded_from_saved:
        source = "resumed run"
    elif view.seeded_from_user_defaults:
        source = "new run · saved defaults"
    else:
        source = "new run · defaults"
    left = text.Text(no_wrap=True, end="")
    left.append("editing ", style=theme.TEXT_MUTED)
    left.append(source, style=theme.TEXT_PRIMARY)
    left.append(f"   {view.working.run.checkpoint_dir}/", style=theme.TEXT_DIM2)
    grid.add_row(left, _status_chip(view))
    return grid


def _status_chip(view: state.ConfiguratorState) -> text.Text:
    status = view.status()
    out = text.Text(no_wrap=True, end="")
    out.append(f"{_STATUS_GLYPH[status]} ", style=_STATUS_COLOR[status])
    out.append(_start_action_label(view), style=_STATUS_COLOR[status])
    return out


#### Form band ####


def _form_panel(view: state.ConfiguratorState, frame: int) -> panel.Panel:
    return panel.Panel(
        _FormView(view, frame),
        title="[b]CONFIGURATION[/b]",
        subtitle="↑↓ move · ←→ adjust · enter edit",
        title_align="left",
        subtitle_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


class _FormView:
    """The scrollable list of editable fields, grouped by ``group_path``. A
    width/height-aware renderable: it reads the panel height each frame and
    slides a viewport so the focused field stays visible, marking clipped rows.
    Headers are emitted depth-first: a section header fires when the first path
    element changes; a group header fires when the second changes; a subgroup
    header when the third changes."""

    def __init__(self, view: state.ConfiguratorState, frame: int):
        self.view = view
        self.frame = frame

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        rows, selected_row = self._rows()
        height = options.height or options.max_height or len(rows)
        window, scroll_up, scroll_down = arch_diagram.viewport(
            rows, selected_row, height
        )
        # end="" to match every other row: the loop below supplies the inter-row
        # newlines, so a default-newline indicator would inject a blank line and
        # push the bottom row (incl. the ▼ indicator itself) off the panel.
        if scroll_up:
            window[0] = text.Text(_SCROLL_UP, style=theme.TEXT_MUTED, end="")
        if scroll_down:
            window[-1] = text.Text(_SCROLL_DOWN, style=theme.TEXT_MUTED, end="")
        for index, line in enumerate(window):
            if index:
                yield text.Text("\n", end="")
            yield line

    def _rows(self) -> tuple[list[text.Text], int]:
        rows: list[text.Text] = []
        selected_row = 0
        visible_specs = [
            spec
            for spec in fields.FIELD_SPECS
            if spec.visible_when is None or spec.visible_when(self.view.working)
        ]
        # Track the last-emitted path so we only emit headers for levels that
        # change (depth 0 = section, 1 = group, 2 = subgroup).
        current_path: tuple[str, ...] = ()
        for spec in visible_specs:
            path = spec.group_path
            for depth in range(len(path)):
                if depth >= len(current_path) or path[depth] != current_path[depth]:
                    rows.append(_depth_header(path[depth], depth))
            current_path = path
            if spec.attr == self.view.selected_attr:
                selected_row = len(rows)
            rows.append(_field_row(self.view, spec, self.frame))
        return rows, selected_row


# Per-depth display style for group_path headers.
_DEPTH_INDENT = ["", "  ", "    "]
_DEPTH_GLYPH = ["", "· ", "▸ "]
_DEPTH_STYLE = [f"bold {theme.TEXT_MUTED}", theme.TEXT_DIM2, theme.TEXT_MUTED]


def _depth_header(name: str, depth: int) -> text.Text:
    capped = min(depth, len(_DEPTH_INDENT) - 1)
    out = text.Text(no_wrap=True, end="")
    out.append(
        f"{_DEPTH_INDENT[capped]}{_DEPTH_GLYPH[capped]}{name}",
        style=_DEPTH_STYLE[capped],
    )
    return out


def _field_row(
    view: state.ConfiguratorState, spec: fields.FieldSpec, frame: int
) -> text.Text:
    focused = spec.attr == view.selected_attr
    changed = fields.is_changed(view.working, view.saved, spec)
    editing = focused and view.mode is state.Mode.EDIT

    line = text.Text(no_wrap=True, end="")
    line.append(f"{_row_marker(focused, changed)} ", style=_marker_color(view, spec))
    line.append(spec.label.ljust(_LABEL_W), style=_label_color(focused, changed, spec))
    line.append(_value_text(view, spec, editing, frame))
    if spec.unit:
        line.append(f" {spec.unit}", style=theme.TEXT_MUTED)
    glyph = _IMPACT_GLYPH[spec.impact]
    if glyph:
        line.append(f"  {glyph}", style=_IMPACT_COLOR[spec.impact])
    return line


def _value_text(
    view: state.ConfiguratorState, spec: fields.FieldSpec, editing: bool, frame: int
) -> text.Text:
    if editing:
        caret = "▏" if (frame // _CARET_BLINK_FRAMES) % 2 == 0 else " "
        value = (view.edit_buffer + caret).ljust(_VALUE_W)
        return text.Text(value, style=f"bold {theme.BORDER_HEADLINE}")
    formatted = fields.format_value(view.working, spec).ljust(_VALUE_W)
    color = theme.TEXT_BRIGHT if spec.attr == view.selected_attr else theme.TEXT_PRIMARY
    return text.Text(formatted, style=color)


def _row_marker(focused: bool, changed: bool) -> str:
    if focused:
        return _MARKER_FOCUS
    return _MARKER_CHANGED if changed else " "


def _marker_color(view: state.ConfiguratorState, spec: fields.FieldSpec) -> str:
    if spec.attr == view.selected_attr:
        return _MODE_COLOR[view.mode]
    return _IMPACT_COLOR[spec.impact]


def _label_color(focused: bool, changed: bool, spec: fields.FieldSpec) -> str:
    if changed:
        return (
            _IMPACT_COLOR[spec.impact]
            if spec.impact is not fields.ChangeImpact.NONE
            else theme.TEXT_BRIGHT
        )
    return theme.TEXT_PRIMARY if focused else theme.TEXT_DIM2


#### Architecture diagram band ####


def _arch_panel(view: state.ConfiguratorState) -> panel.Panel:
    """The model-topology panel: a live box-and-arrow flow diagram of the working
    network (see :class:`arch_diagram.ArchitectureDiagram`), which reacts to the
    edited config and highlights the focused field."""
    return panel.Panel(
        arch_diagram.ArchitectureDiagram(view),
        title="[b]ARCHITECTURE[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


#### Detail band ####


def _detail(view: state.ConfiguratorState) -> panel.Panel:
    spec = view.selected_spec()
    body = rich_console.Group(*_detail_lines(view, spec))
    return panel.Panel(
        body,
        title="[b]FIELD[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _detail_lines(
    view: state.ConfiguratorState, spec: fields.FieldSpec
) -> list[text.Text]:
    lines: list[text.Text] = [_detail_title(spec), _detail_value(view, spec)]
    constraint = _detail_constraint(spec)
    if constraint:
        lines.append(_kv("range", constraint))
    lines.append(_kv("default", fields.default_string(spec)))
    hint = _detail_hint(view, spec)
    if hint:
        lines.append(text.Text(hint, style=theme.TEXT_DIM2))
    lines.append(text.Text(""))
    lines.append(text.Text(spec.help, style=theme.TEXT_MUTED))
    return lines


def _detail_title(spec: fields.FieldSpec) -> text.Text:
    out = text.Text(no_wrap=True)
    out.append(spec.label, style=f"bold {theme.TEXT_BRIGHT}")
    if spec.impact is not fields.ChangeImpact.NONE:
        out.append(
            f"   {_IMPACT_GLYPH[spec.impact]} ", style=_IMPACT_COLOR[spec.impact]
        )
        out.append(_impact_note(spec.impact), style=_IMPACT_COLOR[spec.impact])
    return out


def _detail_value(view: state.ConfiguratorState, spec: fields.FieldSpec) -> text.Text:
    out = text.Text(no_wrap=True)
    out.append(
        fields.format_value(view.working, spec), style=f"bold {theme.TEXT_PRIMARY}"
    )
    if spec.unit:
        out.append(f" {spec.unit}", style=theme.TEXT_MUTED)
    return out


def _impact_note(impact: fields.ChangeImpact) -> str:
    if impact is fields.ChangeImpact.FRESH:
        return "needs a fresh run"
    if impact is fields.ChangeImpact.REGIME:
        return "reinterprets a resumed run"
    return ""


def _detail_constraint(spec: fields.FieldSpec) -> str:
    if isinstance(spec, fields.BootstrapField):
        return "none / random / archive path  ←/→ cycles  enter to type"
    if isinstance(spec, fields.OptionalChoiceField):
        return f"{spec.none_label} / " + " / ".join(spec.choices)
    if isinstance(spec, fields.ChoiceField):
        return " / ".join(spec.choices)
    if isinstance(spec, fields.LayersField):
        return "type widths (e.g. 256,128) · ←/→ adds/removes a layer"
    if isinstance(spec, fields.OptionalIntField):
        return f"step {spec.step}  or '{spec.none_label}' to inherit"
    if isinstance(spec, fields.IntField):
        return f"step {spec.step}"
    if isinstance(spec, fields.OptionalFloatField):
        return f"step {spec.step:g}  or 'none' to inherit"
    if isinstance(spec, fields.FloatField):
        return f"step {spec.step:g}"
    return ""


def _detail_hint(view: state.ConfiguratorState, spec: fields.FieldSpec) -> str:
    cfg = view.working
    if isinstance(spec, fields.BootstrapField):
        return _bootstrap_hint(view)
    if spec.attr == "eval_games":
        return f"→ {cfg.eval_pairs} mirrored pairs = {2 * cfg.eval_pairs} games"
    if spec.attr == "max_iterations" and view.status() is runs.RunStatus.RESUMABLE:
        start = (view.summary.iteration or 0) + 1
        if cfg.run.max_iterations > 0:
            return f"→ resumes at iter {start}, stops at {start + cfg.run.max_iterations - 1}"
        return "→ resumes and runs until you stop it"
    if (
        spec.attr == "device"
        and cfg.misc.device.startswith("cuda")
        and not view.cuda_available
    ):
        return "→ cuda unavailable — will fall back to cpu"
    if (
        spec.attr == "opponent_reset_win_rate"
        and cfg.opponent.opponent_reset_win_rate == 0
    ):
        return "→ 0 disables opponent advancement"
    if spec.attr == "history_len" and cfg.run.history_len < charts.CHART_WINDOW:
        return f"→ below the {charts.CHART_WINDOW}-iter chart window"
    return ""


def _bootstrap_hint(view: state.ConfiguratorState) -> str:
    """Detail-panel hint for the bootstrap_opponent field."""
    value = view.working.opponent.bootstrap_opponent
    if value == "none":
        return "→ no bootstrap phase — starts directly in self-play"
    if value == "random":
        return "→ bootstrap against the random agent (original behaviour)"

    # Path value — try to match against a known archive entry for rich metadata.
    archive_entry = _find_archive_entry(view, value)
    if archive_entry is None:
        return "→ custom checkpoint path"

    # Build the metadata line from the archive entry.
    parts: list[str] = []
    if archive_entry.model_version is not None:
        parts.append(f"v{archive_entry.model_version}")
    if archive_entry.total_games is not None:
        parts.append(f"{archive_entry.total_games:,} games")
    # Show either first-session stamp or archive date.
    if archive_entry.first_session_stamp is not None:
        parts.append(f"started {archive_entry.first_session_stamp}")
    else:
        date_str = time.strftime("%Y-%m-%d", time.localtime(archive_entry.modified))
        parts.append(f"archived {date_str}")
    return "→ " + " · ".join(parts) if parts else "→ archived run"


def _find_archive_entry(
    view: state.ConfiguratorState, checkpoint_path: str
) -> runs.ArchiveEntry | None:
    """Return the :class:`~runs.ArchiveEntry` whose ``last.pt`` equals
    ``checkpoint_path``, or ``None`` if it is not a known archive."""
    checkpoint_dir = pathlib.Path(view.working.run.checkpoint_dir)
    for entry in view.summary.archives:
        expected = str(
            checkpoint_dir
            / artifacts.ARCHIVE_SUBDIR
            / entry.label
            / artifacts.LAST_CKPT
        )
        if checkpoint_path == expected:
            return entry
    return None


#### Run-management band ####


def _runinfo(view: state.ConfiguratorState) -> panel.Panel:
    return panel.Panel(
        rich_console.Group(*_runinfo_lines(view)),
        title="[b]RUN MANAGEMENT[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _runinfo_lines(view: state.ConfiguratorState) -> list[text.Text]:
    status = view.status()
    summary = view.summary
    lines: list[text.Text] = [_status_line(status), _era_line(view, status)]
    if summary.exists and summary.readable:
        lines.extend(_run_detail_lines(summary))
    lines.append(text.Text(""))
    lines.extend(_archive_lines(summary))
    return lines


def _status_line(status: runs.RunStatus) -> text.Text:
    out = text.Text(no_wrap=True)
    out.append(f"{_STATUS_GLYPH[status]} ", style=_STATUS_COLOR[status])
    out.append(_status_text(status), style=_STATUS_COLOR[status])
    return out


def _era_line(view: state.ConfiguratorState, status: runs.RunStatus) -> text.Text:
    """The artifact era the launched run will train at: the saved run's frozen
    era when Start resumes it (highlighted when older than the live version),
    the current MODEL_VERSION for any fresh launch."""
    era = view.working.encoding_version
    label = "resume" if status is runs.RunStatus.RESUMABLE else "new run"
    style = theme.TEXT_DIM2 if era == version.MODEL_VERSION else theme.CAUTION
    return text.Text(f"era {era} ({label})", style=style, no_wrap=True)


def _status_text(status: runs.RunStatus) -> str:
    if status is runs.RunStatus.EMPTY:
        return "empty — Start launches a new run here"
    if status is runs.RunStatus.RESUMABLE:
        return "resume-ready — Start continues this run"
    if status is runs.RunStatus.INCOMPATIBLE:
        return "architecture changed — needs a fresh run"
    return "checkpoint unreadable"


def _run_detail_lines(summary: runs.RunSummary) -> list[text.Text]:
    iteration = summary.iteration if summary.iteration is not None else 0
    games = summary.total_games if summary.total_games is not None else 0
    lines = [_kv("iteration", f"{iteration:04d}"), _kv("games", f"{games:,}")]
    if summary.best_win_rate is not None:
        lines.append(_kv("best win", f"{summary.best_win_rate * 100:.1f}%"))
    opponent = (
        "random"
        if summary.opponent_generation == 0
        else f"self·gen{summary.opponent_generation}"
    )
    lines.append(_kv("opponent", opponent))
    artifacts_present = ", ".join(
        name
        for present, name in (
            (True, "last"),
            (summary.has_best, "best"),
            (summary.has_opponent, "opponent"),
            (summary.has_metrics, "metrics"),
        )
        if present
    )
    lines.append(_kv("artifacts", artifacts_present))
    return lines


def _archive_lines(summary: runs.RunSummary) -> list[text.Text]:
    if not summary.archives:
        return [text.Text("no archived runs", style=theme.TEXT_MUTED)]
    header = text.Text(no_wrap=True)
    header.append(f"{len(summary.archives)} archived", style=theme.TEXT_DIM2)
    header.append("  (archive/)", style=theme.TEXT_MUTED)
    lines = [header]
    for entry in reversed(summary.archives[-_ARCHIVES_SHOWN:]):
        row = text.Text(no_wrap=True)
        row.append("  • ", style=theme.TEXT_MUTED)
        row.append(entry.label, style=theme.TEXT_DIM2)
        lines.append(row)
    return lines


#### Footer band ####


def _footer(view: state.ConfiguratorState) -> panel.Panel:
    return panel.Panel(
        rich_console.Group(*_footer_lines(view)),
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _footer_lines(view: state.ConfiguratorState) -> list[text.Text]:
    return [_action_hints(view), _nav_hints(view), _message_line(view)]


def _action_hints(view: state.ConfiguratorState) -> text.Text:
    if view.mode is state.Mode.EDIT:
        return _hint_row([("enter", "commit"), ("esc", "cancel"), ("⌫", "delete")])
    if view.mode is state.Mode.CONFIRM:
        return _hint_row([("esc", "cancel")])
    return _hint_row(
        [
            ("S", _start_action_label(view)),
            ("N", "new run"),
            ("A", "archive"),
            ("R", "reset"),
            ("D", "save defaults"),
            ("Q", "quit"),
        ]
    )


def _nav_hints(view: state.ConfiguratorState) -> text.Text:
    if view.mode is state.Mode.NAVIGATE:
        return _hint_row(
            [("↑↓", "move"), ("←→", "adjust"), ("enter", "edit"), ("type", "value")],
            muted=True,
        )
    return text.Text("")


def _message_line(view: state.ConfiguratorState) -> text.Text:
    if view.message is None:
        return text.Text("")
    return text.Text(f"  {view.message.text}", style=_MESSAGE_COLOR[view.message.kind])


def _hint_row(pairs: list[tuple[str, str]], muted: bool = False) -> text.Text:
    out = text.Text(no_wrap=True)
    label_color = theme.TEXT_MUTED if muted else theme.TEXT_DIM2
    for index, (key, label) in enumerate(pairs):
        if index:
            out.append("   ")
        out.append(f"[{key}]", style=f"bold {theme.TEXT_PRIMARY}")
        out.append(f" {label}", style=label_color)
    return out


#### Modal ####


class _Modal:
    """The centered confirmation panel. Height-aware: the keyed options and the
    title are always shown; when the body region is too short the explanatory
    prompt lines are elided (with a ``…`` marker) rather than letting the
    safety-critical option rows clip off the bottom (rich ``Align`` crops, it
    does not shrink)."""

    def __init__(self, prompt: state.ConfirmPrompt):
        self.prompt = prompt

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        height = options.height or options.max_height or 24
        width = max(_MODAL_MIN_WIDTH, min(_MODAL_WIDTH, options.max_width - 4))
        option_lines = _modal_option_lines(self.prompt)
        # Reserve: the panel's two border rows, the option rows, and one blank
        # separator. Whatever is left is the budget for the prompt text.
        budget = height - 2 - len(option_lines) - 1
        if budget >= len(self.prompt.lines):
            shown, elided = self.prompt.lines, False
        else:
            shown = self.prompt.lines[: max(0, budget - 1)]  # a row for the ellipsis
            elided = True

        body: list[text.Text] = [
            text.Text(line, style=theme.TEXT_PRIMARY) for line in shown
        ]
        if elided:
            body.append(text.Text("…", style=theme.TEXT_MUTED))
        body.append(text.Text(""))
        body.extend(option_lines)
        inner = panel.Panel(
            rich_console.Group(*body),
            title=f"[b]{self.prompt.title}[/b]",
            title_align="left",
            box=box.HEAVY,
            border_style=_MODE_COLOR[state.Mode.CONFIRM],
            padding=(0, 3),
            width=width,
        )
        yield align.Align.center(inner, vertical="middle")


def _modal(prompt: state.ConfirmPrompt) -> _Modal:
    return _Modal(prompt)


def _modal_option_lines(prompt: state.ConfirmPrompt) -> list[text.Text]:
    """One row per option so long labels never get clipped; the default option
    is marked and a danger option (overwrite) is de-emphasized in the alarm
    color so it is never the reflexive pick."""
    lines: list[text.Text] = []
    for option in prompt.options:
        highlighted = option.key == prompt.default_key
        key_color = theme.BAD if option.danger else theme.TEXT_BRIGHT
        label_color = (
            theme.BAD
            if option.danger
            else (theme.TEXT_BRIGHT if highlighted else theme.TEXT_DIM2)
        )
        row = text.Text(no_wrap=True)
        row.append("▸ " if highlighted else "  ", style=label_color)
        row.append(f"[{option.key.upper()}]", style=f"bold {key_color}")
        row.append(f" {option.label}", style=label_color)
        lines.append(row)
    return lines


#### Shared helpers ####


def _start_action_label(view: state.ConfiguratorState) -> str:
    return "resume" if view.status() is runs.RunStatus.RESUMABLE else "start fresh"


def _kv(label: str, value: str) -> text.Text:
    out = text.Text(no_wrap=True)
    out.append(f"{label:<11}", style=theme.TEXT_MUTED)
    out.append(value, style=theme.TEXT_PRIMARY)
    return out
