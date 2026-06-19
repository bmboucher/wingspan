"""The FLIGHT PLAN input loop — the worker side of the configurator screen.

:func:`run_configurator` opens a ``rich`` ``Live`` on the alternate screen, polls
keys without blocking (so the screen reflows on resize and the edit caret
animates), and dispatches each keypress against the :class:`state.ConfiguratorState`
until the user launches a run (returns the chosen :class:`config.RunConfig`) or
quits (returns ``None``). It builds no torch model itself — the network is only
constructed afterward by ``app._run_training`` — so nothing that can raise a CUDA
or load error runs inside the alternate-screen block.

:func:`build_initial_state` and :func:`dispatch` are the pure, console-free core
(state in, state/outcome out) so they can be unit-tested without a terminal.
"""

from __future__ import annotations

import pathlib
import sys
import time

import rich.console as rich_console
from rich import live

from wingspan import version
from wingspan.training import artifacts, config
from wingspan.training.configure import (
    fields,
    keys,
    runs,
    screen,
    state,
    user_defaults,
)

# Re-render on every handled key, otherwise on this heartbeat so the caret
# blinks and a resize reflows without repainting on every idle poll.
_HEARTBEAT_FRAMES = 4
# Selection jump for PageUp / PageDown.
_PAGE_JUMP = 5
# Characters accepted while typing into a numeric field.
_NUMERIC_CHARS = frozenset("0123456789.eE+-")
# Characters accepted while typing a per-layer width list (digits + separators).
_LAYERS_CHARS = frozenset("0123456789, ")
# Single-letter commands recognized in NAVIGATE mode.
_QUIT_CHARS = frozenset("qQ")
_START_CHARS = frozenset("sS")
_NEW_CHARS = frozenset("nN")
_ARCHIVE_CHARS = frozenset("aA")
_RESET_CHARS = frozenset("rR")
_SAVE_DEFAULTS_CHARS = frozenset("dD")


def run_configurator(
    initial: config.RunConfig,
    console: rich_console.Console,
    cuda_available: bool,
) -> config.RunConfig | None:
    """Run the FLIGHT PLAN screen. Returns the config to launch, or ``None`` if
    the user quit (or the terminal can't host a full-screen TUI)."""
    if not _interactive(console):
        _warn_not_interactive(console)
        return None
    view = build_initial_state(initial, cuda_available)
    return _run_loop(console, view)


def build_initial_state(
    initial: config.RunConfig, cuda_available: bool
) -> state.ConfiguratorState:
    """Inspect the target directory and seed the editor. When a readable run is
    already there, start from *its* saved settings (so the user tunes the actual
    run, not argparse defaults), keeping the directory they pointed at; for a
    fresh target, prefer the user's saved defaults file over factory defaults."""
    summary = runs.inspect_run(initial.run.checkpoint_dir)
    working, seeded = _seed_from_summary(initial, summary)
    seeded_from_user_defaults = False
    defaults_warning: str | None = None
    if not seeded:
        loaded = user_defaults.load_defaults(initial)
        if loaded.train_config is not None:
            working = loaded.train_config
            seeded_from_user_defaults = True
        defaults_warning = loaded.warning
    working = runs.align_era(summary, fields.reset_hidden_fields(working))
    view = state.ConfiguratorState(
        working=working,
        saved=summary.train_config,
        summary=summary,
        cuda_available=cuda_available,
        selected_attr=fields.editable_attrs()[0],
        seeded_from_saved=seeded,
        seeded_from_user_defaults=seeded_from_user_defaults,
    )
    if defaults_warning is not None:
        view.notify(state.MessageKind.WARN, defaults_warning)
    return view


def _seed_from_summary(
    current: config.RunConfig, summary: runs.RunSummary
) -> tuple[config.RunConfig, bool]:
    """Decide the editor's working config for an inspected directory: when it
    holds a run whose saved config still reads cleanly, seed from *those*
    settings (keeping the directory pointed at) so the user tunes the actual
    run rather than argparse defaults; otherwise keep ``current`` for a fresh /
    unreadable / empty target. Returns ``(working, seeded_from_saved)``. Shared
    by the initial build and the re-inspect after a directory change so the two
    never disagree.

    The saved config is seeded regardless of whether its architecture matches
    ``current``: gating on compatibility would discard the saved settings in
    exactly the case where they matter most (a run with a non-default
    architecture), leaving the editor on argparse defaults and reporting a
    spurious "architecture changed" the moment the screen opens. By always
    seeding, ``working`` equals the saved run on entry (so it reads RESUMABLE
    and nothing is marked changed); the INCOMPATIBLE verdict then appears only
    once the user actually edits an architecture field."""
    if (
        summary.exists
        and summary.readable
        and not summary.config_invalid
        and summary.train_config is not None
    ):
        saved = summary.train_config
        updated_run = saved.run.model_copy(
            update={"checkpoint_dir": current.run.checkpoint_dir}
        )
        seeded = saved.model_copy(update={"run": updated_run})
        return seeded, True
    return current, False


def dispatch(view: state.ConfiguratorState, event: keys.KeyEvent) -> state.Outcome:
    """Apply one keypress to ``view``; returns whether to continue, quit, or
    launch. Pure (no console / IO) so it is unit-testable."""
    if event.kind is keys.KeyKind.INTERRUPT:
        return state.Outcome.QUIT
    if view.mode is state.Mode.CONFIRM:
        return _dispatch_confirm(view, event)
    if view.mode is state.Mode.EDIT:
        return _dispatch_edit(view, event)
    return _dispatch_navigate(view, event)


###### PRIVATE #######

#### Live loop ####


def _run_loop(
    console: rich_console.Console, view: state.ConfiguratorState
) -> config.RunConfig | None:
    frame = 0
    last_render = -_HEARTBEAT_FRAMES
    try:
        with (
            live.Live(
                screen.build(view, frame),
                console=console,
                screen=True,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as display,
            keys.KeyReader() as reader,
        ):
            while True:
                if frame - last_render >= _HEARTBEAT_FRAMES:
                    display.update(screen.build(view, frame), refresh=True)
                    last_render = frame
                frame += 1
                event = reader.poll()
                if event is None:
                    continue
                outcome = _safe_dispatch(view, event)
                display.update(screen.build(view, frame), refresh=True)
                last_render = frame
                if outcome is state.Outcome.QUIT:
                    return None
                if outcome is state.Outcome.LAUNCH:
                    return view.working
    except KeyboardInterrupt:
        # An OS-level SIGINT (rather than a decoded \x03 byte) is still a clean
        # quit — the Live / KeyReader context managers have already restored the
        # terminal on the way out.
        return None


def _safe_dispatch(
    view: state.ConfiguratorState, event: keys.KeyEvent
) -> state.Outcome:
    """Dispatch a key, turning any unexpected failure into a status message so
    a stray error never unwinds through the alternate-screen Live."""
    try:
        return dispatch(view, event)
    except Exception as error:  # noqa: BLE001 — never strand the terminal
        view.mode = state.Mode.NAVIGATE
        view.confirm = None
        view.notify(state.MessageKind.ERROR, f"error: {error}")
        return state.Outcome.CONTINUE


def _interactive(console: rich_console.Console) -> bool:
    """Whether the console can host a full-screen alternate-buffer TUI. On POSIX
    a real stdin is also required: the key reader puts stdin into raw mode, which
    fails on a redirected / piped stdin (and Windows reads the console directly,
    so stdin redirection there is irrelevant)."""
    if not (console.is_terminal and not console.legacy_windows):
        return False
    return sys.platform == "win32" or sys.stdin.isatty()


def _warn_not_interactive(console: rich_console.Console) -> None:
    console.print(
        "[bold]FLIGHT PLAN[/bold] needs a VT-capable interactive terminal "
        "(e.g. Windows Terminal).\nRun without [bold]--config[/bold] to start "
        "training with the given flags, or launch from a real terminal."
    )


#### NAVIGATE mode ####


def _dispatch_navigate(
    view: state.ConfiguratorState, event: keys.KeyEvent
) -> state.Outcome:
    handlers = {
        keys.KeyKind.ESCAPE: lambda: state.Outcome.QUIT,
        keys.KeyKind.UP: lambda: _move_selection(view, -1),
        keys.KeyKind.DOWN: lambda: _move_selection(view, 1),
        keys.KeyKind.PAGE_UP: lambda: _move_selection(view, -_PAGE_JUMP),
        keys.KeyKind.PAGE_DOWN: lambda: _move_selection(view, _PAGE_JUMP),
        keys.KeyKind.HOME: lambda: _select_end(view, first=True),
        keys.KeyKind.END: lambda: _select_end(view, first=False),
        keys.KeyKind.LEFT: lambda: _apply_nudge(view, -1),
        keys.KeyKind.RIGHT: lambda: _apply_nudge(view, 1),
        keys.KeyKind.ENTER: lambda: _activate_selected(view),
    }
    handler = handlers.get(event.kind)
    if handler is not None:
        return handler()
    if event.kind is keys.KeyKind.CHAR:
        return _navigate_char(view, event.char)
    return state.Outcome.CONTINUE


def _navigate_char(view: state.ConfiguratorState, char: str) -> state.Outcome:
    if char in _QUIT_CHARS:
        return state.Outcome.QUIT
    if char in _START_CHARS:
        return _start_action(view)
    if char in _NEW_CHARS:
        return _new_run_action(view)
    if char in _ARCHIVE_CHARS:
        return _archive_action(view)
    if char in _RESET_CHARS:
        return _reset_action(view)
    if char in _SAVE_DEFAULTS_CHARS:
        return _save_defaults_action(view)
    spec = view.selected_spec()
    if char in _NUMERIC_CHARS and isinstance(
        spec, (fields.IntField, fields.FloatField)
    ):
        _begin_edit(view, spec, initial=char)
    elif char in _LAYERS_CHARS and isinstance(spec, fields.LayersField):
        _begin_edit(view, spec, initial=char)
    return state.Outcome.CONTINUE


def _move_selection(view: state.ConfiguratorState, delta: int) -> state.Outcome:
    attrs = fields.editable_attrs(view.working)
    index = attrs.index(view.selected_attr)
    view.selected_attr = attrs[min(max(index + delta, 0), len(attrs) - 1)]
    return state.Outcome.CONTINUE


def _select_end(view: state.ConfiguratorState, first: bool) -> state.Outcome:
    attrs = fields.editable_attrs(view.working)
    view.selected_attr = attrs[0] if first else attrs[-1]
    return state.Outcome.CONTINUE


def _snap_if_invisible(view: state.ConfiguratorState) -> None:
    # After a mutation that may change field visibility, snap the cursor to the
    # nearest visible field above the current position so navigation never lands
    # on a hidden spec. Works for any visibility toggle, not just head_layers_mode.
    attrs = fields.editable_attrs(view.working)
    if view.selected_attr in attrs:
        return
    all_attrs = fields.editable_attrs()
    pos = all_attrs.index(view.selected_attr)
    for attr in reversed(all_attrs[:pos]):
        if attr in attrs:
            view.selected_attr = attr
            return
    if attrs:
        view.selected_attr = attrs[0]


def _apply_nudge(view: state.ConfiguratorState, direction: int) -> state.Outcome:
    spec = view.selected_spec()
    if isinstance(spec, (fields.TextField, fields.PathField)):
        view.notify(state.MessageKind.INFO, "press enter to edit this field")
        return state.Outcome.CONTINUE
    if isinstance(spec, fields.BootstrapField):
        return _cycle_bootstrap(view, direction)
    updated, error = fields.nudge(view.working, spec, direction)
    if error is not None:
        view.notify(state.MessageKind.WARN, error)
        return state.Outcome.CONTINUE
    era_moved = _update_working(view, updated)
    view.message = None
    if era_moved:
        _notify_era_moved(view)
    _snap_if_invisible(view)
    return state.Outcome.CONTINUE


def _cycle_bootstrap(view: state.ConfiguratorState, direction: int) -> state.Outcome:
    """Cycle the bootstrap_opponent field through none → random → archived paths."""
    # Build the full choices list: fixed options first, then archives latest-first.
    archive_paths = [
        str(
            pathlib.Path(view.working.run.checkpoint_dir)
            / artifacts.ARCHIVE_SUBDIR
            / entry.label
            / artifacts.LAST_CKPT
        )
        for entry in reversed(view.summary.archives)
        if entry.has_checkpoint
    ]
    choices = ["none", "random"] + archive_paths
    current = str(fields.read_field(view.working, view.selected_spec()))
    index = choices.index(current) if current in choices else 0
    new_value = choices[(index + direction) % len(choices)]
    updated, error = fields.commit(view.working, view.selected_spec(), new_value)
    if error is not None:
        view.notify(state.MessageKind.WARN, error)
        return state.Outcome.CONTINUE
    era_moved = _update_working(view, updated)
    view.message = None
    if era_moved:
        _notify_era_moved(view)
    _snap_if_invisible(view)
    return state.Outcome.CONTINUE


def _activate_selected(view: state.ConfiguratorState) -> state.Outcome:
    spec = view.selected_spec()
    if isinstance(spec, fields.ChoiceField):
        return _apply_nudge(view, 1)  # Enter cycles a choice
    _begin_edit(view, spec, initial="")
    return state.Outcome.CONTINUE


def _begin_edit(
    view: state.ConfiguratorState, spec: fields.FieldSpec, initial: str
) -> None:
    view.mode = state.Mode.EDIT
    if initial:
        view.edit_buffer = initial
    elif isinstance(spec, fields.BootstrapField):
        # format_value trims BootstrapField paths to just "label/last.pt" for
        # display; we need the full stored path as the edit starting point so
        # that pressing Enter without typing doesn't truncate and break it.
        view.edit_buffer = str(fields.read_field(view.working, spec))
    else:
        view.edit_buffer = fields.format_value(view.working, spec)
    view.message = None


#### EDIT mode ####


def _dispatch_edit(
    view: state.ConfiguratorState, event: keys.KeyEvent
) -> state.Outcome:
    if event.kind is keys.KeyKind.ESCAPE:
        view.mode = state.Mode.NAVIGATE
        view.edit_buffer = ""
        view.notify(state.MessageKind.INFO, "edit cancelled")
        return state.Outcome.CONTINUE
    if event.kind is keys.KeyKind.ENTER:
        return _commit_edit(view)
    if event.kind is keys.KeyKind.BACKSPACE:
        view.edit_buffer = view.edit_buffer[:-1]
        return state.Outcome.CONTINUE
    if event.kind is keys.KeyKind.CHAR and _accepts_char(
        view.selected_spec(), event.char
    ):
        view.edit_buffer += event.char
    return state.Outcome.CONTINUE


def _commit_edit(view: state.ConfiguratorState) -> state.Outcome:
    spec = view.selected_spec()
    updated, error = fields.commit(view.working, spec, view.edit_buffer)
    if error is not None:
        view.notify(state.MessageKind.ERROR, error)
        return state.Outcome.CONTINUE
    changed_dir = updated.run.checkpoint_dir != view.working.run.checkpoint_dir
    era_moved = _update_working(view, updated)
    view.mode = state.Mode.NAVIGATE
    view.edit_buffer = ""
    _snap_if_invisible(view)
    if changed_dir:
        _reinspect(view)
    elif era_moved:
        _notify_era_moved(view)
    else:
        view.notify(
            state.MessageKind.INFO,
            f"set {spec.label} = {fields.format_value(updated, spec)}",
        )
    return state.Outcome.CONTINUE


def _accepts_char(spec: fields.FieldSpec, char: str) -> bool:
    if isinstance(spec, (fields.IntField, fields.FloatField)):
        return char in _NUMERIC_CHARS
    if isinstance(spec, fields.LayersField):
        return char in _LAYERS_CHARS
    return char.isprintable()


#### Actions ####


def _start_action(view: state.ConfiguratorState) -> state.Outcome:
    status = view.status()
    if status is runs.RunStatus.EMPTY:
        return _launch(view, resume=False)
    if status is runs.RunStatus.RESUMABLE:
        return _launch(view, resume=True)
    view.confirm = _fresh_confirm(view)  # incompatible / unreadable — must restart
    view.mode = state.Mode.CONFIRM
    return state.Outcome.CONTINUE


def _new_run_action(view: state.ConfiguratorState) -> state.Outcome:
    if view.summary.exists:
        view.confirm = _fresh_confirm(view)
        view.mode = state.Mode.CONFIRM
        return state.Outcome.CONTINUE
    return _launch(view, resume=False)


def _archive_action(view: state.ConfiguratorState) -> state.Outcome:
    if not view.summary.exists:
        view.notify(state.MessageKind.INFO, "nothing to archive — directory is empty")
        return state.Outcome.CONTINUE
    view.confirm = _archive_only_confirm(view)
    view.mode = state.Mode.CONFIRM
    return state.Outcome.CONTINUE


def _reset_action(view: state.ConfiguratorState) -> state.Outcome:
    has_user_defaults = (
        user_defaults.load_defaults(view.working).train_config is not None
    )
    view.confirm = _reset_confirm(has_user_defaults)
    view.mode = state.Mode.CONFIRM
    return state.Outcome.CONTINUE


def _save_defaults_action(view: state.ConfiguratorState) -> state.Outcome:
    try:
        path = user_defaults.save_defaults(view.working)
    except OSError as error:
        view.notify(state.MessageKind.ERROR, f"could not save defaults — {error}")
        return state.Outcome.CONTINUE
    view.notify(state.MessageKind.SUCCESS, f"saved as defaults → {path.name}")
    return state.Outcome.CONTINUE


def _launch(view: state.ConfiguratorState, resume: bool) -> state.Outcome:
    # A fresh launch must never inherit a stale era from a saved-run seed; the
    # loop's ``adopt_checkpoint_era`` is the backstop, but the returned config
    # should already be what the screen claimed would launch.
    cfg = view.working
    if not resume and cfg.encoding_version != version.MODEL_VERSION:
        cfg = config.with_encoding_version(cfg, version.MODEL_VERSION)

    # Validate cross-field constraints before handing off to the training loop.
    problems = config.validate_launchable(cfg)
    if problems:
        view.notify(state.MessageKind.WARN, "cannot launch — " + "; ".join(problems))
        return state.Outcome.CONTINUE

    view.working = cfg.model_copy(
        update={"run": cfg.run.model_copy(update={"resume": resume})}
    )
    return state.Outcome.LAUNCH


#### CONFIRM mode ####


def _dispatch_confirm(
    view: state.ConfiguratorState, event: keys.KeyEvent
) -> state.Outcome:
    prompt = view.confirm
    if prompt is None:  # defensive — never reached with mode CONFIRM
        view.mode = state.Mode.NAVIGATE
        return state.Outcome.CONTINUE
    if event.kind is keys.KeyKind.ESCAPE:
        return _apply_confirm(view, state.ConfirmAction.CANCEL)
    if event.kind is keys.KeyKind.ENTER:
        option = prompt.option_for(prompt.default_key)
        return _apply_confirm(view, option.action) if option else state.Outcome.CONTINUE
    if event.kind is keys.KeyKind.CHAR:
        option = prompt.option_for(event.char.lower())
        if option is not None:
            return _apply_confirm(view, option.action)
    return state.Outcome.CONTINUE


def _apply_confirm(
    view: state.ConfiguratorState, action: state.ConfirmAction
) -> state.Outcome:
    if action is state.ConfirmAction.CANCEL:
        view.mode = state.Mode.NAVIGATE
        view.confirm = None
        view.notify(state.MessageKind.INFO, "cancelled")
        return state.Outcome.CONTINUE
    if action is state.ConfirmAction.ARCHIVE_THEN_FRESH:
        return _archive_then(view, launch=True)
    if action is state.ConfirmAction.ARCHIVE_ONLY:
        return _archive_then(view, launch=False)
    if action is state.ConfirmAction.RESET_TO_DEFAULTS:
        return _apply_reset(view, config.RunConfig(), "factory defaults")
    if action is state.ConfirmAction.RESET_TO_USER_DEFAULTS:
        return _apply_reset_to_user_defaults(view)
    return _overwrite_then_fresh(view)


def _archive_then(view: state.ConfiguratorState, launch: bool) -> state.Outcome:
    label = runs.default_archive_label(view.summary, _timestamp())
    result = runs.archive_run(view.working.run.checkpoint_dir, label)
    view.mode = state.Mode.NAVIGATE
    view.confirm = None
    if not result.ok:
        view.notify(state.MessageKind.ERROR, f"archive failed — {result.errors[0]}")
        return state.Outcome.CONTINUE
    if launch:
        view.notify(state.MessageKind.SUCCESS, f"archived → {label}")
        return _launch(view, resume=False)
    _reinspect(view)
    view.notify(
        state.MessageKind.SUCCESS, f"archived {len(result.moved)} files → {label}"
    )
    return state.Outcome.CONTINUE


def _overwrite_then_fresh(view: state.ConfiguratorState) -> state.Outcome:
    removed = runs.clear_run(view.working.run.checkpoint_dir)
    view.confirm = None
    view.notify(
        state.MessageKind.WARN, f"removed {len(removed)} files — starting fresh"
    )
    return _launch(view, resume=False)


def _fresh_confirm(view: state.ConfiguratorState) -> state.ConfirmPrompt:
    summary = view.summary
    iteration = summary.iteration or 0
    games = summary.total_games or 0
    best = (
        f"{summary.best_win_rate * 100:.1f}%"
        if summary.best_win_rate is not None
        else "—"
    )
    lines = [
        f"A run already exists in {view.working.run.checkpoint_dir}/:",
        f"  iter {iteration:04d} · {games:,} games · best {best}",
        "",
        "Archive moves it to archive/<label>/ — kept and recoverable.",
        "Overwrite deletes last.pt / best.pt / metrics — unrecoverable.",
    ]
    if view.working.encoding_version != version.MODEL_VERSION:
        lines.append(f"The new run starts at the current era {version.MODEL_VERSION}.")
    return state.ConfirmPrompt(
        title="START A NEW RUN",
        lines=lines,
        options=[
            state.ConfirmOption(
                key="a",
                label="archive & start",
                action=state.ConfirmAction.ARCHIVE_THEN_FRESH,
            ),
            state.ConfirmOption(
                key="o",
                label="overwrite & start",
                action=state.ConfirmAction.OVERWRITE_THEN_FRESH,
                danger=True,
            ),
            state.ConfirmOption(
                key="c", label="cancel", action=state.ConfirmAction.CANCEL
            ),
        ],
        default_key="a",
    )


def _archive_only_confirm(view: state.ConfiguratorState) -> state.ConfirmPrompt:
    summary = view.summary
    label_hint = runs.default_archive_label(summary, "<time>")
    return state.ConfirmPrompt(
        title="ARCHIVE THIS RUN",
        lines=[
            f"Move the run in {view.working.run.checkpoint_dir}/ to:",
            f"  archive/{label_hint}",
            "",
            "The directory is then clean; you stay on this screen.",
        ],
        options=[
            state.ConfirmOption(
                key="a", label="archive", action=state.ConfirmAction.ARCHIVE_ONLY
            ),
            state.ConfirmOption(
                key="c", label="cancel", action=state.ConfirmAction.CANCEL
            ),
        ],
        default_key="a",
    )


def _reset_confirm(has_user_defaults: bool) -> state.ConfirmPrompt:
    lines = [
        "Restore all fields to defaults.",
        "Your current edits will be lost.",
    ]
    options: list[state.ConfirmOption] = []
    if has_user_defaults:
        lines.append(f"User defaults: {user_defaults.DEFAULTS_FILENAME}")
        options.append(
            state.ConfirmOption(
                key="u",
                label="user defaults",
                action=state.ConfirmAction.RESET_TO_USER_DEFAULTS,
            )
        )
    options.append(
        state.ConfirmOption(
            key="f",
            label="factory defaults",
            action=state.ConfirmAction.RESET_TO_DEFAULTS,
            danger=True,
        )
    )
    options.append(
        state.ConfirmOption(key="c", label="cancel", action=state.ConfirmAction.CANCEL)
    )
    return state.ConfirmPrompt(
        title="RESET TO DEFAULTS",
        lines=lines,
        options=options,
        default_key="c",
    )


def _apply_reset_to_user_defaults(view: state.ConfiguratorState) -> state.Outcome:
    # Re-load at apply time: the file could have gone bad between the prompt
    # and the keypress, and a stale prompt must not corrupt the working config.
    loaded = user_defaults.load_defaults(view.working)
    if loaded.train_config is None:
        view.mode = state.Mode.NAVIGATE
        view.confirm = None
        view.notify(
            state.MessageKind.WARN,
            loaded.warning or "no saved defaults — keeping current settings",
        )
        return state.Outcome.CONTINUE
    outcome = _apply_reset(view, loaded.train_config, "user defaults")
    view.seeded_from_user_defaults = True
    return outcome


def _apply_reset(
    view: state.ConfiguratorState, defaults: config.RunConfig, label: str
) -> state.Outcome:
    updated_run = defaults.run.model_copy(
        update={"checkpoint_dir": view.working.run.checkpoint_dir}
    )
    _update_working(view, defaults.model_copy(update={"run": updated_run}))
    view.seeded_from_saved = False
    view.seeded_from_user_defaults = False
    view.mode = state.Mode.NAVIGATE
    view.confirm = None
    view.notify(state.MessageKind.SUCCESS, f"reset to {label}")
    return state.Outcome.CONTINUE


#### Shared ####


def _update_working(view: state.ConfiguratorState, updated: config.RunConfig) -> bool:
    """Install a mutated working config: reset newly-hidden fields, then
    re-align the era against the inspected run (the saved run's era while the
    architecture still matches it, the live MODEL_VERSION otherwise). Returns
    whether the era moved, so callers can surface a footer notice instead of
    their normal message."""
    aligned = runs.align_era(view.summary, fields.reset_hidden_fields(updated))
    era_moved = aligned.encoding_version != view.working.encoding_version
    view.working = aligned
    return era_moved


def _notify_era_moved(view: state.ConfiguratorState) -> None:
    """Footer notice for an era move: an edit either broke compatibility with
    the saved run (a fresh run at the live era) or restored it (re-pinned)."""
    era = view.working.encoding_version
    if view.status() is runs.RunStatus.RESUMABLE:
        view.notify(
            state.MessageKind.WARN,
            f"matches the saved run again — era {era} (resume)",
        )
    else:
        view.notify(
            state.MessageKind.WARN,
            f"architecture changed — a fresh run will start at era {era}",
        )


def _reinspect(view: state.ConfiguratorState) -> None:
    """Re-read the (possibly newly-pointed-at) directory; refresh the summary
    and the saved-config baseline without disturbing the user's working edits."""
    view.summary = runs.inspect_run(view.working.run.checkpoint_dir)
    view.saved = view.summary.train_config
    # Re-run the same seeding decision the initial build used, so the header,
    # the changed-field markers, and what Start does all stay consistent with
    # the newly-inspected directory (e.g. pointing at a compatible run re-seeds
    # its saved settings; archiving away the run drops the "resumed" framing).
    view.working, view.seeded_from_saved = _seed_from_summary(
        view.working, view.summary
    )
    # The era is a property of the newly-inspected directory, not of the edits
    # carried over: re-align so e.g. pointing an era-pinned working config at
    # an empty directory un-pins it to the live version.
    view.working = runs.align_era(view.summary, view.working)
    view.notify(state.MessageKind.INFO, f"inspected {view.working.run.checkpoint_dir}/")


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
