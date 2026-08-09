"""Interactive input widgets for the aid session: typeahead card lookup and
numeric per-field counts entry.

Three layers, each testable independently:

1. **Pure reducers/renderers** (:func:`step_typeahead`, :func:`step_counts`,
   :func:`render_typeahead_frame`, :func:`render_counts_frame`) take a widget
   state plus one decoded keystroke and return the next state -- no I/O, so
   every keystroke path is a plain unit test.
2. **Drivers** (:func:`run_typeahead`, :func:`run_counts`) thread an
   injectable event iterator and frame-drawing callback through the
   reducers, so a full widget session can be scripted and asserted on
   headlessly (no real terminal needed).
3. **Tty shells** (:func:`typeahead_pick`, :func:`counts_entry`) are thin
   wrappers that open a real :class:`wingspan.training.configure.keys.KeyReader`
   and feed it to the layer-2 drivers. Callers branch on
   ``wingspan.aid.console.Console.supports_interactive()`` before reaching
   these -- a scripted test console never does, so this layer is deliberately
   kept free of any logic beyond wiring, since it is the one layer a headless
   test suite cannot exercise.
"""

from __future__ import annotations

import collections.abc

from wingspan.agents import interactive
from wingspan.aid import console as console_module
from wingspan.aid import models
from wingspan.training.configure import keys

# How many matches a typeahead frame shows before truncating to a
# "... N more" line.
MAX_VISIBLE_MATCHES = 8

# The in-place frame-renderer contract: given the frame's lines and the
# number of lines drawn last time, redraw in place and return the new line
# count. `wingspan.agents.interactive.draw_frame` satisfies this; tests
# inject a recording stand-in.
type DrawFn = collections.abc.Callable[[list[str], int], int]


def step_typeahead(
    state: models.TypeaheadState,
    n_matches: int,
    event: keys.KeyEvent,
    *,
    allow_blank: bool,
) -> tuple[models.TypeaheadState, models.TypeaheadOutcome]:
    """Apply one decoded keystroke to a typeahead widget's state.

    ``n_matches`` is the number of matches for ``state.query`` *before* this
    event (the driver recomputes matches for the resulting query itself).
    Returns a new state -- ``state`` is never mutated -- with ``highlight``
    always clamped/wrapped into the currently visible match range, plus the
    :class:`~wingspan.aid.models.TypeaheadOutcome` the caller should act on.
    Raises ``KeyboardInterrupt`` on an INTERRUPT event (the aid entry point
    catches it as a session abort).
    """
    visible = min(n_matches, MAX_VISIBLE_MATCHES)
    query = state.query
    highlight = state.highlight
    outcome = models.TypeaheadOutcome.CONTINUE

    if event.kind is keys.KeyKind.CHAR:
        query = query + event.char
        highlight = 0
    elif event.kind is keys.KeyKind.BACKSPACE:
        query = query[:-1]
        highlight = 0
    elif event.kind is keys.KeyKind.UP:
        highlight = highlight - 1
    elif event.kind is keys.KeyKind.DOWN:
        highlight = highlight + 1
    elif event.kind is keys.KeyKind.ENTER:
        if visible > 0:
            outcome = models.TypeaheadOutcome.ACCEPT
        elif allow_blank and query == "":
            outcome = models.TypeaheadOutcome.ACCEPT_BLANK
    elif event.kind is keys.KeyKind.ESCAPE:
        query = ""
        highlight = 0
    elif event.kind is keys.KeyKind.INTERRUPT:
        raise KeyboardInterrupt

    new_state = models.TypeaheadState(
        query=query, highlight=_clamp_highlight(highlight, visible)
    )
    return new_state, outcome


def step_counts(
    state: models.CountsState,
    target_total: int,
    event: keys.KeyEvent,
    *,
    field_caps: collections.abc.Sequence[int] | None = None,
) -> tuple[models.CountsState, bool]:
    """Apply one decoded keystroke to a counts widget's state.

    ``field_caps[focus]`` caps the focused field's value when given, else
    ``target_total`` is the cap. Returns a new state -- ``state`` is never
    mutated -- and whether this event was an ENTER that accepted the entry
    (only when the values sum to ``target_total``; every other event returns
    ``False``). Raises ``KeyboardInterrupt`` on an INTERRUPT event.
    """
    n_fields = len(state.values)
    focus = state.focus
    values = list(state.values)
    accepted = False
    cap = field_caps[focus] if field_caps is not None else target_total

    if event.kind is keys.KeyKind.LEFT:
        focus = (focus - 1) % n_fields
    elif event.kind is keys.KeyKind.RIGHT:
        focus = (focus + 1) % n_fields
    elif event.kind is keys.KeyKind.UP:
        values[focus] = min(values[focus] + 1, cap)
    elif event.kind is keys.KeyKind.DOWN:
        values[focus] = max(values[focus] - 1, 0)
    elif event.kind is keys.KeyKind.CHAR:
        if event.char.isdigit():
            values[focus] = min(int(event.char), cap)
    elif event.kind is keys.KeyKind.BACKSPACE:
        values[focus] = 0
    elif event.kind is keys.KeyKind.ESCAPE:
        values = [0] * n_fields
        focus = 0
    elif event.kind is keys.KeyKind.ENTER:
        accepted = sum(values) == target_total
    elif event.kind is keys.KeyKind.INTERRUPT:
        raise KeyboardInterrupt

    return models.CountsState(values=values, focus=focus), accepted


def render_typeahead_frame(
    prompt: str,
    state: models.TypeaheadState,
    match_labels: collections.abc.Sequence[str],
    *,
    allow_blank: bool,
) -> list[str]:
    """Render a typeahead widget's current state as plain-ASCII frame lines.

    Line 1 is ``prompt``, line 2 echoes the query. With no matches, a single
    hint line follows (worded differently for an empty vs. non-empty query).
    Otherwise up to :data:`MAX_VISIBLE_MATCHES` match lines follow, the
    highlighted one prefixed ``"* "`` and the rest ``"  "``, with a trailing
    ``"... N more"`` line when the match list was truncated.
    """
    lines = [prompt, "> " + state.query]
    if not match_labels:
        if state.query == "":
            hint = (
                "  (type to search; Enter = face-down/unknown)"
                if allow_blank
                else "  (type to search)"
            )
        else:
            hint = "  (no matches — Backspace or Esc to edit)"
        lines.append(hint)
        return lines

    visible_labels = match_labels[:MAX_VISIBLE_MATCHES]
    for index, label in enumerate(visible_labels):
        prefix = "* " if index == state.highlight else "  "
        lines.append(prefix + label)
    hidden_count = len(match_labels) - len(visible_labels)
    if hidden_count > 0:
        lines.append(f"  ... {hidden_count} more (keep typing)")
    return lines


def render_counts_frame(
    prompt: str,
    labels: collections.abc.Sequence[str],
    state: models.CountsState,
    target_total: int,
) -> list[str]:
    """Render a counts widget's current state as plain-ASCII frame lines.

    Line 1 is ``prompt``. Line 2 lists every field as ``label:value``, the
    focused field bracketed (``[seed:2]``) and the rest space-padded
    (`` seed:2 ``). Line 3 shows the running total against ``target_total``,
    worded differently once they match.
    """
    fields = [
        f"[{label}:{value}]" if index == state.focus else f" {label}:{value} "
        for index, (label, value) in enumerate(zip(labels, state.values))
    ]
    total = sum(state.values)
    if total == target_total:
        total_line = f"total {total}/{target_total} — Enter to accept"
    else:
        total_line = (
            f"total {total}/{target_total} (Enter accepts when the total matches)"
        )
    return [prompt, " ".join(fields), total_line]


def run_typeahead[T](
    events: collections.abc.Iterator[keys.KeyEvent],
    prompt: str,
    find: collections.abc.Callable[[str], list[T]],
    render: collections.abc.Callable[[T], str],
    draw: DrawFn,
    *,
    allow_blank: bool = False,
    initial: collections.abc.Sequence[T] | None = None,
) -> T | None:
    """Drive a typeahead widget to completion from an injected event stream.

    Matches for the empty query are ``initial`` (or none, if ``initial`` is
    ``None``); a non-empty query calls ``find(query)``. Draws the initial
    frame before consuming any event, then redraws after every step. Returns
    the accepted match, or ``None`` on an ACCEPT_BLANK. Raises
    ``RuntimeError`` if ``events`` is exhausted without a selection -- a tty
    shell feeds an endless stream, so this only fires from a buggy script.
    """
    state = models.TypeaheadState()
    matches = _typeahead_matches(state.query, find, initial)
    prev_line_count = draw(
        render_typeahead_frame(
            prompt, state, [render(match) for match in matches], allow_blank=allow_blank
        ),
        0,
    )
    for event in events:
        state, outcome = step_typeahead(
            state, len(matches), event, allow_blank=allow_blank
        )
        if outcome is models.TypeaheadOutcome.ACCEPT:
            return matches[state.highlight]
        if outcome is models.TypeaheadOutcome.ACCEPT_BLANK:
            return None
        matches = _typeahead_matches(state.query, find, initial)
        prev_line_count = draw(
            render_typeahead_frame(
                prompt,
                state,
                [render(match) for match in matches],
                allow_blank=allow_blank,
            ),
            prev_line_count,
        )
    raise RuntimeError("typeahead event stream ended without a selection")


def run_counts(
    events: collections.abc.Iterator[keys.KeyEvent],
    prompt: str,
    labels: collections.abc.Sequence[str],
    target_total: int,
    draw: DrawFn,
    *,
    field_caps: collections.abc.Sequence[int] | None = None,
) -> list[int]:
    """Drive a counts widget to completion from an injected event stream.

    Starts from all-zero values with the first field focused, drawing the
    initial frame before consuming any event and redrawing after every step.
    Returns the accepted values on an ENTER whose sum matches
    ``target_total``. Raises ``RuntimeError`` if ``events`` is exhausted
    without an acceptance -- a tty shell feeds an endless stream, so this
    only fires from a buggy script.
    """
    state = models.CountsState(values=[0] * len(labels))
    prev_line_count = draw(render_counts_frame(prompt, labels, state, target_total), 0)
    for event in events:
        state, accepted = step_counts(state, target_total, event, field_caps=field_caps)
        if accepted:
            return list(state.values)
        prev_line_count = draw(
            render_counts_frame(prompt, labels, state, target_total), prev_line_count
        )
    raise RuntimeError("counts event stream ended without a selection")


def typeahead_pick[T](
    con: console_module.Console,
    prompt: str,
    find: collections.abc.Callable[[str], list[T]],
    render: collections.abc.Callable[[T], str],
    *,
    allow_blank: bool = False,
    initial: collections.abc.Sequence[T] | None = None,
) -> T | None:
    """Run a typeahead widget on the real terminal and echo the selection.

    Thin tty shell over :func:`run_typeahead`: opens a raw
    :class:`~wingspan.training.configure.keys.KeyReader`, feeds it an endless
    poll loop, and draws with
    :func:`wingspan.agents.interactive.draw_frame`. All widget logic lives
    in :func:`run_typeahead`/:func:`step_typeahead`; this function is only
    wiring.
    """
    interactive.enable_ansi()
    with keys.KeyReader() as reader:
        picked = run_typeahead(
            _tty_events(reader),
            prompt,
            find,
            render,
            interactive.draw_frame,
            allow_blank=allow_blank,
            initial=initial,
        )
    con.say(
        f"selected: {render(picked)}"
        if picked is not None
        else "selected: (face-down/unknown)"
    )
    return picked


def counts_entry(
    con: console_module.Console,
    prompt: str,
    labels: collections.abc.Sequence[str],
    target_total: int,
    *,
    field_caps: collections.abc.Sequence[int] | None = None,
) -> list[int]:
    """Run a counts widget on the real terminal and echo the entered values.

    Thin tty shell over :func:`run_counts`: opens a raw
    :class:`~wingspan.training.configure.keys.KeyReader`, feeds it an endless
    poll loop, and draws with
    :func:`wingspan.agents.interactive.draw_frame`. All widget logic lives
    in :func:`run_counts`/:func:`step_counts`; this function is only wiring.
    """
    interactive.enable_ansi()
    with keys.KeyReader() as reader:
        values = run_counts(
            _tty_events(reader),
            prompt,
            labels,
            target_total,
            interactive.draw_frame,
            field_caps=field_caps,
        )
    entries = " ".join(f"{label}:{value}" for label, value in zip(labels, values))
    con.say(f"entered: {entries}")
    return values


###### PRIVATE #######


def _clamp_highlight(highlight: int, visible: int) -> int:
    """Wrap ``highlight`` into ``[0, visible)``, or 0 when nothing is visible."""
    if visible <= 0:
        return 0
    return highlight % visible


def _typeahead_matches[T](
    query: str,
    find: collections.abc.Callable[[str], list[T]],
    initial: collections.abc.Sequence[T] | None,
) -> list[T]:
    """The current match list for ``query``: ``initial`` (or none) when
    blank, else ``find(query)``."""
    if query == "":
        return list(initial) if initial is not None else []
    return find(query)


def _tty_events(reader: keys.KeyReader) -> collections.abc.Iterator[keys.KeyEvent]:
    """Endless generator of decoded keypresses from ``reader``, skipping the
    ``None`` results ``poll()`` returns between keystrokes."""
    while True:
        event = reader.poll()
        if event is not None:
            yield event
