"""Terminal selection-form widget for the interactive CLI.

``select_form`` renders one or more sections on a single screen: each section
is either a checkbox group (toggle any subset, optionally capped at an exact
count) or a radio group (exactly one selected). The user navigates every
section's entries with the arrow keys, toggles/selects the focused entry with
the space bar, and confirms the whole form with Enter. This lets a single
screen host, say, "keep any subset of these birds AND pick exactly one bonus
card" as one decision.

The widget renders and re-renders an in-place frame using ANSI cursor
movement; on a non-interactive stream (piped stdin/stdout) it falls back to a
section-by-section numeric prompt so scripted play still works.

The widget is presentation-only: sections carry pre-rendered option strings
and ``select_form`` returns the chosen option indices per section, leaving the
caller to map those back onto whatever model it is assembling.
"""

from __future__ import annotations

import enum
import sys
import typing

import pydantic

# A normalized keypress token. Raw escape sequences / Windows scan codes are
# collapsed onto these by ``_read_key`` so the form loop stays platform-blind.
type Key = typing.Literal["up", "down", "space", "enter", "other"]

# ANSI: move cursor up N rows, then erase from the cursor to end of screen.
# Used to redraw a multi-line frame in place without scrolling the terminal.
_CURSOR_UP = "\x1b[{n}A"
_CLEAR_BELOW = "\x1b[0J"

# ANSI green, applied to a selected entry's lines so the current picks stand out.
_GREEN = "\x1b[32m"
_RESET = "\x1b[0m"

# Windows console flag (and the std-output handle id) needed to make the
# console interpret the ANSI sequences above; enabled once per process.
_STD_OUTPUT_HANDLE = -11
_ENABLE_VT_PROCESSING = 0x0004
_ansi_enabled = False

_INSTRUCTIONS = "  ↑/↓ move · space select · enter accept"

# Arrow-key lookups: Windows getwch scan codes vs. xterm escape tails.
_WIN_ARROWS: dict[str, Key] = {"H": "up", "P": "down", "K": "other", "M": "other"}
_UNIX_ARROWS: dict[str, Key] = {"[A": "up", "[B": "down", "[C": "other", "[D": "other"}


class Mode(enum.StrEnum):
    """How a section's entries may be selected."""

    MULTI = "multi"  # checkbox: toggle any subset (optionally capped)
    SINGLE = "single"  # radio: exactly one entry selected


class Section(pydantic.BaseModel):
    """One group of options within a :func:`select_form` screen.

    ``options`` are pre-rendered display strings (each may span multiple lines;
    continuation lines are indented to align under the first). For a ``MULTI``
    section ``required_count`` pins the exact number that must be checked and
    doubles as a hard cap — the widget refuses to check more than that many;
    leave it ``None`` to allow any subset (including none). A ``SINGLE`` section
    always requires exactly one selection and ignores ``required_count``.
    """

    title: str = ""
    options: list[str]
    mode: Mode = Mode.MULTI
    required_count: int | None = None


def select_form(
    sections: typing.Sequence[Section],
    *,
    header: str,
    instructions: str = _INSTRUCTIONS,
    live_options: typing.Callable[[list[set[int]]], list[list[str]]] | None = None,
    live_footer: typing.Callable[[list[set[int]]], list[str]] | None = None,
) -> list[list[int]]:
    """Resolve a multi-section selection screen.

    Returns one ascending index list per section (parallel to ``sections``).
    ``SINGLE`` sections default to their first entry so the form is always
    confirmable; Enter is honoured only once every section's constraint is met.

    ``live_options`` lets option *text* react to the current selection: it is
    called each frame with the live ``selections`` and returns the display
    strings for every section (one inner list per section). The option *counts*
    must match ``sections`` exactly — only the rendered text may change. When
    omitted, the static ``Section.options`` are shown. The fallback (non-tty)
    path always uses the static text.

    ``live_footer`` lets the caller render extra informational lines below the
    options (above the status line), recomputed each frame from the live
    ``selections`` — e.g. a "Can Play: ..." summary that reacts to the current
    picks. Ignored on the fallback (non-tty) path.
    """
    if not _interactive_supported():
        return _select_form_fallback(sections, header)

    enable_ansi()
    selections = [_initial_selection(sec) for sec in sections]
    positions = [
        (s, i) for s, sec in enumerate(sections) for i in range(len(sec.options))
    ]
    if not positions:
        return [sorted(sel) for sel in selections]

    focus = 0
    drawn_lines = 0

    # Redraw the frame, block for one key, mutate selection/focus, repeat.
    while True:
        option_texts = (
            live_options(selections)
            if live_options is not None
            else [list(sec.options) for sec in sections]
        )
        footer = live_footer(selections) if live_footer is not None else []
        status = _form_status(sections, selections)
        frame = _form_frame(
            sections,
            option_texts,
            selections,
            positions,
            focus,
            header,
            footer,
            status,
            instructions,
        )
        drawn_lines = _draw(frame, drawn_lines)
        key = _read_key()

        if key == "enter" and _form_accepts(sections, selections):
            return [sorted(sel) for sel in selections]
        if key == "up":
            focus = (focus - 1) % len(positions)
        elif key == "down":
            focus = (focus + 1) % len(positions)
        elif key == "space":
            _toggle(sections, selections, positions[focus])


###### PRIVATE #######

#### Selection state ####


def _initial_selection(sec: Section) -> set[int]:
    """Starting selection for a section (radios default to their first entry)."""
    if sec.mode is Mode.SINGLE and sec.options:
        return {0}
    return set()


def _toggle(
    sections: typing.Sequence[Section],
    selections: list[set[int]],
    pos: tuple[int, int],
) -> None:
    """Apply a space-bar press to the focused ``(section, option)`` entry."""
    section_idx, option_idx = pos
    sec = sections[section_idx]
    sel = selections[section_idx]
    if sec.mode is Mode.SINGLE:
        selections[section_idx] = {option_idx}
        return
    # MULTI: un-check freely; only check when below the required-count cap.
    if option_idx in sel:
        sel.discard(option_idx)
    elif sec.required_count is None or len(sel) < sec.required_count:
        sel.add(option_idx)


def _form_accepts(
    sections: typing.Sequence[Section],
    selections: typing.Sequence[set[int]],
) -> bool:
    """Whether every section's selection constraint is currently satisfied."""
    for s, sec in enumerate(sections):
        count = len(selections[s])
        if sec.mode is Mode.SINGLE and count != 1:
            return False
        if (
            sec.mode is Mode.MULTI
            and sec.required_count is not None
            and count != sec.required_count
        ):
            return False
    return True


def _form_status(
    sections: typing.Sequence[Section],
    selections: typing.Sequence[set[int]],
) -> str:
    """One-line status: outstanding requirements, or a ready-to-accept hint."""
    issues: list[str] = []
    for s, sec in enumerate(sections):
        count = len(selections[s])
        if sec.mode is Mode.SINGLE and count != 1:
            issues.append("choose one")
        elif (
            sec.mode is Mode.MULTI
            and sec.required_count is not None
            and count != sec.required_count
        ):
            issues.append(f"select {sec.required_count - count} more")
    return "  enter to accept" if not issues else "  " + " · ".join(issues)


#### Frame construction ####


def _form_frame(
    sections: typing.Sequence[Section],
    option_texts: typing.Sequence[typing.Sequence[str]],
    selections: typing.Sequence[set[int]],
    positions: typing.Sequence[tuple[int, int]],
    focus: int,
    header: str,
    footer: typing.Sequence[str],
    status: str,
    instructions: str,
) -> list[str]:
    """Build the full form frame as a list of physical lines.

    ``option_texts`` carries the per-section display strings to render (which
    may differ from ``Section.options`` when a live renderer is in play); it is
    parallel to ``sections`` in both section and option count. ``footer`` lines
    are rendered below every section, just above the status line.
    """
    focus_section, focus_option = positions[focus]
    lines = [header, ""]
    for s, sec in enumerate(sections):
        if sec.title:
            lines.append(sec.title)
        for i, option in enumerate(option_texts[s]):
            cursor = ">" if (s == focus_section and i == focus_option) else " "
            selected = i in selections[s]
            prefix = _entry_prefix(cursor, sec.mode, selected)
            entry_lines = _with_prefix(prefix, option)
            if selected:
                entry_lines = [f"{_GREEN}{line}{_RESET}" for line in entry_lines]
            lines.extend(entry_lines)
        lines.append("")
    lines.extend(footer)
    lines.extend([status, instructions])
    return lines


def _entry_prefix(cursor: str, mode: Mode, selected: bool) -> str:
    """The ``> [x] `` / ``> (*) `` marker that leads an option's first line."""
    if mode is Mode.SINGLE:
        return f"{cursor} ({'*' if selected else ' '}) "
    return f"{cursor} [{'x' if selected else ' '}] "


def _with_prefix(prefix: str, text: str) -> list[str]:
    """Prefix the first line of ``text`` with ``prefix``; align continuations.

    A multi-line option (e.g. a bird stat line plus a power line) keeps its
    body indented flush under the first line's content.
    """
    body = text.split("\n")
    continuation = " " * len(prefix)
    return [prefix + body[0]] + [continuation + line for line in body[1:]]


#### Terminal I/O ####


def _draw(lines: list[str], prev_line_count: int) -> int:
    """Render ``lines`` in place, overwriting ``prev_line_count`` prior rows.

    Returns the number of lines drawn so the next call knows how far up to
    move the cursor before redrawing.
    """
    chunks: list[str] = []
    if prev_line_count:
        chunks.append(_CURSOR_UP.format(n=prev_line_count))
        chunks.append(_CLEAR_BELOW)
    chunks.append("\n".join(lines))
    sys.stdout.write("".join(chunks) + "\n")
    sys.stdout.flush()
    return len(lines)


def _read_key() -> Key:
    """Block for a single keypress and return a normalized :data:`Key`.

    Reads a raw (unbuffered, un-echoed) key on both Windows (``msvcrt``) and
    POSIX (``termios``/``tty``) terminals, collapsing arrow-key scan codes and
    escape sequences onto the small :data:`Key` vocabulary.
    """
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return _WIN_ARROWS.get(msvcrt.getwch(), "other")
        return _normalize_ascii(ch)
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                return _UNIX_ARROWS.get(sys.stdin.read(2), "other")
            return _normalize_ascii(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _normalize_ascii(ch: str) -> Key:
    """Map a single printable/control character onto a :data:`Key` token."""
    if ch in ("\r", "\n"):
        return "enter"
    if ch == " ":
        return "space"
    if ch in ("\x03", "\x04", "\x1b"):  # Ctrl-C / Ctrl-D / bare Esc
        raise KeyboardInterrupt
    return "other"


def _interactive_supported() -> bool:
    """Whether both stdin and stdout are real terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def enable_ansi() -> None:
    """Enable ANSI escape processing on the Windows console (once per run).

    Idempotent and a no-op off Windows; safe to call from any agent that
    emits ANSI colors (the selection form and the colored power text both
    rely on it).
    """
    global _ansi_enabled
    if _ansi_enabled or sys.platform != "win32":
        _ansi_enabled = True
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
    mode = ctypes.c_uint()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) != 0:
        kernel32.SetConsoleMode(handle, mode.value | _ENABLE_VT_PROCESSING)
    _ansi_enabled = True


#### Non-interactive fallback ####


def _select_form_fallback(
    sections: typing.Sequence[Section],
    header: str,
) -> list[list[int]]:
    """Section-by-section numeric prompt for non-tty streams."""
    print(header)
    out: list[list[int]] = []
    for sec in sections:
        if sec.title:
            print(sec.title)
        for i, option in enumerate(sec.options):
            for line in _with_prefix(f"  [{i}] ", option):
                print(line)
        if sec.mode is Mode.SINGLE:
            out.append(sorted(_read_single_fallback(len(sec.options))))
        else:
            out.append(
                sorted(_read_multi_fallback(len(sec.options), sec.required_count))
            )
    return out


def _read_multi_fallback(n_options: int, required: int | None) -> set[int]:
    """Read a space-separated index list, enforcing ``required`` when set."""
    while True:
        chosen = _parse_index_list(
            input("keep (space-separated numbers)> ").strip(), n_options
        )
        if chosen is None:
            print("  enter valid numbers")
            continue
        if required is not None and len(chosen) != required:
            print(f"  select exactly {required}")
            continue
        return chosen


def _read_single_fallback(n_options: int) -> set[int]:
    """Read exactly one index."""
    while True:
        raw = input("choice> ").strip()
        if raw == "" and n_options == 1:
            return {0}
        try:
            idx = int(raw)
        except ValueError:
            print("  enter a number")
            continue
        if 0 <= idx < n_options:
            return {idx}
        print("  out of range")


def _parse_index_list(raw: str, n_options: int) -> set[int] | None:
    """Parse a space-separated index list, or ``None`` if any token is bad."""
    if raw == "":
        return set()
    out: set[int] = set()
    for token in raw.split():
        try:
            idx = int(token)
        except ValueError:
            return None
        if not 0 <= idx < n_options:
            return None
        out.add(idx)
    return out
