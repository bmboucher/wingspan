"""Raw single-key input for the configurator's full-screen form.

Mirrors the cross-platform pattern proven in :mod:`wingspan.agents.interactive`:
on Windows keys come from ``msvcrt`` (with the two-call ``\\x00`` / ``\\xe0``
lead-byte protocol that arrow and other special keys use), on POSIX from
``termios`` / ``tty`` raw mode plus ``select``. Both collapse onto the small
:class:`KeyKind` vocabulary the controller dispatches on, so the input loop
stays platform-blind. The platform-specific imports live inside ``sys.platform``
branches (never at module top level) so the type checker narrows the unused
branch away and a bare ``import msvcrt`` never trips analysis on another host.

Reading is *non-blocking*: :meth:`KeyReader.poll` waits at most ``timeout``
seconds for a key and otherwise returns ``None``, letting the controller repaint
on a steady cadence — so the screen reflows on a terminal resize and the edit
caret animates instead of the UI freezing between keystrokes.
"""

from __future__ import annotations

import enum
import sys
import time
import typing

import pydantic


class KeyKind(enum.StrEnum):
    """The normalized keypress vocabulary the configurator reacts to."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ENTER = "enter"
    ESCAPE = "escape"
    BACKSPACE = "backspace"
    TAB = "tab"
    HOME = "home"
    END = "end"
    PAGE_UP = "page_up"
    PAGE_DOWN = "page_down"
    CHAR = "char"  # a printable character, carried in ``KeyEvent.char``
    INTERRUPT = "interrupt"  # Ctrl-C
    OTHER = "other"


class KeyEvent(pydantic.BaseModel):
    """One decoded keypress: a :class:`KeyKind` plus, for ``CHAR``, the literal
    character that was typed."""

    kind: KeyKind
    char: str = ""


# How long :meth:`KeyReader.poll` blocks waiting for a key before yielding the
# frame back to the controller (~33 Hz — smooth without pegging a core).
POLL_INTERVAL_SECONDS = 0.03
# Windows busy-wait granularity: how often kbhit is checked inside one poll.
_KBHIT_SLEEP_SECONDS = 0.005

# Windows ``getwch`` lead bytes that introduce a two-character special key.
_WIN_LEAD_BYTES = ("\x00", "\xe0")
# Second ``getwch`` byte of a Windows special key -> normalized kind.
_WIN_SPECIALS: dict[str, KeyKind] = {
    "H": KeyKind.UP,
    "P": KeyKind.DOWN,
    "K": KeyKind.LEFT,
    "M": KeyKind.RIGHT,
    "G": KeyKind.HOME,
    "O": KeyKind.END,
    "I": KeyKind.PAGE_UP,
    "Q": KeyKind.PAGE_DOWN,
}
# POSIX escape-sequence tail (the characters after ESC ``[``) -> normalized kind.
_UNIX_ESCAPES: dict[str, KeyKind] = {
    "A": KeyKind.UP,
    "B": KeyKind.DOWN,
    "C": KeyKind.RIGHT,
    "D": KeyKind.LEFT,
    "H": KeyKind.HOME,
    "F": KeyKind.END,
    "5~": KeyKind.PAGE_UP,
    "6~": KeyKind.PAGE_DOWN,
}


def decode_char(ch: str) -> KeyEvent:
    """Map one single character onto a :class:`KeyEvent` (pure; unit-testable
    without a console)."""
    if ch in ("\r", "\n"):
        return KeyEvent(kind=KeyKind.ENTER)
    if ch == "\t":
        return KeyEvent(kind=KeyKind.TAB)
    if ch in ("\x08", "\x7f"):  # Windows BS / POSIX DEL
        return KeyEvent(kind=KeyKind.BACKSPACE)
    if ch == "\x1b":
        return KeyEvent(kind=KeyKind.ESCAPE)
    if ch == "\x03":  # Ctrl-C — surfaced as a normal quit, never raised
        return KeyEvent(kind=KeyKind.INTERRUPT)
    if len(ch) == 1 and ch.isprintable():
        return KeyEvent(kind=KeyKind.CHAR, char=ch)
    return KeyEvent(kind=KeyKind.OTHER)


def decode_windows_special(code: str) -> KeyEvent:
    """Map the second byte of a Windows two-byte special key onto a KeyEvent."""
    return KeyEvent(kind=_WIN_SPECIALS.get(code, KeyKind.OTHER))


def decode_unix_escape(tail: str) -> KeyEvent:
    """Map a POSIX ``ESC [`` escape tail onto a KeyEvent."""
    return KeyEvent(kind=_UNIX_ESCAPES.get(tail, KeyKind.OTHER))


class KeyReader:
    """A raw single-key source used as a context manager.

    On POSIX the terminal is put in cbreak mode for the duration of the ``with``
    block and restored on exit; on Windows the console already delivers keys to
    ``getwch`` unbuffered, so entry / exit are no-ops. Call :meth:`poll` once per
    frame for the next :class:`KeyEvent`, or ``None`` if none arrived in time.
    """

    def __init__(self) -> None:
        self._posix_fd = -1
        self._posix_saved: object | None = None  # opaque termios attrs blob

    def __enter__(self) -> "KeyReader":
        if sys.platform != "win32":
            import termios
            import tty

            self._posix_fd = sys.stdin.fileno()
            self._posix_saved = termios.tcgetattr(self._posix_fd)
            tty.setcbreak(self._posix_fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if sys.platform != "win32" and self._posix_saved is not None:
            import termios

            saved = typing.cast("list[typing.Any]", self._posix_saved)
            termios.tcsetattr(self._posix_fd, termios.TCSADRAIN, saved)

    def poll(self, timeout: float = POLL_INTERVAL_SECONDS) -> KeyEvent | None:
        """Return the next keypress, or ``None`` after ``timeout`` seconds."""
        if sys.platform == "win32":
            return self._poll_windows(timeout)
        return self._poll_posix(timeout)

    ###### PRIVATE #######

    def _poll_windows(self, timeout: float) -> KeyEvent | None:
        import msvcrt

        deadline = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in _WIN_LEAD_BYTES:
                    return decode_windows_special(msvcrt.getwch())
                return decode_char(ch)
            if time.monotonic() >= deadline:
                return None
            time.sleep(_KBHIT_SLEEP_SECONDS)

    def _poll_posix(self, timeout: float) -> KeyEvent | None:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return decode_char(ch)
        # An ESC may stand alone or introduce a ``[`` escape; peek without
        # blocking to tell the two apart.
        more, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not more or sys.stdin.read(1) != "[":
            return KeyEvent(kind=KeyKind.ESCAPE)
        tail = sys.stdin.read(1)
        if tail in ("5", "6"):  # PgUp / PgDn arrive as ``[5~`` / ``[6~``
            tail += sys.stdin.read(1)
        return decode_unix_escape(tail)
