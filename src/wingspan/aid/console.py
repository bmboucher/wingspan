"""Interactive console primitives for the aid session.

``Console`` wraps the read/write functions every aid prompt flows through,
injectable so tests can script a full session headlessly. ``LogEcho``
mirrors the engine's own game log to the console as narration -- the log
already reads like a play-by-play, including the "skipping decision, only 1
choice" lines the engine writes when ``Engine.ask`` short-circuits a forced
move, so echoing it is the whole narration layer; nothing is re-derived from
the Decision stream.
"""

from __future__ import annotations

import collections.abc

from wingspan import state
from wingspan.agents import display

# Confirm-prompt answers that resolve to "yes"/"no". A bare Enter (empty
# answer) defaults to yes, matching the "[Y/n]" prompt suffix.
_CONFIRM_YES_ANSWERS = frozenset({"", "y", "yes"})
_CONFIRM_NO_ANSWERS = frozenset({"n", "no"})


class Console:
    """The injectable read/write pair every aid prompt flows through.

    ``read``/``write`` default to wrapping the builtins ``input``/``print``;
    tests inject scripted callables instead so a full session can run
    headlessly."""

    def __init__(
        self,
        read: collections.abc.Callable[[], str] | None = None,
        write: collections.abc.Callable[[str], None] | None = None,
    ) -> None:
        self._read = read if read is not None else input
        self._write = write if write is not None else print

    def say(self, text: str) -> None:
        """Write one line of narration/output."""
        self._write(text)

    def ask(self, prompt: str, default: str = "") -> str:
        """Prompt for free text; a blank (or whitespace-only) answer returns
        ``default``."""
        self._write(prompt)
        answer = self._read().strip()
        return answer if answer else default

    def menu(
        self,
        header: str,
        lines: collections.abc.Sequence[str],
        default_idx: int | None = None,
    ) -> int:
        """Print a 1-based numbered ``lines`` menu under ``header`` and loop
        until the user enters a valid index, returned 0-based. A blank
        answer picks ``default_idx`` when one is given."""
        self._write(header)
        for offset, line in enumerate(lines):
            marker = " (default)" if offset == default_idx else ""
            self._write(f"  {offset + 1}. {line}{marker}")
        while True:
            answer = self._read().strip()
            if not answer and default_idx is not None:
                return default_idx
            if answer.isdigit() and 1 <= int(answer) <= len(lines):
                return int(answer) - 1
            self._write(f"Enter a number from 1 to {len(lines)}.")

    def confirm(self, prompt: str) -> bool:
        """Ask a y/n question; a bare Enter (blank answer) means yes."""
        self._write(f"{prompt} [Y/n]")
        while True:
            answer = self._read().strip().casefold()
            if answer in _CONFIRM_YES_ANSWERS:
                return True
            if answer in _CONFIRM_NO_ANSWERS:
                return False
            self._write("Please answer y or n.")


class LogEcho:
    """Flushes new ``GameState.log`` lines to a ``Console`` as narration.

    Attach ``game_state`` once ``oracle_state.build_state`` constructs it;
    :meth:`flush` is a no-op before that (setup dialogs that run before the
    state exists have nothing to echo)."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.game_state: state.GameState | None = None
        self._printed_through = 0

    def flush(self) -> None:
        """Print every ``game_state.log`` line since the last flush,
        stripped of ANSI styling. No-op while unattached."""
        if self.game_state is None:
            return
        for line in self.game_state.log[self._printed_through :]:
            self.console.say(display.strip_ansi(line))
        self._printed_through = len(self.game_state.log)
