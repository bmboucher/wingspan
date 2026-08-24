"""Interactive console primitives for the aid session.

``Console`` wraps the read/write functions every aid prompt flows through,
injectable so tests can script a full session headlessly. ``LogEcho``
mirrors the engine's own game log to the console as narration -- the log
already reads like a play-by-play, including the "skipping decision, only 1
choice" lines the engine writes when ``Engine.ask`` short-circuits a forced
move, so echoing it is the whole narration layer; nothing is re-derived from
the Decision stream. Per-player lines are stripped of their redundant
``[Name]`` prefix and, on an interactive console, color-coded by seat
(green for the user, red for the opponent) instead. When
:meth:`Console.supports_interactive` is true, widget helpers (see
``wingspan.aid.widgets``) may write ANSI frames directly to stdout instead
of routing through :meth:`Console.say`.
"""

from __future__ import annotations

import collections.abc
import sys

from wingspan import state
from wingspan.agents import display

# Confirm-prompt answers that resolve to "yes"/"no". A bare Enter (empty
# answer) defaults to yes, matching the "[Y/n]" prompt suffix.
_CONFIRM_YES_ANSWERS = frozenset({"", "y", "yes"})
_CONFIRM_NO_ANSWERS = frozenset({"n", "no"})

# LogEcho's per-seat narration colors: green for the user's own seat (id 0),
# red for the opponent's (id 1). Global lines (player_id None) get no color.
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"


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
        self._interactive = (
            read is None
            and write is None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )
        self._read = read if read is not None else input
        self._write = write if write is not None else print

    def supports_interactive(self) -> bool:
        """Whether single-keystroke widgets may take over real stdio: both
        read and write are the builtin defaults AND stdin/stdout are real
        ttys. Scripted test consoles (injected read/write) always report
        False."""
        return self._interactive

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
    """Flushes new ``GameState.log_entries`` lines to a ``Console`` as
    narration, one player-attributed color and a stripped ``[Name]`` prefix
    at a time (green for the user's seat, red for the opponent's, plain for
    global lines).

    Attach ``game_state`` once ``oracle_state.build_state`` constructs it;
    :meth:`flush` is a no-op before that (setup dialogs that run before the
    state exists have nothing to echo)."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.game_state: state.GameState | None = None
        self._printed_through = 0

    def flush(self) -> None:
        """Print every ``game_state.log_entries`` line since the last flush,
        stripped of ANSI styling and of its redundant ``[Name]`` prefix, and
        color-coded by ``player_id`` when the console supports it. No-op
        while unattached."""
        game_state = self.game_state
        if game_state is None:
            return
        for entry in game_state.log_entries[self._printed_through :]:
            self.console.say(self._render(entry, game_state))
        self._printed_through = len(game_state.log_entries)

    def _render(self, entry: state.LogEntry, game_state: state.GameState) -> str:
        """Strip ANSI styling and the ``[Name]`` prefix from one entry, then
        color it by seat when the console supports interactive output."""
        text = display.strip_ansi(entry.text)
        if entry.player_id is None:
            return text
        player_name = game_state.players[entry.player_id].name
        text = text.removeprefix(f"[{player_name}] ")
        if not self.console.supports_interactive():
            return text
        color = _GREEN if entry.player_id == 0 else _RED
        return f"{color}{text}{_ANSI_RESET}"
