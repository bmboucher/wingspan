"""Live progress readout with throughput and ETA for shard-streamed studies.

:class:`StudyProgress` is fed the cumulative game count as shards complete and
prints a throttled status line to a text stream (stderr for the CLI), so the
study's data output is never mixed with progress text.
"""

import time
import typing

from wingspan.research import constants

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_PERCENT = 100.0


class StudyProgress:
    """Throttled progress reporter for a study of ``total_games`` games.

    On a TTY the status line is rewritten in place (carriage return, padded to
    the longest line so far) at most once per
    ``constants.PROGRESS_TTY_INTERVAL_SECONDS``; otherwise each status line is
    appended with a newline at most once per
    ``constants.PROGRESS_LOG_INTERVAL_SECONDS``. The ETA uses the overall
    average rate since construction, so early estimates include pool spawn and
    net-load time and converge as the run proceeds. ``now`` is injectable so
    tests can drive a fake clock."""

    def __init__(
        self,
        total_games: int,
        stream: typing.TextIO,
        now: typing.Callable[[], float] = time.monotonic,
    ) -> None:
        self._total_games = total_games
        self._stream = stream
        self._now = now
        self._is_tty = stream.isatty()
        self._interval = (
            constants.PROGRESS_TTY_INTERVAL_SECONDS
            if self._is_tty
            else constants.PROGRESS_LOG_INTERVAL_SECONDS
        )
        self._started = now()
        self._last_print: float | None = None
        self._longest_line = 0

    def update(self, games_done: int) -> None:
        """Report ``games_done`` (cumulative) if the throttle interval has
        elapsed since the last printed line."""
        current = self._now()
        if self._last_print is not None and current - self._last_print < self._interval:
            return
        self._print(games_done, current)

    def finish(self, games_done: int) -> None:
        """Always print a final status line; on a TTY, end the in-place line
        with a newline so later prints start clean."""
        self._print(games_done, self._now())
        if self._is_tty:
            self._stream.write("\n")
            self._stream.flush()

    def _print(self, games_done: int, current: float) -> None:
        line = format_progress(games_done, self._total_games, current - self._started)
        self._last_print = current
        if self._is_tty:
            self._longest_line = max(self._longest_line, len(line))
            self._stream.write("\r" + line.ljust(self._longest_line))
        else:
            self._stream.write(line + "\n")
        self._stream.flush()


def format_progress(games_done: int, total_games: int, elapsed_seconds: float) -> str:
    """One status line, e.g. ``games 2,500/10,000 (25.0%) | 20.0 games/s |
    elapsed 0:02:05 | ETA 0:06:15``. The rate and ETA segments are omitted
    until some games are done and some time has elapsed."""
    percent = _PERCENT * games_done / total_games if total_games > 0 else _PERCENT
    segments = [
        f"games {games_done:,}/{total_games:,} ({percent:.1f}%)",
    ]
    if games_done > 0 and elapsed_seconds > 0:
        rate = games_done / elapsed_seconds
        remaining_seconds = (total_games - games_done) / rate
        segments.append(f"{rate:.1f} games/s")
        segments.append(f"elapsed {_format_duration(elapsed_seconds)}")
        segments.append(f"ETA {_format_duration(remaining_seconds)}")
    else:
        segments.append(f"elapsed {_format_duration(elapsed_seconds)}")
    return " | ".join(segments)


###### PRIVATE #######


def _format_duration(seconds: float) -> str:
    """``H:MM:SS`` with unpadded hours (which may exceed 24); fractional
    seconds are truncated."""
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, _SECONDS_PER_HOUR)
    minutes, secs = divmod(remainder, _SECONDS_PER_MINUTE)
    return f"{hours}:{minutes:02d}:{secs:02d}"
