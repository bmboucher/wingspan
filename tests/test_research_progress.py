"""Progress reporter for shard-streamed studies (``wingspan.research.progress``).

Black-box over the status-line format (rate, elapsed, ETA, duration edge cases)
and the reporter's throttling and rendering on TTY and non-TTY streams, driven
by a fake clock.
"""

from __future__ import annotations

import io

from wingspan.research import constants, progress


class _FakeClock:
    def __init__(self) -> None:
        self.time = 0.0

    def __call__(self) -> float:
        return self.time


class _TtyStream(io.StringIO):
    """A string stream that claims to be a terminal and keeps each write."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, text: str, /) -> int:
        self.writes.append(text)
        return super().write(text)


def test_format_progress_exact():
    assert progress.format_progress(2_500, 10_000, 125.0) == (
        "games 2,500/10,000 (25.0%) | 20.0 games/s | elapsed 0:02:05 | ETA 0:06:15"
    )


def test_format_progress_without_games_omits_rate_and_eta():
    line = progress.format_progress(0, 10_000, 0.0)
    assert line == "games 0/10,000 (0.0%) | elapsed 0:00:00"
    assert "games/s" not in line
    assert "ETA" not in line


def test_format_progress_duration_edges():
    assert "elapsed 0:00:59" in progress.format_progress(1, 2, 59.9)
    assert "elapsed 1:01:01" in progress.format_progress(1, 2, 3661.0)
    assert "elapsed 25:00:00" in progress.format_progress(1, 2, 25 * 3600.0)


def test_log_stream_throttles_and_appends_lines():
    clock = _FakeClock()
    stream = io.StringIO()
    reporter = progress.StudyProgress(100, stream, now=clock)
    clock.time = 1.0
    reporter.update(10)
    clock.time = 2.0
    reporter.update(20)
    assert stream.getvalue().count("\n") == 1

    clock.time = 1.0 + constants.PROGRESS_LOG_INTERVAL_SECONDS
    reporter.update(30)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("games 30/100")

    clock.time += 0.5
    reporter.finish(100)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert lines[2].startswith("games 100/100 (100.0%)")
    assert stream.getvalue().endswith("\n")


def test_tty_rewrites_in_place_pads_and_finishes_with_newline():
    clock = _FakeClock()
    stream = _TtyStream()
    reporter = progress.StudyProgress(10_000, stream, now=clock)
    clock.time = 1.0
    reporter.update(10_000)
    clock.time += constants.PROGRESS_TTY_INTERVAL_SECONDS
    # A shorter line (fewer digits) must still fully overwrite the longer one.
    reporter.finish(1)

    first, second, newline = stream.writes
    assert first.startswith("\r")
    assert second.startswith("\r")
    assert len(second.rstrip()) < len(first.rstrip())
    assert len(second) == len(first)
    assert newline == "\n"


def test_tty_throttle_uses_tty_interval():
    clock = _FakeClock()
    stream = _TtyStream()
    reporter = progress.StudyProgress(100, stream, now=clock)
    clock.time = 0.1
    reporter.update(1)
    clock.time = 0.1 + constants.PROGRESS_TTY_INTERVAL_SECONDS / 2
    reporter.update(2)
    assert len(stream.writes) == 1
    clock.time = 0.1 + constants.PROGRESS_TTY_INTERVAL_SECONDS
    reporter.update(3)
    assert len(stream.writes) == 2
