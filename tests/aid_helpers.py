"""Shared test helpers for the ``wingspan.aid`` test suite.

``scripted_console`` builds a ``Console`` that services prompts from a fixed
queue of pre-scripted answers and records every printed line, so a full aid
dialog can be driven headlessly without real stdin -- shared by the stage-2
oracle tests and the stage-3 advisor/relay/hooks tests.

``responder_console`` is the stage-4 counterpart for the headless end-to-end
session test: rather than a fixed answer queue (which assumes a known dialog
sequence), it answers every read by pattern-matching the most recently
written line(s) against a table of substring rules, so it survives whatever
decision sequence an arbitrary full game produces.
"""

from __future__ import annotations

import collections
import collections.abc

from wingspan.aid import console

# A responder rule pairs a prompt substring with a function from the matched
# line to the answer text -- a function rather than a bare string so a rule
# can compute its answer from the prompt itself (e.g. an N-dice-faces prompt
# whose N varies per call).
type ResponderRule = tuple[str, collections.abc.Callable[[str], str]]


def scripted_console(answers: list[str]) -> tuple[console.Console, list[str]]:
    """A ``Console`` whose ``read`` pops pre-scripted answers in order and
    whose ``write`` records every printed line, for asserting on prompts."""
    queue = collections.deque(answers)
    transcript: list[str] = []

    def read() -> str:
        return queue.popleft()

    def write(text: str) -> None:
        transcript.append(text)

    return console.Console(read=read, write=write), transcript


def responder_console(
    rules: list[ResponderRule], default: str = "0"
) -> tuple[console.Console, list[str]]:
    """A ``Console`` that answers every read by scanning the lines written
    since the previous read, most-recent first, for the first ``rules`` entry
    whose substring appears -- calling its answer function with the matched
    line. Falls back to ``default`` when nothing matches. ``write`` also
    appends every line to the returned transcript (the full session history,
    not just the unconsumed batch), so a test can assert on what the advisor
    or relay actually printed."""
    transcript: list[str] = []
    pending: list[str] = []

    def write(text: str) -> None:
        transcript.append(text)
        pending.append(text)

    def read() -> str:
        batch = list(pending)
        pending.clear()
        for line in reversed(batch):
            for substring, answer in rules:
                if substring in line:
                    return answer(line)
        return default

    return console.Console(read=read, write=write), transcript
