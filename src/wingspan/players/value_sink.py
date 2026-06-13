"""One-slot mailbox for passing critic value outputs from agent to handler.

The policy agent runs the net's forward pass and produces a critic value;
the instrumentation handler needs that value to build the timeline chart.
Since agent and handler are fully decoupled (no shared reference to the net),
:class:`ValueProbe` bridges them: the agent writes via :meth:`record`
immediately after the forward pass, and the handler reads via :meth:`take`
inside ``made_decision``.

This is a plain runtime object — **not** a Pydantic model. It carries a
transient scalar that crosses one decision boundary and is not a data record.
"""

from __future__ import annotations


class ValueProbe:
    """A one-slot mailbox for the critic's raw value output (deciding-player
    POV, divided by ``score_norm``).

    :meth:`take` clears on read so a forced move or setup decision — which
    skip the main forward pass and therefore never call :meth:`record` — leave
    ``None`` for the handler, producing a gap in the value line rather than
    reusing the previous decision's stale value.
    """

    def __init__(self) -> None:
        self._value: float | None = None

    def record(self, value: float) -> None:
        """Store the critic's value for the current decision."""
        self._value = value

    def take(self) -> float | None:
        """Return and clear the stored value; ``None`` if nothing was recorded."""
        result = self._value
        self._value = None
        return result
