"""One-slot mailbox for passing policy data from agent to instrumentation handler.

The policy agent runs the net's forward pass and produces both a critic value
and a policy distribution; the instrumentation handler needs these to build the
timeline chart and the decision-log display.  Since agent and handler are fully
decoupled (no shared reference to the net), :class:`DecisionProbe` bridges them:
the agent writes via :meth:`record` / :meth:`record_policy` immediately after
the forward pass, and the handler reads via :meth:`take` inside ``made_decision``.

This is a plain runtime object — **not** a Pydantic model.  It carries transient
data that crosses one decision boundary and is not a persistent record.
"""

from __future__ import annotations

import pydantic


class PolicyAnnotation(pydantic.BaseModel):
    """The policy-head output for one genuine decision.

    ``probs`` is the softmax distribution over the offered choices (same order
    as ``decision.choices``); ``scores`` is the raw logits when available (main
    net) or ``None`` (setup-net value-only mode); ``chosen_idx`` is the index
    of the option that was actually played.

    ``state_vec`` and ``choice_feats`` carry the raw encoder outputs for the
    encoding-viewer modal; they are present only for main-net decisions (the
    setup path leaves them ``None``). ``include_setup`` and ``card_embed_dim``
    are the two values needed to reconstruct the correct :class:`VectorLayout`
    from the stripe registry."""

    probs: list[float]
    scores: list[float] | None = None
    chosen_idx: int
    state_vec: list[float] | None = None
    choice_feats: list[list[float]] | None = None
    include_setup: bool | None = None
    card_embed_dim: int | None = None


class DecisionProbe:
    """A one-slot mailbox for the critic's value output and policy annotation.

    :meth:`take` clears both slots on read so a forced move or setup decision —
    which skip the main forward pass — leave ``None`` for the handler, producing
    a gap in the value line rather than reusing the previous decision's stale
    value.
    """

    def __init__(self) -> None:
        self._value: float | None = None
        self._policy: PolicyAnnotation | None = None

    def record(self, value: float) -> None:
        """Store the critic's value for the current decision."""
        self._value = value

    def record_policy(self, annotation: PolicyAnnotation) -> None:
        """Store the policy distribution for the current decision."""
        self._policy = annotation

    def take(self) -> tuple[float | None, PolicyAnnotation | None]:
        """Return and clear both slots; each is ``None`` if nothing was recorded."""
        value = self._value
        policy = self._policy
        self._value = None
        self._policy = None
        return value, policy
