"""Plaintext renderer for the structured game-event tree.

Produces a human-readable log from a :class:`~wingspan.gamelog.models.GameEventTree`.
Each phase opens with a ``=== KIND ===`` header.  Events render with a
``[summary]`` header line — the same at-a-glance text the HTML log shows, from
:mod:`wingspan.gamelog.summarize` — followed by their sub-events (decisions as
``→``, forced moves as ``!``, recorded effects as ``·``) and recursively nested
children.

Unlike the HTML log, which shows only decisions and hidden-information reveals
beneath each header, this renderer keeps **every** effect row.  It is the
detailed log: the header says what happened, and the rows below it prove it.

This format is the target of ``wingspan play --log``; the raw ``engine.log``
dump (old ``--log`` behaviour) remains available as ``--debug-log``.
"""

from __future__ import annotations

import typing

from wingspan.gamelog import models, summarize


def render_plaintext(tree: models.GameEventTree) -> str:
    """Render ``tree`` as a human-readable multi-line string.

    Each phase opens with a ``=== KIND ===`` header.  Events are indented by
    nesting depth; sub-events appear as prefix + text lines:
    ``→ text`` for decisions, ``! text`` for forced moves, ``· text`` for
    recorded effects.
    """
    lines: list[str] = []
    for phase in tree.phases:
        lines.append(f"=== {phase.kind.upper()} ===")
        for event in phase.events:
            lines.extend(_render_event(event, indent=0))
    return "\n".join(lines)


###### PRIVATE #######


def _effect_text(effect: models.AnyEffect) -> str:
    """One-line plaintext form of a recorded state mutation.

    Deliberately mechanical — it names the effect kind and its own fields rather
    than composing prose, so a newly added effect class renders something
    truthful without needing a hand-written template here.  The prose form lives
    in :mod:`wingspan.gamelog.summarize` and is what the header line uses."""
    fields: dict[str, object] = effect.model_dump(exclude={"kind", "player_id"})
    parts: list[str] = []
    for name, value in fields.items():
        if value is None or value == [] or value == "":
            continue
        rendered = (
            " ".join(str(item) for item in typing.cast("list[object]", value))
            if isinstance(value, list)
            else str(value)
        )
        parts.append(f"{name}={rendered}")
    return f"{effect.kind}({', '.join(parts)})"


def _render_event(event: models.GameEvent, *, indent: int) -> list[str]:
    """Recursively render one event and its sub-events / children."""
    prefix = "  " * indent
    lines: list[str] = []

    # Header: the same summary line the HTML log collapses this event to.
    lines.append(f"{prefix}[{summarize.summary_text(event)}]")

    # Sub-events: decisions (→), forced (!), effects (·).
    for sub in event.sub_events:
        if isinstance(sub, models.DecisionSubEvent):
            lines.append(f"{prefix}  → {sub.outcome_text}")
        elif isinstance(sub, models.ForcedSubEvent):
            lines.append(f"{prefix}  ! {sub.outcome_text}")
        else:
            lines.append(f"{prefix}  · {_effect_text(sub)}")

    # Children recurse at increased indent (e.g. WhitePowerEvent under PlayBirdEvent).
    for child in event.children:
        lines.extend(_render_event(child, indent=indent + 1))

    return lines
