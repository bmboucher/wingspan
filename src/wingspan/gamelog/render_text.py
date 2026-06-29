"""Plaintext renderer for the structured game-event tree.

Produces a human-readable log from a :class:`~wingspan.gamelog.models.GameEventTree`.
Each phase opens with a ``=== KIND ===`` header.  Events render with a
type-specific label line followed by their sub-events (decisions as ``→``,
forced moves as ``!``, notes undecorated) and recursively nested children.

This format is the target of ``wingspan play --log``; the raw ``engine.log``
dump (old ``--log`` behaviour) remains available as ``--debug-log``.
"""

from __future__ import annotations

from wingspan.gamelog import models


def render_plaintext(tree: models.GameEventTree) -> str:
    """Render ``tree`` as a human-readable multi-line string.

    Each phase opens with a ``=== KIND ===`` header.  Events are indented by
    nesting depth; sub-events appear as prefix + text lines:
    ``→ text`` for decisions, ``! text`` for forced moves, bare text for notes.
    """
    lines: list[str] = []
    for phase in tree.phases:
        lines.append(f"=== {phase.kind.upper()} ===")
        for event in phase.events:
            lines.extend(_render_event(event, indent=0))
    return "\n".join(lines)


###### PRIVATE #######


def _event_label(event: models.GameEvent) -> str:
    """A concise label for an event's header line, including type-specific fields."""
    if isinstance(event, models.ActivateBaseEvent):
        return f"Activate {event.habitat} ({event.action})"
    if isinstance(event, models.ActivateBrownEvent):
        prefix = "Brown" if event.is_brown else "——"
        return f"{prefix}: {event.bird_name}"
    if isinstance(event, models.WhitePowerEvent):
        return f"White power: {event.bird_name}"
    if isinstance(event, models.ReactionEvent):
        return f"Reaction: {event.bird_name}"
    if isinstance(event, models.SetupEvent):
        return _setup_event_label(event)
    if isinstance(event, models.RoundGoalEvent):
        return _round_goal_label(event)
    if isinstance(event, models.FinalScoringEvent):
        return _final_scoring_label(event)
    return type(event).__name__


def _setup_event_label(event: models.SetupEvent) -> str:
    """Label for a setup event, listing kept cards and bonus."""
    parts: list[str] = []
    if event.kept_card_names:
        parts.append("kept: " + ", ".join(event.kept_card_names))
    if event.kept_bonus_name:
        parts.append("bonus: " + event.kept_bonus_name)
    detail = f" ({'; '.join(parts)})" if parts else ""
    return f"Setup{detail}"


def _round_goal_label(event: models.RoundGoalEvent) -> str:
    """Label for a round goal event with per-seat counts and VP."""
    seat_parts: list[str] = []
    for seat_idx, (count, vp) in enumerate(zip(event.counts, event.vps, strict=False)):
        seat_parts.append(f"P{seat_idx}: {count}/{vp}VP")
    detail = f" [{', '.join(seat_parts)}]" if seat_parts else ""
    return f"Round {event.round_idx + 1} goal — {event.description}{detail}"


def _final_scoring_label(event: models.FinalScoringEvent) -> str:
    """Label for the final scoring event with per-seat totals."""
    totals = [str(score.total) for score in event.scores]
    return f"Final scoring [{', '.join(totals)}]"


def _render_event(event: models.GameEvent, *, indent: int) -> list[str]:
    """Recursively render one event and its sub-events / children."""
    prefix = "  " * indent
    lines: list[str] = []

    # Header: bracket with type-specific label.
    lines.append(f"{prefix}[{_event_label(event)}]")

    # Sub-events: decisions (→), forced (!), notes (bare).
    for sub in event.sub_events:
        if isinstance(sub, models.DecisionSubEvent):
            lines.append(f"{prefix}  → {sub.outcome_text}")
        elif isinstance(sub, models.ForcedSubEvent):
            lines.append(f"{prefix}  ! {sub.text}")
        elif isinstance(sub, models.NoteSubEvent):
            lines.append(f"{prefix}  {sub.text}")

    # Children recurse at increased indent (e.g. WhitePowerEvent under PlayBirdEvent).
    for child in event.children:
        lines.extend(_render_event(child, indent=indent + 1))

    return lines
