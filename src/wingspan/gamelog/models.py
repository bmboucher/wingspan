"""Pydantic data models for the structured game-event tree.

Every game produces one :class:`GameEventTree` whose phases and events
replace the text-parsing approach used in earlier versions.  Both the HTML
decision log and the plaintext detailed log are pure renderers over this tree.

This module also holds the display primitives formerly in
:mod:`wingspan.reporting.game_log_html` (``EncodedSubField``, ``EncodedStripe``,
``DecisionOption``) so they can be shared without importing the reporting layer.

**Import discipline:** this module depends only on ``pydantic`` and the standard
library — no engine, state, training, or torch imports — so it can be freely
imported by ``reporting`` without closing any import cycle.
"""

from __future__ import annotations

import typing

import pydantic

# ---------------------------------------------------------------------------
# Encoding-viewer primitives (formerly in reporting.game_log_html)


class EncodedSubField(pydantic.BaseModel):
    """One named element or block within a non-zero feature stripe, for the
    encoding-viewer modal.

    Exactly one of ``active_index``, ``raw_value``, or ``raw_values`` is set,
    depending on the sub-field's encoding: ``"one-hot"`` uses ``active_index``
    (the argmax position); size-1 scalars use ``raw_value``; multi-element
    blocks use ``raw_values`` (non-zero positions only)."""

    name: str
    description: str
    encoding: str
    value_range: str
    notes: str | None = None
    active_index: int | None = None
    raw_value: float | None = None
    raw_values: list[float] | None = None
    decoded_label: str | None = None


class EncodedStripe(pydantic.BaseModel):
    """One non-zero stripe from the state or choice vector, for the
    encoding-viewer modal.

    ``sub_fields`` holds only the non-zero elements (or the whole stripe when
    it carries no named sub-fields). Empty stripes (all-zero) are never
    included in the parent list."""

    name: str
    description: str
    sub_fields: list[EncodedSubField] = []


class DecisionOption(pydantic.BaseModel):
    """One offered option within a decision box in the decision log.

    ``prob`` is the policy's softmax probability (``None`` when unavailable);
    ``score`` is the raw logit used for ranking (``None`` for the setup-net
    value-only mode); ``selected`` marks the option that was actually played.
    ``choice_stripes`` carries the non-zero choice-vector stripes for the
    encoding-viewer modal (``None`` when no model backed this seat)."""

    label: str
    prob: float | None = None
    score: float | None = None
    selected: bool = False
    choice_stripes: list[EncodedStripe] | None = None


# ---------------------------------------------------------------------------
# Sub-events (leaf nodes: decisions and non-decision notes)


class SubEvent(pydantic.BaseModel):
    """Abstract leaf node within a :class:`GameEvent`.

    ``player_id`` is the seat responsible for this sub-event (``None`` for
    global events like birdfeeder rerolls)."""

    player_id: int | None = None


class NoteSubEvent(SubEvent):
    """A non-decision notification line — "draws X from the deck", power outcomes.

    Emitted for game events that are not otherwise captured as decisions or
    forced moves."""

    text: str


class ForcedSubEvent(SubEvent):
    """A forced single-choice auto-resolve — the engine's only option was pre-determined.

    Rendered as a non-collapsible outcome line in the HTML log."""

    text: str


class DecisionSubEvent(SubEvent):
    """A genuine decision with full policy annotation.

    ``outcome_text`` is the humanized summary (the collapsed header text in the
    HTML log).  ``options`` is the top-N list of offered choices (including the
    chosen one).  ``state_stripes`` holds the non-zero state-vector stripes for
    the encoding-viewer modal; ``None`` when no model backed this seat.

    Timeline scalars: ``value`` is the critic's predicted return for the deciding
    seat at this decision (``None`` for random/human seats); ``turn_counter`` and
    ``setup_slot`` together locate the decision on the game clock (see
    :mod:`wingspan.training.timestamps`); ``family_idx`` identifies the decision
    type for timestamp interpolation; ``score_p0`` / ``score_p1`` / ``margin_before``
    are the live scores and relative margin used by the target-line computation."""

    outcome_text: str
    options: list[DecisionOption] = []
    state_stripes: list[EncodedStripe] | None = None
    value: float | None = None
    # Clock fields — turn_counter + setup_slot together give provisional_timestamp.
    turn_counter: int = 0
    setup_slot: int | None = None  # 0=keep, 1=bonus, 2=food; None means in-turn
    family_idx: int = 0
    score_p0: int = 0
    score_p1: int = 0
    margin_before: float = 0.0


# ---------------------------------------------------------------------------
# Top-level game events (one per logical action)


class GameEvent(pydantic.BaseModel):
    """A top-level event or nested sub-event in the game tree.

    ``player_id`` is the acting seat; ``sub_events`` are the leaf nodes
    (decisions and notes) that belong to this event; ``children`` holds nested
    :class:`GameEvent` objects (e.g. white powers under a play-bird event,
    pink reactions under a habitat-row activation)."""

    player_id: int | None = None
    sub_events: list[SubEvent] = []
    children: list[GameEvent] = []


class MainActionEvent(GameEvent):
    """Event #4: the player selects a main action (gain food / lay eggs /
    draw cards / play a bird)."""


class PlayBirdEvent(GameEvent):
    """Event #1: the player plays one bird card (main action or extra play).

    Sub-events include the bird+habitat selection decision, egg-cost removals,
    food payment, and the bird's white 'when played' power (as a child
    :class:`WhitePowerEvent`)."""


class WhitePowerEvent(GameEvent):
    """A white 'when played' power resolution, nested under :class:`PlayBirdEvent`."""

    bird_name: str


class ReactionEvent(GameEvent):
    """A pink reactor firing attributed to the reacting player.

    Placed at phase level when no enclosing action is open (e.g. a gain-food
    reactor fires after the base event closes), or nested under a play-bird or
    predator event when one is still open."""

    bird_name: str


class ActivateBaseEvent(GameEvent):
    """Event #2: the base-ability decisions for one habitat action (gain food /
    lay eggs / draw cards), NOT including the row's brown powers."""

    habitat: str
    action: str


class ActivateBrownEvent(GameEvent):
    """Event #3: one bird's brown-power slot in the activated row.

    Emitted for every bird crossed (right-to-left), including non-brown birds
    (``is_brown=False`` → empty event with a "no brown power" note).  A 3-bird
    row therefore always produces exactly 3 :class:`ActivateBrownEvent`s."""

    bird_name: str
    is_brown: bool


class SetupEvent(GameEvent):
    """Event #5: one player's setup phase (selecting birds, food, and bonus).

    ``kept_card_names`` and ``kept_bonus_name`` are filled in by the recorder
    when the ``SetupChoice`` / ``BonusCardChoice`` decisions resolve."""

    kept_card_names: list[str] = []
    kept_bonus_name: str | None = None


class FinalScoreBreakdown(pydantic.BaseModel):
    """A seat's seven-component final score."""

    birds: int = 0
    eggs: int = 0
    tucked: int = 0
    cached: int = 0
    bonus: int = 0
    goals: int = 0
    total: int = 0


class RoundGoalEvent(GameEvent):
    """Event #6a: one round's goal scoring.

    ``counts`` is a per-seat list of category counts; ``vps`` is a per-seat
    list of VP awarded."""

    round_idx: int
    description: str
    counts: list[int] = []
    vps: list[int] = []


class FinalScoringEvent(GameEvent):
    """Event #6b: the game's final scoring summary.

    ``scores`` holds one :class:`FinalScoreBreakdown` per seat, in seat order."""

    scores: list[FinalScoreBreakdown] = []


class LooseEvent(GameEvent):
    """Catch-all bucket for a decision recorded when no other event is open.

    Used as the auto-wrap target when ``EventRecorder.record_decision`` or
    ``record_forced`` fires outside any explicit ``begin_*/end_event`` bracket
    (e.g. a stray power decision not yet wired into the call-site graph)."""


# ---------------------------------------------------------------------------
# Phase and tree containers


class PhaseNode(pydantic.BaseModel):
    """One navigable phase: a sequential group of :class:`GameEvent` objects.

    ``kind`` matches the phase-boundary strings used by the HTML handler:
    ``"game_start"``, ``"setup"``, ``"round"``, ``"turn"``, or ``"game_end"``."""

    kind: str
    events: list[GameEvent] = []


class GameEventTree(pydantic.BaseModel):
    """The complete event tree for one game, organized as an ordered list of
    :class:`PhaseNode` objects whose positions are 1-to-1 with the HTML
    handler's :class:`~wingspan.reporting.game_log_html.PhaseRecord` list."""

    phases: list[PhaseNode] = []


# ---------------------------------------------------------------------------
# Type alias for the event union (used by type checkers / renderers)

AnySubEvent = typing.Union[NoteSubEvent, ForcedSubEvent, DecisionSubEvent]
AnyGameEvent = typing.Union[
    MainActionEvent,
    PlayBirdEvent,
    WhitePowerEvent,
    ReactionEvent,
    ActivateBaseEvent,
    ActivateBrownEvent,
    SetupEvent,
    RoundGoalEvent,
    FinalScoringEvent,
    LooseEvent,
]
