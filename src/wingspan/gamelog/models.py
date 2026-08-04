"""Pydantic data models for the structured game-event tree.

Every game produces one :class:`GameEventTree` whose phases and events
replace the text-parsing approach used in earlier versions.  Both the HTML
decision log and the plaintext detailed log are pure renderers over this tree.

This module also holds the display primitives formerly in
:mod:`wingspan.reporting.game_log_html` (``EncodedSubField``, ``EncodedStripe``,
``DecisionOption``) so they can be shared without importing the reporting layer.

**Serialization contract.** Every :class:`GameEvent` and :class:`SubEvent`
subclass carries a ``kind`` literal, and the recursive ``children`` /
``sub_events`` fields are typed as *discriminated unions* (:data:`AnyGameEvent`
/ :data:`AnySubEvent`) rather than their base classes.  This is load-bearing:
with base-class annotations, ``model_dump_json`` silently drops every
subclass-declared field (``bird_name``, ``outcome_text``, ``habitat``, …) and
the node type becomes unrecoverable.  ``tests/test_gamelog_serialize.py``
guards the round-trip.

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
    global events like birdfeeder rerolls).  Concrete subclasses each declare a
    distinct ``kind`` literal so :data:`AnySubEvent` can discriminate them."""

    player_id: int | None = None


class NoteSubEvent(SubEvent):
    """A non-decision notification line — "draws X from the deck", power outcomes.

    Emitted for game events that are not otherwise captured as decisions or
    forced moves."""

    kind: typing.Literal["note"] = "note"
    text: str


class ResolvedSubEvent(SubEvent):
    """Abstract base for a decision point the engine resolved — forced or genuine.

    ``outcome_text`` is the humanized summary of what was chosen (the collapsed
    header text in the HTML log).  The remaining fields locate the resolution on
    the game clock so forced moves and genuine decisions are equally joinable to
    the timeline: ``turn_counter`` and ``setup_slot`` together give the
    provisional timestamp (see :mod:`wingspan.training.timestamps`),
    ``family_idx`` identifies the decision type for timestamp interpolation,
    ``scores`` is the live per-seat score list in seat order, and
    ``margin_before`` is the deciding seat's own margin — its score minus the
    *best* other seat's score, reducing to the legacy own-minus-opponent value
    at two seats."""

    outcome_text: str
    turn_counter: int = 0
    setup_slot: int | None = None  # 0=keep, 1=bonus, 2=food; None means in-turn
    family_idx: int = 0
    scores: list[int] = []
    margin_before: float = 0.0


class ForcedSubEvent(ResolvedSubEvent):
    """A forced single-choice auto-resolve — the engine's only option was pre-determined.

    Rendered as a non-collapsible outcome line in the HTML log.  Carries the
    same clock fields as :class:`DecisionSubEvent` but no policy annotation:
    the agent was never consulted, so there is no distribution to show."""

    kind: typing.Literal["forced"] = "forced"


class DecisionSubEvent(ResolvedSubEvent):
    """A genuine decision with full policy annotation.

    ``options`` is the top-N list of offered choices (including the chosen one).
    ``state_stripes`` holds the non-zero state-vector stripes for the
    encoding-viewer modal; ``None`` when no model backed this seat.  ``value``
    is the critic's predicted return for the deciding seat at this decision
    (``None`` for random/human seats)."""

    kind: typing.Literal["decision"] = "decision"
    options: list[DecisionOption] = []
    state_stripes: list[EncodedStripe] | None = None
    value: float | None = None


type AnySubEvent = typing.Annotated[
    NoteSubEvent | ForcedSubEvent | DecisionSubEvent,
    pydantic.Field(discriminator="kind"),
]
"""Discriminated union of every concrete :class:`SubEvent`.

``GameEvent.sub_events`` is annotated with this rather than ``list[SubEvent]``
so serialization preserves each subclass's own fields."""


# ---------------------------------------------------------------------------
# Top-level game events (one per logical action)


class GameEvent(pydantic.BaseModel):
    """A top-level event or nested sub-event in the game tree.

    ``event_id`` is a per-game monotonic identifier assigned by the recorder as
    the event is opened; it gives every node a stable address for the flat
    structured log and for cross-referencing between renderers.  ``player_id``
    is the acting seat; ``sub_events`` are the leaf nodes (decisions and notes)
    that belong to this event; ``children`` holds nested :class:`GameEvent`
    objects (e.g. white powers under a play-bird event, pink reactions under a
    habitat-row activation)."""

    event_id: int = 0
    player_id: int | None = None
    sub_events: list[AnySubEvent] = []
    children: list[AnyGameEvent] = []


class MainActionEvent(GameEvent):
    """Event #4: the player selects a main action (gain food / lay eggs /
    draw cards / play a bird)."""

    kind: typing.Literal["main_action"] = "main_action"


class PlayBirdEvent(GameEvent):
    """Event #1: the player plays one bird card (main action or extra play).

    Sub-events include the bird+habitat selection decision, egg-cost removals,
    food payment, and the bird's white 'when played' power (as a child
    :class:`WhitePowerEvent`)."""

    kind: typing.Literal["play_bird"] = "play_bird"


class WhitePowerEvent(GameEvent):
    """A white 'when played' power resolution, nested under :class:`PlayBirdEvent`."""

    kind: typing.Literal["white_power"] = "white_power"
    bird_name: str


class ReactionEvent(GameEvent):
    """A pink reactor firing attributed to the reacting player.

    Placed at phase level when no enclosing action is open (e.g. a gain-food
    reactor fires after the base event closes), or nested under a play-bird or
    predator event when one is still open."""

    kind: typing.Literal["reaction"] = "reaction"
    bird_name: str


class ActivateBaseEvent(GameEvent):
    """Event #2: the base-ability decisions for one habitat action (gain food /
    lay eggs / draw cards), NOT including the row's brown powers."""

    kind: typing.Literal["activate_base"] = "activate_base"
    habitat: str
    action: str


class ActivateBrownEvent(GameEvent):
    """Event #3: one bird's brown-power slot in the activated row.

    Emitted for every bird crossed (right-to-left), including non-brown birds
    (``is_brown=False`` → an event with no sub-events and no children).  A
    3-bird row therefore always produces exactly 3
    :class:`ActivateBrownEvent`s."""

    kind: typing.Literal["activate_brown"] = "activate_brown"
    bird_name: str
    is_brown: bool


class ExtraPlayEvent(GameEvent):
    """One accrued extra bird play being offered and accepted or declined.

    Wraps the accept/decline ask so a *declined* extra play is still a named
    node rather than a loose decision.  When accepted, the resulting
    :class:`PlayBirdEvent` is a child.  ``habitat`` is the habitat the extra
    play is restricted to, or ``None`` when unrestricted."""

    kind: typing.Literal["extra_play"] = "extra_play"
    habitat: str | None = None


class TurnEndEvent(GameEvent):
    """The end-of-turn obligations block (``DRAW_CARDS_THEN_DISCARD_EOT`` discards)."""

    kind: typing.Literal["turn_end"] = "turn_end"


class SetupEvent(GameEvent):
    """Event #5: one player's setup phase (selecting birds, food, and bonus).

    ``kept_card_names`` and ``kept_bonus_name`` are filled in by the recorder
    when the ``SetupChoice`` / ``BonusCardChoice`` decisions resolve."""

    kind: typing.Literal["setup"] = "setup"
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

    kind: typing.Literal["round_goal"] = "round_goal"
    round_idx: int
    description: str
    counts: list[int] = []
    vps: list[int] = []


class FinalScoringEvent(GameEvent):
    """Event #6b: the game's final scoring summary.

    ``scores`` holds one :class:`FinalScoreBreakdown` per seat, in seat order."""

    kind: typing.Literal["final_scoring"] = "final_scoring"
    scores: list[FinalScoreBreakdown] = []


class LooseEvent(GameEvent):
    """Catch-all bucket for a decision recorded when no other event is open.

    Used as the auto-wrap target when ``EventRecorder.record_decision`` or
    ``record_forced`` fires outside any explicit ``begin_*/end_event`` bracket
    (e.g. a stray power decision not yet wired into the call-site graph)."""

    kind: typing.Literal["loose"] = "loose"


type AnyGameEvent = typing.Annotated[
    MainActionEvent
    | PlayBirdEvent
    | WhitePowerEvent
    | ReactionEvent
    | ActivateBaseEvent
    | ActivateBrownEvent
    | ExtraPlayEvent
    | TurnEndEvent
    | SetupEvent
    | RoundGoalEvent
    | FinalScoringEvent
    | LooseEvent,
    pydantic.Field(discriminator="kind"),
]
"""Discriminated union of every concrete :class:`GameEvent`.

``GameEvent.children`` and ``PhaseNode.events`` are annotated with this rather
than ``list[GameEvent]`` so serialization preserves each subclass's own
fields."""


# ---------------------------------------------------------------------------
# Phase and tree containers


class PhaseNode(pydantic.BaseModel):
    """One navigable phase: a sequential group of :class:`GameEvent` objects.

    ``kind`` matches the phase-boundary strings used by the HTML handler:
    ``"game_start"``, ``"setup"``, ``"round"``, ``"turn"``, or ``"game_end"``.
    ``round_idx`` and ``turn_idx`` make the phase self-describing so consumers
    do not have to recover them by positional ``zip`` against the reporting
    layer's ``PhaseRecord`` list; both are ``None`` on phases where they do not
    apply (``turn_idx`` is the 1-based turn number within the round)."""

    kind: str
    round_idx: int | None = None
    turn_idx: int | None = None
    events: list[AnyGameEvent] = []


class GameEventTree(pydantic.BaseModel):
    """The complete event tree for one game, organized as an ordered list of
    :class:`PhaseNode` objects whose positions are 1-to-1 with the HTML
    handler's :class:`~wingspan.reporting.game_log_html.PhaseRecord` list."""

    phases: list[PhaseNode] = []


# ---------------------------------------------------------------------------
# Resolve the forward reference from GameEvent.children to AnyGameEvent, which
# is only defined once every subclass exists.  Each subclass inherits the
# unresolved ``children`` field and so needs its own rebuild.

GameEvent.model_rebuild()
MainActionEvent.model_rebuild()
PlayBirdEvent.model_rebuild()
WhitePowerEvent.model_rebuild()
ReactionEvent.model_rebuild()
ActivateBaseEvent.model_rebuild()
ActivateBrownEvent.model_rebuild()
ExtraPlayEvent.model_rebuild()
TurnEndEvent.model_rebuild()
SetupEvent.model_rebuild()
RoundGoalEvent.model_rebuild()
FinalScoringEvent.model_rebuild()
LooseEvent.model_rebuild()
PhaseNode.model_rebuild()
