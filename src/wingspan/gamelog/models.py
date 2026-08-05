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

import enum
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


# ---------------------------------------------------------------------------
# Effects: the ledger of what actually changed
#
# NAMING NOTE: :class:`wingspan.cards.schema.EffectKind` is a *different* thing —
# it is the static, per-card declaration of what a bird's power is supposed to
# do.  The classes below record what a game *actually did*, one entry per state
# mutation.  The two never mix: card data is parsed into ``EffectKind``, and
# executing it emits these.
#
# Card-vs-bird field naming is consistent throughout: ``card`` is a card moving
# between zones (hand / deck / tray / board), ``bird`` is a bird already in play
# that is being targeted.


class FoodSource(enum.StrEnum):
    """Where a gained food token came from."""

    FEEDER = "feeder"
    SUPPLY = "supply"
    CACHE = "cache"
    OPPONENT = "opponent"
    DEAL = "deal"


class CardSource(enum.StrEnum):
    """Where a card moving into a zone came from."""

    DECK = "deck"
    TRAY = "tray"
    HAND = "hand"
    DEAL = "deal"
    OPPONENT = "opponent"


class EffectPurpose(enum.StrEnum):
    """Why a resource was spent or gained — the intent behind the mutation.

    Recording intent is the reason the ledger is explicit rather than derived
    from a state diff: a diff can show that two food left the supply, but not
    that one paid a bird's cost and the other bought an egg."""

    PLAY_BIRD = "play_bird"
    CONVERT = "convert"
    POWER = "power"
    SETUP = "setup"
    TURN_END = "turn_end"
    ACTION = "action"


class Effect(SubEvent):
    """Abstract base for one recorded state mutation.

    Every concrete subclass is emitted by :mod:`wingspan.engine.ledger`, which
    performs the mutation and records the effect in the same call so the two
    cannot drift.  ``tests/test_gamelog_ledger.py`` replays the whole ledger
    against the final :class:`~wingspan.state.GameState` to prove it."""


class GainFoodEffect(Effect):
    """``amount`` food tokens of one type entered a player's supply."""

    kind: typing.Literal["gain_food"] = "gain_food"
    food: str
    amount: int
    source: FoodSource


class SpendFoodEffect(Effect):
    """``amount`` food tokens of one type left a player's supply."""

    kind: typing.Literal["spend_food"] = "spend_food"
    food: str
    amount: int
    purpose: EffectPurpose


class CacheFoodEffect(Effect):
    """Food was placed on a bird in play.

    ``from_supply`` distinguishes the two printed forms: caching food the player
    already holds (which also decrements their supply, recorded as its own
    :class:`SpendFoodEffect`) from a power that caches straight off the supply
    without the food ever passing through the player's pool."""

    kind: typing.Literal["cache_food"] = "cache_food"
    bird: str
    food: str
    amount: int
    from_supply: bool = False


class UncacheFoodEffect(Effect):
    """Food was removed from a bird's cache."""

    kind: typing.Literal["uncache_food"] = "uncache_food"
    bird: str
    food: str
    amount: int


class LayEggEffect(Effect):
    """``count`` eggs were added to one bird in play."""

    kind: typing.Literal["lay_egg"] = "lay_egg"
    bird: str
    habitat: str
    slot: int
    count: int = 1
    purpose: EffectPurpose = EffectPurpose.ACTION


class RemoveEggEffect(Effect):
    """``count`` eggs were removed from one bird in play."""

    kind: typing.Literal["remove_egg"] = "remove_egg"
    bird: str
    habitat: str
    slot: int
    count: int = 1
    purpose: EffectPurpose = EffectPurpose.ACTION


class DrawCardEffect(Effect):
    """A card entered a player's hand.

    A ``source`` of :attr:`CardSource.DECK` makes this a **reveal**: the card's
    identity was hidden until this moment, and this record is the only place it
    appears."""

    kind: typing.Literal["draw_card"] = "draw_card"
    card: str
    source: CardSource
    tray_slot: int | None = None


class DiscardCardEffect(Effect):
    """A card left a player's hand for the discard pile."""

    kind: typing.Literal["discard_card"] = "discard_card"
    card: str
    purpose: EffectPurpose = EffectPurpose.POWER


class TuckCardEffect(Effect):
    """A card was tucked behind a bird in play.

    A ``source`` of :attr:`CardSource.DECK` makes this a **reveal** — the card
    goes from the hidden deck straight under a bird without ever being seen."""

    kind: typing.Literal["tuck_card"] = "tuck_card"
    card: str
    bird: str
    source: CardSource


class PlayBirdEffect(Effect):
    """A bird card left the hand and was placed on the board."""

    kind: typing.Literal["play_bird_effect"] = "play_bird_effect"
    card: str
    habitat: str
    slot: int


class MoveBirdEffect(Effect):
    """A bird already in play relocated to another habitat row.

    Distinct from :class:`PlayBirdEffect`: no card leaves a hand, and the bird
    keeps its eggs, tucked cards and cache.  Recording a move as a second
    placement would double-count the bird on any board reconstruction."""

    kind: typing.Literal["move_bird"] = "move_bird"
    card: str
    from_habitat: str
    from_slot: int
    to_habitat: str
    to_slot: int


class PassCardEffect(Effect):
    """A card moved from one player's hand to another's."""

    kind: typing.Literal["pass_card"] = "pass_card"
    card: str
    to_player_id: int


class FeederRerollEffect(Effect):
    """The birdfeeder was rerolled — a **reveal** of the fresh dice faces.

    ``player_id`` is whoever caused the roll, which for a pink reactor is not
    necessarily the current player."""

    kind: typing.Literal["feeder_reroll"] = "feeder_reroll"
    faces: list[str] = []


class DiceRollEffect(Effect):
    """A dice-predator rolled outside the feeder — a **reveal** of the faces.

    The ``ROLL_NOT_IN_FEEDER_CACHE`` birds (Anhinga and similar) roll dice that
    never enter the birdfeeder, so their outcome is visible nowhere else."""

    kind: typing.Literal["dice_roll"] = "dice_roll"
    bird: str
    faces: list[str] = []


class TrayRefillEffect(Effect):
    """A face-up tray slot was restocked from the deck — a **reveal**."""

    kind: typing.Literal["tray_refill"] = "tray_refill"
    slot: int
    card: str


type AnyEffect = (
    GainFoodEffect
    | SpendFoodEffect
    | CacheFoodEffect
    | UncacheFoodEffect
    | LayEggEffect
    | RemoveEggEffect
    | DrawCardEffect
    | DiscardCardEffect
    | TuckCardEffect
    | PlayBirdEffect
    | MoveBirdEffect
    | PassCardEffect
    | FeederRerollEffect
    | DiceRollEffect
    | TrayRefillEffect
)
"""Union of every concrete :class:`Effect`.

The declared parameter type of ``EventRecorder.record_effect``: the abstract
:class:`Effect` base is not assignable to the discriminated ``sub_events`` list,
and naming the union keeps a new effect class from being silently accepted
without joining :data:`AnySubEvent`."""


type AnySubEvent = typing.Annotated[
    ForcedSubEvent
    | DecisionSubEvent
    | GainFoodEffect
    | SpendFoodEffect
    | CacheFoodEffect
    | UncacheFoodEffect
    | LayEggEffect
    | RemoveEggEffect
    | DrawCardEffect
    | DiscardCardEffect
    | TuckCardEffect
    | PlayBirdEffect
    | MoveBirdEffect
    | PassCardEffect
    | FeederRerollEffect
    | DiceRollEffect
    | TrayRefillEffect,
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
    draw cards / play a bird).

    ``action`` is the chosen ``MainAction`` value, stamped by the recorder when
    the decision resolves (the same pattern as ``SetupEvent.kept_card_names``).
    It shares its vocabulary with :attr:`ActivateBaseEvent.action`, so one label
    table serves both."""

    kind: typing.Literal["main_action"] = "main_action"
    action: str | None = None


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


class RefillTrayEvent(GameEvent):
    """The face-up tray being restocked from the deck.

    Its own event because a refill belongs to no player action: it runs at the
    end of every turn and again when a round's tray is reset, both outside any
    open bracket.  ``player_id`` is the seat whose turn ended, or ``None`` for
    the end-of-round reset."""

    kind: typing.Literal["refill_tray"] = "refill_tray"


class DealEvent(GameEvent):
    """One player's opening deal: starting hand, bonus cards, one of each food.

    Sits in the ``game_start`` phase, before that player's ``setup`` phase,
    because both setup paths deal before the setup pick is presented — and the
    fixed-setup path deals *every* seat up front, so the deal cannot be folded
    into the setup bracket.  Its own event keeps the dealt-card reveals off the
    anonymous :class:`LooseEvent` bucket."""

    kind: typing.Literal["deal"] = "deal"


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
    | RefillTrayEvent
    | DealEvent
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
RefillTrayEvent.model_rebuild()
DealEvent.model_rebuild()
SetupEvent.model_rebuild()
RoundGoalEvent.model_rebuild()
FinalScoringEvent.model_rebuild()
LooseEvent.model_rebuild()
PhaseNode.model_rebuild()
