"""Decision-point system.

Wingspan has a branching factor that's awkward to flatten. Instead the engine
exposes a sequence of *decision points*: at each, the agent is given a list of
legal choices and picks one. This makes both human CLI input and RL action
masking straightforward.

The shape of this module is an explicit class hierarchy:

* ``Choice`` is the abstract base of every legal option an agent can pick.
  Each subclass models one *data shape* — e.g. ``BirdChoice`` carries a
  ``Bird`` field, ``HabitatChoice`` carries a ``Habitat``, ``SetupChoice``
  carries the combined hand/food/bonus pick. There is no opaque ``payload``;
  every option's data is reachable through named typed attributes.

* ``Decision`` is generic in the Choice subtype it accepts:
  ``Decision[BirdChoice]`` is a decision whose ``choices`` list contains
  ``BirdChoice`` instances. Each old ``DecisionType`` enum entry now has a
  dedicated ``Decision`` subclass that pins the Choice type and adds any
  extra typed context the decision needs (e.g. ``SetupDecision`` carries
  ``dealt_cards`` and ``dealt_bonus`` so a CLI can present a sub-dialog).

The starting hand / starting food / bonus-card pick is exposed as a single
``SetupDecision`` carrying every legal combination (504 choices for the
standard 5-card / 2-bonus deal) so the RL model sees one fixed-shape action
space at setup. Interactive front-ends are expected to split the pick into
sub-dialogs and assemble the answer as a ``SetupChoice``.
"""

from __future__ import annotations

import enum
import typing

import pydantic

from wingspan import cards, state

# ---------------------------------------------------------------------------
# Main-action enum (the four top-level cube-spend options)
#
# Playing a bird *is* one of these now (``PLAY_BIRD``). ``MainActionDecision``
# picks only the action *type*; choosing which bird to play, in which habitat,
# for which payment is a separate follow-up ``PlayBirdDecision``, so "which
# action?" and "which bird to play?" are scored by different heads. ``PLAY_BIRD``
# is offered only when the player has at least one legal play.


class MainAction(enum.StrEnum):
    GAIN_FOOD = "gain_food"
    LAY_EGGS = "lay_eggs"
    DRAW_CARDS = "draw_cards"
    PLAY_BIRD = "play_bird"


# ---------------------------------------------------------------------------
# Choice hierarchy
#
# One class per data shape. Multiple decision types may reuse the same Choice
# subclass when their options carry the same kind of data — e.g. both
# ``GainFoodDecision`` and ``SpendFoodDecision`` use
# ``FoodChoice`` (modulo the skip variant).


class Choice(pydantic.BaseModel):
    """Abstract base of every legal option presented to an agent.

    ``label`` is a short human-readable description used by the CLI and the
    game log. Subclasses add the typed fields that carry the choice's data.
    """

    label: str


class SkipChoice(Choice):
    """Decline an optional decision. Carries no extra data beyond the label."""


class PayCostChoice(Choice):
    """Accept a fixed, power-defined exchange — pay X to get Y.

    Used for the yes/no "accept exchange?" decisions (``AcceptExchangeDecision``):
    discard 1 egg to draw a card (Wetland trade), or discard 1 food to tuck N
    cards from the deck (Sandhill Crane etc.). Both sides of the trade are
    fully determined by the power/action, so the agent's only decision is
    accept vs. skip — but the trade's *terms* are surfaced as typed fields so
    the commit-to-cost head can weigh what is gained against what is paid
    instead of scoring a featureless token. A field left at its default means
    "this resource is not part of this exchange".

    Distinct from ``FoodChoice`` because the agent isn't picking *which* food;
    they're confirming the offered exchange. The human-readable ``label`` names
    the specific cost."""

    paid_food: cards.Food | None = None  # a specific food token paid, if any
    paid_egg_count: int = 0  # eggs removed as payment
    gained_card_count: int = 0  # cards drawn into hand
    gained_tuck_count: int = 0  # cards tucked behind the bird (VP + tuck count)


class MainActionChoice(Choice):
    action: MainAction


class BirdChoice(Choice):
    """Pick a ``Bird`` (typically from a hand or a drawn pile)."""

    bird: cards.Bird


class PlayBirdChoice(Choice):
    """Play a specific bird in a specific habitat for a specific food payment.

    Offered by ``PlayBirdDecision`` — the menu reached both when the main
    action is ``PLAY_BIRD`` and for each power-granted extra play. A single
    ``PlayBirdChoice`` bundles the bird, the target habitat, and one fully
    specified food payment, so the habitat and food-payment picks are made in
    one step rather than as separate follow-up decisions. A bird playable in
    two habitats, or payable two ways, yields one ``PlayBirdChoice`` per legal
    ``(habitat, payment)`` combination.

    The egg cost is deliberately *not* folded in — it stays a separate
    follow-up decision (``RemoveEggDecision``)."""

    bird: cards.Bird
    habitat: cards.Habitat
    payment: state.FoodPool


class PlayedBirdChoice(Choice):
    """Pick a bird currently in play, by direct reference. Distinct from
    ``BoardTargetChoice`` (which identifies a board cell) because some
    powers operate on the bird object itself rather than its slot."""

    played_bird: state.PlayedBird


class HabitatChoice(Choice):
    habitat: cards.Habitat


class FoodChoice(Choice):
    food: cards.Food


class BoardTargetChoice(Choice):
    """A specific bird on the asking player's board, identified by
    ``(habitat, slot)``."""

    habitat: cards.Habitat
    slot: int


class BonusCardChoice(Choice):
    bonus_card: cards.BonusCard


class DrawSourceChoice(Choice):
    """A draw source for the DRAW_CARDS action: either a tray slot or the
    top of the deck. ``tray_index`` and ``bird`` are set when
    ``source == 'tray'`` (the specific face-up card on offer) and are both
    ``None`` for the deck (a blind draw)."""

    source: typing.Literal["tray", "deck"]
    tray_index: int | None = None
    bird: cards.Bird | None = None


class PlayerIdChoice(Choice):
    player_id: int


class SetupChoice(Choice):
    """Combined starting-hand / starting-food / bonus-card pick.

    See ``SetupDecision`` for the enumeration: the setup phase is exposed as
    a single Decision whose choices cover every legal combination (504 for
    the standard 5-card / 2-bonus deal).

    The player starts with one of each food; keeping a card costs one food,
    so ``kept_foods`` is a subset of distinct ``Food`` values whose size is
    ``len(cards.ALL_FOODS) - len(kept_cards)`` — i.e. the foods the player
    retains after paying for ``kept_cards``. Framing the choice as "keep
    cards AND keep foods" (rather than "keep cards AND discard foods") keeps
    the two subsets symmetric and matches how an interactive UI naturally
    presents the pick.
    """

    kept_cards: tuple[cards.Bird, ...]
    kept_foods: tuple[cards.Food, ...]
    bonus_card: cards.BonusCard | None


# ---------------------------------------------------------------------------
# Decision hierarchy
#
# ``Decision`` is generic in the Choice subtype it accepts. Each old
# ``DecisionType`` enum entry becomes a Decision subclass that pins the
# allowed Choice type(s); decisions that may be skipped union ``SkipChoice``
# into their parameterization so the consumer can branch on type.


class Decision[C: Choice](pydantic.BaseModel):
    """Abstract base of every decision point.

    ``choices`` carries the legal options. The Choice subtype is fixed by
    the subclass parameterization (e.g. ``Decision[BirdChoice]``), so a
    consumer that constructs a ``BirdPowerPickBirdFromHandDecision`` can rely on
    every option being a ``BirdChoice``.
    """

    player_id: int
    prompt: str
    choices: typing.Annotated[list[C], pydantic.Field(min_length=1)]


class MainActionDecision(Decision[MainActionChoice]):
    """Top-of-turn cube-spend pick — the action *type* only.

    The choices are the three habitat-row actions (Gain Food / Lay Eggs / Draw
    Cards) plus ``PLAY_BIRD`` when the player has at least one legal play. This
    decision picks only *which* action; if ``PLAY_BIRD`` is chosen, *which* bird
    to play (where, paid how) is a separate follow-up ``PlayBirdDecision``. The
    split keeps "which action?" and "which bird to play?" as distinct judgments
    on distinct scoring heads (``MACRO_ACTION`` vs ``PLAY_BIRD``)."""


class SetupDecision(Decision[SetupChoice]):
    """Combined hand / food / bonus pick presented as a single decision.

    The dealt cards and bonus cards are surfaced as typed fields so an
    interactive agent (e.g. the CLI) can present a multi-step sub-dialog
    without parsing them out of an opaque context dict.
    """

    dealt_cards: list[cards.Bird]
    dealt_bonus: list[cards.BonusCard]


class PlayBirdDecision(Decision[PlayBirdChoice]):
    """Pick which bird to play, where, and paid how — one ``PlayBirdChoice`` per
    legal ``(bird, habitat, food payment)`` the player can make right now.

    Reached in two contexts: when the turn's main action is ``PLAY_BIRD`` (the
    follow-up to ``MainActionDecision``), and for each power-granted extra play
    (filtered to the granting power's habitat, if any). Both are the same
    judgment — "which bird is worth playing, where, paid how?" — so both route
    to the ``PLAY_BIRD`` head; the egg cost stays a follow-up
    (``RemoveEggDecision``)."""


class RemoveEggDecision(Decision[BoardTargetChoice | SkipChoice]):
    """Pick which played bird to remove an egg from. Used wherever an egg is
    *spent*: the play-bird egg cost, the Wetland egg→card trade, and the
    discard-egg-for-wild power. ``SkipChoice`` is offered when the removal is
    optional. (Formerly ``PlayBirdPickEggToPayDecision`` — renamed because the
    judgment "which egg can I best afford to lose?" is the same across every
    caller, not specific to playing a bird.)"""


class GainFoodDecision(Decision[FoodChoice | SkipChoice]):
    """Pick which food to gain — a birdfeeder die or a token from the supply.

    The single "which food advances my plans?" decision, unified across every
    trigger: the main Gain Food action, the each-player feeder gain, and every
    power that grants a food (a named feeder die, any die, fewest-forest, the
    wild half of discard-egg-for-wild, the gain half of Green Heron's trade).
    ``SkipChoice`` is offered only where the gain is optional (e.g. Green Heron
    may decline the trade); mandatory gains offer food choices only. (Formerly
    ``GainFoodPickDieDecision`` — widened past the feeder die and renamed.)"""


class SpendFoodDecision(Decision[FoodChoice | SkipChoice]):
    """Pick which food to give up — the inverse of ``GainFoodDecision``:
    "which food can I most afford to part with?" Used by the lose half of Green
    Heron's trade (discard 1 food back to the supply). ``SkipChoice`` is offered
    where declining the spend is legal."""


class LayEggDecision(Decision[BoardTargetChoice | SkipChoice]):
    """Pick which played bird to lay an egg on. ``SkipChoice`` is offered
    when the lay is optional (most pink reactors)."""


class DrawCardsPickSourceDecision(Decision[DrawSourceChoice]):
    """Pick whether to draw from the deck or from a specific tray slot."""


class GainExtraFoodDecision(Decision[BirdChoice | SkipChoice]):
    """Optional Forest conversion: discard one card from hand to take one
    extra food die. Each ``BirdChoice`` is a candidate card to discard;
    ``SkipChoice`` declines. Offered once, only when the cube lands on a trade
    space (an odd number of birds in the row). The judgment is *which card to
    give up* — a bird-discard valuation — so it shares the discard head; the
    resulting extra food die is then a separate ``GainFoodDecision``."""


class LayExtraEggsDecision(Decision[FoodChoice | SkipChoice]):
    """Optional Grassland conversion: spend one food to lay one extra egg.
    Each ``FoodChoice`` is a food type the player can spend; ``SkipChoice``
    declines. Offered once, only when the cube lands on a trade space (an odd
    number of birds in the row)."""


class AcceptExchangeDecision(Decision[PayCostChoice | SkipChoice]):
    """Accept a fixed, power-defined exchange — yes/no. Used wherever the terms
    are fully determined and the only judgment is "is this trade worth it given
    my position and the round goal?": the Wetland egg→card conversion (the bird
    the egg comes off is a follow-up ``RemoveEggDecision``) and the
    discard-food-to-tuck powers (Sandhill Crane etc.). The ``PayCostChoice``
    carries the trade terms as typed fields so the commit-to-cost head can weigh
    them; ``SkipChoice`` declines. Offered once for the trade-space conversion,
    only when the cube lands on a trade space (an odd number of birds in the
    row). (Subsumes the former ``DrawCardsConvertDecision`` and the pay-cost
    branch of ``BirdPowerPickFoodDecision``.)"""


class BirdPowerPickBirdFromHandDecision(Decision[BirdChoice]):
    """Power asks for a specific Bird from a drafted or drawn pile (e.g. the
    American Oystercatcher draft) — a bird-*acquisition* judgment."""


class BirdPowerPickPlayedBirdDecision(Decision[PlayedBirdChoice]):
    """Power asks for a bird currently in play by reference (e.g.
    move-bird-if-rightmost, repeat-brown-power)."""


class BirdPowerPickBonusCardDecision(Decision[BonusCardChoice]):
    """Power asks the player to keep one of several drawn bonus cards."""


class BirdPowerTuckFromHandDecision(Decision[BirdChoice | SkipChoice]):
    """Power asks the player to tuck a card from hand (or skip) — a
    bird-*discard* judgment (the card leaves hand to become a tuck)."""


class BirdPowerPickStartingPlayerDecision(Decision[PlayerIdChoice]):
    """Pink/round-start power that designates the next round's starter."""


class BirdPowerPickHabitatDecision(Decision[HabitatChoice]):
    """Power asks the player to designate a habitat target."""


# ---------------------------------------------------------------------------
# Stable iteration order for the encoder's decision-type one-hot stripe.
# Append to the end when adding new decision subclasses so the existing
# stripe ordering is preserved for trained checkpoints that care about it.

ALL_DECISION_CLASSES: tuple[type[Decision[typing.Any]], ...] = (
    MainActionDecision,
    PlayBirdDecision,
    SetupDecision,
    RemoveEggDecision,
    GainFoodDecision,
    SpendFoodDecision,
    LayEggDecision,
    DrawCardsPickSourceDecision,
    BirdPowerPickBirdFromHandDecision,
    BirdPowerPickPlayedBirdDecision,
    BirdPowerPickBonusCardDecision,
    BirdPowerTuckFromHandDecision,
    BirdPowerPickStartingPlayerDecision,
    BirdPowerPickHabitatDecision,
    GainExtraFoodDecision,
    LayExtraEggsDecision,
    AcceptExchangeDecision,
)


# ---------------------------------------------------------------------------
# Judgment-family taxonomy
#
# The RL model groups the decision classes above into *judgment families* —
# one per distinct skill a player exercises (see ``DECISIONS.md`` §5). Each
# family becomes one scoring head on the shared trunk: several as-built
# ``Decision`` classes collapse onto one family when they ask the same
# underlying judgment (e.g. every "is this bird worth keeping/playing?"
# decision), so the policy specializes per skill rather than per trigger.
#
# The family of a decision is a pure function of its class
# (``family_for`` / ``family_index_for``). ``ALL_DECISION_FAMILIES`` pins the
# stable head order — append, never reorder, when adding a family, so existing
# trained checkpoints keep their head→family alignment (mirrors the
# ``ALL_DECISION_CLASSES`` contract for the decision-type one-hot).


class DecisionFamily(enum.StrEnum):
    """The judgment a decision exercises — the unit of policy specialization.

    Several as-built ``Decision`` classes map to one family when they share an
    underlying judgment, and the RL model trains one scoring head per family
    rather than one per decision class. See ``DECISIONS.md`` §5 for the
    rationale and the full per-class mapping.
    """

    SETUP = "setup"
    MACRO_ACTION = "macro_action"
    BIRD_ACQUISITION = "bird_acquisition"
    BIRD_DISCARD = "bird_discard"
    GAIN_FOOD = "gain_food"
    SPEND_FOOD = "spend_food"
    EGG_PLACEMENT = "egg_placement"
    EGG_REMOVAL = "egg_removal"
    COMMIT_TO_COST = "commit_to_cost"
    BONUS_VALUATION = "bonus_valuation"
    HABITAT_PLACEMENT = "habitat_placement"
    MISC_RARE = "misc_rare"
    PLAY_BIRD = "play_bird"


ALL_DECISION_FAMILIES: tuple[DecisionFamily, ...] = (
    DecisionFamily.SETUP,
    DecisionFamily.MACRO_ACTION,
    DecisionFamily.BIRD_ACQUISITION,
    DecisionFamily.BIRD_DISCARD,
    DecisionFamily.GAIN_FOOD,
    DecisionFamily.SPEND_FOOD,
    DecisionFamily.EGG_PLACEMENT,
    DecisionFamily.EGG_REMOVAL,
    DecisionFamily.COMMIT_TO_COST,
    DecisionFamily.BONUS_VALUATION,
    DecisionFamily.HABITAT_PLACEMENT,
    DecisionFamily.MISC_RARE,
    DecisionFamily.PLAY_BIRD,
)

# Per-class assignment. Keyed on the concrete decision class so routing is a
# pure function of the class. Bird valuation is split by direction (DECISIONS.md
# §3.3): *acquiring* a bird ("which do I take?") and *giving one up* ("which do
# I lose?") are opposite judgments and route to separate heads. Choosing the
# turn's action *type* (``MainActionDecision`` -> ``MACRO_ACTION``) is split from
# choosing *which bird to play* (``PlayBirdDecision`` -> ``PLAY_BIRD``); the
# latter serves both the main-action PLAY_BIRD branch and power-granted extra
# plays, since "which bird, where, paid how?" is one judgment in both.
_DECISION_FAMILY: dict[type[Decision[typing.Any]], DecisionFamily] = {
    SetupDecision: DecisionFamily.SETUP,
    MainActionDecision: DecisionFamily.MACRO_ACTION,
    PlayBirdDecision: DecisionFamily.PLAY_BIRD,
    DrawCardsPickSourceDecision: DecisionFamily.BIRD_ACQUISITION,
    BirdPowerPickBirdFromHandDecision: DecisionFamily.BIRD_ACQUISITION,
    BirdPowerTuckFromHandDecision: DecisionFamily.BIRD_DISCARD,
    GainExtraFoodDecision: DecisionFamily.BIRD_DISCARD,
    GainFoodDecision: DecisionFamily.GAIN_FOOD,
    SpendFoodDecision: DecisionFamily.SPEND_FOOD,
    LayExtraEggsDecision: DecisionFamily.SPEND_FOOD,
    LayEggDecision: DecisionFamily.EGG_PLACEMENT,
    RemoveEggDecision: DecisionFamily.EGG_REMOVAL,
    AcceptExchangeDecision: DecisionFamily.COMMIT_TO_COST,
    BirdPowerPickBonusCardDecision: DecisionFamily.BONUS_VALUATION,
    BirdPowerPickHabitatDecision: DecisionFamily.HABITAT_PLACEMENT,
    BirdPowerPickPlayedBirdDecision: DecisionFamily.MISC_RARE,
    BirdPowerPickStartingPlayerDecision: DecisionFamily.MISC_RARE,
}

_DECISION_FAMILY_INDEX: dict[type[Decision[typing.Any]], int] = {
    cls: ALL_DECISION_FAMILIES.index(family) for cls, family in _DECISION_FAMILY.items()
}


def family_for(decision_class: type[Decision[typing.Any]]) -> DecisionFamily:
    """Return the judgment family a decision class belongs to.

    Pure function of the class — a decision always routes to the same policy
    head. Raises ``KeyError`` for an unregistered class, which is the intended
    failure mode: a new ``Decision`` subclass must be assigned a family here
    before it can be trained.
    """
    return _DECISION_FAMILY[decision_class]


def family_index_for(decision_class: type[Decision[typing.Any]]) -> int:
    """Return the index of a decision class's family in ``ALL_DECISION_FAMILIES``.

    This is the scoring-head index the RL model routes the decision through.
    """
    return _DECISION_FAMILY_INDEX[decision_class]
