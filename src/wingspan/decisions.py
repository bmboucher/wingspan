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
# Main-action enum (the three habitat-row cube-spend options)
#
# Playing a bird is *not* one of these. Each legal way to play a bird is
# surfaced as its own ``PlayBirdChoice`` at the main-action stage (see
# ``MainActionDecision``), so the main-action pick is "3 habitat actions + one
# option per (bird, habitat, food payment) the player can play right now".


class MainAction(enum.StrEnum):
    GAIN_FOOD = "gain_food"
    LAY_EGGS = "lay_eggs"
    DRAW_CARDS = "draw_cards"


# ---------------------------------------------------------------------------
# Choice hierarchy
#
# One class per data shape. Multiple decision types may reuse the same Choice
# subclass when their options carry the same kind of data — e.g. both
# ``GainFoodPickDieDecision`` and ``BirdPowerPickFoodDecision`` use
# ``FoodChoice`` (modulo the skip / pay-cost variants).


class Choice(pydantic.BaseModel):
    """Abstract base of every legal option presented to an agent.

    ``label`` is a short human-readable description used by the CLI and the
    game log. Subclasses add the typed fields that carry the choice's data.
    """

    label: str


class SkipChoice(Choice):
    """Decline an optional decision. Carries no extra data beyond the label."""


class PayCostChoice(Choice):
    """Accept the fixed, power-defined cost of an optional power.

    Used for powers like 'discard 1 [seed] to tuck 2 cards from the deck':
    both the food to be paid and the resulting effect are fully determined
    by the bird's power, so the agent's only decision is yes-pay vs. skip.
    This choice carries no fields — its *type* (alongside ``SkipChoice``) is
    the entire signal. The human-readable ``label`` names the specific cost.

    Distinct from ``FoodChoice`` because the agent isn't picking *which*
    food; they're confirming the offered exchange."""


class MainActionChoice(Choice):
    action: MainAction


class BirdChoice(Choice):
    """Pick a ``Bird`` (typically from a hand or a drawn pile)."""

    bird: cards.Bird


class PlayBirdChoice(Choice):
    """Play a specific bird in a specific habitat for a specific food payment.

    Surfaced at the main-action stage (see ``MainActionDecision``): a single
    ``PlayBirdChoice`` bundles the bird, the target habitat, and one fully
    specified food payment, so the habitat- and food-payment sub-decisions
    that ``BirdChoice`` used to defer are folded into the main-action pick. A
    bird playable in two habitats, or payable two ways, yields one
    ``PlayBirdChoice`` per legal ``(habitat, payment)`` combination.

    The egg cost is deliberately *not* folded in — it stays a separate
    follow-up decision (``PlayBirdPickEggToPayDecision``)."""

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


class FoodPaymentChoice(Choice):
    """A fully-specified payment for paying a bird's cost."""

    payment: state.FoodPool


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
    consumer that constructs a ``PlayBirdPickCardDecision`` can rely on
    every option being a ``BirdChoice``.
    """

    player_id: int
    prompt: str
    choices: typing.Annotated[list[C], pydantic.Field(min_length=1)]


class MainActionDecision(Decision[MainActionChoice | PlayBirdChoice]):
    """Top-of-turn cube-spend pick.

    The choices are the three habitat-row actions (``MainActionChoice``) plus
    one ``PlayBirdChoice`` per legal ``(bird, habitat, food payment)`` the
    player can play right now. Picking a ``PlayBirdChoice`` commits to that
    bird, habitat, and payment in one step (only the egg cost remains a
    follow-up decision); picking a ``MainActionChoice`` runs the corresponding
    habitat action."""


class SetupDecision(Decision[SetupChoice]):
    """Combined hand / food / bonus pick presented as a single decision.

    The dealt cards and bonus cards are surfaced as typed fields so an
    interactive agent (e.g. the CLI) can present a multi-step sub-dialog
    without parsing them out of an opaque context dict.
    """

    dealt_cards: list[cards.Bird]
    dealt_bonus: list[cards.BonusCard]


class PlayBirdPickCardDecision(Decision[BirdChoice]):
    """Choose which bird (from hand) to play."""


class PlayBirdPickHabitatDecision(Decision[HabitatChoice]):
    """Choose which habitat to place a multi-habitat bird in."""


class PlayBirdPickFoodPaymentDecision(Decision[FoodPaymentChoice]):
    """Choose a specific food-payment combination for a played bird."""


class PlayBirdPickEggToPayDecision(Decision[BoardTargetChoice | SkipChoice]):
    """Pick which played bird to remove an egg from when paying the egg
    cost. ``SkipChoice`` is offered when the cost is optional."""


class GainFoodPickDieDecision(Decision[FoodChoice]):
    """Pick which face of the birdfeeder die to take."""


class LayEggPickBirdDecision(Decision[BoardTargetChoice | SkipChoice]):
    """Pick which played bird to lay an egg on. ``SkipChoice`` is offered
    when the lay is optional (most pink reactors)."""


class DrawCardsPickSourceDecision(Decision[DrawSourceChoice]):
    """Pick whether to draw from the deck or from a specific tray slot."""


class GainFoodConvertDecision(Decision[BirdChoice | SkipChoice]):
    """Optional Forest conversion: discard one card from hand to take one
    extra food die. Each ``BirdChoice`` is a candidate card to discard;
    ``SkipChoice`` declines. Offered once, only when the cube lands on a trade
    space (an odd number of birds in the row)."""


class LayEggsConvertDecision(Decision[FoodChoice | SkipChoice]):
    """Optional Grassland conversion: spend one food to lay one extra egg.
    Each ``FoodChoice`` is a food type the player can spend; ``SkipChoice``
    declines. Offered once, only when the cube lands on a trade space (an odd
    number of birds in the row)."""


class DrawCardsConvertDecision(Decision[PayCostChoice | SkipChoice]):
    """Optional Wetland conversion: discard one egg to draw one extra card.
    ``PayCostChoice`` accepts the fixed egg cost (the bird the egg comes off
    is a follow-up pick); ``SkipChoice`` declines. Offered once, only when the
    cube lands on a trade space (an odd number of birds in the row)."""


class BirdPowerPickFoodDecision(Decision[FoodChoice | SkipChoice | PayCostChoice]):
    """A power-driven food pick. ``PayCostChoice`` covers the 'accept the
    offered cost' branch of tuck-from-deck-paid powers, where the food and
    reward are both fixed by the bird's power text."""


class BirdPowerPickBirdFromHandDecision(Decision[BirdChoice]):
    """Power asks for a specific Bird from a drafted or drawn pile."""


class BirdPowerPickPlayedBirdDecision(Decision[PlayedBirdChoice]):
    """Power asks for a bird currently in play by reference (e.g.
    move-bird-if-rightmost, repeat-brown-power)."""


class BirdPowerPickBonusCardDecision(Decision[BonusCardChoice]):
    """Power asks the player to keep one of several drawn bonus cards."""


class BirdPowerTuckFromHandDecision(Decision[BirdChoice | SkipChoice]):
    """Power asks the player to tuck a card from hand (or skip)."""


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
    SetupDecision,
    PlayBirdPickCardDecision,
    PlayBirdPickHabitatDecision,
    PlayBirdPickFoodPaymentDecision,
    PlayBirdPickEggToPayDecision,
    GainFoodPickDieDecision,
    LayEggPickBirdDecision,
    DrawCardsPickSourceDecision,
    BirdPowerPickFoodDecision,
    BirdPowerPickBirdFromHandDecision,
    BirdPowerPickPlayedBirdDecision,
    BirdPowerPickBonusCardDecision,
    BirdPowerTuckFromHandDecision,
    BirdPowerPickStartingPlayerDecision,
    BirdPowerPickHabitatDecision,
    GainFoodConvertDecision,
    LayEggsConvertDecision,
    DrawCardsConvertDecision,
)
