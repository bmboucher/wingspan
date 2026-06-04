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
import random
import typing

import pydantic

from wingspan import cards, state

# ---------------------------------------------------------------------------
# Main-action enum (the four top-level cube-spend options)
#
# Playing a bird *is* one of these now (``PLAY_BIRD``). ``MainActionDecision``
# picks only the action *type*; choosing which bird to play and in which habitat
# is a separate follow-up ``PlayBirdDecision``, so "which action?" and "which
# bird to play?" are scored by different heads. The costs are further follow-ups
# (``RemoveEggDecision`` then ``PayBirdFoodDecision``). ``PLAY_BIRD`` is offered
# only when the player has at least one legal play.


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

    Read the label through :meth:`display_label` rather than ``.label``
    directly: a subclass whose label is expensive to build (``SetupChoice``)
    can leave it empty at construction time and render it on first access.
    """

    label: str

    def display_label(self) -> str:
        """The human-readable label, computed on demand if not stored."""
        return self.label


class SkipChoice(Choice):
    """Decline an optional decision. Carries no extra data beyond the label."""


class PayCostChoice(Choice):
    """Accept a fixed, power-defined exchange — pay X to get Y.

    Used for the yes/no "accept exchange?" decisions (``AcceptExchangeDecision``):
    discard 1 egg to draw a card (Wetland trade), discard 1 food to tuck N
    cards from the deck (Sandhill Crane etc.), or use a power-granted extra
    bird play. Both sides of the trade are fully determined by the
    power/action, so the agent's only decision is accept vs. skip — but the
    trade's *terms* are surfaced as typed fields so the skip-optional head can
    weigh what is gained against what is paid instead of scoring a featureless
    token. A field left at its default means "this resource is not part of
    this exchange".

    The terms form a symmetric ``pay -> gain`` ledger over the game's
    resources (cards, food, eggs, bird plays). The deciding player's own flows
    are the ``paid_*`` / ``gained_*`` fields; the ``opp_gained_*`` fields capture
    what a shared-benefit power additionally grants the *opponent* (e.g. an
    optional "each player gains food" trade), so the skip-optional head can weigh
    that the trade also helps the opponent. The opponent only ever *receives* in
    such powers, so there are no opp-pay fields. Most fields are 0 for any given
    exchange.

    Distinct from ``FoodChoice`` because the agent isn't picking *which* food;
    they're confirming the offered exchange. The human-readable ``label`` names
    the specific cost."""

    # Self side — what the deciding player gives up / receives.
    paid_food: cards.Food | None = None  # a specific food token paid, if any
    paid_food_count: int = 0  # unspecified food tokens paid (when type is a follow-up)
    paid_card_count: int = 0  # cards discarded from hand as payment
    paid_egg_count: int = 0  # eggs removed as payment
    gained_food_count: int = 0  # food gained from the supply
    gained_egg_count: int = 0  # eggs laid
    gained_card_count: int = 0  # cards drawn into hand
    gained_tuck_count: int = 0  # cards tucked behind the bird (VP + tuck count)
    gained_play_count: int = 0  # extra bird plays unlocked (the extra-play accept)
    # Opponent side — what a shared-benefit power also grants the opponent.
    opp_gained_food_count: int = 0
    opp_gained_egg_count: int = 0
    opp_gained_card_count: int = 0
    opp_gained_tuck_count: int = 0


class TuckActivateChoice(Choice):
    """Commit to tucking ``cards_to_tuck`` card(s) from hand.

    The "yes" half of an ``ActivateTuckDecision`` gate — precedes the card
    selection (``BirdPowerTuckFromHandDecision``) so the skip/activate judgment
    is scored separately from the "which card?" judgment. ``cards_to_tuck``
    carries how many cards the player is committing to tuck (almost always 1)
    so the skip-optional head can weigh the tuck's value without inspecting the
    follow-up decision."""

    cards_to_tuck: int = 1


class ResetBirdfeederChoice(Choice):
    """Affirm the optional birdfeeder reset — "yes, reroll all the dice".

    Offered by ``ResetBirdfeederDecision`` as the yes side of a yes/no pick; the
    no side reuses ``SkipChoice``. Carries no data beyond the label because the
    reset itself is fully determined (reroll every die) — the only judgment is
    whether to take it, so a bare affirmative choice is all the head needs."""


class MainActionChoice(Choice):
    action: MainAction


class BirdChoice(Choice):
    """Pick a ``Bird`` (typically from a hand or a drawn pile)."""

    bird: cards.Bird


class PlayBirdChoice(Choice):
    """Play a specific bird in a specific habitat.

    Offered by ``PlayBirdDecision`` — the menu reached both when the main
    action is ``PLAY_BIRD`` and for each power-granted extra play. A bird
    playable in two habitats yields one ``PlayBirdChoice`` per legal habitat.

    The costs are deliberately *not* folded in: once the play is committed,
    the egg cost resolves via ``RemoveEggDecision`` and the food payment via
    ``PayBirdFoodDecision``, in that order. Only (bird, habitat) pairs with at
    least one legal payment and an affordable egg cost are offered, so a
    chosen play is always completable."""

    bird: cards.Bird
    habitat: cards.Habitat


class FoodPaymentChoice(Choice):
    """One complete food payment — a multiset of tokens covering a bird's
    printed cost.

    Offered by ``PayBirdFoodDecision`` after a play is committed: one choice
    per distinct legal payment from ``helpers.enumerate_payments`` (1-for-1
    matching, 2-for-1 substitution, and 1-of-any wild fills). Distinct from
    ``FoodChoice`` (a single token) because a payment is chosen as a whole —
    "pay fish+fish" vs "pay fish and 2 fruit for the fish slot" are competing
    complete payments, not independent token picks."""

    payment: state.FoodPool


class PlayedBirdChoice(Choice):
    """Pick a bird currently in play, by direct reference. Distinct from
    ``BoardTargetChoice`` (which identifies a board cell) because some
    powers operate on the bird object itself rather than its slot."""

    played_bird: state.PlayedBird


class HabitatChoice(Choice):
    habitat: cards.Habitat


class FoodChoice(Choice):
    """Pick a food to gain (or spend).

    ``from_choice_die`` distinguishes the two ways the invertebrate/seed
    *choice die* (the birdfeeder's combo face) can be taken from an otherwise
    identical plain-die gain: when ``True`` the food is taken specifically from
    a choice die rather than a single-food face. It is only ever ``True`` for
    ``INVERTEBRATE``/``SEED`` at a feeder gain where a choice die is showing, so
    the model can weigh burning a flexible choice die against spending a rigid
    single face (e.g. to deny an opponent the flexible die). Every other use of
    ``FoodChoice`` leaves it ``False``."""

    food: cards.Food
    from_choice_die: bool = False


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

    # ``label`` defaults empty: the setup deal enumerates 504 choices but the
    # agent only ever reads one (and self-play reads none), so the human label
    # is rendered lazily by ``display_label`` from the typed fields below
    # instead of being built for every option up front.
    label: str = ""
    kept_cards: tuple[cards.Bird, ...]
    kept_foods: tuple[cards.Food, ...]
    bonus_card: cards.BonusCard | None

    def display_label(self) -> str:
        """Render — and cache — the keep-cards / keep-foods / bonus summary on
        first access. Only the CLI and the illegal-choice error path need it."""
        if not self.label:
            kept_names = [bird.name for bird in self.kept_cards] or ["none"]
            food_names = [food.value for food in self.kept_foods] or ["none"]
            bonus = self.bonus_card.name if self.bonus_card is not None else "(none)"
            self.label = (
                f"keep:[{','.join(kept_names)}] foods:[{','.join(food_names)}] "
                f"bonus:{bonus}"
            )
        return self.label


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
    on distinct scoring heads (``MAIN_ACTION`` vs ``PLAY_BIRD``)."""


class SetupDecision(Decision[SetupChoice]):
    """Combined hand / food / bonus pick presented as a single decision.

    The dealt cards and bonus cards are surfaced as typed fields so an
    interactive agent (e.g. the CLI) can present a multi-step sub-dialog
    without parsing them out of an opaque context dict.
    """

    dealt_cards: list[cards.Bird]
    dealt_bonus: list[cards.BonusCard]


class PlayBirdDecision(Decision[PlayBirdChoice]):
    """Pick which bird to play and where — one ``PlayBirdChoice`` per legal
    ``(bird, habitat)`` pair the player can complete right now.

    Reached in two contexts: when the turn's main action is ``PLAY_BIRD`` (the
    follow-up to ``MainActionDecision``), and for each power-granted extra play
    (filtered to the granting power's habitat, if any). Both are the same
    judgment — "which bird is worth playing, and where?" — so both route to the
    ``PLAY_BIRD`` head. The costs are follow-ups, eggs then food:
    ``RemoveEggDecision`` (the ``PAY_EGG`` head) then ``PayBirdFoodDecision``
    (the ``SPEND_FOOD`` head), keeping the strategic pick separate from the
    logistics of paying for it."""


class RemoveEggDecision(Decision[BoardTargetChoice | SkipChoice]):
    """Pick which played bird to remove an egg from. Used wherever an egg is
    *spent*: the play-bird egg cost, the Wetland egg→card trade, and the
    discard-egg-for-wild power. ``SkipChoice`` is offered when the removal is
    optional. (Formerly ``PlayBirdPickEggToPayDecision`` — renamed because the
    judgment "which egg can I best afford to lose?" is the same across every
    caller, not specific to playing a bird.)"""


class PayBirdFoodDecision(Decision[FoodPaymentChoice]):
    """Pick how to pay a committed bird play's printed food cost — one
    ``FoodPaymentChoice`` per distinct legal payment multiset.

    The final play-bird follow-up (after the egg cost's ``RemoveEggDecision``):
    the bird and habitat are already settled, so this is pure spend logistics —
    "which tokens can I most afford to part with?" — and routes to the
    ``SPEND_FOOD`` head. Mandatory: no ``SkipChoice``, because the commitment
    happened upstream (``MainActionDecision``'s ``PLAY_BIRD`` pick or the
    extra-play accept). When only one payment is legal the decision is forced
    and ``Engine.ask`` auto-resolves it without consulting the agent.

    ``bird`` and ``habitat`` carry the committed play as typed context (every
    choice pays for the same play), so the encoder and an interactive UI can
    show what is being paid for without re-deriving it."""

    bird: cards.Bird
    habitat: cards.Habitat


class GainFoodDecision(Decision[FoodChoice | SkipChoice]):
    """Pick which food to gain — a birdfeeder die or a token from the supply.

    The single "which food advances my plans?" decision, unified across every
    trigger: the main Gain Food action, the each-player feeder gain, and every
    power that grants a food (a named feeder die, any die, fewest-forest, the
    wild half of discard-egg-for-wild, step 3 of Green Heron's wild-food trade).
    ``SkipChoice`` is offered only where the gain is optional; mandatory gains
    offer food choices only. (Formerly ``GainFoodPickDieDecision`` — widened
    past the feeder die and renamed.)"""


class SpendFoodDecision(Decision[FoodChoice | SkipChoice]):
    """Pick which food to give up — the inverse of ``GainFoodDecision``:
    "which food can I most afford to part with?" Used by step 2 of Green
    Heron's wild-food trade (discard 1 food back to the supply; mandatory after
    the upstream SKIP_OPTIONAL activation gate). ``SkipChoice`` is offered
    where declining the spend is legal."""


class LayEggDecision(Decision[BoardTargetChoice | SkipChoice]):
    """Pick which played bird to lay an egg on. ``SkipChoice`` is offered
    when the lay is optional (most pink reactors)."""


class DrawCardsPickSourceDecision(Decision[DrawSourceChoice]):
    """Pick whether to draw from the deck or from a specific tray slot."""


class DiscardBirdForFoodDecision(Decision[BirdChoice]):
    """Forest conversion step 2: mandatory bird discard from hand after the
    player committed to the exchange via a preceding ``AcceptExchangeDecision``.
    Each ``BirdChoice`` is a candidate card to discard; no ``SkipChoice`` is
    offered because the commitment already happened. The food-die pick that
    follows is a separate ``GainFoodDecision``."""


class SpendFoodForEggDecision(Decision[FoodChoice]):
    """Grassland conversion step 2: mandatory food spend after the player
    committed to the exchange via a preceding ``AcceptExchangeDecision``.
    Each ``FoodChoice`` is a food type the player can spend; no ``SkipChoice``
    is offered because the commitment already happened. The egg laid in step 3
    follows as a ``LayEggDecision``."""


class AcceptExchangeDecision(Decision[PayCostChoice | SkipChoice]):
    """Take a fixed, fully-determined optional exchange — yes/no. Used wherever
    the terms are settled up front and the only judgment is "is taking this
    worth it given my position and the round goal?": the Forest card→food
    conversion (step 1; the card to discard is a follow-up
    ``DiscardBirdForFoodDecision``), the Grassland food→egg conversion (step 1;
    the food to spend is a follow-up ``SpendFoodForEggDecision``), the Wetland
    egg→card conversion (the egg comes off via a follow-up
    ``RemoveEggDecision``), the discard-food-to-tuck powers (Sandhill Crane
    etc.), and the power-granted extra bird play (accept commits to the
    ``PlayBirdDecision`` menu; skip forfeits the credit). The ``PayCostChoice``
    carries the trade terms as typed fields so the skip-optional head can
    weigh them; ``SkipChoice`` declines. (Subsumes the former
    ``DrawCardsConvertDecision`` and the pay-cost branch of
    ``BirdPowerPickFoodDecision``.)"""


class BirdPowerPickBirdFromHandDecision(Decision[BirdChoice]):
    """Power asks for a specific Bird from a drafted or drawn pile —
    a bird-*acquisition* judgment. Currently has no call sites in the engine
    (retained in ``ALL_DECISION_CLASSES`` for checkpoint compatibility —
    do not remove or reorder it)."""


class BirdPowerPickPlayedBirdDecision(Decision[PlayedBirdChoice]):
    """Power asks for a bird currently in play by reference (e.g.
    move-bird-if-rightmost, repeat-brown-power)."""


class BirdPowerPickBonusCardDecision(Decision[BonusCardChoice]):
    """Power asks the player to keep one of several drawn bonus cards."""


class ActivateTuckDecision(Decision[TuckActivateChoice | SkipChoice]):
    """Gate before ``BirdPowerTuckFromHandDecision``: does the player want to tuck?

    The "yes" option is a ``TuckActivateChoice``; the "no" option is a
    ``SkipChoice``. By separating the activate/skip judgment from the card
    selection, the ``SKIP_OPTIONAL`` head scores whether tucking is worthwhile
    at all, while the ``DISCARD_BIRD`` head scores which card to give up.
    Offered before every tuck power — both white/brown and pink (Horned Lark)."""


class BirdPowerTuckFromHandDecision(Decision[BirdChoice]):
    """Mandatory card selection after ``ActivateTuckDecision`` accepted.

    Choices are the candidate cards from hand — no ``SkipChoice``, because the
    activate/skip judgment already happened upstream. The card that is chosen
    leaves hand to become a tucked card under the triggering bird."""


class BirdPowerDiscardFromHandDecision(Decision[BirdChoice]):
    """Power requires the player to discard a card from hand — a mandatory
    bird-discard judgment. Used wherever a power moves cards *out* of a
    player's hand as part of a draft/pass mechanic (e.g. the American
    Oystercatcher pass-and-return draft: active player passes 2 of 3 drawn
    cards to the opponent, opponent returns 1). No ``SkipChoice``: the
    commitment happened upstream (accepting the power's
    ``AcceptExchangeDecision``). Distinct from ``DiscardBirdForFoodDecision``
    (the Forest-trade step 2) and ``BirdPowerTuckFromHandDecision``
    (which is optional)."""


class BirdPowerPickGainOrderDecision(Decision[PlayerIdChoice]):
    """Pick which player takes food first for an "each player gains a die from the
    birdfeeder, starting with the player of your choice" power (Anna's /
    Ruby-throated Hummingbird). Resolves on the active player's own turn; each
    ``PlayerIdChoice`` is a candidate starter and the choice vector's ``is_self``
    flag marks the option that is the deciding player. (Not the next round's
    starter — there is no such core-set power.)"""


class BirdPowerPickHabitatDecision(Decision[HabitatChoice]):
    """Power asks the player to designate a habitat target."""


class ResetBirdfeederDecision(Decision[ResetBirdfeederChoice | SkipChoice]):
    """Offer the optional birdfeeder reset before a player takes food.

    Wingspan lets a player reroll the whole feeder *before* gaining food
    whenever every die shows the same face (one single food, or all on the
    invertebrate/seed choice face). It is purely a player option — the separate
    "reroll an empty feeder" rule is automatic and never surfaces as a decision.
    The two choices are ``ResetBirdfeederChoice`` (yes, reroll) and
    ``SkipChoice`` (no, take from the feeder as-is). Offered at every feeder
    gain — the main Gain Food action and every bird power that pulls from the
    feeder — so the judgment "is a fresh roll worth more than what's showing?"
    is scored on its own head."""


# ---------------------------------------------------------------------------
# Stable iteration order for the encoder's decision-type one-hot stripe.
# Append to the end when adding new decision subclasses — reordering or removing
# entries shifts the stripe indices, which is a FRESH (checkpoint-invalidating)
# change. See CLAUDE.md "Checkpoint compatibility policy".
#
# ``SetupDecision`` is kept LAST on purpose: the main model's encoding is
# config-driven (``encode.EncodingSpec.include_setup``), and excluding setup
# drops exactly the trailing decision-type column. Keeping it last means every
# other decision's index is invariant to whether setup is included.

ALL_DECISION_CLASSES: tuple[type[Decision[typing.Any]], ...] = (
    MainActionDecision,
    PlayBirdDecision,
    RemoveEggDecision,
    GainFoodDecision,
    SpendFoodDecision,
    LayEggDecision,
    DrawCardsPickSourceDecision,
    BirdPowerPickBirdFromHandDecision,
    BirdPowerPickPlayedBirdDecision,
    BirdPowerPickBonusCardDecision,
    BirdPowerTuckFromHandDecision,
    BirdPowerPickGainOrderDecision,
    BirdPowerPickHabitatDecision,
    AcceptExchangeDecision,
    ResetBirdfeederDecision,
    DiscardBirdForFoodDecision,
    SpendFoodForEggDecision,
    PayBirdFoodDecision,
    BirdPowerDiscardFromHandDecision,
    ActivateTuckDecision,  # FRESH: appended after BirdPowerDiscardFromHandDecision, before SetupDecision
    SetupDecision,
)


# ---------------------------------------------------------------------------
# Judgment-family taxonomy
#
# The RL model groups the decision classes above into *judgment families* —
# one per distinct skill a player exercises (see ``DECISIONS.md`` §2). Each
# family becomes one scoring head on the shared trunk: several as-built
# ``Decision`` classes collapse onto one family when they ask the same
# underlying judgment (e.g. every "is this bird worth keeping/playing?"
# decision), so the policy specializes per skill rather than per trigger.
#
# The family of a decision is a pure function of its class
# (``family_for`` / ``family_index_for``). ``ALL_DECISION_FAMILIES`` pins the
# stable head order — append, never reorder, when adding a family: the
# head→family alignment is part of the checkpoint format (mirrors the
# ``ALL_DECISION_CLASSES`` contract for the decision-type one-hot, and the same
# FRESH-change rule applies).


class DecisionFamily(enum.StrEnum):
    """The judgment a decision exercises — the unit of policy specialization.

    Several as-built ``Decision`` classes map to one family when they share an
    underlying judgment, and the RL model trains one scoring head per family
    rather than one per decision class. See ``DECISIONS.md`` §0 for the full
    per-class mapping and §2 for the per-family rationale.
    """

    SETUP = "setup"
    MAIN_ACTION = "main_action"
    DRAW_BIRD = "draw_bird"
    DISCARD_BIRD = "discard_bird"
    GAIN_FOOD = "gain_food"
    SPEND_FOOD = "spend_food"
    LAY_EGG = "lay_egg"
    PAY_EGG = "pay_egg"
    SKIP_OPTIONAL = "skip_optional"
    CHOOSE_BONUS = "choose_bonus"
    MISC_RARE = "misc_rare"
    PLAY_BIRD = "play_bird"
    RESET_BIRDFEEDER = "reset_birdfeeder"


ALL_DECISION_FAMILIES: tuple[DecisionFamily, ...] = (
    DecisionFamily.MAIN_ACTION,
    DecisionFamily.DRAW_BIRD,
    DecisionFamily.DISCARD_BIRD,
    DecisionFamily.GAIN_FOOD,
    DecisionFamily.SPEND_FOOD,
    DecisionFamily.LAY_EGG,
    DecisionFamily.PAY_EGG,
    DecisionFamily.SKIP_OPTIONAL,
    DecisionFamily.CHOOSE_BONUS,
    DecisionFamily.MISC_RARE,
    DecisionFamily.PLAY_BIRD,
    DecisionFamily.RESET_BIRDFEEDER,
    # SETUP is kept LAST so excluding it (the config-driven setup-model path,
    # ``encode.EncodingSpec.include_setup=False``) drops exactly the trailing
    # scoring head and leaves every other family's head index unchanged.
    DecisionFamily.SETUP,
)

# Per-class assignment. Keyed on the concrete decision class so routing is a
# pure function of the class. Bird valuation is split by direction (DECISIONS.md
# §2.2/§2.3): *acquiring* a bird ("which do I take?") and *giving one up* ("which do
# I lose?") are opposite judgments and route to separate heads. Choosing the
# turn's action *type* (``MainActionDecision`` -> ``MAIN_ACTION``) is split from
# choosing *which bird to play, where* (``PlayBirdDecision`` -> ``PLAY_BIRD``);
# the latter serves both the main-action PLAY_BIRD branch and power-granted
# extra plays. The play's costs route to the generic spend heads —
# ``RemoveEggDecision`` -> ``PAY_EGG``, ``PayBirdFoodDecision`` ->
# ``SPEND_FOOD`` — so paying for a bird trains the same judgments as every
# other egg / food spend.
_DECISION_FAMILY: dict[type[Decision[typing.Any]], DecisionFamily] = {
    SetupDecision: DecisionFamily.SETUP,
    MainActionDecision: DecisionFamily.MAIN_ACTION,
    PlayBirdDecision: DecisionFamily.PLAY_BIRD,
    DrawCardsPickSourceDecision: DecisionFamily.DRAW_BIRD,
    BirdPowerPickBirdFromHandDecision: DecisionFamily.DRAW_BIRD,
    BirdPowerTuckFromHandDecision: DecisionFamily.DISCARD_BIRD,
    BirdPowerDiscardFromHandDecision: DecisionFamily.DISCARD_BIRD,
    DiscardBirdForFoodDecision: DecisionFamily.DISCARD_BIRD,
    GainFoodDecision: DecisionFamily.GAIN_FOOD,
    SpendFoodDecision: DecisionFamily.SPEND_FOOD,
    SpendFoodForEggDecision: DecisionFamily.SPEND_FOOD,
    PayBirdFoodDecision: DecisionFamily.SPEND_FOOD,
    LayEggDecision: DecisionFamily.LAY_EGG,
    RemoveEggDecision: DecisionFamily.PAY_EGG,
    AcceptExchangeDecision: DecisionFamily.SKIP_OPTIONAL,
    ActivateTuckDecision: DecisionFamily.SKIP_OPTIONAL,
    BirdPowerPickBonusCardDecision: DecisionFamily.CHOOSE_BONUS,
    BirdPowerPickHabitatDecision: DecisionFamily.MISC_RARE,
    BirdPowerPickPlayedBirdDecision: DecisionFamily.MISC_RARE,
    BirdPowerPickGainOrderDecision: DecisionFamily.MISC_RARE,
    ResetBirdfeederDecision: DecisionFamily.RESET_BIRDFEEDER,
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
    Stable across the setup axis: because ``SETUP`` is the *last* family,
    excluding it (``include_setup=False``) drops only the trailing head, so every
    other family keeps the same index whether or not setup is in the main model.
    """
    return _DECISION_FAMILY_INDEX[decision_class]


# ---------------------------------------------------------------------------
# Setup axis: which decision classes / families belong solely to the separate
# setup model. When the main model delegates the opening to that model
# (``encode.EncodingSpec.include_setup=False``), these are excluded from its
# decision-type one-hot and its scoring heads. Both are kept LAST in their
# stable orders, so excluding them is a clean truncation that leaves every other
# index unchanged (see ``ALL_DECISION_CLASSES`` / ``ALL_DECISION_FAMILIES``).

_SETUP_ONLY_CLASSES: frozenset[type[Decision[typing.Any]]] = frozenset({SetupDecision})
_SETUP_ONLY_FAMILIES: frozenset[DecisionFamily] = frozenset({DecisionFamily.SETUP})


def active_decision_classes(
    include_setup: bool = True,
) -> tuple[type[Decision[typing.Any]], ...]:
    """The decision classes the main model's decision-type one-hot covers, in
    stable order. Excludes the setup-only classes when ``include_setup`` is
    ``False`` (the opening is scored by the separate setup model instead)."""
    if include_setup:
        return ALL_DECISION_CLASSES
    return tuple(cls for cls in ALL_DECISION_CLASSES if cls not in _SETUP_ONLY_CLASSES)


def active_decision_families(include_setup: bool = True) -> tuple[DecisionFamily, ...]:
    """The judgment families the main model trains a scoring head for, in stable
    order. Excludes ``SETUP`` when ``include_setup`` is ``False``."""
    if include_setup:
        return ALL_DECISION_FAMILIES
    return tuple(
        family for family in ALL_DECISION_FAMILIES if family not in _SETUP_ONLY_FAMILIES
    )


def random_choice[C: Choice](decision: Decision[C], rng: random.Random) -> C:
    """A uniform-random legal choice, deterministic in ``rng``.

    Used to resolve a decision *off* the main policy — specifically a
    ``SetupDecision`` for a net whose encoding excludes setup
    (``EncodingSpec.include_setup=False``): the opening is the separate setup
    model's responsibility, so eval / self-play just need *a* reproducible
    opening to play the rest of the game on."""
    return decision.choices[rng.randrange(len(decision.choices))]


def is_setup_decision(decision: Decision[typing.Any]) -> bool:
    """Whether ``decision`` is the combined opening pick (``SetupDecision``).

    Deliberately a plain ``bool`` predicate rather than a ``TypeGuard``: callers
    gate on it *without* narrowing the decision's ``Choice`` type, so they can
    return ``random_choice(decision, rng)`` as the generic ``C`` with no cast."""
    return isinstance(decision, SetupDecision)
