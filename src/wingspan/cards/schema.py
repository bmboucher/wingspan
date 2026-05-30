"""Schema for bird, bonus-card, and end-of-round-goal cards.

All structured data models are Pydantic v2 ``BaseModel`` subclasses. The
public API of this module is:

- the enums (``Habitat``, ``Food``, ``NestType``, ``PowerColor``,
  ``EffectKind``);
- the ``Effect`` / ``Power`` intermediate representation;
- the three top-level card models (``Bird``, ``BonusCard``,
  ``EndRoundGoal``);
- the three "input" record models (``BirdRecord``, ``BonusRecord``,
  ``GoalRecord``) that mirror the raw wingsearch JSON shape and expose a
  ``.load()`` method that returns the corresponding parsed card model.
"""

from __future__ import annotations

import enum

import pydantic

# ---------------------------------------------------------------------------
# Enums


class Habitat(enum.StrEnum):
    FOREST = "forest"
    GRASSLAND = "grassland"
    WETLAND = "wetland"


ALL_HABITATS = [Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND]


class Food(enum.StrEnum):
    INVERTEBRATE = "invertebrate"
    SEED = "seed"
    FISH = "fish"
    FRUIT = "fruit"
    RODENT = "rodent"


ALL_FOODS = [Food.INVERTEBRATE, Food.SEED, Food.FISH, Food.FRUIT, Food.RODENT]
N_FOODS = len(ALL_FOODS)

_FOOD_INDEX = {food: i for i, food in enumerate(ALL_FOODS)}


def food_index(food: Food) -> int:
    """Canonical position of ``food`` in :data:`ALL_FOODS` — the index used
    throughout the codebase for the vector / tuple representation of
    per-food amounts."""
    return _FOOD_INDEX[food]


class NestType(enum.StrEnum):
    BOWL = "bowl"
    CAVITY = "cavity"
    GROUND = "ground"
    PLATFORM = "platform"
    STAR = "star"  # wildcard
    NONE = "none"  # birds with no nest icon (rare in core)


class PowerColor(enum.StrEnum):
    BROWN = "brown"  # When activated (column power)
    WHITE = "white"  # When played (one-shot)
    PINK = "pink"  # Once between turns
    YELLOW = "yellow"  # End of round (not in core set per data sample)
    NONE = "none"


class EffectKind(enum.StrEnum):
    """A small library of generic power patterns. Any bird whose printed
    text does not match a known pattern gets ``UNIMPLEMENTED`` and is run
    as a no-op (with a logged warning the first time it triggers)."""

    GAIN_FOOD_SUPPLY = "gain_food_supply"
    GAIN_FOOD_BIRDFEEDER = "gain_food_birdfeeder"
    GAIN_FOOD_FROM_FEEDER_CHOICE = "gain_food_from_feeder_choice"
    GAIN_DIE_ANY = "gain_die_any"
    LAY_EGG_ON_THIS = "lay_egg_on_this"
    LAY_EGG_ANY = "lay_egg_any"
    DRAW_CARDS = "draw_cards"
    CACHE_FOOD = "cache_food"
    TUCK_FROM_HAND = "tuck_from_hand"
    TUCK_FROM_DECK = "tuck_from_deck"
    PLAY_ADDITIONAL_BIRD = "play_additional_bird"
    ALL_PLAYERS_GAIN_FOOD = "all_players_gain_food"
    ALL_PLAYERS_DRAW = "all_players_draw"
    DRAW_BONUS = "draw_bonus"
    DISCARD_EGG_FOR_WILD = "discard_egg_for_wild"
    EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER = "each_player_gains_die_choose_order"
    ALL_PLAYERS_LAY_EGG_ON_NEST = "all_players_lay_egg_on_nest"
    DRAW_FROM_TRAY_ALL = "draw_from_tray_all"
    TRADE_WILD_FOOD = "trade_wild_food"
    FEWEST_FOREST_GAINS_DIE = "fewest_forest_gains_die"
    PLAY_ADDITIONAL_BIRD_HERE = "play_additional_bird_here"
    DRAW_N_PLUS_ONE_DRAFT = "draw_n_plus_one_draft"
    DRAW_BONUS_KEEP = "draw_bonus_keep"
    LAY_EGG_ALL_NEST = "lay_egg_all_nest"
    GAIN_ALL_FOOD_FEEDER = "gain_all_food_feeder"
    TUCK_FROM_DECK_PAID = "tuck_from_deck_paid"
    PREDATOR_HUNT = "predator_hunt"
    MOVE_BIRD_IF_RIGHTMOST = "move_bird_if_rightmost"
    REPEAT_BROWN_POWER = "repeat_brown_power"
    REPEAT_PREDATOR_POWER = "repeat_predator_power"
    PINK_LAY_EGG_ON_NEST = "pink_lay_egg_on_nest"
    PINK_PREDATOR_FEEDER = "pink_predator_feeder"
    UNIMPLEMENTED = "unimplemented"


# ---------------------------------------------------------------------------
# Food cost (immutable 6-vector: 5 specific + 1 wild)


class BirdCost(pydantic.BaseModel):
    """The food cost printed on a bird card.

    Stored as a 6-tuple aligned to :data:`ALL_FOODS` plus a final wild slot:
    ``counts[food_index(f)]`` for each specific food, and ``counts[-1]`` for
    the wild count. Frozen / hashable so it can sit on the immutable
    :class:`Bird` card model."""

    model_config = pydantic.ConfigDict(frozen=True)

    counts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)

    @property
    def specific(self) -> tuple[int, int, int, int, int]:
        """The 5 specific-food slots, in :data:`ALL_FOODS` order."""
        return self.counts[:N_FOODS]  # type: ignore[return-value]

    @property
    def wild(self) -> int:
        """Number of wild (any-food) slots."""
        return self.counts[N_FOODS]

    @property
    def total(self) -> int:
        """Total printed food cost (specific + wild)."""
        return sum(self.counts)

    def specific_of(self, food: Food) -> int:
        """Specific-food count for ``food``."""
        return self.counts[food_index(food)]

    def is_free(self) -> bool:
        return self.total == 0

    @classmethod
    def from_specific(cls, specific: dict[Food, int], wild: int = 0) -> "BirdCost":
        """Build a cost from a sparse ``{food: count}`` dict + wild count."""
        vec = [0] * N_FOODS
        for food, count in specific.items():
            vec[food_index(food)] = count
        return cls(counts=(vec[0], vec[1], vec[2], vec[3], vec[4], wild))


# ---------------------------------------------------------------------------
# Power IR


class Effect(pydantic.BaseModel):
    """Structured representation of a single power effect.

    Carriers are named, typed fields rather than a positional ``extra`` tuple.
    Each ``EffectKind`` documents which fields it consumes; unused fields stay
    ``None``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    kind: EffectKind
    amount: int = 0
    food: Food | None = None
    habitat: Habitat | None = None
    raw_text: str = ""

    # --- typed carriers (replace the old untyped ``extra: tuple``) ----------
    keep_count: int | None = None  # DRAW_BONUS_KEEP: # to keep
    max_wingspan_cm: int | None = None  # PREDATOR_TUCK: hunt threshold
    nest: NestType | None = None  # LAY_EGG_ALL_NEST, ALL_PLAYERS_LAY_ON_NEST
    food_a: Food | None = None  # GAIN_FOOD_BIRDFEEDER_CHOICE: first option
    food_b: Food | None = None  # GAIN_FOOD_BIRDFEEDER_CHOICE: second option


class Power(pydantic.BaseModel):
    """A bird's parsed power: the trigger color and the list of effects it
    applies. ``raw_text`` is preserved for logging."""

    model_config = pydantic.ConfigDict(frozen=True)

    color: PowerColor
    effects: tuple[Effect, ...] = ()
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Cards


class Bird(pydantic.BaseModel):
    """A single bird card. Immutable after load — all per-game mutable state
    lives on ``PlayedBird`` in ``wingspan.state``."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: int
    name: str
    scientific_name: str
    color: PowerColor  # power color (a.k.a. trigger time)
    points: int
    nest: NestType
    egg_limit: int
    wingspan_cm: int
    habitats: tuple[Habitat, ...]  # one or more habitats the bird may live in
    food_cost: BirdCost
    flocking: bool  # food may be paid with any subset including wild
    predator: bool
    is_swift_start: bool
    raw_power_text: str
    power: Power
    # which bonus-card categories include this bird (set of bonus IDs)
    bonus_categories: tuple[str, ...] = ()

    @property
    def primary_habitat(self) -> Habitat:
        return self.habitats[0]


class BonusCard(pydantic.BaseModel):
    """A bonus card whose point payout depends on counts of qualifying birds.

    A card pays one of two mutually exclusive ways, expanded from the printed
    VP text by the loader:

    - **Tiered** — ``thresholds`` holds ``(count, vp)`` pairs ascending by
      count; the engine awards the highest matched pair (e.g. ``5 to 7 birds:
      3[point]; 8+ birds: 7[point]``).
    - **Per-bird** — ``per_bird_vp`` is the VP each qualifying bird earns
      (e.g. ``2[point] per bird``); the engine awards ``per_bird_vp * count``.

    Exactly one applies: a per-bird card has ``per_bird_vp`` set and empty
    ``thresholds``; a tiered card has ``per_bird_vp is None``."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: int
    name: str
    condition: str
    explanatory: str
    vp_text: str
    # tiered payout: number_of_qualifying_birds -> vp; expanded from the printed
    # text. Empty for per-bird cards (see ``per_bird_vp``).
    thresholds: tuple[tuple[int, int], ...] = ()
    # per-bird payout: VP awarded per qualifying bird; None for tiered cards.
    per_bird_vp: int | None = None


class EndRoundGoal(pydantic.BaseModel):
    """One of the 16 end-of-round goals.

    ``category`` is a short tag the scoring engine dispatches on (set by the
    loader). The placement payout is intentionally not stored here: it depends
    on which round the goal is scored in, not on the goal card itself — see
    ``state.ROUND_GOAL_PAYOUTS_2P`` and ``engine.scoring.score_round_goal``."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: int
    description: str  # e.g. "[bird] in [forest]"
    # canonical category enum-string for easy dispatch
    category: str


# ---------------------------------------------------------------------------
# Input records (raw wingsearch JSON shapes)
#
# Each record models one row of a wingsearch JSON file. The ``Field(alias=...)``
# declarations map the JSON column names (which contain spaces, parentheses,
# and capitalisation) to Pythonic attribute names. Each record exposes a
# ``.load()`` method that returns the corresponding parsed card model defined
# above. The conversion helpers themselves live in :mod:`wingspan.cards.parse`
# and are imported lazily inside ``.load()`` to avoid a top-level circular
# import (``parse`` imports this module).


class BirdRecord(pydantic.BaseModel):
    """Raw bird record as it appears in ``master.json``.

    Bonus-category columns (one per bonus card, e.g. ``"Anatomist": "X"``)
    are not declared here; ``extra="allow"`` captures them into
    :attr:`model_extra` so the dynamic lookup in ``.load()`` can find them.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    id: int
    card_set: str = pydantic.Field(alias="Set")
    common_name: str = pydantic.Field(alias="Common name")
    scientific_name: str | None = pydantic.Field(default=None, alias="Scientific name")
    color: str | None = pydantic.Field(default=None, alias="Color")
    power_text: str | None = pydantic.Field(default=None, alias="Power text")
    victory_points: float | None = pydantic.Field(default=None, alias="Victory points")
    nest_type: str | None = pydantic.Field(default=None, alias="Nest type")
    egg_limit: float | None = pydantic.Field(default=None, alias="Egg limit")
    wingspan_cm: float | None = pydantic.Field(default=None, alias="Wingspan")
    forest: str | None = pydantic.Field(default=None, alias="Forest")
    grassland: str | None = pydantic.Field(default=None, alias="Grassland")
    wetland: str | None = pydantic.Field(default=None, alias="Wetland")
    invertebrate: float | None = pydantic.Field(default=None, alias="Invertebrate")
    seed: float | None = pydantic.Field(default=None, alias="Seed")
    fish: float | None = pydantic.Field(default=None, alias="Fish")
    fruit: float | None = pydantic.Field(default=None, alias="Fruit")
    rodent: float | None = pydantic.Field(default=None, alias="Rodent")
    wild_food: float | None = pydantic.Field(default=None, alias="Wild (food)")
    total_food_cost: float | None = pydantic.Field(
        default=None, alias="Total food cost"
    )
    flocking: str | None = pydantic.Field(default=None, alias="Flocking")
    predator: str | None = pydantic.Field(default=None, alias="Predator")
    swift_start: str | None = pydantic.Field(default=None, alias="Swift Start")

    def load(self, bonuses: list[BonusRecord]) -> Bird | None:
        """Build the parsed :class:`Bird` from this raw record.

        Returns ``None`` for malformed records (e.g. no habitats marked).
        ``bonuses`` is the list of all known bonus cards; it is used to
        compute :attr:`Bird.bonus_categories` by looking up each bonus
        card's name in this record's extra columns."""
        from wingspan.cards import parse

        color = parse.parse_power_color(self.color)
        habitats = parse.parse_habitats(self)
        if not habitats:
            return None  # malformed record - skip
        cost = parse.parse_food_cost(self)
        power_text = self.power_text or ""
        return Bird(
            id=self.id,
            name=self.common_name,
            scientific_name=self.scientific_name or "",
            color=color,
            points=int(self.victory_points or 0),
            nest=parse.parse_nest(self.nest_type),
            egg_limit=int(self.egg_limit or 0),
            wingspan_cm=int(self.wingspan_cm or 0),
            habitats=tuple(habitats),
            food_cost=cost,
            flocking=(self.flocking == "X"),
            predator=(self.predator == "X"),
            is_swift_start=(self.swift_start == "X"),
            raw_power_text=power_text,
            power=parse.parse_power(color, power_text),
            bonus_categories=parse.bonus_categories_for_bird(self, bonuses),
        )


class BonusRecord(pydantic.BaseModel):
    """Raw bonus-card record as it appears in ``bonus.json``."""

    id: int
    card_set: str = pydantic.Field(alias="Set")
    bonus_card: str = pydantic.Field(alias="Bonus card")
    condition: str | None = pydantic.Field(default=None, alias="Condition")
    explanatory_text: str | None = pydantic.Field(
        default=None, alias="Explanatory text"
    )
    vp_text: str | None = pydantic.Field(default=None, alias="VP")

    def load(self) -> BonusCard:
        """Build the parsed :class:`BonusCard` from this raw record."""
        from wingspan.cards import parse

        vp_text = self.vp_text or ""
        return BonusCard(
            id=self.id,
            name=self.bonus_card,
            condition=self.condition or "",
            explanatory=self.explanatory_text or "",
            vp_text=vp_text,
            thresholds=parse.parse_bonus_thresholds(vp_text),
            per_bird_vp=parse.parse_bonus_per_bird(vp_text),
        )


class GoalRecord(pydantic.BaseModel):
    """Raw end-of-round goal record as it appears in ``goals.json``."""

    id: int
    card_set: str = pydantic.Field(alias="Set")
    goal: str | None = pydantic.Field(default=None, alias="Goal")

    def load(self) -> EndRoundGoal:
        """Build the parsed :class:`EndRoundGoal` from this raw record."""
        from wingspan.cards import parse

        desc = self.goal or ""
        return EndRoundGoal(
            id=self.id,
            description=desc,
            category=parse.goal_category(desc),
        )
