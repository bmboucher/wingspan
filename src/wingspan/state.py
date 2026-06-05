"""Core game state.

All mutable state lives here. The engine never holds a reference to a player
or bird that bypasses this module: ``GameState`` is the single object the
engine reads and writes, and it is designed to be cheap to ``copy.deepcopy``
for RL rollouts and MCTS branching.

All data carriers are Pydantic v2 ``BaseModel``s. Mutation is intentionally
permitted (``validate_assignment`` defaults to ``False``); the engine relies
on in-place mutation for turn-by-turn play.
"""

from __future__ import annotations

import collections.abc
import random
import typing

import pydantic

from wingspan import cards

# ---------------------------------------------------------------------------
# Public constants

ROUND_CUBES = [8, 7, 6, 5]  # action cubes per player per round (rounds 1..4)
ROW_SLOTS = 5  # 5 columns per habitat row
N_HABITATS = 3
BIRDFEEDER_DICE = 5
TRAY_SIZE = 3
HAND_LIMIT = None  # Wingspan has no hand limit
STARTING_HAND_SIZE = 5
STARTING_FOOD_TOTAL = 5  # players choose food+cards totalling 5
STARTING_BONUS_CARDS_DEAL = 2  # deal 2, keep 1

# Wingspan's printed reward tracks per column (core, simplified):
#   Forest:    take 1, 1, 2, 2, 3 dice (and rolling-extras after)
#   Grassland: lay 2, 2, 3, 3, 4 eggs
#   Wetland:   draw 1, 1, 2, 2, 3 cards (+1 between turns)
# Reward count is keyed on the number of birds in the activated row (1..5).
GAIN_FOOD_TRACK = (1, 1, 2, 2, 3)
LAY_EGGS_TRACK = (2, 2, 3, 3, 4)
DRAW_CARDS_TRACK = (1, 1, 2, 2, 3)

# Egg cost to play the Nth bird (N = current row length, 0..4) into any
# habitat. The printed player mat shows the same ladder across all three
# habitats — the action reward differs by habitat, not the egg cost.
EGG_COSTS = (0, 1, 1, 2, 2)

# Sentinel egg cost when the row is full, so callers can compare numerically
# without a separate "row full" branch.
FULL_ROW_EGG_COST = 99

# Round-goal placement payouts for a 2-player game, indexed by round (0..3) as
# ``(1st_place_vp, 2nd_place_vp)``. The printed goal board pays more in later
# rounds. The payout belongs to the round slot, not the goal card. A player
# whose category count is 0 does not place and scores nothing (see
# ``engine.scoring.score_round_goal``).
ROUND_GOAL_PAYOUTS_2P = ((4, 1), (5, 2), (6, 3), (7, 4))


def _new_bird_list() -> list[cards.Bird]:
    return []


def _new_bonus_card_list() -> list[cards.BonusCard]:
    return []


def _new_round_goal_list() -> list[cards.EndRoundGoal]:
    return []


# ---------------------------------------------------------------------------
# Food pool — mutable 5-vector aligned to cards.ALL_FOODS


class FoodPool(pydantic.BaseModel):
    """Mutable per-food count pool indexed by :class:`cards.Food`.

    Internally a length-5 ``list[int]`` aligned to :data:`cards.ALL_FOODS`.
    The pool exposes the small dict-like surface used by the engine —
    ``pool[food]`` get/set/``+=``, ``items()``, ``values()``, ``get()``,
    ``in``, iteration by food, ``total()`` — so call sites that previously
    used ``dict[cards.Food, int]`` keep reading naturally."""

    counts: list[int] = pydantic.Field(
        default_factory=lambda: [0] * cards.N_FOODS,
        min_length=cards.N_FOODS,
        max_length=cards.N_FOODS,
    )

    # ---- access ----------------------------------------------------------

    def __getitem__(self, food: cards.Food) -> int:
        return self.counts[cards.food_index(food)]

    def __setitem__(self, food: cards.Food, amount: int) -> None:
        self.counts[cards.food_index(food)] = amount

    def __contains__(self, food: object) -> bool:
        return isinstance(food, cards.Food)

    def __iter__(self) -> typing.Iterator[cards.Food]:  # type: ignore[override]
        return iter(cards.ALL_FOODS)

    def keys(self) -> list[cards.Food]:
        return list(cards.ALL_FOODS)

    def values(self) -> list[int]:
        return list(self.counts)

    def items(self) -> list[tuple[cards.Food, int]]:
        return list(zip(cards.ALL_FOODS, self.counts))

    def get(self, food: cards.Food, default: int = 0) -> int:
        return self.counts[cards.food_index(food)]

    def as_dict(self) -> dict[cards.Food, int]:
        """Materialize as a plain ``dict[Food, int]`` (e.g. for logging /
        snapshotting). Includes every food, even zero entries."""
        return dict(zip(cards.ALL_FOODS, self.counts))

    def format(self) -> str:
        """Compact human-readable rendering for logs.

        Lists only non-zero foods, e.g. ``1fish+1rodent``. Returns
        ``(empty)`` when every count is zero."""
        parts = [
            f"{amount}{food.value}"
            for food, amount in zip(cards.ALL_FOODS, self.counts)
            if amount > 0
        ]
        return "+".join(parts) if parts else "(empty)"

    # ---- ops -------------------------------------------------------------

    def total(self) -> int:
        return sum(self.counts)

    def is_empty(self) -> bool:
        return all(count == 0 for count in self.counts)

    def types_with_positive(self) -> list[cards.Food]:
        """Foods whose count is currently positive (in canonical order)."""
        return [food for food, count in zip(cards.ALL_FOODS, self.counts) if count > 0]

    def zero(self) -> None:
        """Reset every count to zero (in place)."""
        for i in range(cards.N_FOODS):
            self.counts[i] = 0

    @classmethod
    def from_dict(cls, mapping: dict[cards.Food, int]) -> "FoodPool":
        """Build a pool from a sparse ``{food: count}`` dict."""
        pool = cls()
        for food, amount in mapping.items():
            pool[food] = amount
        return pool

    @classmethod
    def uniform(cls, count: int) -> "FoodPool":
        """Pool with the same count for every food (e.g. ``uniform(99)`` for
        the effectively-infinite supply)."""
        return cls(counts=[count] * cards.N_FOODS)


# ---------------------------------------------------------------------------
# Player-owned mutable state


class PlayedBird(pydantic.BaseModel):
    """A single bird in play on a player's board. Tracks per-game mutable
    state separately from the immutable ``Bird`` card data."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    bird: cards.Bird
    eggs: int = 0
    # cached food, tracked per food type so the encoder can see the composition
    cached_food: FoodPool = pydantic.Field(default_factory=FoodPool)
    tucked_cards: int = 0
    # number of times the bird has been activated this round (used by some rules)
    activations: int = 0


class Board(pydantic.BaseModel):
    """A player's three habitat rows (forest, grassland, wetland) plus the
    derived gameplay counts driven off them.

    Indexed by ``cards.Habitat`` for ergonomic row access — ``board[h]``,
    ``board.row(h)``, ``board.items()``, ``board.values()`` and
    ``board.get(h, default)`` all behave like the dict this class replaced.
    """

    rows: dict[cards.Habitat, list[PlayedBird]] = pydantic.Field(
        default_factory=lambda: {habitat: [] for habitat in cards.ALL_HABITATS},
    )

    # ---- access ----------------------------------------------------------

    def row(self, habitat: cards.Habitat) -> list[PlayedBird]:
        """The ordered list of PlayedBirds in ``habitat`` (left to right)."""
        return self.rows[habitat]

    def __getitem__(self, habitat: cards.Habitat) -> list[PlayedBird]:
        return self.rows[habitat]

    def __setitem__(self, habitat: cards.Habitat, row: list[PlayedBird]) -> None:
        self.rows[habitat] = row

    def items(self) -> typing.ItemsView[cards.Habitat, list[PlayedBird]]:
        return self.rows.items()

    def values(self) -> typing.ValuesView[list[PlayedBird]]:
        return self.rows.values()

    def keys(self) -> typing.KeysView[cards.Habitat]:
        return self.rows.keys()

    def get(
        self,
        habitat: cards.Habitat,
        default: list[PlayedBird] | None = None,
    ) -> list[PlayedBird] | None:
        return self.rows.get(habitat, default)

    # ---- derived gameplay counts ----------------------------------------

    def gain_food_count(self) -> int:
        """Number of birdfeeder dice the Forest action grants this turn."""
        return _track_count(len(self.rows[cards.Habitat.FOREST]), GAIN_FOOD_TRACK)

    def lay_eggs_count(self) -> int:
        """Number of eggs the Grassland action lays this turn."""
        return _track_count(len(self.rows[cards.Habitat.GRASSLAND]), LAY_EGGS_TRACK)

    def draw_cards_count(self) -> int:
        """Number of cards the Wetland action draws this turn."""
        return _track_count(len(self.rows[cards.Habitat.WETLAND]), DRAW_CARDS_TRACK)

    def action_offers_convert(self, habitat: cards.Habitat) -> bool:
        """Whether ``habitat``'s action space offers its one-step resource
        trade this turn.

        The printed mat puts a trade arrow (Forest: discard a card for a food;
        Grassland: spend a food for an egg; Wetland: discard an egg for a card)
        on every other action space, so the cube lands on a trade space exactly
        when the row holds an odd number of birds (1, 3, or 5). The trade is a
        single exchange, not repeatable."""
        return _track_offers_convert(len(self.rows[habitat]))

    def next_egg_cost(self, habitat: cards.Habitat) -> int:
        """Egg cost to play the next bird into ``habitat``. Returns
        :data:`FULL_ROW_EGG_COST` when the row is already full."""
        col = len(self.rows[habitat])
        if col >= len(EGG_COSTS):
            return FULL_ROW_EGG_COST
        return EGG_COSTS[col]


class Player(pydantic.BaseModel):
    """One player's complete per-game state."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    id: int
    name: str
    hand: list[cards.Bird] = pydantic.Field(default_factory=_new_bird_list)
    bonus_cards: list[cards.BonusCard] = pydantic.Field(
        default_factory=_new_bonus_card_list
    )
    food: FoodPool = pydantic.Field(default_factory=FoodPool)
    board: Board = pydantic.Field(default_factory=Board)
    action_cubes_left: int = 0
    # cumulative end-of-round goal points (in 2P play, accumulate per round)
    round_goal_points: int = 0
    # set by the scoring phase; ``None`` until the game ends
    final_score: int | None = None

    @property
    def total_eggs(self) -> int:
        return sum(pb.eggs for row in self.board.values() for pb in row)

    @property
    def total_tucked(self) -> int:
        return sum(pb.tucked_cards for row in self.board.values() for pb in row)

    @property
    def total_cached(self) -> int:
        return sum(pb.cached_food.total() for row in self.board.values() for pb in row)

    def total_food(self) -> int:
        return self.food.total()

    def can_play_in(self, habitat: cards.Habitat) -> bool:
        return len(self.board[habitat]) < ROW_SLOTS

    def row_activation_count(self, habitat: cards.Habitat) -> int:
        """How many birds sit in ``habitat`` (the count drives column-power
        triggers and the printed action-reward track)."""
        return len(self.board[habitat])


# ---------------------------------------------------------------------------
# Shared game state


# The two foods on the birdfeeder die's sixth (choice) face — the taker picks
# one. Kept here so ``Birdfeeder.take`` and the reroll share one definition.
_CHOICE_FACE_FOODS = (cards.Food.INVERTEBRATE, cards.Food.SEED)


class Birdfeeder(pydantic.BaseModel):
    """The communal birdfeeder dice tower.

    Wingspan's five dice each show one of six faces: the five single foods, plus
    a sixth face that is *invertebrate or seed — the taker chooses*. The single
    faces currently showing are held in ``counts``; ``choice_dice`` counts the
    dice on the invertebrate/seed face, which resolve to one of those two foods
    only when a player takes them. So fish / fruit / rodent each come up 1/6 of
    the time, while invertebrate and seed are each obtainable 1/3 of the time
    (their own face plus the shared choice face)."""

    counts: FoodPool = pydantic.Field(default_factory=FoodPool)
    # Dice currently showing the invertebrate/seed choice face (unresolved until
    # a player takes one and picks which of the two foods it yields).
    choice_dice: int = 0

    def total(self) -> int:
        return self.counts.total() + self.choice_dice

    def is_empty(self) -> bool:
        return self.total() == 0

    def distinct_faces(self) -> int:
        """Number of distinct die faces currently showing — each single food in
        play counts once, and the invertebrate/seed face counts once when any
        die shows it. Drives the reroll-when-one-face-left rule."""
        return len(self.counts.types_with_positive()) + (1 if self.choice_dice else 0)

    def reset_available(self) -> bool:
        """Whether the optional pre-gain reset (Rule 2) is on offer: every die
        shows the same face — one single food, or all dice on the
        invertebrate/seed choice face. The single shared predicate behind the
        engine's reset offer and the state encoder's reset-availability flag."""
        return self.distinct_faces() == 1

    def gainable_foods(self) -> list[cards.Food]:
        """Foods a player could take right now, in canonical ``ALL_FOODS`` order:
        every food with a single face showing, plus invertebrate and seed
        whenever a choice die is showing."""
        present = set(self.counts.types_with_positive())
        if self.choice_dice > 0:
            present.update(_CHOICE_FACE_FOODS)
        return [food for food in cards.ALL_FOODS if food in present]

    def gainable_count(self, food: cards.Food) -> int:
        """How many dice could yield ``food`` — its single faces plus the shared
        choice dice when it is invertebrate or seed (the choice dice count toward
        both, since one die can be taken as either). For display only."""
        extra = self.choice_dice if food in _CHOICE_FACE_FOODS else 0
        return self.counts[food] + extra

    def gain_options(
        self, allowed: collections.abc.Iterable[cards.Food] | None = None
    ) -> list[tuple[cards.Food, bool]]:
        """The distinct ways to take one die right now, as ``(food, from_choice_die)``
        pairs in canonical order: each single food face showing
        (``from_choice_die=False``), then — when a choice die is showing —
        invertebrate and seed taken from the choice die (``from_choice_die=True``).

        Unlike :meth:`gainable_foods`, the plain and choice-die ways to take
        invertebrate/seed are listed *separately*, because they spend different
        dice and the model scores them apart. ``allowed`` restricts the foods (a
        power that grants only specific foods); ``None`` offers every option."""
        allow = None if allowed is None else set(allowed)
        options: list[tuple[cards.Food, bool]] = []
        for food in cards.ALL_FOODS:
            if self.counts[food] > 0 and (allow is None or food in allow):
                options.append((food, False))
        if self.choice_dice > 0:
            for food in _CHOICE_FACE_FOODS:
                if allow is None or food in allow:
                    options.append((food, True))
        return options

    def gain_option_label(self, food: cards.Food, from_choice_die: bool) -> str:
        """Human-readable label for one :meth:`gain_options` entry (CLI / log only)."""
        if from_choice_die:
            return f"{food.value} (choice die ×{self.choice_dice})"
        return f"{food.value}({self.counts[food]})"

    def format(self) -> str:
        """Compact human-readable rendering of the full die state for logs.

        Combines named-food dice (via :meth:`FoodPool.format`) with the choice-die
        count, e.g. ``"2fish+1fruit+2choice"`` or ``"3choice"``.  Returns
        ``"(empty)"`` when no dice are showing."""
        plain = self.counts.format()  # "2fish+1fruit" or "(empty)"
        if not self.choice_dice:
            return plain
        choice_part = f"{self.choice_dice}choice"
        return choice_part if plain == "(empty)" else f"{plain}+{choice_part}"

    def take(self, food: cards.Food, *, from_choice_die: bool = False) -> None:
        """Consume one die yielding ``food``.

        With ``from_choice_die=False`` (the default), spend a matching single
        face first, falling back to a choice die only when ``food`` is
        invertebrate/seed and no single face is showing. With
        ``from_choice_die=True``, consume a choice die specifically (raising if
        none is showing) — used when the gain was offered as the invertebrate/
        seed choice-die option. Raises if ``food`` is not gainable the requested
        way; callers gate on :meth:`gain_options` / :meth:`gainable_foods`."""
        if from_choice_die:
            if self.choice_dice > 0 and food in _CHOICE_FACE_FOODS:
                self.choice_dice -= 1
            else:
                raise ValueError(f"{food.value} is not gainable from a choice die")
            return
        if self.counts[food] > 0:
            self.counts[food] -= 1
        elif self.choice_dice > 0 and food in _CHOICE_FACE_FOODS:
            self.choice_dice -= 1
        else:
            raise ValueError(f"{food.value} is not gainable from the birdfeeder")

    def reroll(self, rng: random.Random) -> None:
        """Reroll all five dice over the six equally-likely faces — the five
        single foods plus the invertebrate/seed choice face."""
        self.counts.zero()
        self.choice_dice = 0
        for _ in range(BIRDFEEDER_DICE):
            face = rng.randint(
                0, cards.N_FOODS
            )  # 0..N_FOODS-1 -> food; N_FOODS -> choice
            if face < cards.N_FOODS:
                self.counts[cards.ALL_FOODS[face]] += 1
            else:
                self.choice_dice += 1


class RoundGoalResult(pydantic.BaseModel):
    """The frozen outcome of one round goal at the moment it was scored.

    Once a round is scored its goal standings never change, no matter how the
    boards evolve afterwards — readers of past-round standings (the state
    encoder, the CLI display) must consult this record rather than recomputing
    from the live board. Both lists are per-seat, indexed by ``Player.id``."""

    counts: list[int]
    vp_awarded: list[int]


def _new_round_goal_result_list() -> list[RoundGoalResult]:
    return []


class GameState(pydantic.BaseModel):
    """The full game state: players, decks, board, supplies, and the
    cumulative game log."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    rng: random.Random
    players: list[Player]
    current_player: int
    # Player who takes the first turn of round 1, chosen randomly at game
    # start. Turn order rotates each round: round ``r`` is started by
    # ``(start_player + r) % len(players)``.
    start_player: int = 0
    round_idx: int  # 0..3
    bird_deck: list[cards.Bird]  # remaining
    bird_discard: list[cards.Bird] = pydantic.Field(default_factory=_new_bird_list)
    bonus_deck: list[cards.BonusCard] = pydantic.Field(
        default_factory=_new_bonus_card_list
    )
    bonus_discard: list[cards.BonusCard] = pydantic.Field(
        default_factory=_new_bonus_card_list
    )
    # The face-up bird tray. Always exactly ``TRAY_SIZE`` slots; ``None``
    # marks an empty slot. A card taken from the tray sets that slot to
    # ``None`` until ``refill_tray`` runs (at end of turn, or on an explicit
    # power). Position is meaningful: slot 0 is the left card, slot 2 is the
    # right card, matching "the middle card in the bird tray" power wording.
    tray: list[cards.Bird | None] = pydantic.Field(
        default_factory=lambda: [None] * TRAY_SIZE
    )
    birdfeeder: Birdfeeder = pydantic.Field(default_factory=Birdfeeder)
    food_supply: FoodPool = pydantic.Field(default_factory=lambda: FoodPool.uniform(99))
    # All 16 goals shuffled — first 4 are the per-round goals for rounds 1..4
    round_goals: list[cards.EndRoundGoal] = pydantic.Field(
        default_factory=_new_round_goal_list
    )
    # Frozen per-round goal outcomes, appended by ``scoring.score_round_goal``
    # the moment each round pays out. Starts empty and grows to 4 entries;
    # entry ``r`` exists exactly when round ``r`` has been scored, so
    # ``len(scored_goals)`` is the number of scored rounds.
    scored_goals: list[RoundGoalResult] = pydantic.Field(
        default_factory=_new_round_goal_result_list
    )
    game_over: bool = False
    log: list[str] = pydantic.Field(default_factory=list)

    # Per-turn scratch state. Reset at the start of every turn by the engine.
    # ``turn_extra_play_habitat``, when set, restricts the next extra play to a
    # single habitat (e.g. House Wren grants +1 play in this bird's habitat)
    # and is cleared by the extra-plays loop after consuming one play.
    turn_extra_plays: int = 0
    turn_extra_play_habitat: cards.Habitat | None = None
    # How many end-of-turn discard obligations the active player has accrued
    # from "draw N, then discard 1 at end of turn" birds this turn. Cleared
    # by reset_turn_state at the start of each turn.
    turn_end_discards: int = 0

    # ----- helpers ------------------------------------------------------

    def opponent(self) -> Player:
        return self.players[1 - self.current_player]

    def me(self) -> Player:
        return self.players[self.current_player]

    def draw_bird(self) -> cards.Bird | None:
        if not self.bird_deck:
            if not self.bird_discard:
                return None
            self.bird_deck = self.bird_discard
            self.bird_discard = []
            self.rng.shuffle(self.bird_deck)
        return self.bird_deck.pop()

    def refill_tray(self) -> None:
        """Fill each empty (``None``) tray slot from the deck, left-to-right.

        Called at the end of each turn and by powers that explicitly refill
        (e.g. Brant) — never automatically when a card is drawn mid-turn.
        Stops early if the deck (and discard) run dry, leaving later slots
        empty."""
        for slot_idx in range(TRAY_SIZE):
            if self.tray[slot_idx] is None:
                drawn = self.draw_bird()
                if drawn is None:
                    break
                self.tray[slot_idx] = drawn

    def reset_tray(self) -> None:
        """Discard all face-up tray cards and replenish with fresh ones.

        Called at the end of each round. The discarded cards enter
        ``bird_discard`` so they can be shuffled back into the deck when it
        runs dry later in the game."""
        self.bird_discard.extend(bird for bird in self.tray if bird is not None)
        self.tray = [None] * TRAY_SIZE
        self.refill_tray()

    def reset_turn_state(self) -> None:
        """Clear per-turn scratch fields. Called at the start of every turn."""
        self.turn_extra_plays = 0
        self.turn_extra_play_habitat = None
        self.turn_end_discards = 0


# ---------------------------------------------------------------------------
# Construction


def new_game(
    rng: random.Random,
    birds: list[cards.Bird],
    bonuses: list[cards.BonusCard],
    goals: list[cards.EndRoundGoal],
    player_names: tuple[str, str] = ("P0", "P1"),
) -> GameState:
    """Shuffle the decks, deal an empty board for two players, and return a
    ready-to-play ``GameState``. Side effects: rerolls the birdfeeder."""
    deck = list(birds)
    rng.shuffle(deck)
    bonus_deck = list(bonuses)
    rng.shuffle(bonus_deck)
    round_goals = _select_round_goals(rng, goals)

    players = [Player(id=i, name=player_names[i]) for i in range(2)]
    start_player = rng.randint(0, len(players) - 1)
    state = GameState(
        rng=rng,
        players=players,
        current_player=start_player,
        start_player=start_player,
        round_idx=0,
        bird_deck=deck,
        bonus_deck=bonus_deck,
        round_goals=round_goals,
    )
    state.refill_tray()
    state.birdfeeder.reroll(rng)
    return state


###### PRIVATE #######


_N_ROUND_GOALS = 4


def _select_round_goals(
    rng: random.Random, goals: list[cards.EndRoundGoal]
) -> list[cards.EndRoundGoal]:
    """Pick four goals that come from four distinct physical tiles.

    Goals are grouped by ``tile_id``; four tile groups are chosen without
    replacement, then one side (goal) is chosen at random from each group."""
    # Group goals by tile_id, preserving only tiles with at least one goal.
    tiles: dict[int, list[cards.EndRoundGoal]] = {}
    for goal in goals:
        tiles.setdefault(goal.tile_id, []).append(goal)

    available_tile_ids = list(tiles.keys())
    rng.shuffle(available_tile_ids)
    chosen_tile_ids = available_tile_ids[:_N_ROUND_GOALS]

    return [rng.choice(tiles[tile_id]) for tile_id in chosen_tile_ids]


def _track_count(num_birds: int, track: tuple[int, ...]) -> int:
    """Map a row's bird count to its printed action-reward count.

    The reward track is indexed by column (1..5 birds → indices 0..4); when
    a row exceeds 5 birds (unreachable under printed rules but allowed by
    expansions / power moves), each extra bird adds one more reward."""
    return track[min(num_birds, len(track) - 1)] + max(0, num_birds - len(track))


def _track_offers_convert(num_birds: int) -> bool:
    """Whether a row with ``num_birds`` birds lands the action cube on a trade
    space. Trade arrows sit on every other action space, so they are reached
    when the cube lands on an odd-indexed slot (1, 3, 5 birds played)."""
    return num_birds % 2 == 1
