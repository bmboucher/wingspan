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
    # generic count; we don't track specific cached food types
    cached_food: int = 0
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
        return sum(pb.cached_food for row in self.board.values() for pb in row)

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


class Birdfeeder(pydantic.BaseModel):
    """The communal birdfeeder dice tower."""

    counts: FoodPool = pydantic.Field(default_factory=FoodPool)

    def total(self) -> int:
        return self.counts.total()

    def reroll(self, rng: random.Random) -> None:
        """Reroll all five dice. Wingspan uses five dice each showing one of
        six faces (5 foods + 1 wild face that grants 2 foods of choice from
        the feeder); we simplify by treating the wild face as a random food."""
        self.counts.zero()
        for _ in range(BIRDFEEDER_DICE):
            self.counts[rng.choice(cards.ALL_FOODS)] += 1


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
    # The face-up bird tray. Holds up to ``TRAY_SIZE`` cards but may carry
    # fewer mid-turn: a card taken from the tray leaves its slot empty until
    # ``refill_tray`` runs (at the end of the turn, or when a bird power
    # explicitly refills). The list is kept compact — an empty slot is simply
    # an absent entry — so ``len(tray)`` is the number of cards still face-up.
    tray: list[cards.Bird] = pydantic.Field(default_factory=_new_bird_list)
    birdfeeder: Birdfeeder = pydantic.Field(default_factory=Birdfeeder)
    food_supply: FoodPool = pydantic.Field(default_factory=lambda: FoodPool.uniform(99))
    # All 16 goals shuffled — first 4 are the per-round goals for rounds 1..4
    round_goals: list[cards.EndRoundGoal] = pydantic.Field(
        default_factory=_new_round_goal_list
    )
    game_over: bool = False
    log: list[str] = pydantic.Field(default_factory=list)

    # Per-turn scratch state. Reset at the start of every turn by the engine.
    # ``turn_extra_play_habitat``, when set, restricts the next extra play to a
    # single habitat (e.g. House Wren grants +1 play in this bird's habitat)
    # and is cleared by the extra-plays loop after consuming one play.
    turn_extra_plays: int = 0
    turn_extra_play_habitat: cards.Habitat | None = None

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
        """Top the tray back up to ``TRAY_SIZE`` from the deck.

        Called at the end of each turn and by powers that explicitly refill
        (e.g. Brant) — never automatically when a card is drawn mid-turn.
        Stops early if the deck (and discard) run dry, leaving the tray short."""
        while len(self.tray) < TRAY_SIZE:
            drawn = self.draw_bird()
            if drawn is None:
                break
            self.tray.append(drawn)

    def reset_turn_state(self) -> None:
        """Clear per-turn scratch fields. Called at the start of every turn."""
        self.turn_extra_plays = 0
        self.turn_extra_play_habitat = None


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
    goal_pool = list(goals)
    rng.shuffle(goal_pool)
    round_goals = goal_pool[:4]

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
