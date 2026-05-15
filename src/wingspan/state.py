"""Core game state.

Designed so a full game state can be deep-copied cheaply (no Python heavy
structures, no references back to the engine). All mutation goes through
methods on the engine so we have a single audit trail for the game log.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .cards import (
    ALL_FOODS, ALL_HABITATS, Bird, BonusCard, EndRoundGoal, Food, Habitat,
)

ROUND_CUBES = [8, 7, 6, 5]   # action cubes per player per round (rounds 1..4)
ROW_SLOTS = 5                # 5 columns per habitat row
N_HABITATS = 3
BIRDFEEDER_DICE = 5
TRAY_SIZE = 3
HAND_LIMIT = None            # Wingspan has no hand limit
STARTING_HAND_SIZE = 5
STARTING_FOOD_TOTAL = 5      # players choose food+cards totalling 5
STARTING_BONUS_CARDS_DEAL = 2  # deal 2, keep 1


@dataclass
class PlayedBird:
    bird: Bird
    eggs: int = 0
    cached_food: int = 0          # generic count; we don't track specific cached food types
    tucked_cards: int = 0
    # number of times the bird has been activated this round (used by some rules)
    activations: int = 0


@dataclass
class Player:
    id: int
    name: str
    hand: list[Bird] = field(default_factory=list)
    bonus_cards: list[BonusCard] = field(default_factory=list)
    food: dict[Food, int] = field(default_factory=lambda: {f: 0 for f in ALL_FOODS})
    # board[h] is the ordered list of PlayedBird in habitat h (left to right)
    board: dict[Habitat, list[PlayedBird]] = field(
        default_factory=lambda: {h: [] for h in ALL_HABITATS}
    )
    action_cubes_left: int = 0
    # cumulative end-of-round goal points (in 2P play, accumulate per round)
    round_goal_points: int = 0

    # Lay-eggs / draw-cards / gain-food bonus tracking (number of "extra" steps
    # granted by columns occupied on the row) — recomputed each activation.

    @property
    def total_eggs(self) -> int:
        return sum(pb.eggs for r in self.board.values() for pb in r)

    @property
    def total_tucked(self) -> int:
        return sum(pb.tucked_cards for r in self.board.values() for pb in r)

    @property
    def total_cached(self) -> int:
        return sum(pb.cached_food for r in self.board.values() for pb in r)

    def total_food(self) -> int:
        return sum(self.food.values())

    def can_play_in(self, habitat: Habitat) -> bool:
        return len(self.board[habitat]) < ROW_SLOTS


@dataclass
class Birdfeeder:
    counts: dict[Food, int] = field(default_factory=lambda: {f: 0 for f in ALL_FOODS})

    def total(self) -> int:
        return sum(self.counts.values())

    def reroll(self, rng: random.Random) -> None:
        # Reroll all dice. Wingspan uses 5 dice each showing one of 6 faces
        # (5 foods + 1 wild face that grants 2 foods of choice from feeder);
        # we simplify by treating the wild face as a random food.
        for f in ALL_FOODS:
            self.counts[f] = 0
        for _ in range(BIRDFEEDER_DICE):
            self.counts[rng.choice(ALL_FOODS)] += 1


@dataclass
class GameState:
    rng: random.Random
    players: list[Player]
    current_player: int
    round_idx: int                          # 0..3
    bird_deck: list[Bird]                   # remaining
    bird_discard: list[Bird] = field(default_factory=list)
    bonus_deck: list[BonusCard] = field(default_factory=list)
    bonus_discard: list[BonusCard] = field(default_factory=list)
    tray: list[Bird] = field(default_factory=list)
    birdfeeder: Birdfeeder = field(default_factory=Birdfeeder)
    food_supply: dict[Food, int] = field(default_factory=lambda: {f: 99 for f in ALL_FOODS})
    # All 16 goals shuffled — first 4 are the per-round goals for rounds 1..4
    round_goals: list[EndRoundGoal] = field(default_factory=list)
    game_over: bool = False
    log: list[str] = field(default_factory=list)

    # ------ helpers ----------------------------------------------------
    def opponent(self) -> Player:
        return self.players[1 - self.current_player]

    def me(self) -> Player:
        return self.players[self.current_player]

    def draw_bird(self) -> Optional[Bird]:
        if not self.bird_deck:
            if not self.bird_discard:
                return None
            self.bird_deck = self.bird_discard
            self.bird_discard = []
            self.rng.shuffle(self.bird_deck)
        return self.bird_deck.pop()

    def refill_tray(self) -> None:
        while len(self.tray) < TRAY_SIZE:
            b = self.draw_bird()
            if b is None:
                break
            self.tray.append(b)


def new_game(
    rng: random.Random,
    birds: list[Bird],
    bonuses: list[BonusCard],
    goals: list[EndRoundGoal],
    player_names: tuple[str, str] = ("P0", "P1"),
) -> GameState:
    deck = list(birds)
    rng.shuffle(deck)
    bonus_deck = list(bonuses)
    rng.shuffle(bonus_deck)
    goal_pool = list(goals)
    rng.shuffle(goal_pool)
    round_goals = goal_pool[:4]

    players = [Player(id=i, name=player_names[i]) for i in range(2)]
    state = GameState(
        rng=rng,
        players=players,
        current_player=0,
        round_idx=0,
        bird_deck=deck,
        bonus_deck=bonus_deck,
        round_goals=round_goals,
    )
    state.refill_tray()
    state.birdfeeder.reroll(rng)
    return state
