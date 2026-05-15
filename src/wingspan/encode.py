"""State and action encoders for RL.

The state encoder produces a fixed-size dense feature vector summarizing the
game from the active player's POV. We avoid the temptation to give the agent
the full deck composition (too large for the first training cycle to learn);
instead it sees aggregate hand statistics, board summaries, food, supply
counts, round number, and round-goal category as a one-hot.

The action encoder maps a Decision into a *fixed-size policy slot* by
combining the decision type with a small per-type index. Because Wingspan
decisions have wildly different cardinalities (4 main actions vs picking
which of 180 birds in your hand to play), we cap each type at a reasonable
maximum and present the legal mask alongside the slot list.
"""
from __future__ import annotations

import numpy as np

from .actions import Decision, DecisionType, MainAction
from .cards import ALL_FOODS, ALL_HABITATS, Bird, Food, Habitat, PowerColor
from .state import GameState, Player

# Caps for variable-length decision slots
MAX_HAND_PICKS = 10        # hand can grow; we surface the first 10 options
MAX_BOARD_TARGETS = 15     # up to 5 birds * 3 habitats
MAX_TRAY_PICKS = 4         # 3 tray slots + 1 deck
MAX_FOOD_TYPES = 5
MAX_HABITATS = 3
MAX_PAYMENT_PICKS = 8      # food-payment options
MAX_GOAL_CATEGORIES = 18

# Decision slot table: each decision type gets a contiguous block of action ids
_TYPE_SLOTS: list[tuple[DecisionType, int]] = [
    (DecisionType.SETUP_KEEP_FOOD_OR_DISCARD_CARD, 2),
    (DecisionType.SETUP_PICK_BONUS, 2),
    (DecisionType.MAIN_ACTION, 4),
    (DecisionType.PLAY_BIRD_PICK_CARD, MAX_HAND_PICKS),
    (DecisionType.PLAY_BIRD_PICK_HABITAT, MAX_HABITATS),
    (DecisionType.PLAY_BIRD_PICK_FOOD_PAYMENT, MAX_PAYMENT_PICKS),
    (DecisionType.PLAY_BIRD_PICK_EGG_TO_PAY, MAX_BOARD_TARGETS),
    (DecisionType.GAIN_FOOD_PICK_DIE, MAX_FOOD_TYPES),
    (DecisionType.LAY_EGG_PICK_BIRD, MAX_BOARD_TARGETS),
    (DecisionType.DRAW_CARDS_PICK_SOURCE, MAX_TRAY_PICKS),
    (DecisionType.BIRD_POWER_PICK_FOOD, MAX_FOOD_TYPES),
    (DecisionType.BIRD_POWER_PICK_BIRD, MAX_BOARD_TARGETS),
    (DecisionType.BIRD_POWER_TUCK_FROM_HAND, MAX_HAND_PICKS + 1),
    (DecisionType.BIRD_POWER_PICK_STARTING_PLAYER, 2),
    (DecisionType.BIRD_POWER_PICK_HABITAT, MAX_HABITATS),
    (DecisionType.SKIP_OPTIONAL, 2),
]
TYPE_OFFSET: dict[DecisionType, int] = {}
_offset = 0
for t, w in _TYPE_SLOTS:
    TYPE_OFFSET[t] = _offset
    _offset += w
TOTAL_ACTION_SLOTS = _offset


# State encoding -----------------------------------------------------------

def _summary_food(player: Player) -> np.ndarray:
    return np.array([player.food[f] for f in ALL_FOODS], dtype=np.float32)


def _summary_board(player: Player) -> np.ndarray:
    parts = []
    for h in ALL_HABITATS:
        row = player.board[h]
        parts.append(np.array([
            len(row),
            sum(pb.eggs for pb in row),
            sum(pb.bird.points for pb in row),
            sum(pb.tucked_cards for pb in row),
            sum(pb.cached_food for pb in row),
            sum(1 for pb in row if pb.bird.color == PowerColor.BROWN),
        ], dtype=np.float32))
    return np.concatenate(parts)


def _summary_hand(player: Player) -> np.ndarray:
    if not player.hand:
        return np.zeros(8, dtype=np.float32)
    pts = [b.points for b in player.hand]
    costs = [b.total_food_cost for b in player.hand]
    eggs = [b.egg_limit for b in player.hand]
    return np.array([
        len(player.hand),
        np.mean(pts), np.max(pts),
        np.mean(costs), np.min(costs),
        np.mean(eggs),
        sum(1 for b in player.hand if Habitat.FOREST in b.habitats),
        sum(1 for b in player.hand if Habitat.WETLAND in b.habitats),
    ], dtype=np.float32)


def _summary_birdfeeder(state: GameState) -> np.ndarray:
    return np.array([state.birdfeeder.counts[f] for f in ALL_FOODS], dtype=np.float32)


# Stable global ordering of goal categories
_GOAL_CATEGORIES = [
    "birds_forest","birds_grassland","birds_wetland",
    "eggs_forest","eggs_grassland","eggs_wetland",
    "eggs_bowl","eggs_cavity","eggs_ground","eggs_platform",
    "bowl_birds_with_eggs","cavity_birds_with_eggs",
    "ground_birds_with_eggs","platform_birds_with_eggs",
    "tucked_cards","wingspan_under_30","wingspan_over_65",
]


def encode_state(state: GameState) -> np.ndarray:
    me = state.me()
    opp = state.opponent()
    parts: list[np.ndarray] = []
    parts.append(_summary_food(me))                         # 5
    parts.append(_summary_food(opp))                        # 5
    parts.append(_summary_board(me))                        # 18
    parts.append(_summary_board(opp))                       # 18
    parts.append(_summary_hand(me))                         # 8
    parts.append(np.array([len(opp.hand)], dtype=np.float32))
    parts.append(_summary_birdfeeder(state))                # 5
    parts.append(np.array([
        state.round_idx,
        me.action_cubes_left,
        opp.action_cubes_left,
        me.round_goal_points,
        opp.round_goal_points,
        len(state.tray),
        len(state.bird_deck),
    ], dtype=np.float32))                                   # 7
    # Round goal one-hot (current + remaining)
    rg = np.zeros(MAX_GOAL_CATEGORIES, dtype=np.float32)
    if state.round_idx < len(state.round_goals):
        cat = state.round_goals[state.round_idx].category
        if cat in _GOAL_CATEGORIES:
            rg[_GOAL_CATEGORIES.index(cat)] = 1.0
    parts.append(rg)
    return np.concatenate(parts).astype(np.float32)


def state_size() -> int:
    # Match what encode_state produces. Hand-computed; asserted at runtime.
    return 5 + 5 + 18 + 18 + 8 + 1 + 5 + 7 + MAX_GOAL_CATEGORIES


# Action encoding ----------------------------------------------------------

def encode_decision(decision: Decision) -> tuple[np.ndarray, list[int]]:
    """Return (mask, action_ids).

    ``mask`` is a TOTAL_ACTION_SLOTS-length 0/1 array marking legal actions.
    ``action_ids`` is the list of slot ids for the legal choices, in the same
    order as ``decision.choices`` (so the agent can map an action id back to a
    Choice object).
    """
    mask = np.zeros(TOTAL_ACTION_SLOTS, dtype=np.float32)
    offset = TYPE_OFFSET[decision.type]
    ids: list[int] = []
    # We pack choices by their order in the Decision, capped at the slot size.
    block_size = next(w for (t, w) in _TYPE_SLOTS if t == decision.type)
    for i, _c in enumerate(decision.choices[:block_size]):
        slot = offset + i
        mask[slot] = 1.0
        ids.append(slot)
    return mask, ids
