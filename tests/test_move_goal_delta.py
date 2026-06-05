# pyright: reportPrivateUsage=false
# (reads the layout's package-private stripe constants to slice choice rows)
"""Tests for move-bird consequence pricing on habitat rows.

When a ``BirdPowerPickHabitatDecision`` carries the move context
(``moving_bird`` / ``from_habitat``), each destination row prices relocating
that bird: habitat bird counts, the egg block riding along (including the
egg-set minimum), and the habitat-spread bonus card. The "stay" row's deltas
are all-zero, and a context-free habitat designation stays a bare one-hot.
"""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, state  # noqa: E402
from wingspan.encode import layout  # noqa: E402

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}


class _Approx:
    """Tolerant float comparator (pytest.approx is untyped under strict pyright)."""

    def __init__(self, expected: float) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isclose(
            float(other), self.expected, rel_tol=1e-6, abs_tol=1e-9
        )


def _game_with_goals(categories: list[str]) -> state.GameState:
    game_state = state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)
    game_state.round_goals = [
        cards.EndRoundGoal(id=i, description=cat, category=cat, tile_id=i)
        for i, cat in enumerate(categories)
    ]
    return game_state


def _goal_delta_slot(row: np.ndarray, goal_idx: int) -> tuple[float, float]:
    base = layout._OFF_GOAL_DELTA + goal_idx * layout._GOAL_DELTA_SLOT_DIM
    return (
        float(row[base + layout._GOAL_DELTA_COUNT]),
        float(row[base + layout._GOAL_DELTA_VP]),
    )


def _bonus_delta(row: np.ndarray) -> tuple[float, float, float]:
    base = layout._OFF_BONUS_DELTA
    return (
        float(row[base + layout._BONUS_DELTA_QUAL]),
        float(row[base + layout._BONUS_DELTA_STEPPED]),
        float(row[base + layout._BONUS_DELTA_LINEAR]),
    )


def _move_rows(
    game_state: state.GameState,
    moving_bird: state.PlayedBird,
    from_habitat: cards.Habitat,
    destinations: list[cards.Habitat],
) -> np.ndarray:
    decision = decisions.BirdPowerPickHabitatDecision(
        player_id=0,
        prompt="move",
        choices=[
            decisions.HabitatChoice(label=habitat.value, habitat=habitat)
            for habitat in destinations
        ],
        moving_bird=moving_bird,
        from_habitat=from_habitat,
    )
    return encode.encode_choices(decision, game_state)


def test_move_rows_price_bird_count_eggs_and_sets():
    """Moving a 2-egg bird out of the forest prices the bird-count loss, the
    egg block leaving / arriving, and the egg-set minimum it completes."""
    game_state = _game_with_goals(
        ["birds_forest", "eggs_forest", "eggs_wetland", "egg_sets_3habitats"]
    )
    player = game_state.players[0]
    roomy = next(bird for bird in _BIRDS if bird.egg_limit >= 3)
    static_pb = state.PlayedBird(bird=roomy, eggs=1)
    mover = state.PlayedBird(bird=roomy, eggs=2)
    player.board[cards.Habitat.FOREST].extend([static_pb, mover])
    player.board[cards.Habitat.GRASSLAND].append(state.PlayedBird(bird=roomy, eggs=1))
    player.board[cards.Habitat.WETLAND].append(state.PlayedBird(bird=roomy, eggs=0))

    rows = _move_rows(
        game_state,
        mover,
        cards.Habitat.FOREST,
        [cards.Habitat.FOREST, cards.Habitat.GRASSLAND, cards.Habitat.WETLAND],
    )
    stay_row, grass_row, wetland_row = rows[0], rows[1], rows[2]

    # Every row carries the moving bird's identity; the stay row prices zero.
    for row in rows:
        assert row[layout._OFF_BIRD_ID + cards.bird_index(roomy)] == 1.0
    for goal_idx in range(4):
        assert _goal_delta_slot(stay_row, goal_idx) == (0.0, 0.0)

    # To wetland: forest loses a bird (-1) and 2 eggs (-2); wetland gains the
    # 2 eggs; the per-habitat egg minimum rises 0 -> 1 (a completed set).
    assert _goal_delta_slot(wetland_row, 0)[0] == _Approx(-1 / 5)
    assert _goal_delta_slot(wetland_row, 1)[0] == _Approx(-2 / 5)
    count, vp = _goal_delta_slot(wetland_row, 2)
    assert count == _Approx(2 / 5)
    assert vp == _Approx(6 / 10)  # 0 -> 2 vs opp 0 takes round-3 first (6)
    count, vp = _goal_delta_slot(wetland_row, 3)
    assert count == _Approx(1 / 5)
    assert vp == _Approx(7 / 10)  # 0 -> 1 set vs opp 0 takes round-4 first (7)

    # To grassland: same forest losses, but no wetland eggs and no set raised.
    assert _goal_delta_slot(grass_row, 0)[0] == _Approx(-1 / 5)
    assert _goal_delta_slot(grass_row, 1)[0] == _Approx(-2 / 5)
    assert _goal_delta_slot(grass_row, 2) == (0.0, 0.0)
    assert _goal_delta_slot(grass_row, 3) == (0.0, 0.0)


def test_move_rows_price_habitat_spread_bonus():
    """With Ecologist held, evening out the rows (+1 to the unique minimum)
    pays 2 VP per bird on the new minimum; a sideways move prices nothing."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.bonus_cards = [_BONUS_BY_NAME["Ecologist"]]
    any_bird = _BIRDS[0]
    for habitat, count in zip(cards.ALL_HABITATS, (3, 2, 1)):
        player.board[habitat] = [state.PlayedBird(bird=any_bird) for _ in range(count)]
    mover = player.board[cards.Habitat.FOREST][-1]

    rows = _move_rows(
        game_state,
        mover,
        cards.Habitat.FOREST,
        [cards.Habitat.WETLAND, cards.Habitat.GRASSLAND],
    )
    # Forest -> wetland: rows become 2/2/2, the minimum rises 1 -> 2 (+2 VP).
    qual, stepped, linear = _bonus_delta(rows[0])
    assert qual == _Approx(1 / 5)
    assert stepped == _Approx(2 / 7)
    assert linear == _Approx(2 / 7)
    # Forest -> grassland: rows become 2/3/1, the minimum stays 1.
    assert _bonus_delta(rows[1]) == (0.0, 0.0, 0.0)


def test_context_free_habitat_rows_stay_bare():
    """A habitat designation with no move context carries no identity and no
    deltas — just the habitat one-hot."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    decision = decisions.BirdPowerPickHabitatDecision(
        player_id=0,
        prompt="designate",
        choices=[decisions.HabitatChoice(label="forest", habitat=cards.Habitat.FOREST)],
    )
    row = encode.encode_choices(decision, game_state)[0]
    bird_id_block = row[layout._OFF_BIRD_ID : layout._OFF_BIRD_ID + layout._BIRD_ID_DIM]
    assert not bird_id_block.any()
    for goal_idx in range(4):
        assert _goal_delta_slot(row, goal_idx) == (0.0, 0.0)
    assert _bonus_delta(row) == (0.0, 0.0, 0.0)
