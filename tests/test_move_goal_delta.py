# pyright: reportPrivateUsage=false
# (reads the layout's package-private stripe constants to slice choice rows)
"""Tests for move-bird consequence pricing on habitat rows.

A ``BirdPowerPickHabitatDecision`` carries the move context (``moving_bird`` /
``from_habitat``): each destination row prices relocating that bird — habitat
bird counts, the egg block riding along (including the egg-set minimum), and
the habitat-spread bonus card — and marks the bird's landing slot in the
board-index block (the destination row's next free slot; the "stay" row marks
the bird's current slot). The "stay" row's deltas are all-zero.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pydantic
import pytest

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


def _board_location(row: np.ndarray) -> tuple[int | None, int | None]:
    """Return (hab_idx, col_idx) indicated by the board_hab/board_col one-hots,
    or (None, None) if either block is all-zero."""
    hab_block = row[
        layout._OFF_BOARD_HAB : layout._OFF_BOARD_HAB + layout._BOARD_HAB_DIM
    ]
    col_block = row[
        layout._OFF_BOARD_COL : layout._OFF_BOARD_COL + layout._BOARD_COL_DIM
    ]
    hab_hits = [idx for idx, val in enumerate(hab_block) if val != 0.0]
    col_hits = [idx for idx, val in enumerate(col_block) if val != 0.0]
    hab_idx: int | None = hab_hits[0] if len(hab_hits) == 1 else None
    col_idx: int | None = col_hits[0] if len(col_hits) == 1 else None
    return hab_idx, col_idx


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
        assert row[layout._OFF_BIRD_ID] == cards.bird_index(roomy) + 1
    for goal_idx in range(4):
        assert _goal_delta_slot(stay_row, goal_idx) == (0.0, 0.0)

    # Each row marks exactly one board location — the bird's landing slot
    # via the board_hab and board_col one-hots.
    # The mover sits at forest slot 1 (rightmost of [static, mover]); the
    # grassland and wetland rows each hold one bird, so it would land at slot 1.
    hab_indices = list(cards.ALL_HABITATS)
    assert _board_location(stay_row) == (hab_indices.index(cards.Habitat.FOREST), 1)
    assert _board_location(grass_row) == (hab_indices.index(cards.Habitat.GRASSLAND), 1)
    assert _board_location(wetland_row) == (hab_indices.index(cards.Habitat.WETLAND), 1)

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


def test_move_context_is_required():
    """The decision's move context is required by construction — the plain
    habitat-designation mode (no ``moving_bird``) no longer exists, so the
    encoder never sees a context-free habitat row."""
    with pytest.raises(pydantic.ValidationError):
        decisions.BirdPowerPickHabitatDecision.model_validate(
            {
                "player_id": 0,
                "prompt": "designate",
                "choices": [
                    decisions.HabitatChoice(
                        label="forest", habitat=cards.Habitat.FOREST
                    )
                ],
            }
        )
