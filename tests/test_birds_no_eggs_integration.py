# pyright: reportPrivateUsage=false
# (reads the layout's package-private stripe constants and the state
# encoder's private round-goal stripe builder)
"""Tests that the ``birds_no_eggs`` goal is a first-class encoder citizen.

The category joined the stable ``GOAL_CATEGORIES`` ordering (filling the last
``MAX_GOAL_CATEGORIES`` slot), so a dealt no-egg goal is visible by identity
in the state vector, and bird-carrying choice rows price the eggless newcomer
a play would add to the count. The egg-event and commitment-bound pricing for
this goal lives with its families (``test_egg_goal_delta``,
``test_commit_row_pricing``).
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, state  # noqa: E402
from wingspan.encode import layout, state_encode  # noqa: E402

_BIRDS, _BONUSES, _GOALS = cards.load_all()


class _Approx:
    """Tolerant float comparator (pytest.approx is untyped under strict pyright)."""

    def __init__(self, expected: float) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isclose(
            float(other), self.expected, rel_tol=1e-6, abs_tol=1e-9
        )


def _no_eggs_game() -> state.GameState:
    game_state = state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)
    goal = cards.EndRoundGoal(
        id=0, description="[bird] with no [egg]", category="birds_no_eggs", tile_id=0
    )
    game_state.round_goals = [goal] * 4
    return game_state


def test_category_fills_the_last_one_hot_slot():
    """The category sits at the stable ordering's final headroom slot
    (append-only contract). The core loader never deals the tile (its
    goals.json row is ``Set: european``), but the engine already branches on
    the category (``powers/grants``, ``powers/multi_actor``), so the encoders
    must price and identify it wherever it appears."""
    assert all(goal.category != "birds_no_eggs" for goal in _GOALS)
    assert layout.GOAL_CATEGORIES.index("birds_no_eggs") == 19
    assert len(layout.GOAL_CATEGORIES) <= layout.MAX_GOAL_CATEGORIES


def test_state_round_goal_identity_is_visible():
    """A dealt no-egg goal sets its category one-hot in the state vector
    (before the append it encoded as an all-zero identity)."""
    game_state = _no_eggs_game()
    vec = state_encode._round_goals_all_rounds(game_state, game_state.players[0])
    assert float(vec[layout.GOAL_CATEGORIES.index("birds_no_eggs")]) == 1.0


def test_bird_rows_price_the_eggless_newcomer():
    """Every bird-carrying row prices +1 toward the no-egg count: a played
    bird arrives without eggs."""
    game_state = _no_eggs_game()
    any_bird = _BIRDS[0]
    decision = decisions.BirdPowerTuckFromHandDecision(
        player_id=0,
        prompt="tuck",
        choices=[decisions.BirdChoice(label=any_bird.name, bird=any_bird)],
    )
    row = encode.encode_choices(decision, game_state)[0]
    base = layout._OFF_GOAL_DELTA
    assert float(row[base + layout._GOAL_DELTA_COUNT]) == _Approx(1 / 5)
    assert float(row[base + layout._GOAL_DELTA_VP]) == _Approx(4 / 10)
