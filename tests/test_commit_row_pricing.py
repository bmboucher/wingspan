# pyright: reportPrivateUsage=false
# (reads the layout's package-private stripe constants to slice choice rows)
"""Tests for consequence pricing on commitment rows.

The *whether* choices — the main-action menu and the accept-exchange rows —
commit to resource flows whose targets are picked in follow-up decisions, so
they carry aggregate pricing: net hand-card flow against the hand-counting
bonus card (``bonus_delta``), and a capacity-capped optimistic round-goal
bound for committed egg gains / payments (``goal_delta``).
"""

from __future__ import annotations

import math
import random

import numpy as np

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


def _main_action_rows(
    game_state: state.GameState,
) -> dict[decisions.MainAction, np.ndarray]:
    actions = list(decisions.MainAction)
    decision = decisions.MainActionDecision(
        player_id=0,
        prompt="act",
        choices=[
            decisions.MainActionChoice(label=action.value, action=action)
            for action in actions
        ],
    )
    feats = encode.encode_choices(decision, game_state)
    return {action: feats[i] for i, action in enumerate(actions)}


def _accept_row(
    game_state: state.GameState, accept: decisions.PayCostChoice
) -> np.ndarray:
    decision = decisions.AcceptExchangeDecision(
        player_id=0,
        prompt="trade",
        choices=[accept, decisions.SkipChoice(label="skip")],
    )
    return encode.encode_choices(decision, game_state)[0]


def test_draw_cards_action_prices_hand_growth():
    """With Visionary Leader held at 4 cards, the 1-card wetland draw crosses
    the 5-card tier (+4 VP stepped); the other actions price nothing."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.bonus_cards = [_BONUS_BY_NAME["Visionary Leader"]]
    player.hand = list(_BIRDS[:4])

    rows = _main_action_rows(game_state)
    qual, stepped, linear = _bonus_delta(rows[decisions.MainAction.DRAW_CARDS])
    assert qual == _Approx(1 / 5)
    assert stepped == _Approx(4 / 7)  # score(5) - score(4) = 4 - 0
    assert linear == _Approx(0.8 / 7)  # linear(5) - linear(4) = 4.0 - 3.2
    assert _bonus_delta(rows[decisions.MainAction.GAIN_FOOD]) == (0.0, 0.0, 0.0)
    assert _bonus_delta(rows[decisions.MainAction.PLAY_BIRD]) == (0.0, 0.0, 0.0)


def test_lay_eggs_action_prices_capacity_capped_bound():
    """The LAY_EGGS row advertises the best the committed eggs could do for an
    egg goal, capped by real room in qualifying slots."""
    game_state = _game_with_goals(["eggs_forest"] * 4)
    player = game_state.players[0]
    roomy = next(bird for bird in _BIRDS if bird.egg_limit >= 4)
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=roomy))
    # Empty grassland row -> the action lays 2 eggs; forest room >= 2.
    count, vp = _goal_delta_slot(
        _main_action_rows(game_state)[decisions.MainAction.LAY_EGGS], 0
    )
    assert count == _Approx(2 / 5)
    assert vp == _Approx(4 / 10)  # 0 -> 2 vs opp 0 takes round-1 first (4)

    # Cap the forest room at one egg: the bound drops to +1.
    player.board[cards.Habitat.FOREST][0].eggs = roomy.egg_limit - 1
    count, _ = _goal_delta_slot(
        _main_action_rows(game_state)[decisions.MainAction.LAY_EGGS], 0
    )
    assert count == _Approx(1 / 5)


def test_lay_eggs_action_prices_no_egg_overflow():
    """With the birds_no_eggs anti-goal active, the LAY_EGGS row prices the
    forced overflow: spare room on already-egged birds absorbs eggs for free,
    and only the remainder costs an eggless bird its status."""
    game_state = _game_with_goals(["birds_no_eggs"] * 4)
    player = game_state.players[0]
    roomy = next(bird for bird in _BIRDS if bird.egg_limit >= 4)
    player.board[cards.Habitat.FOREST].append(
        state.PlayedBird(bird=roomy, eggs=roomy.egg_limit - 1)  # spare room 1
    )
    player.board[cards.Habitat.WETLAND].append(state.PlayedBird(bird=roomy))

    # Empty grassland row -> the action lays 2 eggs; one overflows onto the
    # only eggless bird, forfeiting the goal entirely (count 1 -> 0).
    count, vp = _goal_delta_slot(
        _main_action_rows(game_state)[decisions.MainAction.LAY_EGGS], 0
    )
    assert count == _Approx(-1 / 5)
    assert vp == _Approx(-4 / 10)

    # Free up the egged bird's room: both eggs now land beside existing eggs.
    player.board[cards.Habitat.FOREST][0].eggs = 1
    silent = _goal_delta_slot(
        _main_action_rows(game_state)[decisions.MainAction.LAY_EGGS], 0
    )
    assert silent == (0.0, 0.0)


def test_accept_egg_gain_prices_optimistic_bound():
    """The Grassland-conversion accept row (pay food -> +1 egg) advertises the
    one egg's best case; the skip row stays silent."""
    game_state = _game_with_goals(["eggs_forest"] * 4)
    player = game_state.players[0]
    player.board[cards.Habitat.FOREST].append(
        state.PlayedBird(bird=next(bird for bird in _BIRDS if bird.egg_limit >= 2))
    )
    accept = decisions.PayCostChoice(
        label="pay 1 food", paid_food_count=1, gained_egg_count=1
    )
    row = _accept_row(game_state, accept)
    count, vp = _goal_delta_slot(row, 0)
    assert count == _Approx(1 / 5)
    assert vp == _Approx(4 / 10)


def test_accept_egg_payment_prices_least_damage():
    """The Wetland-conversion accept row (pay 1 egg -> +1 card): forced to
    break the goal when every egg sits in the goal's habitat, free when an
    expendable egg exists elsewhere."""
    game_state = _game_with_goals(["eggs_forest"] * 4)
    player = game_state.players[0]
    any_bird = next(bird for bird in _BIRDS if bird.egg_limit >= 2)
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=any_bird, eggs=1))
    accept = decisions.PayCostChoice(
        label="pay 1 egg", paid_egg_count=1, gained_card_count=1
    )

    forced = _goal_delta_slot(_accept_row(game_state, accept), 0)
    assert forced[0] == _Approx(-1 / 5)
    assert forced[1] == _Approx(-4 / 10)

    # An egg outside the goal's habitat makes the payment dodgeable.
    player.board[cards.Habitat.WETLAND].append(state.PlayedBird(bird=any_bird, eggs=1))
    dodgeable = _goal_delta_slot(_accept_row(game_state, accept), 0)
    assert dodgeable == (0.0, 0.0)


def test_accept_rows_price_net_hand_flow():
    """Card flows on accept rows price the hand-counting bonus card in both
    directions: the Forest-conversion discard shrinks the hand below a tier,
    the Oystercatcher double-draw climbs toward one."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.bonus_cards = [_BONUS_BY_NAME["Visionary Leader"]]

    player.hand = list(_BIRDS[:5])  # exactly at the 5-card tier (4 VP)
    discard = decisions.PayCostChoice(
        label="discard a card", paid_card_count=1, gained_food_count=1
    )
    qual, stepped, linear = _bonus_delta(_accept_row(game_state, discard))
    assert qual == _Approx(1 / 5)
    assert stepped == _Approx(-4 / 7)  # score(4) - score(5) = 0 - 4
    assert linear == _Approx(-0.8 / 7)

    player.hand = list(_BIRDS[:4])
    double_draw = decisions.PayCostChoice(
        label="draw 2, opp draws 1", gained_card_count=2, opp_gained_card_count=1
    )
    qual, stepped, linear = _bonus_delta(_accept_row(game_state, double_draw))
    assert qual == _Approx(1 / 5)
    assert stepped == _Approx(4 / 7)  # score(6) - score(4) = 4 - 0
    assert linear == _Approx(1.8 / 7)  # linear(6) - linear(4) = 5.0 - 3.2


def test_accept_rows_silent_without_consequences():
    """No held bonus and no egg terms -> the trade rows carry no deltas (the
    extra-play accept and the tuck-from-deck trade are priced downstream)."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    extra_play = decisions.PayCostChoice(label="extra play", gained_play_count=1)
    row = _accept_row(game_state, extra_play)
    assert _bonus_delta(row) == (0.0, 0.0, 0.0)
    for goal_idx in range(4):
        assert _goal_delta_slot(row, goal_idx) == (0.0, 0.0)
