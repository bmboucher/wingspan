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
from wingspan.engine import scoring  # noqa: E402

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


def _goal_delta_ignoring_eggs_slot(
    row: np.ndarray, goal_idx: int
) -> tuple[float, float]:
    base = layout._OFF_GOAL_DELTA_IGNORING_EGGS + goal_idx * layout._GOAL_DELTA_SLOT_DIM
    return (
        float(row[base + layout._GOAL_DELTA_COUNT]),
        float(row[base + layout._GOAL_DELTA_VP]),
    )


def _exchange_slice(row: np.ndarray) -> np.ndarray:
    return row[layout._OFF_EXCHANGE : layout._OFF_EXCHANGE + layout._EXCHANGE_DIM]


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


# ---------------------------------------------------------------------------
# MAIN_ACTION row pricing: exchange forecast (GAIN_FOOD/LAY_EGGS/DRAW_CARDS)
# and PLAY_BIRD's best-case bonus_delta / goal_delta over every legal play.


def test_draw_cards_row_prices_worked_example_exchange():
    """The DRAW_CARDS row's exchange forecast matches the Stage 1 worked
    example at stripe level: a single wetland Common Yellowthroat
    (DRAW_CARDS_THEN_DISCARD_EOT amount=2) with an egg fires the wetland
    conversion, so the row prices 4 cards drawn (1 base + 1 conversion + 2
    power), 1 egg paid (the conversion), and 1 card discarded (the EOT
    side)."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    eot_bird = next(bird for bird in _BIRDS if bird.name == "Common Yellowthroat")
    player.hand = [eot_bird]
    player.board[cards.Habitat.WETLAND] = [state.PlayedBird(bird=eot_bird, eggs=1)]

    row = _main_action_rows(game_state)[decisions.MainAction.DRAW_CARDS]
    assert row[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_DRAW] == _Approx(
        4 / layout._EXCHANGE_SCALE
    )
    assert row[layout._OFF_EXCHANGE + layout._EXCHANGE_EGGS_TO_PAY] == _Approx(
        1 / layout._EXCHANGE_SCALE
    )
    assert row[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_DISCARD] == _Approx(
        1 / layout._EXCHANGE_SCALE
    )


def test_gain_food_and_lay_eggs_forecast_nonzero_play_bird_exchange_zero():
    """GAIN_FOOD and LAY_EGGS rows carry a nonzero base-track exchange
    forecast; the PLAY_BIRD row's exchange stripe stays all zero even when
    legal plays exist — its resource flows come from the played bird's
    power (not fired yet at the pick point), not from a habitat-row track."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.food = state.FoodPool(counts=[5, 5, 5, 5, 5])
    playable = next(bird for bird in _BIRDS if cards.Habitat.FOREST in bird.habitats)
    player.hand = [playable]
    # LAY_EGGS needs spare egg room on the board to have anywhere to lay.
    roomy = next(bird for bird in _BIRDS if bird.egg_limit >= 2)
    player.board[cards.Habitat.WETLAND] = [state.PlayedBird(bird=roomy)]

    rows = _main_action_rows(game_state)
    gain_food_row = rows[decisions.MainAction.GAIN_FOOD]
    assert gain_food_row[layout._OFF_EXCHANGE + layout._EXCHANGE_FOOD_TO_GAIN] > 0.0

    lay_eggs_row = rows[decisions.MainAction.LAY_EGGS]
    assert lay_eggs_row[layout._OFF_EXCHANGE + layout._EXCHANGE_EGGS_TO_GAIN] > 0.0

    play_bird_row = rows[decisions.MainAction.PLAY_BIRD]
    assert not _exchange_slice(play_bird_row).any()


def test_play_bird_bonus_delta_static_tag_and_per_card_independent_max():
    """A held bonus card statically tagged on a playable candidate prices
    nonzero; two held cards served by two DIFFERENT candidate birds are each
    priced at their own best (per-card independent max, not a single joint
    pick — a bug that evaluated only one "best overall" play would silently
    zero out whichever card that play doesn't tag)."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.food = state.FoodPool(counts=[5, 5, 5, 5, 5])

    bird_feeder = _BONUS_BY_NAME["Bird Feeder"]
    photographer = _BONUS_BY_NAME["Photographer"]
    tagged_a = next(
        bird
        for bird in _BIRDS
        if bird_feeder.name in bird.bonus_categories
        and photographer.name not in bird.bonus_categories
    )
    tagged_b = next(
        bird
        for bird in _BIRDS
        if photographer.name in bird.bonus_categories
        and bird_feeder.name not in bird.bonus_categories
    )
    player.hand = [tagged_a, tagged_b]
    player.bonus_cards = [bird_feeder, photographer]

    expected_stepped = sum(
        scoring.bonus_vp_deltas_for_count_change(bonus_card, 0, 1)[0]
        for bonus_card in (bird_feeder, photographer)
    )
    expected_linear = sum(
        scoring.bonus_vp_deltas_for_count_change(bonus_card, 0, 1)[1]
        for bonus_card in (bird_feeder, photographer)
    )

    row = _main_action_rows(game_state)[decisions.MainAction.PLAY_BIRD]
    qual, stepped, linear = _bonus_delta(row)
    assert qual == _Approx(2 / layout._BONUS_COUNT_SCALE)
    assert stepped == _Approx(expected_stepped / layout._BONUS_VALUE_SCALE)
    assert linear == _Approx(expected_linear / layout._BONUS_VALUE_SCALE)


def test_play_bird_goal_delta_habitat_specific_egg_ignoring_and_scored_freeze():
    """A birds_<habitat> goal moves only via a candidate landing in that
    habitat; an egg-driven goal stays silent in goal_delta (a freshly played
    bird has no eggs yet) but nonzero in goal_delta_ignoring_eggs (the
    played-and-optimally-egg-populated bound); a scored goal reads zero in
    both regardless of any candidate."""
    game_state = _game_with_goals(
        ["birds_wetland", "birds_forest", "eggs_forest", "total_birds"]
    )
    game_state.scored_goals.append(
        state.RoundGoalResult(counts=[0, 0], vp_awarded=[0, 0])
    )
    player = game_state.players[0]
    player.food = state.FoodPool(counts=[5, 5, 5, 5, 5])
    forest_bird = next(
        bird
        for bird in _BIRDS
        if set(bird.habitats) == {cards.Habitat.FOREST} and bird.egg_limit > 0
    )
    player.hand = [forest_bird]

    row = _main_action_rows(game_state)[decisions.MainAction.PLAY_BIRD]

    # Goal 0 (birds_wetland) is frozen by scored_goals — silent either way.
    assert _goal_delta_slot(row, 0) == (0.0, 0.0)
    assert _goal_delta_ignoring_eggs_slot(row, 0) == (0.0, 0.0)

    # Goal 1 (birds_forest) moves: the only candidate lands in forest.
    forest_count, _ = _goal_delta_slot(row, 1)
    assert forest_count == _Approx(1 / layout._GOAL_COUNT_SCALE)

    # Goal 2 (eggs_forest) is silent in goal_delta (no eggs yet) but nonzero
    # in goal_delta_ignoring_eggs (played-and-egg-populated bound).
    eggs_count, _ = _goal_delta_slot(row, 2)
    assert eggs_count == 0.0
    eggs_ignoring_count, _ = _goal_delta_ignoring_eggs_slot(row, 2)
    assert eggs_ignoring_count > 0.0


def test_lay_eggs_bonus_delta_dynamic_egg_card_priced_and_silent():
    """LAY_EGGS additionally prices the capacity-capped best case against the
    held dynamic egg-counting bonus cards: Oologist crosses its 1-egg
    threshold on both eggless birds when the row's 2 fresh eggs are
    committed; without a dynamic egg card the stripe stays silent."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    oologist = _BONUS_BY_NAME["Oologist"]
    roomy = next(bird for bird in _BIRDS if bird.egg_limit >= 4)
    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=roomy),
        state.PlayedBird(bird=roomy),
    ]
    player.bonus_cards = [oologist]

    row = _main_action_rows(game_state)[decisions.MainAction.LAY_EGGS]
    qual, stepped, linear = _bonus_delta(row)
    assert qual == _Approx(1 / layout._BONUS_COUNT_SCALE)
    expected_stepped, expected_linear = scoring.bonus_vp_deltas_for_count_change(
        oologist, 0, 2
    )
    assert stepped == _Approx(expected_stepped / layout._BONUS_VALUE_SCALE)
    assert linear == _Approx(expected_linear / layout._BONUS_VALUE_SCALE)

    static_card = next(
        bonus_card
        for bonus_card in _BONUSES
        if bonus_card.name not in ("Oologist", "Breeding Manager")
    )
    player.bonus_cards = [static_card]
    silent_row = _main_action_rows(game_state)[decisions.MainAction.LAY_EGGS]
    assert _bonus_delta(silent_row) == (0.0, 0.0, 0.0)


def test_play_bird_row_all_zero_without_legal_plays():
    """An empty hand -> no legal plays -> every PLAY_BIRD stripe (exchange,
    bonus_delta, goal_delta, goal_delta_ignoring_eggs) stays at zero."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.bonus_cards = [_BONUS_BY_NAME["Bird Feeder"]]
    assert not player.hand

    row = _main_action_rows(game_state)[decisions.MainAction.PLAY_BIRD]
    assert _bonus_delta(row) == (0.0, 0.0, 0.0)
    for goal_idx in range(4):
        assert _goal_delta_slot(row, goal_idx) == (0.0, 0.0)
        assert _goal_delta_ignoring_eggs_slot(row, goal_idx) == (0.0, 0.0)
    assert not _exchange_slice(row).any()
