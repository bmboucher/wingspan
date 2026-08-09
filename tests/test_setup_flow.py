"""Tests for ``wingspan.engine.setup_flow`` — the setup-resolution free
functions split out of ``engine.core.Engine`` (``apply_setup_choice``,
``resolve_deferred_setup_bonus``, ``resolve_deferred_setup_food``,
``clear_setup_food``).

Drives each function directly (bypassing the full setup phase) with scripted
agents, mirroring the pattern in ``test_combine_gain_food.py``: a generic
``__call__[C: decisions.Choice]`` agent, typed against the ``Decision`` it is
handed, with ``typing.cast`` where the return type can't be narrowed for free.
"""

from __future__ import annotations

import random
import typing

import pytest

from wingspan import cards, decisions, engine, state
from wingspan.engine import setup_flow


def _new_game(seed: int = 0) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(seed), birds, bonuses, goals)


def _deal_one_of_each_food(player: state.Player) -> None:
    """Mimic the post-deal starting food pool: one of each food type, the
    state ``resolve_deferred_setup_food`` sees when ``defer_food=True`` (the
    combined setup choice skips the food update on that path)."""
    for food in cards.ALL_FOODS:
        player.food[food] = 1


def _recording_agent() -> tuple[list[decisions.Decision[typing.Any]], "engine.Agent"]:
    """A scripted agent that records every decision it is asked (in order)
    and always answers with the first offered choice."""
    seen: list[decisions.Decision[typing.Any]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        seen.append(decision)
        return decision.choices[0]

    return seen, agent


# ---------------------------------------------------------------------------
# resolve_deferred_setup_food — decision table across n_kept (defer_food=True,
# combine_gain_food=False)


@pytest.mark.parametrize(
    "n_kept,expected_count,expected_kind",
    [
        (0, 0, None),
        (1, 1, "spend"),
        (2, 2, "spend"),
        (3, 2, "gain"),
        (4, 1, "gain"),
        (5, 0, None),
    ],
)
def test_resolve_deferred_setup_food_decision_table(
    n_kept: int, expected_count: int, expected_kind: str | None
) -> None:
    """For each ``n_kept`` from 0..5, ``resolve_deferred_setup_food`` asks
    exactly the documented count and kind of decisions."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    _deal_one_of_each_food(player)
    seen, agent = _recording_agent()

    eng = engine.Engine(gs)
    setup_flow.resolve_deferred_setup_food(eng, player, agent, n_kept, defer_food=True)

    assert len(seen) == expected_count
    if expected_kind == "spend":
        assert all(isinstance(d, decisions.SpendFoodDecision) for d in seen)
    elif expected_kind == "gain":
        assert all(isinstance(d, decisions.GainFoodDecision) for d in seen)


def test_resolve_deferred_setup_food_spend_path_no_repeated_food() -> None:
    """n_kept=2: two sequential ``SpendFoodDecision`` asks that never offer
    (or pick) the same food twice."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    _deal_one_of_each_food(player)
    picked: list[cards.Food] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.SpendFoodDecision)
        choice = decision.choices[0]
        assert isinstance(choice, decisions.FoodChoice)
        picked.append(choice.food)
        return typing.cast(C, choice)

    eng = engine.Engine(gs)
    setup_flow.resolve_deferred_setup_food(eng, player, agent, 2, defer_food=True)

    assert len(picked) == 2
    assert len(set(picked)) == 2  # no repeats
    assert player.food.total() == 3  # 5 dealt - 2 spent


def test_resolve_deferred_setup_food_gain_path_clears_then_gains() -> None:
    """n_kept=3: the food pool is cleared first, then two sequential
    ``GainFoodDecision`` asks that never offer (or pick) the same food twice."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    _deal_one_of_each_food(player)
    picked: list[cards.Food] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.GainFoodDecision)
        choice = decision.choices[0]
        assert isinstance(choice, decisions.FoodChoice)
        picked.append(choice.food)
        return typing.cast(C, choice)

    eng = engine.Engine(gs)
    setup_flow.resolve_deferred_setup_food(eng, player, agent, 3, defer_food=True)

    assert len(picked) == 2
    assert len(set(picked)) == 2  # no repeats
    # The pre-deal food was cleared before the two gains landed, so the final
    # total is exactly the two gains, not 5 (dealt) + 2 (gained).
    assert player.food.total() == 2


def test_resolve_deferred_setup_food_noop_when_not_deferred() -> None:
    """``defer_food=False`` is a no-op regardless of ``n_kept`` — nothing is
    asked and the dealt food pool is left untouched."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    _deal_one_of_each_food(player)
    seen, agent = _recording_agent()

    eng = engine.Engine(gs)
    setup_flow.resolve_deferred_setup_food(eng, player, agent, 2, defer_food=False)

    assert seen == []
    assert player.food.total() == len(cards.ALL_FOODS)


# ---------------------------------------------------------------------------
# combine_gain_food — collapses to one combined subset decision


def test_resolve_deferred_setup_food_combine_gain_food_single_decision() -> None:
    """Under ``combine_gain_food``, any deferred ``n_kept`` (not only the
    high-keep gain path) resolves via ONE combined ``FoodSubsetChoice``
    decision instead of sequential Spend/Gain asks."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    _deal_one_of_each_food(player)
    seen, agent = _recording_agent()

    eng = engine.Engine(gs, combine_gain_food=True)
    setup_flow.resolve_deferred_setup_food(eng, player, agent, 3, defer_food=True)

    assert len(seen) == 1
    decision = seen[0]
    assert isinstance(decision, decisions.GainFoodDecision)
    assert all(
        isinstance(choice, decisions.FoodSubsetChoice) for choice in decision.choices
    )
    # n_keep = len(ALL_FOODS) - n_kept = 5 - 3 = 2.
    assert player.food.total() == 2


# ---------------------------------------------------------------------------
# resolve_deferred_setup_bonus — return value across its three paths


def test_resolve_deferred_setup_bonus_returns_choice_when_not_deferred() -> None:
    """When ``sc.bonus_card`` is already set (the combined, non-deferred
    setup choice), the function returns it directly without asking anyone."""
    _, bonuses, _ = cards.load_all()
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    dealt_bonus = bonuses[:2]
    sc = decisions.SetupChoice(kept_cards=(), kept_foods=(), bonus_card=dealt_bonus[0])

    eng = engine.Engine(gs)
    result = setup_flow.resolve_deferred_setup_bonus(eng, player, dealt_bonus, sc)

    assert result is dealt_bonus[0]


def test_resolve_deferred_setup_bonus_returns_none_when_no_bonus_dealt() -> None:
    """When no bonus cards were dealt at all, the function returns ``None``."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    sc = decisions.SetupChoice(kept_cards=(), kept_foods=(), bonus_card=None)

    eng = engine.Engine(gs)
    result = setup_flow.resolve_deferred_setup_bonus(eng, player, [], sc)

    assert result is None


def test_resolve_deferred_setup_bonus_returns_agent_pick_when_deferred() -> None:
    """When the setup keep deferred the bonus pick (``bonus_card is None``
    while bonus cards were dealt), the function asks the in-game
    ``BirdPowerPickBonusCardDecision`` and returns whatever the agent picked."""
    _, bonuses, _ = cards.load_all()
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = player.id
    dealt_bonus = bonuses[:2]
    sc = decisions.SetupChoice(kept_cards=(), kept_foods=(), bonus_card=None)

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.BirdPowerPickBonusCardDecision)
        return typing.cast(C, decision.choices[1])

    eng = engine.Engine(gs, agents=[agent, agent])
    result = setup_flow.resolve_deferred_setup_bonus(eng, player, dealt_bonus, sc)

    assert result is dealt_bonus[1]
    assert dealt_bonus[1] in player.bonus_cards
    assert dealt_bonus[0] in gs.bonus_discard
