"""Tests for the CACHE_FOOD power now that cached food is tracked per food type.

``PlayedBird.cached_food`` was a scalar count; it is now a ``FoodPool`` so the
encoder can see *which* foods a bird has cached. These guard that the cache
handler records the food type, and that ``Player.total_cached`` still sums
correctly across types and birds (so scoring is unchanged)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def _no_agent[C: decisions.Choice](
    _engine: engine.Engine,
    _decision: decisions.Decision[C],
) -> C:
    """An ``Agent``-typed stub; caching resolves without consulting an agent."""
    raise AssertionError(
        f"agent should not be consulted (got {type(_decision).__name__})"
    )


def _engine(seed: int = 0) -> tuple[engine.Engine, list[cards.Bird]]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    game_state = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(game_state, agents=[_no_agent, _no_agent]), birds


def _cache(
    eng: engine.Engine,
    player: state.Player,
    pb: state.PlayedBird,
    food: cards.Food,
    amount: int,
) -> None:
    effect = cards.Effect(kind=cards.EffectKind.CACHE_FOOD, food=food, amount=amount)
    powers.apply_effect(
        eng, _no_agent, player, pb, cards.Habitat.FOREST, effect, "activate"
    )


def test_cache_food_records_the_food_type():
    """Caching a specific food bumps that food's slot on the bird's pool, leaving
    the other foods at zero — the per-type tracking the encoder relies on."""
    eng, birds = _engine()
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=birds[0])
    _cache(eng, player, pb, cards.Food.SEED, 2)
    assert pb.cached_food[cards.Food.SEED] == 2
    assert pb.cached_food[cards.Food.FISH] == 0
    assert pb.cached_food.total() == 2


def test_total_cached_sums_across_types_and_birds():
    """``Player.total_cached`` (which feeds final scoring) sums every cached food
    across every bird, regardless of type."""
    eng, birds = _engine()
    player = eng.state.players[0]
    pb_forest = state.PlayedBird(bird=birds[0])
    pb_wetland = state.PlayedBird(bird=birds[1])
    player.board[cards.Habitat.FOREST] = [pb_forest]
    player.board[cards.Habitat.WETLAND] = [pb_wetland]

    _cache(eng, player, pb_forest, cards.Food.SEED, 2)
    _cache(eng, player, pb_forest, cards.Food.FISH, 1)
    _cache(eng, player, pb_wetland, cards.Food.SEED, 2)

    assert pb_forest.cached_food.total() == 3
    assert pb_wetland.cached_food[cards.Food.SEED] == 2
    assert player.total_cached == 5
