"""Coverage tests for round-goal scoring.

Every bundled core goal must resolve to a real scoring category — a goal that
maps to an ``unknown:`` tag silently scores 0 for both players, which distorts
the round-goal portion of the training reward. This pins that, and unit-tests
the two goals (total birds, sets of eggs across the three habitats) that were
previously unscored.
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wingspan import cards, state  # noqa: E402
from wingspan.engine import scoring  # noqa: E402


def _bird(
    name: str, habitats: tuple[cards.Habitat, ...] = (cards.Habitat.FOREST,)
) -> cards.Bird:
    return cards.Bird(
        id=abs(hash(name)) % 100000,
        name=name,
        scientific_name="Testus birdus",
        color=cards.PowerColor.NONE,
        points=1,
        nest=cards.NestType.BOWL,
        egg_limit=5,
        wingspan_cm=50,
        habitats=habitats,
        food_cost=cards.BirdCost(),
        flocking=False,
        predator=False,
        is_swift_start=False,
        raw_power_text="",
        power=cards.Power(color=cards.PowerColor.NONE),
        bonus_categories=(),
    )


def _player_with(
    entries: list[tuple[cards.Bird, int, cards.Habitat]],
) -> state.Player:
    player = state.Player(id=0, name="P0")
    for bird, eggs, habitat in entries:
        player.board[habitat].append(state.PlayedBird(bird=bird, eggs=eggs))
    return player


def _goal(category: str) -> cards.EndRoundGoal:
    return cards.EndRoundGoal(id=1, description=category, category=category, tile_id=0)


def test_new_game_goals_come_from_distinct_tiles() -> None:
    """The four round goals in any game must come from four different tiles."""
    birds, bonuses, goals = cards.load_all()
    core_goals = [goal for goal in goals if goal.category not in ("unknown",)]
    for seed in range(50):
        game_state = state.new_game(random.Random(seed), birds, bonuses, core_goals)
        tile_ids = [goal.tile_id for goal in game_state.round_goals]
        assert (
            len(set(tile_ids)) == 4
        ), f"seed {seed}: round goals share a tile — tile_ids {tile_ids}"


def test_no_core_goal_is_unknown() -> None:
    """No bundled core round goal may map to an ``unknown:`` (zero-scoring) tag."""
    _, _, goals = cards.load_all()
    unknown = [goal.category for goal in goals if goal.category.startswith("unknown")]
    assert unknown == [], f"unscored round goals: {unknown}"


def test_total_birds_counts_all_habitats() -> None:
    player = _player_with(
        [
            (_bird("A"), 0, cards.Habitat.FOREST),
            (_bird("B"), 0, cards.Habitat.GRASSLAND),
            (_bird("C"), 0, cards.Habitat.WETLAND),
            (_bird("D"), 0, cards.Habitat.WETLAND),
        ]
    )
    assert scoring.eval_goal(player, _goal("total_birds")) == 4


def test_egg_sets_three_habitats_is_min_across_habitats() -> None:
    # 1 egg in forest, 3 in grassland, 2 in wetland -> 1 complete set (the min).
    player = _player_with(
        [
            (_bird("F", (cards.Habitat.FOREST,)), 1, cards.Habitat.FOREST),
            (_bird("G", (cards.Habitat.GRASSLAND,)), 3, cards.Habitat.GRASSLAND),
            (_bird("W", (cards.Habitat.WETLAND,)), 2, cards.Habitat.WETLAND),
        ]
    )
    assert scoring.eval_goal(player, _goal("egg_sets_3habitats")) == 1


def test_egg_sets_is_zero_when_a_habitat_has_no_eggs() -> None:
    player = _player_with(
        [
            (_bird("F", (cards.Habitat.FOREST,)), 5, cards.Habitat.FOREST),
            (_bird("G", (cards.Habitat.GRASSLAND,)), 5, cards.Habitat.GRASSLAND),
        ]
    )
    assert scoring.eval_goal(player, _goal("egg_sets_3habitats")) == 0
