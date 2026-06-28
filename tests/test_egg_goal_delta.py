# pyright: reportPrivateUsage=false
# (reads the layout's package-private stripe constants to slice choice rows)
"""Tests for egg-event consequence pricing on lay / remove board-target rows.

Every ``LayEggDecision`` / ``RemoveEggDecision`` candidate row prices the
targeted slot's exact effect on the unscored round goals (``goal_delta``) and
on the held egg-counting dynamic bonus cards (``bonus_delta``): per-habitat
and per-nest egg totals, the has-eggs threshold crossings behind the
``*_birds_with_eggs`` goals, the egg-set minimum, star nests counting as every
nest, and the Oologist / Breeding Manager egg thresholds. Black-box through
``encode.encode_choices``.
"""

from __future__ import annotations

import math
import random

import numpy as np

from wingspan import cards, decisions, encode, engine, state  # noqa: E402
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


def _bird_with_nest(nest: cards.NestType) -> cards.Bird:
    return next(bird for bird in _BIRDS if bird.nest == nest and bird.egg_limit >= 2)


def _target(habitat: cards.Habitat, slot: int) -> decisions.BoardTargetChoice:
    return decisions.BoardTargetChoice(
        label=f"{habitat.value}[{slot}]", habitat=habitat, slot=slot
    )


def _lay_rows(
    game_state: state.GameState, targets: list[decisions.BoardTargetChoice]
) -> np.ndarray:
    choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = list(targets)
    decision = decisions.LayEggDecision(player_id=0, prompt="lay", choices=choices)
    return encode.encode_choices(decision, game_state)


def _remove_rows(
    game_state: state.GameState, targets: list[decisions.BoardTargetChoice]
) -> np.ndarray:
    choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = list(targets)
    decision = decisions.RemoveEggDecision(player_id=0, prompt="pay", choices=choices)
    return encode.encode_choices(decision, game_state)


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


def test_eggs_in_habitat_goal_prices_matching_target_only():
    """An eggs_forest goal moves only when the egg lands in the forest row —
    and the marginal VP is the full first-place payout from a standing start."""
    game_state = _game_with_goals(["eggs_forest"] * 4)
    player = game_state.players[0]
    any_bird = _BIRDS[0]
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=any_bird))
    player.board[cards.Habitat.GRASSLAND].append(state.PlayedBird(bird=any_bird))

    rows = _lay_rows(
        game_state,
        [_target(cards.Habitat.FOREST, 0), _target(cards.Habitat.GRASSLAND, 0)],
    )
    count, vp = _goal_delta_slot(rows[0], 0)
    assert count == _Approx(1 / 5)
    assert vp == _Approx(4 / 10)  # 0 -> 1 vs opp 0: round 1 first place (4)
    assert _goal_delta_slot(rows[1], 0) == (0.0, 0.0)


def test_removal_prices_negative_deltas():
    game_state = _game_with_goals(["eggs_forest"] * 4)
    player = game_state.players[0]
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=_BIRDS[0], eggs=1))
    rows = _remove_rows(game_state, [_target(cards.Habitat.FOREST, 0)])
    count, vp = _goal_delta_slot(rows[0], 0)
    assert count == _Approx(-1 / 5)
    assert vp == _Approx(-4 / 10)  # 1 -> 0 forfeits the uncontested 4 VP


def test_star_nest_counts_toward_nest_egg_goal():
    """Star nests are wild: laying on a star bird advances eggs_bowl, while a
    concrete non-bowl nest does not."""
    game_state = _game_with_goals(["eggs_bowl"] * 4)
    player = game_state.players[0]
    star_bird = _BIRDS[0].model_copy(update={"nest": cards.NestType.STAR})
    ground_bird = _bird_with_nest(cards.NestType.GROUND)
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=star_bird))
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=ground_bird))

    rows = _lay_rows(
        game_state,
        [_target(cards.Habitat.FOREST, 0), _target(cards.Habitat.FOREST, 1)],
    )
    assert _goal_delta_slot(rows[0], 0)[0] == _Approx(1 / 5)
    assert _goal_delta_slot(rows[1], 0) == (0.0, 0.0)


def test_birds_with_eggs_goal_prices_only_threshold_crossings():
    """bowl_birds_with_eggs moves on the 0↔1 egg crossing only: laying on an
    empty bowl bird (+1), laying on one that already has eggs (0); removing
    the last egg (−1), removing one of two (0)."""
    game_state = _game_with_goals(["bowl_birds_with_eggs"] * 4)
    player = game_state.players[0]
    bowl_bird = _bird_with_nest(cards.NestType.BOWL)
    habitat = bowl_bird.habitats[0]
    player.board[habitat].append(state.PlayedBird(bird=bowl_bird, eggs=0))
    player.board[habitat].append(state.PlayedBird(bird=bowl_bird, eggs=2))

    lay_rows = _lay_rows(game_state, [_target(habitat, 0), _target(habitat, 1)])
    assert _goal_delta_slot(lay_rows[0], 0)[0] == _Approx(1 / 5)
    assert _goal_delta_slot(lay_rows[1], 0) == (0.0, 0.0)

    player.board[habitat][0].eggs = 1
    remove_rows = _remove_rows(game_state, [_target(habitat, 0), _target(habitat, 1)])
    assert _goal_delta_slot(remove_rows[0], 0)[0] == _Approx(-1 / 5)
    assert _goal_delta_slot(remove_rows[1], 0) == (0.0, 0.0)


def test_egg_sets_goal_prices_the_minimum_habitat():
    """The user's example: egg_sets_3habitats moves only when the egg lands in
    (or leaves) the habitat at the minimum egg count."""
    game_state = _game_with_goals(["egg_sets_3habitats"] * 4)
    player = game_state.players[0]
    any_bird = next(bird for bird in _BIRDS if bird.egg_limit >= 3)
    for habitat, eggs in zip(cards.ALL_HABITATS, (0, 1, 1)):
        player.board[habitat].append(state.PlayedBird(bird=any_bird, eggs=eggs))

    # Laying into the unique minimum (forest, 0 eggs) completes a set; laying
    # into grassland leaves the minimum at 0.
    rows = _lay_rows(
        game_state,
        [_target(cards.Habitat.FOREST, 0), _target(cards.Habitat.GRASSLAND, 0)],
    )
    count, vp = _goal_delta_slot(rows[0], 0)
    assert count == _Approx(1 / 5)
    assert vp == _Approx(4 / 10)
    assert _goal_delta_slot(rows[1], 0) == (0.0, 0.0)

    # With a tie at the minimum, one egg cannot raise it.
    player.board[cards.Habitat.FOREST][0].eggs = 1
    player.board[cards.Habitat.WETLAND][0].eggs = 2
    tie_rows = _lay_rows(game_state, [_target(cards.Habitat.FOREST, 0)])
    assert _goal_delta_slot(tie_rows[0], 0) == (0.0, 0.0)

    # Removing from a habitat at the minimum breaks a set; removing from one
    # above it does not.
    remove_rows = _remove_rows(
        game_state,
        [_target(cards.Habitat.FOREST, 0), _target(cards.Habitat.WETLAND, 0)],
    )
    assert _goal_delta_slot(remove_rows[0], 0)[0] == _Approx(-1 / 5)
    assert _goal_delta_slot(remove_rows[1], 0) == (0.0, 0.0)


def test_birds_no_eggs_goal_prices_inverted_crossings():
    """birds_no_eggs moves opposite to the has-eggs crossing: laying on an
    eggless bird costs the goal a bird, removing a bird's last egg re-earns
    one, and targets that don't cross the threshold are silent."""
    game_state = _game_with_goals(["birds_no_eggs"] * 4)
    player = game_state.players[0]
    nester = next(bird for bird in _BIRDS if bird.egg_limit >= 2)
    habitat = nester.habitats[0]
    player.board[habitat].append(state.PlayedBird(bird=nester, eggs=0))
    player.board[habitat].append(state.PlayedBird(bird=nester, eggs=1))

    # Laying on the eggless bird forfeits its no-egg status (and the
    # uncontested first place); laying beside existing eggs is free.
    lay_rows = _lay_rows(game_state, [_target(habitat, 0), _target(habitat, 1)])
    count, vp = _goal_delta_slot(lay_rows[0], 0)
    assert count == _Approx(-1 / 5)
    assert vp == _Approx(-4 / 10)  # 1 -> 0 forfeits the uncontested 4 VP
    assert _goal_delta_slot(lay_rows[1], 0) == (0.0, 0.0)

    # With no eggless birds left, removing a bird's *last* egg re-earns first
    # place; removing one of two is silent.
    player.board[habitat][0].eggs = 2
    remove_rows = _remove_rows(game_state, [_target(habitat, 0), _target(habitat, 1)])
    assert _goal_delta_slot(remove_rows[0], 0) == (0.0, 0.0)
    count, vp = _goal_delta_slot(remove_rows[1], 0)
    assert count == _Approx(1 / 5)
    assert vp == _Approx(4 / 10)  # 0 -> 1 vs opp 0 re-takes round 1 (4)


def test_egg_counting_bonus_thresholds_price_on_lay_and_remove():
    """Breeding Manager (≥ 4 eggs, 1 VP/bird): laying the fourth egg is +1 VP,
    removing back below the threshold is −1 VP, and a far-from-threshold
    target prices nothing. Bird-count goals keep the goal stripes quiet."""
    game_state = _game_with_goals(["birds_forest"] * 4)
    player = game_state.players[0]
    player.bonus_cards = [_BONUS_BY_NAME["Breeding Manager"]]
    big_nester = next(bird for bird in _BIRDS if bird.egg_limit >= 5)
    habitat = big_nester.habitats[0]
    player.board[habitat].append(state.PlayedBird(bird=big_nester, eggs=3))
    player.board[habitat].append(state.PlayedBird(bird=big_nester, eggs=1))

    lay_rows = _lay_rows(game_state, [_target(habitat, 0), _target(habitat, 1)])
    qual, stepped, linear = _bonus_delta(lay_rows[0])
    assert qual == _Approx(1 / 5)
    assert stepped == _Approx(1 / 7)  # per-bird card: +1 VP
    assert linear == _Approx(1 / 7)
    assert _bonus_delta(lay_rows[1]) == (0.0, 0.0, 0.0)

    player.board[habitat][0].eggs = 4
    remove_rows = _remove_rows(game_state, [_target(habitat, 0)])
    qual, stepped, linear = _bonus_delta(remove_rows[0])
    assert qual == _Approx(1 / 5)
    assert stepped == _Approx(-1 / 7)
    assert linear == _Approx(-1 / 7)


def test_lay_rows_skip_scored_rounds():
    game_state = _game_with_goals(["eggs_forest"] * 4)
    player = game_state.players[0]
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=_BIRDS[0]))
    eng = engine.Engine(game_state)
    scoring.score_round_goal(eng, 0)

    rows = _lay_rows(game_state, [_target(cards.Habitat.FOREST, 0)])
    assert _goal_delta_slot(rows[0], 0) == (0.0, 0.0)
    count, vp = _goal_delta_slot(rows[0], 1)
    assert count == _Approx(1 / 5)
    assert vp == _Approx(5 / 10)  # round 2 first place pays 5
