"""Tests for the four dynamic bonus cards' live-state scoring.

Breeding Manager (birds with ≥ 4 eggs), Oologist (birds with ≥ 1 egg, tiered),
Visionary Leader (cards in hand at game end, tiered), and Ecologist (birds in
the habitat with the fewest birds) tag no birds in ``bonus_categories`` —
their qualifying counts come from live game state through the dynamic-counter
dispatch in ``scoring.bonus_qualifying_count``. These tests pin each card's
count and payout, including the tier boundaries and the Ecologist tie rule
its printed note spells out ("if all of your habitats have 3 birds in them,
your habitat with the fewest birds has 3 birds in it").
"""

from __future__ import annotations

import random

from wingspan import cards, engine, state  # noqa: E402
from wingspan.engine import scoring  # noqa: E402

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}


def _fresh_game() -> state.GameState:
    return state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)


def _place_birds_with_eggs(player: state.Player, egg_counts: list[int]) -> None:
    """Spread one bird per entry across the habitats, carrying the given eggs."""
    any_bird = _BIRDS[0]
    for i, eggs in enumerate(egg_counts):
        habitat = cards.ALL_HABITATS[i % len(cards.ALL_HABITATS)]
        player.board[habitat].append(state.PlayedBird(bird=any_bird, eggs=eggs))


def test_breeding_manager_counts_birds_with_four_or_more_eggs():
    breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
    player = _fresh_game().players[0]
    _place_birds_with_eggs(player, [4, 3, 5, 0])
    assert scoring.bonus_qualifying_count(player, breeding_manager) == 2
    assert scoring.bonus_score(player, breeding_manager) == 2  # 1 VP per bird


def test_oologist_tiers_on_birds_with_any_egg():
    oologist = _BONUS_BY_NAME["Oologist"]
    for bird_count, expected_vp in ((6, 0), (7, 3), (8, 3), (9, 6)):
        player = _fresh_game().players[0]
        _place_birds_with_eggs(player, [1] * bird_count)
        assert scoring.bonus_qualifying_count(player, oologist) == bird_count
        assert scoring.bonus_score(player, oologist) == expected_vp


def test_visionary_leader_tiers_on_hand_size():
    visionary = _BONUS_BY_NAME["Visionary Leader"]
    for hand_size, expected_vp in ((4, 0), (5, 4), (7, 4), (8, 7)):
        player = _fresh_game().players[0]
        player.hand = list(_BIRDS[:hand_size])
        assert scoring.bonus_qualifying_count(player, visionary) == hand_size
        assert scoring.bonus_score(player, visionary) == expected_vp


def test_ecologist_counts_fewest_habitat_including_ties():
    ecologist = _BONUS_BY_NAME["Ecologist"]
    any_bird = _BIRDS[0]
    cases = [
        # (forest, grassland, wetland) bird counts -> qualifying count
        ((3, 3, 3), 3),  # the printed tie example: all tied at 3 -> count 3
        ((2, 3, 4), 2),
        ((0, 3, 4), 0),  # an empty habitat scores nothing
    ]
    for row_counts, expected_count in cases:
        player = _fresh_game().players[0]
        for habitat, count in zip(cards.ALL_HABITATS, row_counts):
            player.board[habitat] = [
                state.PlayedBird(bird=any_bird) for _ in range(count)
            ]
        assert scoring.bonus_qualifying_count(player, ecologist) == expected_count
        assert scoring.bonus_score(player, ecologist) == 2 * expected_count


def test_final_scoring_includes_dynamic_bonus():
    """A dynamic card's VP flows through final scoring like any other bonus."""
    game_state = _fresh_game()
    player = game_state.players[0]
    player.hand = list(_BIRDS[:8])  # Visionary Leader: 8+ cards -> 7 VP
    player.bonus_cards = [_BONUS_BY_NAME["Visionary Leader"]]
    eng = engine.Engine(game_state)
    scoring.final_scoring(eng)
    assert player.final_score == 7

    assert scoring.running_score(player) == 7


def test_static_bonus_counting_is_unchanged():
    """The dynamic dispatch must not disturb the static tag-based counting."""
    bird_counter = _BONUS_BY_NAME["Bird Counter"]
    tagged = [bird for bird in _BIRDS if bird_counter.name in bird.bonus_categories]
    player = _fresh_game().players[0]
    for bird in tagged[:2]:
        player.board[bird.habitats[0]].append(state.PlayedBird(bird=bird))
    assert scoring.bonus_qualifying_count(player, bird_counter) == 2
