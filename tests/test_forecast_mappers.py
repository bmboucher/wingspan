"""Tests for the v1.6 fixes to ``engine.forecast.effect_exchange_ledger`` — the
one table that maps a bird power onto exchange-ledger fields.

Seven mapper bugs were fixed (the tuck-from-hand kinds never gained the tuck;
``TUCK_FROM_DECK_PAID`` paid an egg instead of the food; ``LAY_EGG_ALL_NEST``
credited the opponent; ``FEWEST_WETLAND_DRAWS_CARD`` gained food;
``PREDATOR_HUNT`` was unmapped; ``PINK_PLAY_BIRD_TUCK`` paid no card;
``EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER`` used a literal 1). These tests pin the
new ledger per real card, the legacy (era-<=1.5) mapping behind
``legacy_v1_5_mappers``, the agreement of the tuck gate's yes row with the
forecast, and that the fix reaches the budget-gated habitat forecast.
"""

from __future__ import annotations

import random

import pytest

from wingspan import cards, decisions, state
from wingspan.engine import forecast, tuck_gate

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BIRD_BY_NAME = {bird.name: bird for bird in _BIRDS}


def _effect(bird_name: str, kind: cards.EffectKind) -> cards.Effect:
    """``bird_name``'s effect of ``kind``."""
    return next(
        effect
        for effect in _BIRD_BY_NAME[bird_name].power.effects
        if effect.kind == kind
    )


def _nonzero(ledger: decisions.ExchangeLedger) -> dict[str, int]:
    """The ledger's nonzero count fields (``paid_food`` / ``gained_food`` are
    types, not counts, and ``effect_exchange_ledger`` never sets them)."""
    return {
        name: value
        for name, value in ledger.model_dump().items()
        if isinstance(value, int) and value
    }


def _live(effect: cards.Effect) -> dict[str, int]:
    return _nonzero(
        forecast.effect_exchange_ledger(
            effect, include_eot_discard=True, legacy_v1_5_mappers=False
        )
    )


def _legacy(effect: cards.Effect) -> dict[str, int]:
    return _nonzero(
        forecast.effect_exchange_ledger(
            effect, include_eot_discard=True, legacy_v1_5_mappers=True
        )
    )


# (bird, effect kind, live ledger, legacy ledger) — one row per fixed kind, the
# legacy column being the explicit pre-v1.6 mapping.
_CASES: list[tuple[str, cards.EffectKind, dict[str, int], dict[str, int]]] = [
    (
        "Cedar Waxwing",
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY,
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_food_count": 1},
        {"paid_card_count": 1, "gained_food_count": 1},
    ),
    (
        "Pygmy Nuthatch",
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE,
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_food_count": 1},
        {"paid_card_count": 1},  # parsed amount is 0
    ),
    (
        "American Coot",
        cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW,
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_card_count": 1},
        {"paid_card_count": 1, "gained_card_count": 1},
    ),
    (
        "Brewer's Blackbird",
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS,
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_egg_count": 1},
        {"paid_card_count": 1, "gained_egg_count": 1},
    ),
    (
        "White-Throated Swift",
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY,
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_egg_count": 1},
        {"paid_card_count": 1, "gained_egg_count": 1},
    ),
    (
        "Sandhill Crane",
        cards.EffectKind.TUCK_FROM_DECK_PAID,
        {"paid_food_count": 1, "gained_tuck_count": 2},
        {"paid_egg_count": 1, "gained_tuck_count": 2},
    ),
    (
        "Say's Phoebe",
        cards.EffectKind.LAY_EGG_ALL_NEST,
        {"gained_egg_count": 1},
        {"gained_egg_count": 1, "opp_gained_egg_count": 1},
    ),
    (
        "American Bittern",
        cards.EffectKind.FEWEST_WETLAND_DRAWS_CARD,
        {"gained_card_count": 1, "opp_gained_card_count": 1},
        {},  # gained food * parsed amount 0
    ),
    (
        "American Oystercatcher",
        cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
        {"gained_card_count": 2, "opp_gained_card_count": 1},
        {},  # parsed amount 0
    ),
    (
        "American Avocet",
        cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        {"gained_egg_count": 1},
        {},  # parsed amount 0
    ),
    (
        "Hermit Thrush",
        cards.EffectKind.FEWEST_FOREST_GAINS_DIE,
        {"gained_food_count": 1, "opp_gained_food_count": 1},
        {},  # gained food * parsed amount 0
    ),
    (
        "Barred Owl",
        cards.EffectKind.PREDATOR_HUNT,
        {"gained_tuck_count": 1},
        {},
    ),
    (
        "Horned Lark",
        cards.EffectKind.PINK_PLAY_BIRD_TUCK,
        {"paid_card_count": 1, "gained_tuck_count": 1},
        {"gained_tuck_count": 1},
    ),
    (
        "Anna's Hummingbird",
        cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER,
        {"gained_food_count": 1, "opp_gained_food_count": 1},
        {"gained_food_count": 1, "opp_gained_food_count": 1},
    ),
]


@pytest.mark.parametrize(
    ("bird_name", "kind", "live", "_legacy_expected"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_live_mapping_per_real_card(
    bird_name: str,
    kind: cards.EffectKind,
    live: dict[str, int],
    _legacy_expected: dict[str, int],
) -> None:
    assert _live(_effect(bird_name, kind)) == live


@pytest.mark.parametrize(
    ("bird_name", "kind", "_live_expected", "legacy"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_legacy_mapping_reproduces_the_pre_v1_6_values(
    bird_name: str,
    kind: cards.EffectKind,
    _live_expected: dict[str, int],
    legacy: dict[str, int],
) -> None:
    assert _legacy(_effect(bird_name, kind)) == legacy


def test_plain_tuck_from_hand_pays_a_card_and_gains_the_tuck() -> None:
    effect = cards.Effect(kind=cards.EffectKind.TUCK_FROM_HAND, amount=1)
    assert _live(effect) == {"paid_card_count": 1, "gained_tuck_count": 1}
    assert _legacy(effect) == {"paid_card_count": 1}


def test_each_player_gains_die_uses_the_effect_amount() -> None:
    effect = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=2
    )
    assert _live(effect) == {"gained_food_count": 2, "opp_gained_food_count": 2}
    assert _legacy(effect) == {"gained_food_count": 1, "opp_gained_food_count": 1}


def test_unaffected_kinds_map_identically_in_both_eras() -> None:
    """The legacy flag changes only the fourteen fixed kinds."""
    fixed = {kind for _, kind, _, _ in _CASES} | {
        cards.EffectKind.TUCK_FROM_HAND,
    }
    for bird in _BIRDS:
        for effect in bird.power.effects:
            if effect.kind in fixed:
                continue
            assert _live(effect) == _legacy(effect), (bird.name, effect.kind)


# ---------------------------------------------------------------------------
# Agreement with the tuck gate's yes row


_TUCK_GATE_BIRDS = [
    ("Cedar Waxwing", cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY),
    ("Pygmy Nuthatch", cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE),
    ("American Coot", cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW),
    ("Brewer's Blackbird", cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS),
    ("White-Throated Swift", cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY),
    ("Horned Lark", cards.EffectKind.PINK_PLAY_BIRD_TUCK),
]

_SHARED_FIELDS = (
    "paid_card_count",
    "gained_tuck_count",
    "gained_card_count",
    "gained_egg_count",
    "gained_food_count",
)


@pytest.mark.parametrize(
    ("bird_name", "kind"), _TUCK_GATE_BIRDS, ids=[name for name, _ in _TUCK_GATE_BIRDS]
)
def test_tuck_gate_row_agrees_with_the_forecast_ledger(
    bird_name: str, kind: cards.EffectKind
) -> None:
    """The gate's yes row and the forecast ledger carry the same shared fields
    for every tuck kind (the gate builds from the forecast mapping)."""
    game_state = state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)
    player = game_state.players[0]
    played = state.PlayedBird(bird=_BIRD_BY_NAME[bird_name])
    player.board[cards.Habitat.FOREST] = [played]
    effect = _effect(bird_name, kind)

    gate_row = tuck_gate.tuck_activate_choice(player, played, effect)
    ledger = forecast.effect_exchange_ledger(
        effect, include_eot_discard=True, legacy_v1_5_mappers=False
    )
    for field in _SHARED_FIELDS:
        assert getattr(gate_row, field) == getattr(ledger, field), field
    assert gate_row.gained_tuck_count == 1
    assert gate_row.paid_card_count == 1


def test_plain_tuck_gate_row_agrees_with_the_forecast_ledger() -> None:
    game_state = state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)
    player = game_state.players[0]
    played = state.PlayedBird(bird=_BIRD_BY_NAME["Cedar Waxwing"])
    effect = cards.Effect(kind=cards.EffectKind.TUCK_FROM_HAND, amount=1)
    gate_row = tuck_gate.tuck_activate_choice(player, played, effect)
    ledger = forecast.effect_exchange_ledger(
        effect, include_eot_discard=True, legacy_v1_5_mappers=False
    )
    for field in _SHARED_FIELDS:
        assert getattr(gate_row, field) == getattr(ledger, field), field


# ---------------------------------------------------------------------------
# The fix reaches the budget-gated habitat forecast


def _crane_game(food: int) -> tuple[state.Player, state.GameState]:
    """Player 0 with Sandhill Crane in the wetland row, ``food`` seed in
    supply, and no eggs anywhere (so the old egg-paying mapper cannot fire)."""
    game_state = state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)
    player = game_state.players[0]
    player.hand = []
    player.food = state.FoodPool.from_dict({cards.Food.SEED: food})
    player.board[cards.Habitat.WETLAND] = [
        state.PlayedBird(bird=_BIRD_BY_NAME["Sandhill Crane"])
    ]
    return player, game_state


def test_paid_tuck_row_power_is_gated_on_food_not_eggs() -> None:
    player, game_state = _crane_game(food=1)
    live = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.DRAW_CARDS
    )
    assert (live.paid_food_count, live.paid_egg_count, live.gained_tuck_count) == (
        1,
        0,
        2,
    )

    legacy = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.DRAW_CARDS, legacy_v1_5_mappers=True
    )
    # The old mapper charged an egg the player does not have: the power was skipped.
    assert (
        legacy.paid_food_count,
        legacy.paid_egg_count,
        legacy.gained_tuck_count,
    ) == (
        0,
        0,
        0,
    )


def test_paid_tuck_row_power_is_skipped_without_food() -> None:
    player, game_state = _crane_game(food=0)
    live = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.DRAW_CARDS
    )
    assert live.gained_tuck_count == 0
