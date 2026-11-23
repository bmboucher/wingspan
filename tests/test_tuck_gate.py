"""Tests for the tuck-gate yes-row builder (``wingspan.engine.tuck_gate``).

Every ``ActivateTuckDecision`` yes row is a ``TuckActivateChoice`` carrying the
exchange ledger, filled optimistically per effect kind. These tests pin the
exact ledger values for each of the seven gate kinds, and that the engine
sites actually offer them.
"""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers, tuck_gate

_HABITAT = cards.Habitat.FOREST


def _player_and_bird(egg_limit: int = 3) -> tuple[state.Player, state.PlayedBird]:
    """A fresh game's player 0 with one real bird on the forest row."""
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(0), birds, bonuses, goals)
    player = game_state.players[0]
    bird = next(candidate for candidate in birds if candidate.egg_limit == egg_limit)
    played = state.PlayedBird(bird=bird)
    player.board[_HABITAT] = [played]
    return player, played


def _gate(
    kind: cards.EffectKind,
    player: state.Player,
    played: state.PlayedBird,
    *,
    amount: int = 1,
    food: cards.Food | None = None,
) -> decisions.TuckActivateChoice:
    effect = cards.Effect(kind=kind, amount=amount, food=food)
    return tuck_gate.tuck_activate_choice(player, played, effect)


def _ledger_fields(choice: decisions.TuckActivateChoice) -> dict[str, typing.Any]:
    """Every ledger field that differs from the all-default ledger."""
    default = decisions.ExchangeLedger()
    return {
        name: getattr(choice, name)
        for name in decisions.ExchangeLedger.model_fields
        if getattr(choice, name) != getattr(default, name)
    }


def test_plain_tuck_carries_only_the_card_and_the_tuck():
    player, played = _player_and_bird()
    choice = _gate(cards.EffectKind.TUCK_FROM_HAND, player, played)
    assert _ledger_fields(choice) == {"paid_card_count": 1, "gained_tuck_count": 1}
    assert choice.label == tuck_gate.TUCK_GATE_LABEL


def test_horned_lark_pink_tuck_is_plain():
    player, played = _player_and_bird()
    choice = _gate(cards.EffectKind.PINK_PLAY_BIRD_TUCK, player, played)
    assert _ledger_fields(choice) == {"paid_card_count": 1, "gained_tuck_count": 1}


def test_then_draw_gains_eff_amount_cards():
    player, played = _player_and_bird()
    choice = _gate(cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW, player, played, amount=2)
    assert _ledger_fields(choice) == {
        "paid_card_count": 1,
        "gained_tuck_count": 1,
        "gained_card_count": 2,
    }


def test_then_lay_on_this_gains_an_egg_only_with_room():
    player, played = _player_and_bird()
    roomy = _gate(cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS, player, played)
    assert roomy.gained_egg_count == 1

    played.eggs = played.bird.egg_limit
    full = _gate(cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS, player, played)
    assert full.gained_egg_count == 0
    assert (full.paid_card_count, full.gained_tuck_count) == (1, 1)


def test_then_lay_any_is_capped_by_board_egg_room():
    player, played = _player_and_bird()
    room = played.bird.egg_limit
    kind = cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY

    assert _gate(kind, player, played, amount=1).gained_egg_count == 1
    assert _gate(kind, player, played, amount=room + 5).gained_egg_count == room

    played.eggs = played.bird.egg_limit - 1
    assert _gate(kind, player, played, amount=3).gained_egg_count == 1

    played.eggs = played.bird.egg_limit
    assert _gate(kind, player, played, amount=3).gained_egg_count == 0


def test_then_gain_food_supply_names_the_food_and_count():
    player, played = _player_and_bird()
    choice = _gate(
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY,
        player,
        played,
        amount=2,
        food=cards.Food.FRUIT,
    )
    assert _ledger_fields(choice) == {
        "paid_card_count": 1,
        "gained_tuck_count": 1,
        "gained_food_count": 2,
        "gained_food": cards.Food.FRUIT,
    }


def test_then_gain_food_choice_gains_one_untyped_food():
    player, played = _player_and_bird()
    choice = _gate(
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE, player, played, amount=0
    )
    assert choice.gained_food_count == 1
    assert choice.gained_food is None


# ---------------------------------------------------------------------------
# The engine sites offer these rows


def _gate_row_for(bird_name: str) -> decisions.TuckActivateChoice:
    """Activate ``bird_name`` (on player 0's forest row, one card in hand) and
    return the yes row of the tuck gate it offers (the agent then skips)."""
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(0), birds, bonuses, goals)
    offered: list[decisions.TuckActivateChoice] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C], /
    ) -> C:
        assert isinstance(decision, decisions.ActivateTuckDecision)
        yes_row = next(
            choice
            for choice in decision.choices
            if isinstance(choice, decisions.TuckActivateChoice)
        )
        offered.append(yes_row)
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.SkipChoice)
            ),
        )

    eng = engine.Engine(game_state, agents=[agent, agent])
    player = eng.state.players[0]
    bird = next(candidate for candidate in birds if candidate.name == bird_name)
    played = state.PlayedBird(bird=bird)
    player.board[_HABITAT] = [played]
    player.hand = [candidate for candidate in birds if candidate.name != bird_name][:2]
    powers.dispatch_power(eng, agent, player, played, _HABITAT, "activate")
    assert len(offered) == 1
    return offered[0]


def test_cedar_waxwing_gate_advertises_the_fruit():
    row = _gate_row_for("Cedar Waxwing")
    assert row.paid_card_count == 1
    assert row.gained_tuck_count == 1
    assert row.gained_food_count == 1
    assert row.gained_food == cards.Food.FRUIT
