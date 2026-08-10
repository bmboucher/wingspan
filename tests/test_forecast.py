"""Tests for ``wingspan.engine.forecast`` — the optimistic per-action exchange
forecast and the effect->ledger branch table it shares with the state
encoder's power-exchange stripe.
"""

from __future__ import annotations

import random

from wingspan import cards, decisions, state
from wingspan.engine import forecast, scoring

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BIRD_BY_NAME = {bird.name: bird for bird in _BIRDS}
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}


def _new_game_state() -> state.GameState:
    """A fresh 2-player game with an empty hand/board/food for player 0 —
    the forecast tests build up exactly the board/hand/egg state each
    scenario needs from this baseline."""
    return state.new_game(random.Random(0), _BIRDS, _BONUSES, _GOALS)


def test_worked_example_draw_cards_with_and_without_egg():
    """Must-hold worked example: a single wetland brown bird with
    ``DRAW_CARDS_THEN_DISCARD_EOT`` amount=2 (a real catalog bird — 5
    core-set birds match this text exactly; Common Yellowthroat is used
    here, so no synthetic bird is needed). With >=1 egg the forecast draws 4
    cards (1 base + 1 egg conversion + 2 power) and pays 1 egg; with 0 eggs
    the conversion is gated off and the total drops to 3. The end-of-turn
    discard side (``paid_card_count=1``) is unaffected either way — it never
    draws down the hand budget the power's own draw just grew."""
    game_state = _new_game_state()
    player = game_state.players[0]
    eot_bird = _BIRD_BY_NAME["Common Yellowthroat"]
    player.hand = [eot_bird, eot_bird, eot_bird]

    player.board[cards.Habitat.WETLAND] = [state.PlayedBird(bird=eot_bird, eggs=1)]
    with_egg = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.DRAW_CARDS
    )
    assert with_egg.gained_card_count == 4
    assert with_egg.paid_egg_count == 1
    assert with_egg.paid_card_count == 1

    player.board[cards.Habitat.WETLAND] = [state.PlayedBird(bird=eot_bird, eggs=0)]
    without_egg = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.DRAW_CARDS
    )
    assert without_egg.gained_card_count == 3
    assert without_egg.paid_egg_count == 0
    assert without_egg.paid_card_count == 1


def test_eot_draw_power_fires_from_an_empty_hand():
    """The EOT discard is not an up-front cost: a ``DRAW_CARDS_THEN_DISCARD_
    EOT`` power activated with an empty hand still draws — the discard
    resolves at end of turn, funded by the cards the power itself just drew.
    The bird sits in the forest row here (no base draw beforehand) so the hand
    budget really is 0 when the power is gated."""
    game_state = _new_game_state()
    player = game_state.players[0]
    eot_bird = _BIRD_BY_NAME["Common Yellowthroat"]
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=eot_bird)]
    assert not player.hand

    ledger = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert ledger.gained_card_count == 2  # the power's draw fired
    assert ledger.paid_card_count == 1  # the EOT discard is still charged


def test_gain_food_conversion_and_empty_row_base_case():
    """A forest row with a card in hand fires the trade-arrow conversion; an
    empty hand silences it; an empty row (even count) prices the base track
    gain only regardless of hand."""
    game_state = _new_game_state()
    player = game_state.players[0]
    no_power_bird = _BIRD_BY_NAME["Hooded Warbler"]  # forest, PowerColor.NONE
    assert no_power_bird.color == cards.PowerColor.NONE
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=no_power_bird)]

    player.hand = [no_power_bird]
    fires = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert fires.gained_food_count == state.GAIN_FOOD_TRACK[1] + 1
    assert fires.paid_card_count == 1

    player.hand = []
    silent = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert silent.gained_food_count == state.GAIN_FOOD_TRACK[1]
    assert silent.paid_card_count == 0

    player.board[cards.Habitat.FOREST] = []
    player.hand = [no_power_bird]  # even a nonzero hand can't convert on row 0
    empty_row = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert empty_row.gained_food_count == state.GAIN_FOOD_TRACK[0]
    assert empty_row.paid_card_count == 0


def test_lay_eggs_capacity_cap_and_conversion_gates():
    """The base lay is capped at real spare egg room; the conversion needs
    both food and room to fire, and is gated off when either is 0."""
    game_state = _new_game_state()
    player = game_state.players[0]

    # Empty grassland row (even -> no conversion offered regardless): the
    # base LAY_EGGS_TRACK[0]=2 lay is capped to the single spare slot.
    tight = _BIRD_BY_NAME["Baltimore Oriole"]  # egg_limit 2
    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=tight, eggs=tight.egg_limit - 1)
    ]
    capped = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.LAY_EGGS
    )
    assert capped.gained_egg_count == 1

    # One grassland bird (odd row -> conversion offered) but no food yet ->
    # gated off even with plenty of room.
    roomy = _BIRD_BY_NAME["Wild Turkey"]  # egg_limit 5, PowerColor.NONE
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=roomy)]
    player.board[cards.Habitat.GRASSLAND] = [state.PlayedBird(bird=roomy)]
    assert player.food.total() == 0
    no_food = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.LAY_EGGS
    )
    assert no_food.paid_food_count == 0

    # Food available, but every slot on the board is already full -> gated
    # off by room instead.
    player.food[cards.Food.FISH] = 1
    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=roomy, eggs=roomy.egg_limit)
    ]
    player.board[cards.Habitat.GRASSLAND] = [
        state.PlayedBird(bird=roomy, eggs=roomy.egg_limit)
    ]
    no_room = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.LAY_EGGS
    )
    assert no_room.gained_egg_count == 0
    assert no_room.paid_food_count == 0


def test_brown_tuck_power_gated_by_hand_budget_after_conversion():
    """A forest ``TUCK_FROM_HAND_THEN_DRAW`` power is skipped when the
    trade-arrow conversion already spent the only card in hand, and fires
    alongside the conversion when a second card remains."""
    game_state = _new_game_state()
    player = game_state.players[0]
    tuck_bird = _BIRD_BY_NAME["Yellow-Rumped Warbler"]
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=tuck_bird)]

    player.hand = [tuck_bird]
    skipped = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert skipped.paid_card_count == 1  # conversion only
    assert skipped.gained_card_count == 0  # power never fired

    player.hand = [tuck_bird, tuck_bird]
    fired = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert fired.paid_card_count == 2  # conversion + power's tuck
    assert fired.gained_card_count == 1  # power's draw


def test_brown_egg_layer_capped_by_remaining_room():
    """A brown ``LAY_EGG_ON_THIS`` power's self-gain is capped by whatever
    spare egg room the base lay left behind."""
    game_state = _new_game_state()
    player = game_state.players[0]
    layer = _BIRD_BY_NAME["Northern Bobwhite"]  # grassland, egg_limit 6

    # Board room exactly matches what the base lay consumes -> the power's
    # own +1 egg is capped to 0.
    player.board[cards.Habitat.GRASSLAND] = [
        state.PlayedBird(bird=layer, eggs=layer.egg_limit - state.LAY_EGGS_TRACK[1])
    ]
    exhausted = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.LAY_EGGS
    )
    assert exhausted.gained_egg_count == state.LAY_EGGS_TRACK[1]

    # A spare forest bird leaves room after the base lay -> the power adds
    # its full +1.
    roomy = _BIRD_BY_NAME["Wild Turkey"]
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=roomy)]
    with_room = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.LAY_EGGS
    )
    assert with_room.gained_egg_count == state.LAY_EGGS_TRACK[1] + 1


def test_all_players_gain_food_forecast_moves_opponent_slot():
    """A brown ``ALL_PLAYERS_GAIN_FOOD`` power on the forest row prices both
    the deciding player's own gain and the shared opponent gain."""
    game_state = _new_game_state()
    player = game_state.players[0]
    oriole = _BIRD_BY_NAME["Baltimore Oriole"]
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=oriole)]

    ledger = forecast.habitat_action_exchange_forecast(
        player, game_state, decisions.MainAction.GAIN_FOOD
    )
    assert ledger.gained_food_count == state.GAIN_FOOD_TRACK[1] + 1
    assert ledger.opp_gained_food_count == 1


def test_effect_exchange_ledger_parity_spot_checks():
    """Spot-check representative ``EffectKind``s across every branch group
    against the documented slot semantics (``layout._EXCHANGE_SLOT_FOR_LEDGER_
    FIELD``): a plain gain, a tuck-then-X compound, a paid deck tuck, an
    egg-cost draw, a wild-food trade, an extra bird play, a shared
    all-players effect, and the EOT-discard seam at both flag values."""

    def effect(kind: cards.EffectKind, amount: int = 0) -> cards.Effect:
        return cards.Effect(kind=kind, amount=amount)

    plain_gain = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.GAIN_FOOD_SUPPLY, 1), include_eot_discard=False
    )
    assert plain_gain.gained_food_count == 1

    tuck_then_draw = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW, 1), include_eot_discard=False
    )
    assert tuck_then_draw.paid_card_count == 1
    assert tuck_then_draw.gained_card_count == 1

    tuck_from_deck_paid = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.TUCK_FROM_DECK_PAID, 2), include_eot_discard=False
    )
    assert tuck_from_deck_paid.paid_egg_count == 1
    assert tuck_from_deck_paid.gained_tuck_count == 2

    discard_egg_for_cards = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.DISCARD_EGG_FOR_CARDS, 2), include_eot_discard=False
    )
    assert discard_egg_for_cards.paid_egg_count == 1
    assert discard_egg_for_cards.gained_card_count == 2

    wild_trade = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.TRADE_WILD_FOOD), include_eot_discard=False
    )
    assert wild_trade.paid_food_count == 1
    assert wild_trade.gained_food_count == 1

    extra_play = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.PLAY_ADDITIONAL_BIRD), include_eot_discard=False
    )
    assert extra_play.gained_play_count == 1

    shared_eggs = forecast.effect_exchange_ledger(
        effect(cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST, 1),
        include_eot_discard=False,
    )
    assert shared_eggs.gained_egg_count == 1
    assert shared_eggs.opp_gained_egg_count == 1

    eot_effect = effect(cards.EffectKind.DRAW_CARDS_THEN_DISCARD_EOT, 2)
    without_eot = forecast.effect_exchange_ledger(eot_effect, include_eot_discard=False)
    assert without_eot.gained_card_count == 2
    assert without_eot.paid_card_count == 0
    with_eot = forecast.effect_exchange_ledger(eot_effect, include_eot_discard=True)
    assert with_eot.gained_card_count == 2
    assert with_eot.paid_card_count == 1


def test_bonus_best_case_count_delta_for_eggs():
    """Greedy cheapest-first best case for the two dynamic egg-counting bonus
    cards; capacity-capped by construction (a bird whose ``egg_limit`` can't
    reach the threshold never enters the candidate list) and zero for every
    non-egg card."""
    game_state = _new_game_state()
    player = game_state.players[0]
    oologist = _BONUS_BY_NAME["Oologist"]
    breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
    roomy = next(bird for bird in _BIRDS if bird.egg_limit >= 4)

    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=roomy),
        state.PlayedBird(bird=roomy),
    ]
    assert scoring.bonus_best_case_count_delta_for_eggs(oologist, player, 2) == 2

    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=roomy, eggs=1)]
    assert (
        scoring.bonus_best_case_count_delta_for_eggs(breeding_manager, player, 2) == 0
    )
    assert (
        scoring.bonus_best_case_count_delta_for_eggs(breeding_manager, player, 3) == 1
    )

    tight = next(bird for bird in _BIRDS if bird.egg_limit < 4)
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=tight)]
    assert (
        scoring.bonus_best_case_count_delta_for_eggs(breeding_manager, player, 100) == 0
    )

    non_egg_card = next(
        bc for bc in _BONUSES if bc.name not in ("Oologist", "Breeding Manager")
    )
    assert scoring.bonus_best_case_count_delta_for_eggs(non_egg_card, player, 5) == 0
