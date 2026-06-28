"""Tests for the HTML decision-log humanizer.

Covers :func:`humanize_choice`, :func:`humanize_outcome`, :func:`humanize_note`,
and :func:`humanize_forced`.  Tests prepend ``src/`` to ``sys.path`` to match
``test_smoke.py``.
"""

from __future__ import annotations

from wingspan import cards, decisions, state
from wingspan.reporting import humanize

# ---------------------------------------------------------------------------
# Helpers — minimal game state and card stubs


def _empty_gs() -> state.GameState:
    """A two-player GameState, suitable for choice label lookup tests."""
    from wingspan import engine as engine_module

    eng, *_ = engine_module.Engine.create(seed=42)
    return eng.state


# ---------------------------------------------------------------------------
# humanize_choice


def test_humanize_choice_draw_source_tray():
    gs = _empty_gs()
    bird = cards.Bird.__new__(cards.Bird)
    object.__setattr__(bird, "name", "Mallard")
    choice = decisions.DrawSourceChoice(
        label="tray[1]=Mallard", source="tray", tray_index=1, bird=bird
    )
    result = humanize.humanize_choice(choice, gs)
    assert result == "Mallard (tray)"


def test_humanize_choice_draw_source_deck():
    gs = _empty_gs()
    choice = decisions.DrawSourceChoice(label="deck", source="deck")
    assert humanize.humanize_choice(choice, gs) == "Draw from the deck"


def test_humanize_choice_skip():
    gs = _empty_gs()
    choice = decisions.SkipChoice(label="skip")
    assert humanize.humanize_choice(choice, gs) == "Decline"


def test_humanize_choice_food():
    gs = _empty_gs()
    choice = decisions.FoodChoice(label="seed", food=cards.Food.SEED)
    assert humanize.humanize_choice(choice, gs) == "seed"


def test_humanize_choice_main_action():
    gs = _empty_gs()
    choice = decisions.MainActionChoice(
        label="lay_eggs", action=decisions.MainAction.LAY_EGGS
    )
    assert humanize.humanize_choice(choice, gs) == "Lay eggs"


# ---------------------------------------------------------------------------
# humanize_outcome


def test_humanize_outcome_main_action_lay_eggs():
    gs = _empty_gs()
    choice = decisions.MainActionChoice(
        label="lay_eggs", action=decisions.MainAction.LAY_EGGS
    )
    decision = decisions.MainActionDecision(player_id=0, prompt="", choices=[choice])
    assert humanize.humanize_outcome(decision, choice, gs) == "Lay eggs"


def test_humanize_outcome_draw_source_deck():
    gs = _empty_gs()
    choice = decisions.DrawSourceChoice(label="deck", source="deck")
    decision = decisions.DrawCardsPickSourceDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Draws from the deck"


def test_humanize_outcome_skip():
    # Without a PayCostChoice sibling, SkipChoice still returns "Declines".
    gs = _empty_gs()
    choice = decisions.SkipChoice(label="skip")
    decision = decisions.AcceptExchangeDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Declines"


def test_humanize_outcome_skip_with_pay_cost():
    # Item 10: when a PayCostChoice sibling exists, Declines mirrors the label.
    gs = _empty_gs()
    skip = decisions.SkipChoice(label="skip")
    pay = decisions.PayCostChoice(
        label="discard egg for card", paid_egg_count=1, gained_card_count=1
    )
    decision = decisions.AcceptExchangeDecision(
        player_id=0, prompt="", choices=[pay, skip]
    )
    assert (
        humanize.humanize_outcome(decision, skip, gs)
        == "Declines: discard egg for card"
    )


# ---------------------------------------------------------------------------
# humanize_note


def test_humanize_note_strips_player_prefix():
    result = humanize.humanize_note("[P0] lays 2 eggs")
    # Should not begin with the player tag
    assert not result.startswith("[P0]")


def test_humanize_note_play_bird():
    text = "[P0] plays American Kestrel into forest (paid seed, 0 eggs)"
    result = humanize.humanize_note(text)
    assert result == "Plays American Kestrel in Forest"


def test_humanize_note_lay_eggs():
    # Item 1: row-count lines are dropped entirely.
    result = humanize.humanize_note("[P0] lay eggs: row has 3 birds, lay 2 eggs")
    assert result == ""


def test_humanize_note_lay_eggs_singular():
    result = humanize.humanize_note("[P1] lay eggs: row has 1 bird, lay 1 egg")
    assert result == ""


def test_humanize_note_draw_cards():
    result = humanize.humanize_note("[P0] draw cards: row has 4 birds, draw 2 cards")
    assert result == ""


def test_humanize_note_drew_from_deck():
    result = humanize.humanize_note("[P1] drew from deck: American Kestrel")
    assert result == "Draws American Kestrel from the deck"


def test_humanize_note_gain_food_token():
    result = humanize.humanize_note("  +1 seed")
    assert result == "Gains seed"


def test_humanize_note_power_activation():
    result = humanize.humanize_note('[P0] @ American Robin - "Draw 1 card."')
    assert "American Robin" in result
    assert "Draw 1 card" in result


def test_humanize_note_birdfeeder_rerolled():
    result = humanize.humanize_note("  birdfeeder empty; rerolled to 2seed 1fish")
    assert "rerolled" in result.lower() or "birdfeeder" in result.lower()


def test_humanize_note_unknown_falls_back_to_sentence_case():
    result = humanize.humanize_note("some totally unknown log line")
    assert result and result[0].isupper()


# ---------------------------------------------------------------------------
# humanize_forced


def test_humanize_forced_deck():
    assert humanize.humanize_forced("deck") == "Draw from the deck"


def test_humanize_forced_tray_slot():
    result = humanize.humanize_forced("tray[1]=American Kestrel")
    assert result == "American Kestrel (tray)"


def test_humanize_forced_board_target():
    result = humanize.humanize_forced("American Robin@forest[0]")
    assert "American Robin" in result
    assert "Forest" in result


def test_humanize_forced_generic_sentence_case():
    result = humanize.humanize_forced("some label")
    assert result == "Some label"


def test_humanize_forced_food_value():
    assert humanize.humanize_forced("seed") == "seed"
    assert humanize.humanize_forced("fish") == "fish"


# ---------------------------------------------------------------------------
# humanize_choice — additional branches


def _stub_bird(name: str) -> cards.Bird:
    bird = cards.Bird.__new__(cards.Bird)
    object.__setattr__(bird, "name", name)
    return bird


def _stub_bonus(name: str) -> cards.BonusCard:
    bonus = cards.BonusCard.__new__(cards.BonusCard)
    object.__setattr__(bonus, "name", name)
    object.__setattr__(bonus, "condition", "stub condition")
    return bonus


def test_humanize_choice_play_bird():
    gs = _empty_gs()
    choice = decisions.PlayBirdChoice(
        label="mallard-wetland",
        bird=_stub_bird("Mallard"),
        habitat=cards.Habitat.WETLAND,
    )
    result = humanize.humanize_choice(choice, gs)
    assert "Mallard" in result
    assert "Wetland" in result


def test_humanize_choice_bonus_card():
    gs = _empty_gs()
    choice = decisions.BonusCardChoice(
        label="ethologist", bonus_card=_stub_bonus("Ethologist")
    )
    assert humanize.humanize_choice(choice, gs) == "Ethologist"


def test_humanize_choice_bird():
    gs = _empty_gs()
    choice = decisions.BirdChoice(label="mallard", bird=_stub_bird("Mallard"))
    assert humanize.humanize_choice(choice, gs) == "Mallard"


def test_humanize_choice_habitat():
    gs = _empty_gs()
    choice = decisions.HabitatChoice(label="forest", habitat=cards.Habitat.FOREST)
    assert humanize.humanize_choice(choice, gs) == "Forest"


def test_humanize_choice_player_id():
    gs = _empty_gs()
    choice = decisions.PlayerIdChoice(label="p1", player_id=1)
    assert humanize.humanize_choice(choice, gs) == "P1"


def test_humanize_choice_tuck_activate_plural():
    gs = _empty_gs()
    choice = decisions.TuckActivateChoice(label="tuck2", cards_to_tuck=2)
    assert humanize.humanize_choice(choice, gs) == "Tuck 2 cards"


def test_humanize_choice_tuck_activate_singular():
    gs = _empty_gs()
    choice = decisions.TuckActivateChoice(label="tuck1", cards_to_tuck=1)
    assert humanize.humanize_choice(choice, gs) == "Tuck 1 card"


def test_humanize_choice_reset_birdfeeder():
    gs = _empty_gs()
    choice = decisions.ResetBirdfeederChoice(label="reset")
    assert humanize.humanize_choice(choice, gs) == "Reset birdfeeder"


def test_humanize_choice_setup_three_birds():
    gs = _empty_gs()
    birds = tuple(_stub_bird(n) for n in ["Mallard", "Robin", "Wren", "Jay"])
    choice = decisions.SetupChoice(
        kept_cards=birds, kept_foods=(cards.Food.SEED,), bonus_card=None
    )
    result = humanize.humanize_choice(choice, gs)
    assert "Mallard" in result
    assert "…" in result  # truncated at 3


def test_humanize_choice_setup_no_birds():
    gs = _empty_gs()
    choice = decisions.SetupChoice(
        kept_cards=(), kept_foods=tuple(cards.ALL_FOODS), bonus_card=None
    )
    assert humanize.humanize_choice(choice, gs) == "Keep no birds"


def test_humanize_choice_played_bird():
    from wingspan import state as state_module

    gs = _empty_gs()
    bird = _stub_bird("American Robin")
    played = state_module.PlayedBird.model_construct(bird=bird, eggs=0)
    choice = decisions.PlayedBirdChoice(label="robin", played_bird=played)
    assert humanize.humanize_choice(choice, gs) == "American Robin"


def test_humanize_choice_food_payment():
    from wingspan import state as state_module

    gs = _empty_gs()
    pool = state_module.FoodPool()
    pool[cards.Food.FISH] = 1
    choice = decisions.FoodPaymentChoice(label="1fish", payment=pool)
    result = humanize.humanize_choice(choice, gs)
    assert "fish" in result


def test_humanize_choice_board_target_with_bird():
    gs = _empty_gs()
    # Pick a habitat that has a bird in slot 0 after game start.
    # forest slot 0 — may or may not have a bird; test the slot-found path
    # by using the first player's board and checking all habitats.
    player = gs.players[0]
    found = False
    for hab in [cards.Habitat.FOREST, cards.Habitat.GRASSLAND, cards.Habitat.WETLAND]:
        row = player.board[hab]
        if row:
            choice = decisions.BoardTargetChoice(label="slot", habitat=hab, slot=0)
            result = humanize.humanize_choice(choice, gs, player_id=0)
            assert row[0].bird.name in result
            found = True
            break
    if not found:
        # Empty board path: returns habitat+slot text, not an error.
        choice = decisions.BoardTargetChoice(
            label="slot", habitat=cards.Habitat.FOREST, slot=0
        )
        result = humanize.humanize_choice(choice, gs, player_id=0)
        assert "Forest" in result or "slot" in result.lower()


# ---------------------------------------------------------------------------
# humanize_outcome — additional branches


def test_humanize_outcome_play_bird():
    gs = _empty_gs()
    bird = _stub_bird("Mallard")
    choice = decisions.PlayBirdChoice(
        label="mallard", bird=bird, habitat=cards.Habitat.WETLAND
    )
    decision = decisions.PlayBirdDecision(player_id=0, prompt="", choices=[choice])
    assert humanize.humanize_outcome(decision, choice, gs) == "Plays Mallard in Wetland"


def test_humanize_outcome_draw_source_tray():
    gs = _empty_gs()
    bird = _stub_bird("Mallard")
    choice = decisions.DrawSourceChoice(
        label="tray[0]=Mallard", source="tray", tray_index=0, bird=bird
    )
    decision = decisions.DrawCardsPickSourceDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert (
        humanize.humanize_outcome(decision, choice, gs) == "Draws Mallard from the tray"
    )


def test_humanize_outcome_food():
    gs = _empty_gs()
    choice = decisions.FoodChoice(label="fish", food=cards.Food.FISH)
    decision = decisions.GainFoodDecision(player_id=0, prompt="", choices=[choice])
    assert humanize.humanize_outcome(decision, choice, gs) == "Gains fish"


def test_humanize_outcome_bonus_card():
    # Item 4: "bonus card" is included in the outcome label.
    gs = _empty_gs()
    choice = decisions.BonusCardChoice(
        label="ethologist", bonus_card=_stub_bonus("Ethologist")
    )
    decision = decisions.BirdPowerPickBonusCardDecision(
        player_id=0, prompt="", choices=[choice]
    )
    result = humanize.humanize_outcome(decision, choice, gs)
    assert result == "Keeps bonus card Ethologist"


def test_humanize_outcome_bird_choice_discard():
    # Item 8: DiscardBirdForFoodDecision → "Discards" (not "Picks").
    gs = _empty_gs()
    choice = decisions.BirdChoice(label="mallard", bird=_stub_bird("Mallard"))
    decision = decisions.DiscardBirdForFoodDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Discards Mallard"


def test_humanize_outcome_bird_choice_pass():
    # Item 8: BirdPowerDiscardFromHandDecision with "pass" in prompt → "Passes".
    gs = _empty_gs()
    choice = decisions.BirdChoice(label="mallard", bird=_stub_bird("Mallard"))
    decision = decisions.BirdPowerDiscardFromHandDecision(
        player_id=0, prompt="pass a card to opponent", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Passes Mallard"


def test_humanize_outcome_bird_choice_other():
    # Non-discard BirdChoice decisions still return "Picks".
    gs = _empty_gs()
    choice = decisions.BirdChoice(label="mallard", bird=_stub_bird("Mallard"))
    decision = decisions.BirdPowerTuckFromHandDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Picks Mallard"


def test_humanize_outcome_setup():
    gs = _empty_gs()
    birds = tuple(_stub_bird(n) for n in ["Mallard", "Robin"])
    choice = decisions.SetupChoice(
        kept_cards=birds, kept_foods=(cards.Food.SEED,), bonus_card=None
    )
    # humanize_outcome only uses decision.player_id for BoardTargetChoice; use
    # model_construct to avoid triggering choice-type validation on the decision.
    decision = decisions.MainActionDecision.model_construct(
        player_id=0, prompt="", choices=[choice]
    )
    result = humanize.humanize_outcome(decision, choice, gs)
    assert "Mallard" in result


def test_humanize_outcome_habitat():
    gs = _empty_gs()
    choice = decisions.HabitatChoice(label="grassland", habitat=cards.Habitat.GRASSLAND)
    decision = decisions.MainActionDecision.model_construct(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Grassland"


def test_humanize_outcome_reset_birdfeeder():
    gs = _empty_gs()
    choice = decisions.ResetBirdfeederChoice(label="reset")
    decision = decisions.ResetBirdfeederDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Resets birdfeeder"


def test_humanize_outcome_tuck_activate():
    gs = _empty_gs()
    choice = decisions.TuckActivateChoice(label="tuck", cards_to_tuck=1)
    decision = decisions.ActivateTuckDecision(player_id=0, prompt="", choices=[choice])
    assert humanize.humanize_outcome(decision, choice, gs) == "Tucks 1 card"


def test_humanize_outcome_pay_cost():
    gs = _empty_gs()
    choice = decisions.PayCostChoice(
        label="discard egg for card", paid_egg_count=1, gained_card_count=1
    )
    decision = decisions.AcceptExchangeDecision(
        player_id=0, prompt="", choices=[choice]
    )
    result = humanize.humanize_outcome(decision, choice, gs)
    assert "discard egg for card" in result


def test_humanize_outcome_food_payment():
    from wingspan import state as state_module

    gs = _empty_gs()
    pool = state_module.FoodPool()
    pool[cards.Food.SEED] = 1
    choice = decisions.FoodPaymentChoice(label="1seed", payment=pool)
    decision = decisions.MainActionDecision.model_construct(
        player_id=0, prompt="", choices=[choice]
    )
    result = humanize.humanize_outcome(decision, choice, gs)
    assert "seed" in result


def test_humanize_outcome_played_bird():
    from wingspan import state as state_module

    gs = _empty_gs()
    bird = _stub_bird("American Robin")
    played = state_module.PlayedBird.model_construct(bird=bird, eggs=0)
    choice = decisions.PlayedBirdChoice(label="robin", played_bird=played)
    decision = decisions.BirdPowerPickPlayedBirdDecision(
        player_id=0, prompt="", choices=[choice]
    )
    assert humanize.humanize_outcome(decision, choice, gs) == "Uses American Robin"


# ---------------------------------------------------------------------------
# humanize_note — additional branches


def test_humanize_note_no_brown_power():
    result = humanize.humanize_note("[P0] @ American Robin - no brown power")
    assert "American Robin" in result
    assert "no brown power" in result


def test_humanize_note_birdfeeder_reset():
    result = humanize.humanize_note("  resets the birdfeeder")
    assert "birdfeeder" in result.lower() or "reset" in result.lower()


def test_humanize_note_convert_discard_for_food():
    # Item 9: conversion echoes are dropped.
    result = humanize.humanize_note("convert: discard seed for +1 food")
    assert result == ""


def test_humanize_note_convert_spend_for_egg():
    result = humanize.humanize_note("convert: spend fish for +1 egg")
    assert result == ""


def test_humanize_note_convert_egg_for_card():
    result = humanize.humanize_note("convert: discard 1 egg for +1 card")
    assert result == ""


def test_humanize_note_declines_extra_play():
    result = humanize.humanize_note("[P0] declines the extra play")
    assert "extra play" in result.lower()


def test_humanize_note_takes_extra_play():
    result = humanize.humanize_note("[P0] takes an EXTRA play")
    assert "extra play" in result.lower()


def test_humanize_note_no_playable_bird():
    result = humanize.humanize_note("[P0] has no playable bird — wasted action")
    assert "bird" in result.lower() or "wasted" in result.lower()


def test_humanize_note_gain_food_row_count_dropped():
    # Item 1: gain-food row-count line is dropped.
    result = humanize.humanize_note("[P0] gain food: row has 2 birds, gain 2 food")
    assert result == ""


# ---------------------------------------------------------------------------
# humanize_outcome — Item 6 (RemoveEggDecision header)


def test_humanize_outcome_remove_egg_with_board():
    # Item 6: RemoveEggDecision + BoardTargetChoice → "Remove 1 egg from {bird}".
    gs = _empty_gs()
    player = gs.players[0]
    # Find a habitat with a bird on the board for a full test.
    found = False
    for hab in [cards.Habitat.FOREST, cards.Habitat.GRASSLAND, cards.Habitat.WETLAND]:
        row = player.board[hab]
        if row:
            choice = decisions.BoardTargetChoice(label="slot", habitat=hab, slot=0)
            decision = decisions.RemoveEggDecision(
                player_id=0, prompt="", choices=[choice]
            )
            result = humanize.humanize_outcome(decision, choice, gs)
            assert result == f"Remove 1 egg from {row[0].bird.name}"
            found = True
            break
    if not found:
        # Empty board: falls back gracefully.
        choice = decisions.BoardTargetChoice(
            label="slot", habitat=cards.Habitat.FOREST, slot=0
        )
        decision = decisions.RemoveEggDecision(player_id=0, prompt="", choices=[choice])
        result = humanize.humanize_outcome(decision, choice, gs)
        assert "Remove 1 egg" in result


def test_humanize_outcome_board_target_non_remove_egg():
    # Non-RemoveEggDecision BoardTargetChoice still uses "Targets".
    gs = _empty_gs()
    for hab in [cards.Habitat.FOREST, cards.Habitat.GRASSLAND, cards.Habitat.WETLAND]:
        row = gs.players[0].board[hab]
        if row:
            choice = decisions.BoardTargetChoice(label="slot", habitat=hab, slot=0)
            decision = decisions.LayEggDecision(
                player_id=0, prompt="", choices=[choice]
            )
            result = humanize.humanize_outcome(decision, choice, gs)
            assert result.startswith("Targets")
            break


# ---------------------------------------------------------------------------
# humanize_choice / humanize_outcome — Item 7 (choice die)


def test_humanize_choice_food_from_choice_die():
    # Item 7: choice-die food gains display the die pair.
    gs = _empty_gs()
    choice = decisions.FoodChoice(
        label="seed", food=cards.Food.SEED, from_choice_die=True
    )
    result = humanize.humanize_choice(choice, gs)
    assert result == "seed (from seed/invertebrate)"


def test_humanize_outcome_food_choice_die():
    gs = _empty_gs()
    choice = decisions.FoodChoice(
        label="seed", food=cards.Food.SEED, from_choice_die=True
    )
    decision = decisions.GainFoodDecision(player_id=0, prompt="", choices=[choice])
    result = humanize.humanize_outcome(decision, choice, gs)
    assert result == "Gains seed (choice die)"


# ---------------------------------------------------------------------------
# humanize_choice — Item 10 (symmetric decline)


def test_humanize_choice_skip_with_accept_exchange():
    # Item 10: SkipChoice under AcceptExchangeDecision mirrors the PayCostChoice label.
    gs = _empty_gs()
    skip = decisions.SkipChoice(label="skip")
    pay = decisions.PayCostChoice(
        label="discard egg for card", paid_egg_count=1, gained_card_count=1
    )
    decision = decisions.AcceptExchangeDecision(
        player_id=0, prompt="", choices=[pay, skip]
    )
    result = humanize.humanize_choice(skip, gs, decision=decision)
    assert result == "Decline (discard egg for card)"


def test_humanize_choice_skip_no_decision():
    # Without decision context, SkipChoice still returns "Decline".
    gs = _empty_gs()
    choice = decisions.SkipChoice(label="skip")
    assert humanize.humanize_choice(choice, gs) == "Decline"


# ---------------------------------------------------------------------------
# DecisionProbe — probe records and clears correctly


def test_decision_probe_take_returns_recorded_values():
    from wingspan.players import decision_probe as dp

    probe = dp.DecisionProbe()
    annotation = dp.PolicyAnnotation(probs=[0.7, 0.3], chosen_idx=0)
    probe.record(1.5)
    probe.record_policy(annotation)
    value, policy = probe.take()
    assert value == 1.5
    assert policy is not None and policy.chosen_idx == 0
    # Second take clears both slots.
    assert probe.take() == (None, None)
