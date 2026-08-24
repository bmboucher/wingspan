"""Tests for ``wingspan.aid.relay`` (the seat-1 agent) and
``wingspan.aid.hooks.AidHandler`` (the opponent-turn dialog + final-round
bonus entry). Torch-free throughout.
"""

from __future__ import annotations

import aid_helpers
from wingspan import cards, decisions, state
from wingspan.aid import console, hooks, models
from wingspan.aid import oracle as oracle_module
from wingspan.aid import placeholders, relay
from wingspan.cards.parse import catalog
from wingspan.engine import core as engine_core
from wingspan.setup_model import candidates as setup_candidates


def _build_setup_decision(
    registry: placeholders.PlaceholderRegistry,
    *,
    include_bonus: bool,
    include_food: bool,
) -> decisions.SetupDecision:
    """A full opponent ``SetupDecision`` over placeholder-carrying dealt
    inputs, in the requested regime shape. Built from
    ``setup_model.candidates.enumerate_setup_candidates`` -- the same
    504-candidate enumeration ``Engine._build_setup_choices`` wraps -- so the
    test exercises the public candidate API rather than an engine internal."""
    dealt_cards = [registry.mint_bird() for _ in range(state.STARTING_HAND_SIZE)]
    dealt_bonus = [
        registry.mint_bonus() for _ in range(state.STARTING_BONUS_CARDS_DEAL)
    ]
    choices = [
        candidate.to_setup_choice()
        for candidate in setup_candidates.enumerate_setup_candidates(
            dealt_cards,
            dealt_bonus,
            include_bonus=include_bonus,
            include_food=include_food,
        )
    ]
    return decisions.SetupDecision(
        player_id=1,
        prompt="opponent setup",
        choices=choices,
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
    )


# ---------------------------------------------------------------------------
# Opponent setup matching


def test_opponent_setup_matches_kept_count_and_foods_full_regime() -> None:
    registry = placeholders.PlaceholderRegistry()
    decision = _build_setup_decision(registry, include_bonus=True, include_food=True)
    con, _ = aid_helpers.scripted_console(["3", "seed, fish"])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=1)

    agent = relay.relay_agent(con, echo, registry, notes)
    chosen = agent(eng, decision)

    assert chosen in decision.choices
    assert len(chosen.kept_cards) == 3
    assert sorted(chosen.kept_foods) == sorted([cards.Food.SEED, cards.Food.FISH])


def test_opponent_setup_matches_kept_count_and_foods_no_bonus_regime() -> None:
    registry = placeholders.PlaceholderRegistry()
    decision = _build_setup_decision(registry, include_bonus=False, include_food=True)
    con, _ = aid_helpers.scripted_console(["2", "rodent, invertebrate, fruit"])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=1)

    agent = relay.relay_agent(con, echo, registry, notes)
    chosen = agent(eng, decision)

    assert chosen in decision.choices
    assert len(chosen.kept_cards) == 2
    assert chosen.bonus_card is None
    assert sorted(chosen.kept_foods) == sorted(
        [cards.Food.RODENT, cards.Food.INVERTEBRATE, cards.Food.FRUIT]
    )


def test_opponent_setup_matches_kept_count_only_no_food_regime() -> None:
    registry = placeholders.PlaceholderRegistry()
    decision = _build_setup_decision(registry, include_bonus=True, include_food=False)
    con, _ = aid_helpers.scripted_console(["4"])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=1)

    agent = relay.relay_agent(con, echo, registry, notes)
    chosen = agent(eng, decision)

    assert chosen in decision.choices
    assert len(chosen.kept_cards) == 4
    assert chosen.kept_foods == ()


# ---------------------------------------------------------------------------
# Turn-note auto-answers


def test_turn_notes_auto_answer_main_action_then_play_bird() -> None:
    registry = placeholders.PlaceholderRegistry()
    noted_bird = catalog.birds_ordered()[4]
    noted_habitat = noted_bird.habitats[0]
    notes = models.TurnNotes()
    notes.main_action = decisions.MainAction.PLAY_BIRD
    notes.plays.append(models.OpponentPlayNote(bird=noted_bird, habitat=noted_habitat))
    con, transcript = aid_helpers.scripted_console([])  # no input should be consumed
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=2)

    agent = relay.relay_agent(con, echo, registry, notes)

    main_decision = decisions.MainActionDecision(
        player_id=1,
        prompt="choose action",
        choices=[
            decisions.MainActionChoice(
                label="gain food", action=decisions.MainAction.GAIN_FOOD
            ),
            decisions.MainActionChoice(
                label="play bird", action=decisions.MainAction.PLAY_BIRD
            ),
        ],
    )
    chosen_action = agent(eng, main_decision)
    assert chosen_action == main_decision.choices[1]
    assert notes.play_consumed_count == 0

    other_bird = catalog.birds_ordered()[9]
    play_decision = decisions.PlayBirdDecision(
        player_id=1,
        prompt="play a bird",
        choices=[
            decisions.PlayBirdChoice(
                label="other", bird=other_bird, habitat=other_bird.habitats[0]
            ),
            decisions.PlayBirdChoice(
                label="noted", bird=noted_bird, habitat=noted_habitat
            ),
        ],
    )
    chosen_play = agent(eng, play_decision)
    assert chosen_play == play_decision.choices[1]
    assert notes.play_consumed_count == 1
    assert any("opponent plays" in line for line in transcript)


def test_turn_notes_auto_answer_non_play_main_action() -> None:
    """A non-``PLAY_BIRD`` ``notes.main_action`` (as the new turn-start menu
    always records, even when no bird was played) auto-answers the real
    ``MainActionDecision`` without prompting, and leaves no play-note state
    behind to confuse anything that comes after."""
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()
    notes.main_action = decisions.MainAction.GAIN_FOOD
    con, transcript = aid_helpers.scripted_console([])  # no input should be consumed
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=20)

    agent = relay.relay_agent(con, echo, registry, notes)

    main_decision = decisions.MainActionDecision(
        player_id=1,
        prompt="choose action",
        choices=[
            decisions.MainActionChoice(
                label="gain food (forest)", action=decisions.MainAction.GAIN_FOOD
            ),
            decisions.MainActionChoice(
                label="play a bird", action=decisions.MainAction.PLAY_BIRD
            ),
        ],
    )
    chosen_action = agent(eng, main_decision)

    assert chosen_action == main_decision.choices[0]
    assert any("auto-selected from your report" in line for line in transcript)
    assert notes.plays == []
    assert notes.play_consumed_count == 0


# ---------------------------------------------------------------------------
# AcceptExchangeDecision extra-play auto-answer


def test_extra_play_auto_accepted_when_note_unconsumed() -> None:
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()
    first_bird = catalog.birds_ordered()[4]
    second_bird = catalog.birds_ordered()[9]
    notes.plays.append(
        models.OpponentPlayNote(bird=first_bird, habitat=first_bird.habitats[0])
    )
    notes.plays.append(
        models.OpponentPlayNote(bird=second_bird, habitat=second_bird.habitats[0])
    )
    notes.play_consumed_count = 1  # the first play already resolved this turn
    con, transcript = aid_helpers.scripted_console([])  # no input should be consumed
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=21)

    agent = relay.relay_agent(con, echo, registry, notes)

    accept_choice = decisions.PayCostChoice(label="play a bird", gained_play_count=1)
    skip_choice = decisions.SkipChoice(label="forfeit the extra play")
    decision = decisions.AcceptExchangeDecision(
        player_id=1,
        prompt="[Opponent] play another bird?",
        choices=[accept_choice, skip_choice],
    )
    chosen = agent(eng, decision)

    assert chosen == accept_choice
    assert any("auto-selected from your report" in line for line in transcript)


def test_extra_play_auto_declined_when_no_unconsumed_note() -> None:
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()  # no plays recorded this turn
    con, transcript = aid_helpers.scripted_console([])  # no input should be consumed
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=22)

    agent = relay.relay_agent(con, echo, registry, notes)

    accept_choice = decisions.PayCostChoice(label="play a bird", gained_play_count=1)
    skip_choice = decisions.SkipChoice(label="forfeit the extra play")
    decision = decisions.AcceptExchangeDecision(
        player_id=1,
        prompt="[Opponent] play another bird?",
        choices=[accept_choice, skip_choice],
    )
    chosen = agent(eng, decision)

    assert chosen == skip_choice
    assert any("auto-selected from your report" in line for line in transcript)


def test_unrelated_accept_exchange_decision_falls_through_to_generic_prompt() -> None:
    """A fixed exchange unrelated to the extra-play credit (e.g. the Forest
    card->food trade) must not be auto-answered -- its accept choice carries
    no ``gained_play_count``, so it still reaches the generic fallback even
    with an unconsumed play note sitting around."""
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()
    noted_bird = catalog.birds_ordered()[4]
    notes.plays.append(
        models.OpponentPlayNote(bird=noted_bird, habitat=noted_bird.habitats[0])
    )
    con, transcript = aid_helpers.scripted_console(["0"])
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=23)

    agent = relay.relay_agent(con, echo, registry, notes)

    accept_choice = decisions.PayCostChoice(
        label="discard 1 card -> +1 food", paid_card_count=1, gained_food_count=1
    )
    skip_choice = decisions.SkipChoice(label="keep cards")
    decision = decisions.AcceptExchangeDecision(
        player_id=1,
        prompt="[Opponent] discard a card to gain 1 extra food?",
        choices=[accept_choice, skip_choice],
    )
    chosen = agent(eng, decision)

    assert chosen == accept_choice
    assert any("opponent's move>" in line for line in transcript)


# ---------------------------------------------------------------------------
# Hidden-card auto-pick / rendering


def test_all_placeholder_choices_auto_pick_index_zero() -> None:
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()
    con, transcript = aid_helpers.scripted_console([])  # no input should be consumed
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=3)

    decision = decisions.BirdPowerPickBirdFromHandDecision(
        player_id=1,
        prompt="pick a bird",
        choices=[
            decisions.BirdChoice(label="a", bird=registry.mint_bird()),
            decisions.BirdChoice(label="b", bird=registry.mint_bird()),
        ],
    )

    agent = relay.relay_agent(con, echo, registry, notes)
    chosen = agent(eng, decision)

    assert chosen == decision.choices[0]
    assert any("opponent used a hidden card" in line for line in transcript)


def test_all_placeholder_bonus_choices_auto_pick_index_zero() -> None:
    """The split_setup_bonus regime defers the opponent's bonus keep to a
    decision over two placeholder bonus cards — auto-picked, never asked."""
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()
    con, transcript = aid_helpers.scripted_console([])  # no input should be consumed
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=3)

    decision = decisions.BirdPowerPickBonusCardDecision(
        player_id=1,
        prompt="keep a bonus card",
        choices=[
            decisions.BonusCardChoice(label="a", bonus_card=registry.mint_bonus()),
            decisions.BonusCardChoice(label="b", bonus_card=registry.mint_bonus()),
        ],
    )

    agent = relay.relay_agent(con, echo, registry, notes)
    chosen = agent(eng, decision)

    assert chosen == decision.choices[0]
    assert any("opponent used a hidden card" in line for line in transcript)


def test_generic_fallback_renders_placeholder_choices_as_face_down() -> None:
    registry = placeholders.PlaceholderRegistry()
    notes = models.TurnNotes()
    con, transcript = aid_helpers.scripted_console(["1"])
    echo = console.LogEcho(con)
    eng, *_ = engine_core.Engine.create(seed=4)

    real_bird = catalog.birds_ordered()[6]
    decision = decisions.BirdPowerPickBirdFromHandDecision(
        player_id=1,
        prompt="pick a bird",
        choices=[
            decisions.BirdChoice(label="hidden", bird=registry.mint_bird()),
            decisions.BirdChoice(label="visible", bird=real_bird),
        ],
    )

    agent = relay.relay_agent(con, echo, registry, notes)
    chosen = agent(eng, decision)

    assert chosen == decision.choices[1]
    assert any("(face-down card)" in line for line in transcript)


# ---------------------------------------------------------------------------
# AidHandler.turn_start: opponent play report + hand surgery


def test_turn_start_reports_a_play_and_swaps_the_placeholder() -> None:
    """Picking PLAY_BIRD (menu index 4) from the new main-action menu still
    drives the identify-bird + pick-habitat sub-flow, and records the picked
    action onto ``notes.main_action``. The opponent's board is empty, so no
    bird there could grant an extra play -- the "another bird?" question is
    never asked, and the scripted answer queue carries no entry for it."""
    registry = placeholders.PlaceholderRegistry()
    single_habitat_bird = next(
        bird for bird in catalog.birds_ordered() if len(bird.habitats) == 1
    )
    con, _ = aid_helpers.scripted_console(["4", single_habitat_bird.name])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=5)
    echo.game_state = eng.state
    session_oracle = oracle_module.SessionOracle(con, echo, registry)
    eng.state.players[1].hand = [registry.mint_bird()]

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_start(engine=eng, player=eng.state.players[1])

    assert notes.main_action == decisions.MainAction.PLAY_BIRD
    assert eng.state.players[1].hand == [single_habitat_bird]
    assert len(notes.plays) == 1
    assert notes.plays[0].bird == single_habitat_bird
    assert notes.plays[0].habitat == single_habitat_bird.habitats[0]


def test_turn_start_menu_offers_all_four_main_actions_and_records_choice() -> None:
    """The pre-turn menu lists all 4 ``MainAction`` values with the engine's
    own labels; picking a non-``PLAY_BIRD`` option records it onto
    ``notes.main_action`` and never opens the bird-identify sub-flow."""
    registry = placeholders.PlaceholderRegistry()
    con, transcript = aid_helpers.scripted_console(["1"])  # gain food (forest)
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=24)
    session_oracle = oracle_module.SessionOracle(con, echo, registry)

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_start(engine=eng, player=eng.state.players[1])

    assert notes.main_action == decisions.MainAction.GAIN_FOOD
    assert notes.plays == []
    assert any("gain food (forest)" in line for line in transcript)
    assert any("lay eggs (grassland)" in line for line in transcript)
    assert any("draw cards (wetland)" in line for line in transcript)
    assert any("play a bird" in line for line in transcript)


def test_turn_start_no_extra_play_ask_without_plays_another_bird_power() -> None:
    """After identifying one bird play, a board bird whose power does NOT
    grant an extra play must not trigger the "another bird?" re-ask -- the
    scripted answer queue carries no entry for it, so an accidental ask
    would raise ``IndexError`` on the empty queue."""
    registry = placeholders.PlaceholderRegistry()
    normal_bird = next(
        bird for bird in catalog.birds_ordered() if not bird.plays_another_bird
    )
    single_habitat_bird = next(
        bird for bird in catalog.birds_ordered() if len(bird.habitats) == 1
    )
    eng, *_ = engine_core.Engine.create(seed=25)
    eng.state.players[1].board[normal_bird.habitats[0]].append(
        state.PlayedBird(bird=normal_bird)
    )
    eng.state.players[1].hand = [registry.mint_bird()]
    con, transcript = aid_helpers.scripted_console(["4", single_habitat_bird.name])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    session_oracle = oracle_module.SessionOracle(con, echo, registry)

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_start(engine=eng, player=eng.state.players[1])

    assert len(notes.plays) == 1
    assert not any("Did they play ANOTHER bird" in line for line in transcript)


def test_turn_start_asks_another_bird_when_board_has_extra_play_power() -> None:
    """A board bird whose power DOES grant an extra play (House Wren --
    ``PLAY_ADDITIONAL_BIRD_HERE``) must trigger the "another bird?" re-ask
    after the first play is identified."""
    registry = placeholders.PlaceholderRegistry()
    extra_play_bird = next(
        bird for bird in catalog.birds_ordered() if bird.name == "House Wren"
    )
    assert extra_play_bird.plays_another_bird
    single_habitat_bird = next(
        bird for bird in catalog.birds_ordered() if len(bird.habitats) == 1
    )
    eng, *_ = engine_core.Engine.create(seed=26)
    eng.state.players[1].board[extra_play_bird.habitats[0]].append(
        state.PlayedBird(bird=extra_play_bird)
    )
    eng.state.players[1].hand = [registry.mint_bird()]
    con, transcript = aid_helpers.scripted_console(["4", single_habitat_bird.name, "n"])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    session_oracle = oracle_module.SessionOracle(con, echo, registry)

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_start(engine=eng, player=eng.state.players[1])

    assert len(notes.plays) == 1
    assert any("Did they play ANOTHER bird" in line for line in transcript)


def test_turn_start_no_op_for_our_own_seat() -> None:
    registry = placeholders.PlaceholderRegistry()
    con, _ = aid_helpers.scripted_console([])  # confirm loop never runs
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=6)
    session_oracle = oracle_module.SessionOracle(con, echo, registry)

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_start(engine=eng, player=eng.state.players[0])

    assert notes.plays == []


# ---------------------------------------------------------------------------
# AidHandler.round_end: final-round opponent-bonus entry


def test_round_end_final_round_swaps_placeholder_bonus() -> None:
    registry = placeholders.PlaceholderRegistry()
    real_bonus = catalog.bonus_cards_ordered()[3]
    con, _ = aid_helpers.scripted_console(["y", real_bonus.name])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=7)
    echo.game_state = eng.state
    session_oracle = oracle_module.SessionOracle(con, echo, registry)
    eng.state.players[1].bonus_cards = [registry.mint_bonus()]

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.round_end(engine=eng, round_num=len(state.ROUND_CUBES) - 1)

    assert eng.state.players[1].bonus_cards == [real_bonus]
    assert handler.opponent_bonus_entered is True


def test_round_end_before_final_round_does_nothing() -> None:
    registry = placeholders.PlaceholderRegistry()
    con, _ = aid_helpers.scripted_console([])  # no dialog should run
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=8)
    session_oracle = oracle_module.SessionOracle(con, echo, registry)
    placeholder_bonus = registry.mint_bonus()
    eng.state.players[1].bonus_cards = [placeholder_bonus]

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.round_end(engine=eng, round_num=0)

    assert eng.state.players[1].bonus_cards == [placeholder_bonus]
    assert handler.opponent_bonus_entered is False


def test_round_end_decline_first_card_still_offers_and_enters_the_second() -> None:
    """A decline on the first placeholder bonus card must not abort the loop
    (``continue``, not ``break``): the second card must still be offered, and
    an accept on it must still stick ``opponent_bonus_entered`` at ``True``
    (never reset back to ``False`` by a later iteration).

    ``registry.swap_bonus`` always replaces the *first* placeholder it finds
    in the list (mirroring ``swap_bird``) rather than the specific card the
    loop is currently on -- a separate, pre-existing quirk out of scope for
    this fix -- so this test only asserts the two things the fix actually
    guarantees (both cards offered, the flag stays ``True``), not which slot
    ends up holding the identified card."""
    registry = placeholders.PlaceholderRegistry()
    real_bonus = catalog.bonus_cards_ordered()[3]
    con, transcript = aid_helpers.scripted_console(["n", "y", real_bonus.name])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=27)
    echo.game_state = eng.state
    session_oracle = oracle_module.SessionOracle(con, echo, registry)
    eng.state.players[1].bonus_cards = [registry.mint_bonus(), registry.mint_bonus()]

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.round_end(engine=eng, round_num=len(state.ROUND_CUBES) - 1)

    assert handler.opponent_bonus_entered is True
    assert real_bonus in eng.state.players[1].bonus_cards
    assert sum("Game over" in line for line in transcript) == 2


# ---------------------------------------------------------------------------
# AidHandler.turn_end: pause after our own seat's turn


def test_turn_end_pauses_after_our_own_seat() -> None:
    registry = placeholders.PlaceholderRegistry()
    con, transcript = aid_helpers.scripted_console([""])
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=9)
    session_oracle = oracle_module.SessionOracle(con, echo, registry)

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_end(engine=eng, player=eng.state.players[0])

    assert any("Press Enter when ready to continue" in line for line in transcript)


def test_turn_end_no_pause_after_opponent_seat() -> None:
    registry = placeholders.PlaceholderRegistry()
    con, transcript = aid_helpers.scripted_console([])  # no prompt should be asked
    echo = console.LogEcho(con)
    notes = models.TurnNotes()
    eng, *_ = engine_core.Engine.create(seed=10)
    session_oracle = oracle_module.SessionOracle(con, echo, registry)

    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=session_oracle, notes=notes, registry=registry
    )
    handler.turn_end(engine=eng, player=eng.state.players[1])

    assert not any("Press Enter when ready to continue" in line for line in transcript)
