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
    registry = placeholders.PlaceholderRegistry()
    single_habitat_bird = next(
        bird for bird in catalog.birds_ordered() if len(bird.habitats) == 1
    )
    con, _ = aid_helpers.scripted_console(["y", single_habitat_bird.name, "n"])
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

    assert eng.state.players[1].hand == [single_habitat_bird]
    assert len(notes.plays) == 1
    assert notes.plays[0].bird == single_habitat_bird
    assert notes.plays[0].habitat == single_habitat_bird.habitats[0]


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
