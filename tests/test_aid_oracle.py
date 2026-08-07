"""Tests for the ``wingspan.aid`` oracle machinery: ``models``, ``console``,
``placeholders``, ``oracle``, and ``oracle_state.build_state``.

No agents, no app, no torch -- this stage is the oracle layer only. Every
interactive test builds a scripted ``Console`` (a fixed queue of answers
plus a printed-line transcript) so prompts resolve without real stdin.
"""

from __future__ import annotations

import random

import pydantic
import pytest

import aid_helpers
from wingspan import cards, state
from wingspan.aid import console, models, oracle, oracle_state, placeholders
from wingspan.cards import lookup
from wingspan.cards.parse import catalog

# A valid 5-die feeder reading (1 each of invertebrate/seed/fish/fruit, one
# choice die) reused across the build_state tests.
_SAMPLE_FEEDER = models.FeederEntry(counts=[1, 1, 1, 1, 0], choice_dice=1)


def _fresh_oracle(
    answers: list[str] | None = None,
) -> tuple[oracle.SessionOracle, list[str]]:
    """A ``SessionOracle`` wired to a scripted console and a fresh
    placeholder registry."""
    console_instance, transcript = aid_helpers.scripted_console(answers or [])
    echo = console.LogEcho(console_instance)
    registry = placeholders.PlaceholderRegistry()
    return oracle.SessionOracle(console_instance, echo, registry), transcript


def _catalog_setup_entry(start_player: int = 0) -> models.SetupEntry:
    """A ``SetupEntry`` built from real catalog cards (non-overlapping hand
    and tray slices, since the physical deal never repeats a card)."""
    birds = catalog.birds_ordered()
    bonuses = catalog.bonus_cards_ordered()
    _, _, goals = cards.load_all()
    return models.SetupEntry(
        hand=tuple(birds[0:5]),
        bonus_pair=tuple(bonuses[0:2]),
        goals=tuple(goals[0:4]),
        tray=tuple(birds[10:13]),
        feeder=_SAMPLE_FEEDER,
        start_player=start_player,
    )


def _build(
    start_player: int = 0,
) -> tuple[
    oracle_state.OracleGameState,
    models.SetupEntry,
    oracle.SessionOracle,
    placeholders.PlaceholderRegistry,
]:
    setup = _catalog_setup_entry(start_player)
    session_oracle, _ = _fresh_oracle()
    gs = oracle_state.build_state(random.Random(0), setup, session_oracle)
    return gs, setup, session_oracle, session_oracle.registry


def _minimal_game_state() -> state.GameState:
    return state.GameState(
        rng=random.Random(0),
        players=[state.Player(id=0, name="P0"), state.Player(id=1, name="P1")],
        current_player=0,
        round_idx=0,
        bird_deck=[],
    )


# ---------------------------------------------------------------------------
# FeederEntry


def test_feeder_entry_rejects_wrong_total_dice() -> None:
    with pytest.raises(pydantic.ValidationError):
        models.FeederEntry(counts=[1, 1, 1, 1, 1], choice_dice=1)  # totals 6, not 5


def test_feeder_entry_rejects_wrong_counts_length() -> None:
    with pytest.raises(pydantic.ValidationError):
        models.FeederEntry(counts=[1, 1, 1], choice_dice=2)


def test_feeder_entry_to_food_pool_round_trips() -> None:
    entry = models.FeederEntry(counts=[1, 1, 1, 1, 0], choice_dice=1)
    pool = entry.to_food_pool()
    assert pool.counts == [1, 1, 1, 1, 0]


# ---------------------------------------------------------------------------
# build_state


def test_build_state_tray_matches_entry() -> None:
    gs, setup, *_ = _build()
    assert list(gs.tray) == list(setup.tray)


def test_build_state_feeder_matches_entry() -> None:
    gs, setup, *_ = _build()
    assert gs.birdfeeder.counts.counts == setup.feeder.counts
    assert gs.birdfeeder.choice_dice == setup.feeder.choice_dice


def test_build_state_deck_length_accounts_for_tray_draws() -> None:
    gs, *_ = _build()
    assert len(gs.bird_deck) == catalog.n_birds() - state.TRAY_SIZE


def test_build_state_goals_start_player_and_current_player() -> None:
    gs, setup, *_ = _build(start_player=1)
    assert list(gs.round_goals) == list(setup.goals)
    assert gs.start_player == 1
    assert gs.current_player == 1


def test_build_state_attaches_oracle_backrefs() -> None:
    gs, _, session_oracle, _ = _build()
    assert session_oracle.game_state is gs
    assert session_oracle.echo.game_state is gs


def test_build_state_never_consults_rng() -> None:
    """The feeder/tray reveals are scripted facts, not random draws -- the
    rng object must come out of build_state in the exact state it went in."""
    setup = _catalog_setup_entry()
    session_oracle, _ = _fresh_oracle()
    rng = random.Random(0)
    state_before = rng.getstate()

    oracle_state.build_state(rng, setup, session_oracle)

    assert rng.getstate() == state_before


# ---------------------------------------------------------------------------
# Deal-order queue consumption


def test_deal_order_hand_then_placeholders() -> None:
    gs, setup, _, registry = _build()

    drawn_hand = [gs.draw_bird() for _ in range(state.STARTING_HAND_SIZE)]
    assert drawn_hand == list(setup.hand)

    opponent_hand = [gs.draw_bird() for _ in range(state.STARTING_HAND_SIZE)]
    assert all(
        bird is not None and registry.is_placeholder(bird) for bird in opponent_hand
    )


def test_deal_order_bonus_then_placeholders() -> None:
    gs, setup, _, registry = _build()
    # The bird and bonus queues are independent, so the bonus deal order
    # (bonus_pair, then two placeholders) holds regardless of bird draws.
    drawn_bonus = [gs.draw_bonus() for _ in range(state.STARTING_BONUS_CARDS_DEAL)]
    assert drawn_bonus == list(setup.bonus_pair)

    opponent_bonus = [gs.draw_bonus() for _ in range(state.STARTING_BONUS_CARDS_DEAL)]
    assert all(
        bonus is not None and registry.is_placeholder(bonus) for bonus in opponent_bonus
    )


def test_deck_lengths_decrement_by_one_per_draw() -> None:
    gs, *_ = _build()

    bird_before = len(gs.bird_deck)
    gs.draw_bird()
    assert len(gs.bird_deck) == bird_before - 1

    bonus_before = len(gs.bonus_deck)
    gs.draw_bonus()
    assert len(gs.bonus_deck) == bonus_before - 1


# ---------------------------------------------------------------------------
# Interactive bird/bonus reveals (empty queue -> prompt)


def test_reveal_bird_interactive_fuzzy_match_resolves() -> None:
    candidates = lookup.find_birds("Barn Owel")
    if len(candidates) == 1:
        answers = ["Barn Owel"]
    else:
        target_index = next(
            i for i, bird in enumerate(candidates) if bird.name == "Barn Owl"
        )
        answers = ["Barn Owel", str(target_index + 1)]
    session_oracle, _ = _fresh_oracle(answers)

    bird = session_oracle.reveal_bird()

    assert bird.name == "Barn Owl"


def test_reveal_bird_interactive_empty_answer_mints_placeholder() -> None:
    session_oracle, _ = _fresh_oracle([""])

    bird = session_oracle.reveal_bird()

    assert session_oracle.registry.is_placeholder(bird)


def test_reveal_bird_interactive_ambiguous_disambiguates_via_menu() -> None:
    candidates = lookup.find_birds("Barn")
    assert len(candidates) > 1
    target_index = next(
        i for i, bird in enumerate(candidates) if bird.name == "Barn Swallow"
    )
    session_oracle, transcript = _fresh_oracle(["Barn", str(target_index + 1)])

    bird = session_oracle.reveal_bird()

    assert bird.name == "Barn Swallow"
    assert any("Multiple matches" in line for line in transcript)


def test_reveal_bird_interactive_no_match_reprompts_then_resolves() -> None:
    session_oracle, transcript = _fresh_oracle(["Zzzznotabirdatall", "Barn Owl"])

    bird = session_oracle.reveal_bird()

    assert bird.name == "Barn Owl"
    assert any("No match" in line for line in transcript)


def test_reveal_bonus_interactive_resolves_by_name() -> None:
    session_oracle, _ = _fresh_oracle(["Anatomist"])

    bonus_card = session_oracle.reveal_bonus()

    assert bonus_card.name == "Anatomist"


def test_reveal_bonus_interactive_empty_answer_mints_placeholder() -> None:
    session_oracle, _ = _fresh_oracle([""])

    bonus_card = session_oracle.reveal_bonus()

    assert session_oracle.registry.is_placeholder(bonus_card)


# ---------------------------------------------------------------------------
# feeder_roll / dice_roll prompting


def test_feeder_roll_consumes_a_queued_entry_without_prompting() -> None:
    session_oracle, transcript = _fresh_oracle([])
    session_oracle.queue_feeder_roll(_SAMPLE_FEEDER)

    entry = session_oracle.feeder_roll()

    assert entry == _SAMPLE_FEEDER
    assert transcript == []


def test_feeder_roll_interactive_parses_scripted_entry() -> None:
    session_oracle, _ = _fresh_oracle(["seed seed fish choice choice"])

    entry = session_oracle.feeder_roll()

    assert entry.counts == [0, 2, 1, 0, 0]  # invertebrate, seed, fish, fruit, rodent
    assert entry.choice_dice == 2


def test_feeder_roll_reprompts_on_invalid_entry() -> None:
    # First answer has only 4 tokens (wrong count); second is valid.
    session_oracle, transcript = _fresh_oracle(
        ["seed seed fish choice", "seed seed fish choice choice"]
    )

    entry = session_oracle.feeder_roll()

    assert entry.choice_dice == 2
    assert any("Enter exactly" in line for line in transcript)


def test_dice_roll_parses_two_faces() -> None:
    session_oracle, _ = _fresh_oracle(["fish rodent"])

    counts, choice_dice = session_oracle.dice_roll(2)

    assert counts.counts == [0, 0, 1, 0, 1]
    assert choice_dice == 0


# ---------------------------------------------------------------------------
# Reshuffle bookkeeping stays truthful under a queued reveal


def test_draw_bird_reshuffles_filler_and_keeps_lengths_truthful() -> None:
    gs, _, session_oracle, _ = _build()
    # Drain the pre-queued setup deal (hand + 5 opponent placeholders) so
    # the bird_queue is empty before exercising the reshuffle path below.
    for _ in range(2 * state.STARTING_HAND_SIZE):
        gs.draw_bird()

    # Drain the deck directly (no oracle involved) and move it to discard,
    # simulating a game that has run the deck dry.
    gs.bird_discard = gs.bird_deck
    gs.bird_deck = []
    discard_len_before = len(gs.bird_discard)

    queued_bird = catalog.birds_ordered()[42]
    session_oracle.queue_bird_reveals([queued_bird])

    revealed = gs.draw_bird()

    assert revealed == queued_bird
    assert gs.bird_discard == []
    assert len(gs.bird_deck) == discard_len_before - 1


def test_draw_bird_returns_none_when_deck_and_discard_are_both_empty() -> None:
    gs, *_ = _build()
    gs.bird_deck = []
    gs.bird_discard = []

    assert gs.draw_bird() is None


def test_draw_bonus_returns_none_on_empty_deck() -> None:
    gs, *_ = _build()
    gs.bonus_deck = []

    assert gs.draw_bonus() is None


# ---------------------------------------------------------------------------
# Placeholder identity semantics


def test_fresh_equal_valued_catalog_card_is_not_a_placeholder() -> None:
    registry = placeholders.PlaceholderRegistry()
    placeholder = registry.mint_bird()
    source = catalog.birds_ordered()[0]
    equal_copy = source.model_copy(deep=True)  # equal by value, distinct identity

    assert not registry.is_placeholder(equal_copy)
    assert equal_copy == placeholder
    assert equal_copy is not placeholder


def test_count_and_first_bird_in() -> None:
    registry = placeholders.PlaceholderRegistry()
    real_bird = catalog.birds_ordered()[3]
    placeholder = registry.mint_bird()
    hand = [real_bird, placeholder]

    assert registry.count_birds_in(hand) == 1
    assert registry.first_bird_in(hand) is placeholder


def test_swap_bird_replaces_first_placeholder_only() -> None:
    registry = placeholders.PlaceholderRegistry()
    real_bird = catalog.birds_ordered()[7]
    placeholder_a = registry.mint_bird()
    placeholder_b = registry.mint_bird()
    hand = [placeholder_a, placeholder_b]

    registry.swap_bird(hand, real_bird)

    assert hand[0] is real_bird
    assert hand[1] is placeholder_b


def test_swap_bird_raises_when_no_placeholder_present() -> None:
    registry = placeholders.PlaceholderRegistry()
    hand = [catalog.birds_ordered()[3]]

    with pytest.raises(ValueError):
        registry.swap_bird(hand, catalog.birds_ordered()[4])


# ---------------------------------------------------------------------------
# Console / LogEcho


def test_console_ask_strips_and_falls_back_to_default() -> None:
    console_instance, _ = aid_helpers.scripted_console(["  ", "typed answer"])
    assert console_instance.ask("prompt", default="fallback") == "fallback"
    assert console_instance.ask("prompt") == "typed answer"


def test_console_confirm_defaults_to_yes_on_blank_answer() -> None:
    console_instance, _ = aid_helpers.scripted_console([""])
    assert console_instance.confirm("continue?") is True


def test_console_menu_rejects_out_of_range_then_accepts() -> None:
    console_instance, transcript = aid_helpers.scripted_console(["9", "2"])
    choice = console_instance.menu("pick one", ["a", "b", "c"])
    assert choice == 1
    assert any("Enter a number from 1 to 3" in line for line in transcript)


def test_log_echo_is_a_no_op_while_unattached() -> None:
    console_instance, transcript = aid_helpers.scripted_console([])
    echo = console.LogEcho(console_instance)
    echo.flush()
    assert transcript == []


def test_log_echo_flushes_only_new_lines_since_last_flush() -> None:
    console_instance, transcript = aid_helpers.scripted_console([])
    echo = console.LogEcho(console_instance)
    gs = _minimal_game_state()
    echo.game_state = gs

    gs.log.append("\x1b[31mline one\x1b[0m")
    echo.flush()
    assert transcript == ["line one"]

    gs.log.append("line two")
    echo.flush()
    assert transcript == ["line one", "line two"]
