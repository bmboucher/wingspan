"""Tests for the bird-tray draw / deferred-refill rules.

A card taken from the face-up bird tray is *not* replaced immediately. The
tray slot stays empty for the rest of the acting player's turn; the tray is
refilled to ``TRAY_SIZE`` only at the end of the turn (or by a bird power that
explicitly refills, e.g. Brant). Two consequences are pinned here:

* Within a turn the tray shrinks as cards are drawn, and every subsequent draw
  is offered only the cards still face-up (plus the deck) — no phantom refill.
* Between turns the tray is topped back up, so every turn opens on a full tray.

The mid-turn behaviour is exercised through the public ``actions`` free
functions; the end-of-turn refill is verified as a live-game invariant via
``Engine.play_one_game``.
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import agents, cards, decisions, engine, state  # noqa: E402
from wingspan.engine import actions  # noqa: E402
from wingspan.engine import core as engine_core  # noqa: E402


def _make_engine(seed: int = 0) -> engine.Engine:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    gs = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(gs)


def _non_brown_bird() -> cards.Bird:
    """A bird whose row-power activation is a no-op, so filling the wetland row
    with it leaves only the draw/refill logic under test."""
    birds, _, _ = cards.load_all()
    return next(bird for bird in birds if bird.color != cards.PowerColor.BROWN)


def _fill_row(
    board: state.Board, habitat: cards.Habitat, count: int, bird: cards.Bird
) -> None:
    board[habitat] = [state.PlayedBird(bird=bird) for _ in range(count)]


def _tray_picking_agent(
    sink: list[decisions.Decision[typing.Any]],
) -> engine_core.Agent:
    """Records every decision and, for a draw-source pick, always takes the
    first remaining tray slot (falling back to the deck when the tray is
    empty). Any other decision takes its first non-skip option."""

    def agent[C: decisions.Choice](
        _eng: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        for choice in decision.choices:
            if isinstance(choice, decisions.DrawSourceChoice) and (
                choice.source == "tray"
            ):
                return choice
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    return agent


# ---------------------------------------------------------------------------
# Mid-turn: drawing from the tray does not refill it


def test_draw_action_does_not_refill_tray_mid_turn():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.me()
    # Two wetland birds -> draw 2 cards, on an even (non-trade) slot.
    _fill_row(player.board, cards.Habitat.WETLAND, 2, _non_brown_bird())
    deck_before = len(eng.state.bird_deck)

    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_draw_cards(eng, _tray_picking_agent(sink))

    # Both cards came out of the tray and it was left short, not topped up.
    assert len(eng.state.tray) == state.TRAY_SIZE - 2
    assert len(player.hand) == 2
    # No card was pulled from the deck (no refill happened during the action).
    assert len(eng.state.bird_deck) == deck_before


def test_draw_options_shrink_as_tray_empties():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.me()
    _fill_row(player.board, cards.Habitat.WETLAND, 2, _non_brown_bird())
    tray_before = [bird.name for bird in eng.state.tray]

    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_draw_cards(eng, _tray_picking_agent(sink))

    source_decisions = [
        decision
        for decision in sink
        if isinstance(decision, decisions.DrawCardsPickSourceDecision)
    ]
    assert len(source_decisions) == 2
    # First draw sees all three face-up cards; the second sees only the two
    # that remain after the first was taken.
    tray_counts = [
        sum(1 for choice in decision.choices if choice.source == "tray")
        for decision in source_decisions
    ]
    assert tray_counts == [3, 2]
    # The card taken first is no longer an option on the second draw.
    second_tray_names = [
        choice.bird.name
        for choice in source_decisions[1].choices
        if choice.source == "tray" and choice.bird is not None
    ]
    assert tray_before[0] not in second_tray_names
    # The deck is still an option at every step.
    assert all(
        any(choice.source == "deck" for choice in decision.choices)
        for decision in source_decisions
    )


def test_empty_tray_offers_only_the_deck():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.me()
    # Simulate a turn that has already drained every tray slot.
    eng.state.tray = []
    deck_before = len(eng.state.bird_deck)

    sink: list[decisions.Decision[typing.Any]] = []
    actions.draw_one_card(eng, _tray_picking_agent(sink), player)

    source_decisions = [
        decision
        for decision in sink
        if isinstance(decision, decisions.DrawCardsPickSourceDecision)
    ]
    assert len(source_decisions) == 1
    assert [choice.source for choice in source_decisions[0].choices] == ["deck"]
    # The blind deck draw still lands in hand.
    assert len(player.hand) == 1
    assert len(eng.state.bird_deck) == deck_before - 1


# ---------------------------------------------------------------------------
# End of turn: the tray is refilled back to full


def _recording_turn_agent(
    rng: random.Random,
    main_tray_snaps: list[tuple[int, bool]],
    tray_draws: list[decisions.DrawSourceChoice],
) -> engine_core.Agent:
    """Plays randomly but records, at each of its own main-action decisions, the
    tray size and whether any cards remain to be drawn — plus every tray draw it
    actually makes, so the test can confirm the refill is genuinely exercised."""
    inner = agents.random_agent(rng)

    def agent[C: decisions.Choice](
        eng: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        choice = inner(eng, decision)
        if isinstance(decision, decisions.MainActionDecision):
            has_cards = bool(
                eng.state.tray or eng.state.bird_deck or eng.state.bird_discard
            )
            main_tray_snaps.append((len(eng.state.tray), has_cards))
        if isinstance(choice, decisions.DrawSourceChoice) and choice.source == "tray":
            tray_draws.append(choice)
        return choice

    return agent


def test_tray_is_full_at_the_start_of_every_turn():
    eng, *_ = engine.Engine.create(seed=123)
    rng = random.Random(123)
    main_tray_snaps: list[tuple[int, bool]] = []
    tray_draws: list[decisions.DrawSourceChoice] = []
    engine.Engine.play_one_game(
        eng.state,
        (
            _recording_turn_agent(rng, main_tray_snaps, tray_draws),
            agents.random_agent(rng),
        ),
    )

    assert main_tray_snaps, "expected the recorded player to take some turns"
    # Each turn opens on a full tray (the previous turn refilled it) unless the
    # deck and discard are both exhausted, in which case it stays short.
    for tray_len, cards_remain in main_tray_snaps:
        assert tray_len == state.TRAY_SIZE or not cards_remain
    # And the refill is not vacuous: at least one turn actually drew from the
    # tray, leaving a slot that had to be topped back up.
    assert tray_draws, "expected at least one tray draw to exercise the refill"
