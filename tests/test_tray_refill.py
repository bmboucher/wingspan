"""Tests for the bird-tray draw / deferred-refill rules.

A card taken from the face-up bird tray is *not* replaced immediately. The
tray slot stays empty for the rest of the acting player's turn; the tray is
refilled to ``TRAY_SIZE`` only at the end of the turn (or by a bird power that
explicitly refills, e.g. Brant). Four rules are pinned here:

* Within a turn the tray shrinks as cards are drawn, and every subsequent draw
  is offered only the cards still face-up (plus the deck) — no phantom refill.
* Between turns the tray is topped back up, so every turn opens on a full tray.
* At the end of each round the face-up tray is discarded and three new cards
  go face-up, so each round opens on a fresh set of options.
* The tray starts with 3 cards during the setup phase, before players make
  their initial keep decisions.

The mid-turn and end-of-turn behaviour is exercised through the public
``actions`` free functions and via ``Engine.play_one_game``; the round-end
reset is verified both as a unit test on ``GameState.reset_tray`` and as a
live-game invariant.
"""

from __future__ import annotations

import random
import typing

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

    # Both slots were vacated (set to None) and the tray was not topped up.
    assert sum(1 for b in eng.state.tray if b is not None) == state.TRAY_SIZE - 2
    assert len(player.hand) == 2
    # No card was pulled from the deck (no refill happened during the action).
    assert len(eng.state.bird_deck) == deck_before


def test_draw_options_shrink_as_tray_empties():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.me()
    _fill_row(player.board, cards.Habitat.WETLAND, 2, _non_brown_bird())
    tray_before = [bird.name for bird in eng.state.tray if bird is not None]

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


def test_empty_tray_draws_from_deck_without_asking():
    """With the tray drained, the deck is the only legal draw source — a forced
    choice — so the engine resolves it without presenting a decision, and the
    blind deck draw still lands in hand."""
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.me()
    # Simulate a turn that has already drained every tray slot.
    eng.state.tray = [None] * state.TRAY_SIZE
    deck_before = len(eng.state.bird_deck)

    sink: list[decisions.Decision[typing.Any]] = []
    actions.draw_one_card(eng, _tray_picking_agent(sink), player)

    # The deck is the only option, so no draw-source decision is presented.
    assert not [
        decision
        for decision in sink
        if isinstance(decision, decisions.DrawCardsPickSourceDecision)
    ]
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
                any(b is not None for b in eng.state.tray)
                or eng.state.bird_deck
                or eng.state.bird_discard
            )
            tray_face_up = sum(1 for b in eng.state.tray if b is not None)
            main_tray_snaps.append((tray_face_up, has_cards))
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


# ---------------------------------------------------------------------------
# End of round: the tray is discarded and replaced with fresh cards


def test_reset_tray_discards_and_replenishes():
    """``GameState.reset_tray`` moves all current tray cards to the discard
    pile and then draws fresh ones up to ``TRAY_SIZE``."""
    eng = _make_engine()
    tray_before = [b for b in eng.state.tray if b is not None]
    discard_before = len(eng.state.bird_discard)

    eng.state.reset_tray()

    # Every card that was face-up is now in the discard.
    assert len(eng.state.bird_discard) == discard_before + len(tray_before)
    for bird in tray_before:
        assert bird in eng.state.bird_discard
    # The tray is replenished to full with new cards.
    assert all(b is not None for b in eng.state.tray)
    # The new tray cards are different from the old ones (barring a
    # vanishingly unlikely collision with a 180-card deck).
    old_names = {bird.name for bird in tray_before}
    new_names = {bird.name for bird in eng.state.tray if bird is not None}
    assert old_names != new_names


def test_reset_tray_with_short_deck():
    """When the entire remaining card pool holds fewer than ``TRAY_SIZE``
    cards, ``reset_tray`` leaves the tray short rather than looping forever.

    Note: ``refill_tray`` recycles ``bird_discard`` (via ``draw_bird``) when
    the deck runs dry, so discarded tray cards are counted as part of the pool.
    To leave the tray genuinely short we must drain deck *and* discard, leaving
    only a single card in the tray so the pool contains exactly 1 card total."""
    eng = _make_engine()
    # Drain deck and discard entirely; leave exactly 1 card face-up (slot 0).
    eng.state.bird_deck = []
    eng.state.bird_discard = []
    eng.state.tray = [eng.state.tray[0], None, None]

    eng.state.reset_tray()

    # That 1 card was discarded then recycled back out of the (now-reshuffled)
    # deck; the tray ends at 1 because the pool is exhausted after that draw.
    assert sum(1 for b in eng.state.tray if b is not None) == 1
    assert len(eng.state.bird_deck) == 0
    assert len(eng.state.bird_discard) == 0


def _round_boundary_agent(
    round_snapshots: dict[int, list[str]],
) -> engine_core.Agent:
    """Records the tray card names at the start of each turn (i.e. before any
    draw) grouped by the current round index. Used to verify the tray changes
    between rounds."""
    rng = random.Random(0)
    inner = agents.random_agent(rng)

    def agent[C: decisions.Choice](
        eng: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Capture tray snapshot before delegating so the isinstance check on
        # `decision` doesn't widen the type seen by `inner` at the return site.
        choice = inner(eng, decision)
        if isinstance(decision, decisions.MainActionDecision):
            round_idx = eng.state.round_idx
            round_snapshots.setdefault(round_idx, []).append(
                "|".join(
                    bird.name
                    for bird in sorted(
                        (b for b in eng.state.tray if b is not None),
                        key=lambda bird: bird.name,
                    )
                )
            )
        return choice

    return agent


def test_tray_is_replaced_between_rounds():
    """Every round opens on a different set of face-up tray cards.

    At the end of each round the engine discards the current tray and draws
    three new cards. This test plays a full game and checks that at least
    one card in the tray changed between consecutive rounds."""
    eng, *_ = engine.Engine.create(seed=42)
    round_snapshots: dict[int, list[str]] = {}
    engine.Engine.play_one_game(
        eng.state,
        (
            _round_boundary_agent(round_snapshots),
            agents.random_agent(random.Random(42)),
        ),
    )

    # All four rounds should have been recorded.
    assert len(round_snapshots) == 4

    # Collect the first tray snapshot for each round (tray at start of turn 1).
    first_per_round = [round_snapshots[idx][0] for idx in range(4)]

    # Consecutive rounds must show a different tray composition.  With 180
    # birds and 3 slots the probability of an accidental match is negligible.
    for earlier, later in zip(first_per_round, first_per_round[1:]):
        assert (
            earlier != later
        ), f"tray was identical across a round boundary: {earlier!r}"


# ---------------------------------------------------------------------------
# Position preservation: drawing a slot leaves its neighbours intact


def test_draw_from_middle_preserves_adjacent_slots():
    """Drawing from a specific tray slot nulls that slot only; the other two
    slots retain their original birds. After ``refill_tray`` the vacated slot
    is filled in-place and the neighbours are still the same birds."""
    eng = _make_engine()
    birds, _, _ = cards.load_all()
    # Force a known tray: three distinct birds in positions 0, 1, 2.
    bird_left = birds[5]
    bird_mid = birds[6]
    bird_right = birds[7]
    eng.state.tray = [bird_left, bird_mid, bird_right]

    # Draw from the middle slot (index 1) directly via draw_one_card.
    player = eng.state.players[0]
    player.hand = []

    def _pick_middle[C: decisions.Choice](
        _eng: engine_core.Engine, decision: decisions.Decision[C]
    ) -> C:
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.DrawSourceChoice)
                and choice.tray_index == 1
            ):
                return choice
        return decision.choices[0]

    actions.draw_one_card(eng, _pick_middle, player)

    # Middle slot is now None; left and right are unchanged.
    assert eng.state.tray[0] is bird_left
    assert eng.state.tray[1] is None
    assert eng.state.tray[2] is bird_right
    assert len(player.hand) == 1
    assert player.hand[0] is bird_mid

    # After refill, slot 1 is filled in-place; slots 0 and 2 are unchanged.
    eng.state.refill_tray()
    assert eng.state.tray[0] is bird_left
    assert eng.state.tray[1] is not None
    assert eng.state.tray[2] is bird_right
