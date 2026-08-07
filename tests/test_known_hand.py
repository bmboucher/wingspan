"""Public hand-knowledge (``Player.known_hand``) tests — Stage 1.

Covers the ledger bookkeeping documented on ``wingspan.engine.ledger``: a card
becomes known the moment it arrives face-up (a tray draw, a face-up deck draw,
a public hand-to-hand pass) and is forgotten wholesale the instant *any* card
leaves the hand face-down (a discard or a tuck) — observers can't tell which
known card left, so the whole set is invalidated rather than guessed at.

Also covers the two ledger bypasses fixed alongside the bookkeeping — Brant's
bulk tray take (``tray_trade._h_draw_from_tray_all``) and the American
Oystercatcher draft's arrival sides (``drafting._h_draw_n_plus_one_draft``) —
and a full-game invariant that ``known_hand`` never runs ahead of ``hand``.

Stage 1 is engine-only: no state/choice encoder reads ``known_hand`` yet, so
none of these tests touch encoding, and the golden encoding fixture
(``tests/test_encoding_golden_n2.py``) is untouched by this file.
"""

from __future__ import annotations

import collections
import random
import typing

import stub_agents
from wingspan import agents, cards, decisions, engine, state
from wingspan.engine import ledger, powers, reactors
from wingspan.gamelog import models as gamelog_models
from wingspan.gamelog import recorder as gamelog_recorder


def _bird_named(name: str, birds: list[cards.Bird]) -> cards.Bird:
    """Look up one loaded ``Bird`` by name."""
    return next(bird for bird in birds if bird.name == name)


def _all_effects(
    tree: gamelog_models.GameEventTree,
) -> list[gamelog_models.AnyEffect]:
    """Every recorded effect in ``tree``, in chronological (DFS) order.

    Mirrors the walker in ``tests/test_gamelog_ledger.py``; duplicated locally
    so this file has no cross-test-module import."""
    found: list[gamelog_models.AnyEffect] = []

    def walk(events: typing.Sequence[gamelog_models.GameEvent]) -> None:
        for event in events:
            for sub in event.sub_events:
                if isinstance(sub, gamelog_models.Effect):
                    found.append(sub)
            walk(event.children)

    for phase in tree.phases:
        walk(phase.events)
    return found


def _fresh_engine(seed: int, num_players: int = 2) -> engine.Engine:
    """A bare ``Engine`` over a freshly-dealt game, seat 0 active."""
    eng, *_ = engine.Engine.create(seed=seed, num_players=num_players)
    eng.state.current_player = 0
    return eng


# ---------------------------------------------------------------------------
# Ledger primitives: which departures/arrivals touch known_hand


def test_tray_draw_marks_known_deck_draw_does_not():
    """A tray draw joins ``known_hand``; a plain deck draw leaves it alone."""
    eng = _fresh_engine(seed=1)
    player = eng.state.me()
    player.hand = []
    player.known_hand = []

    tray_index = next(
        index for index, bird in enumerate(eng.state.tray) if bird is not None
    )
    from_tray = ledger.take_from_tray(eng, player, tray_index)
    assert from_tray is not None
    assert player.known_hand == [from_tray]

    from_deck = ledger.draw_from_deck(eng, player)
    assert from_deck is not None
    assert [bird.name for bird in player.hand] == [from_tray.name, from_deck.name]
    assert player.known_hand == [from_tray]  # unchanged by the hidden deck draw


def test_draw_from_deck_face_up_marks_known():
    """``draw_from_deck(..., face_up=True)`` behaves like a reveal for
    knowledge purposes even though the card came off the deck."""
    eng = _fresh_engine(seed=2)
    player = eng.state.me()
    player.hand = []
    player.known_hand = []

    drawn = ledger.draw_from_deck(eng, player, face_up=True)
    assert drawn is not None
    assert player.known_hand == [drawn]


def test_place_bird_forgets_exactly_the_played_card():
    """Playing a bird forgets only that card; other known cards survive."""
    eng = _fresh_engine(seed=3)
    birds, _, _ = cards.load_all()
    player = eng.state.me()
    played_card, other_known, other_unknown = birds[0], birds[1], birds[2]
    player.hand = [played_card, other_known, other_unknown]
    player.known_hand = [played_card, other_known]

    ledger.place_bird(eng, player, played_card, played_card.habitats[0])

    assert played_card not in player.hand
    assert [bird.name for bird in player.known_hand] == [other_known.name]


def test_discard_from_hand_clears_entire_known_set():
    """A face-down discard wipes ``known_hand`` wholesale, not just the
    discarded card."""
    eng = _fresh_engine(seed=4)
    birds, _, _ = cards.load_all()
    player = eng.state.me()
    card_a, card_b, card_c = birds[0], birds[1], birds[2]
    player.hand = [card_a, card_b, card_c]
    player.known_hand = [card_a, card_b]

    ledger.discard_from_hand(eng, player, card_a)

    assert player.known_hand == []


def test_tuck_from_hand_clears_entire_known_set():
    """A face-down tuck wipes ``known_hand`` wholesale, not just the tucked
    card."""
    eng = _fresh_engine(seed=5)
    birds, _, _ = cards.load_all()
    player = eng.state.me()
    card_a, card_b, card_c, host_card = birds[0], birds[1], birds[2], birds[3]
    host = state.PlayedBird(bird=host_card)
    player.board[host_card.habitats[0]] = [host]
    player.hand = [card_a, card_b, card_c]
    player.known_hand = [card_a, card_b]

    ledger.tuck_from_hand(eng, player, card_a, host)

    assert player.known_hand == []


def test_pass_card_moves_knowledge_sender_to_recipient():
    """``pass_card`` has no engine call site today (this test also protects
    the coverage ratchet), but is the public-transfer primitive: knowledge
    moves with the card, single-card precision on both ends."""
    eng = _fresh_engine(seed=6)
    birds, _, _ = cards.load_all()
    sender, recipient = eng.state.players
    passed_card, sender_other = birds[0], birds[1]
    sender.hand = [passed_card, sender_other]
    sender.known_hand = [passed_card, sender_other]
    recipient.hand = []
    recipient.known_hand = []

    ledger.pass_card(eng, sender, recipient, passed_card)

    assert [bird.name for bird in sender.known_hand] == [sender_other.name]
    assert [bird.name for bird in recipient.known_hand] == [passed_card.name]
    assert recipient.hand == [passed_card]


def test_receive_passed_cards_extends_hand_marks_known_records_no_effect():
    """The arrival counterpart to ``take_into_pile``: extends the hand, marks
    every received card known, and records no gamelog effect of its own."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(7), birds, bonuses, goals)
    rec = gamelog_recorder.EventRecorder(probes=(None, None))
    eng = engine.Engine(gs, event_recorder=rec)
    rec.begin_game()
    to_player = gs.me()
    to_player.hand = []
    to_player.known_hand = []
    received = [birds[0], birds[1]]

    effects_before = len(_all_effects(rec.root))
    ledger.receive_passed_cards(eng, to_player, received)
    effects_after = len(_all_effects(rec.root))

    assert [bird.name for bird in to_player.hand] == [bird.name for bird in received]
    assert [bird.name for bird in to_player.known_hand] == [
        bird.name for bird in received
    ]
    assert effects_after == effects_before


# ---------------------------------------------------------------------------
# Ledger bypasses fixed alongside the bookkeeping


def test_brant_marks_all_tray_cards_known_and_ledgers_each_draw():
    """Brant (``DRAW_FROM_TRAY_ALL``) now goes through ``take_from_tray`` per
    slot: every previously-non-empty slot's card lands in hand *and*
    known_hand, and a ``DrawCardEffect(source=TRAY)`` is recorded for each —
    fixing the pre-existing gamelog hole where the bulk take bypassed the
    ledger entirely."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(8), birds, bonuses, goals)
    rec = gamelog_recorder.EventRecorder(probes=(None, None))
    eng = engine.Engine(gs, event_recorder=rec)
    rec.begin_game()
    gs.current_player = 0
    player = gs.me()
    player.hand = []
    player.known_hand = []
    original_tray = [bird for bird in gs.tray if bird is not None]
    pb = state.PlayedBird(bird=_bird_named("Brant", birds))

    powers.dispatch_power(
        eng, stub_agents.no_agent, player, pb, cards.Habitat.WETLAND, "play"
    )

    expected_names = [bird.name for bird in original_tray]
    assert [bird.name for bird in player.hand] == expected_names
    assert [bird.name for bird in player.known_hand] == expected_names

    tray_draw_effects = [
        effect
        for effect in _all_effects(rec.root)
        if isinstance(effect, gamelog_models.DrawCardEffect)
        and effect.source == gamelog_models.CardSource.TRAY
    ]
    assert len(tray_draw_effects) == len(original_tray)


def test_oystercatcher_draft_marks_all_arrived_cards_known_at_2p():
    """American Oystercatcher (``DRAW_N_PLUS_ONE_DRAFT``) at 2 players: every
    card that arrived via the draft — the opponent's kept card, and the
    active player's kept + returned cards — ends up known to its final
    holder."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(9), birds, bonuses, goals, num_players=2)
    eng = engine.Engine(gs)
    gs.current_player = 0
    p0, p1 = gs.players
    p0.hand = []
    p0.known_hand = []
    p1.hand = []
    p1.known_hand = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        # Always take the first offered choice (accept / first card).
        return decision.choices[0]

    eng.agents = [agent, agent]
    pb = state.PlayedBird(bird=_bird_named("American Oystercatcher", birds))

    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")

    assert len(p0.hand) == 2
    assert len(p1.hand) == 1
    assert sorted(bird.name for bird in p0.known_hand) == sorted(
        bird.name for bird in p0.hand
    )
    assert sorted(bird.name for bird in p1.known_hand) == sorted(
        bird.name for bird in p1.hand
    )


def test_horned_lark_off_turn_tuck_clears_reacting_players_own_known_set():
    """An off-turn pink tuck (Horned Lark, triggered by an opponent's play)
    still goes through ``tuck_from_hand`` and clears the REACTING player's own
    known set — the generic tuck test above covers the mechanism, this pins
    that it also fires correctly off-turn, attributed to the reactor."""
    eng = _fresh_engine(seed=10)
    birds, _, _ = cards.load_all()
    eng.agents = [stub_agents.no_agent, stub_agents.accept_agent]
    lark = state.PlayedBird(bird=_bird_named("Horned Lark", birds))
    reacting = eng.state.players[1]
    reacting.board[cards.Habitat.GRASSLAND] = [lark]
    known_card = _bird_named("Belted Kingfisher", birds)
    other_card = _bird_named("Wood Duck", birds)
    reacting.hand = [known_card, other_card]
    reacting.known_hand = [known_card, other_card]

    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )

    assert lark.tucked_cards == 1
    assert reacting.known_hand == []


# ---------------------------------------------------------------------------
# Full-game invariant


def _known_hand_is_subset_of_hand(live_engine: engine.Engine) -> None:
    """Assert every seat's ``known_hand`` is a sub-multiset of its ``hand``."""
    for player in live_engine.state.players:
        hand_counts = collections.Counter(bird.name for bird in player.hand)
        known_counts = collections.Counter(bird.name for bird in player.known_hand)
        for name, count in known_counts.items():
            assert count <= hand_counts[name], (
                f"player {player.id}: known_hand has {count} of {name!r}, "
                f"hand only has {hand_counts[name]}"
            )


def test_known_hand_never_outruns_hand_over_a_full_game():
    """Seeded 2-player random-agent game: at every decision point, and again
    at game end, every known card is actually in its owner's hand."""
    eng, *_ = engine.Engine.create(seed=11, num_players=2)
    rng = random.Random(11)
    base_agents = [agents.random_agent(rng), agents.random_agent(rng)]

    def _wrap(base: engine.Agent) -> engine.Agent:
        def wrapped[C: decisions.Choice](
            live_engine: engine.Engine, decision: decisions.Decision[C]
        ) -> C:
            _known_hand_is_subset_of_hand(live_engine)
            return base(live_engine, decision)

        return wrapped

    wrapped_agents = [_wrap(base) for base in base_agents]
    engine.Engine.play_one_game(eng.state, wrapped_agents)

    assert eng.state.game_over
    _known_hand_is_subset_of_hand(eng)


# ---------------------------------------------------------------------------
# Serialization


def test_known_hand_round_trips_through_model_dump():
    """A ``Player`` with a non-empty ``known_hand`` survives a
    dump/validate round trip unchanged."""
    birds, _, _ = cards.load_all()
    sample = birds[:3]
    player = state.Player(
        id=0, name="P0", hand=list(sample), known_hand=list(sample[:2])
    )

    restored = state.Player.model_validate(player.model_dump())

    assert restored == player


def test_known_hand_defaults_empty_when_absent_from_dump():
    """A dump from before ``known_hand`` existed (an older persisted log)
    validates to an empty list rather than failing."""
    birds, _, _ = cards.load_all()
    player = state.Player(id=0, name="P0", hand=list(birds[:2]))
    dumped = player.model_dump()
    del dumped["known_hand"]

    restored = state.Player.model_validate(dumped)

    assert restored.known_hand == []
