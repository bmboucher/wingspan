"""Tests for the per-opponent known-hand state stripes (v1.8).

Covers ``state.Player.known_hand`` (maintained by ``engine.ledger``, Stage 1)
and the ``known_hand_opp{k}`` identity multi-hot stripes populated by
``encode.encode_state`` (Stage 2): a 180-wide multi-hot per opponent, appended
at the tail of the multi-hot region (after ``hand_playable_eggs_me``, before
``decision_type``), plus the ``encode.n_extra_hand_multihots`` accessor that
counts them.
"""

from __future__ import annotations

import numpy as np

from wingspan import cards, decisions, encode, engine
from wingspan.encode import layout

# ---------------------------------------------------------------------------
# Helpers


def _main_action_decision(player_id: int = 0) -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=player_id,
        prompt="action",
        choices=[
            decisions.MainActionChoice(
                label="food", action=decisions.MainAction.GAIN_FOOD
            )
        ],
    )


def _known_hand_stripe(vec: np.ndarray) -> np.ndarray:
    """The nearest opponent's ``known_hand_opp`` stripe, at N=2's frozen
    ``[1109, 1289)`` span (``encode.STATE_KNOWN_HAND_OPP_OFFSET`` /
    ``_DIM``)."""
    return vec[
        encode.STATE_KNOWN_HAND_OPP_OFFSET : encode.STATE_KNOWN_HAND_OPP_OFFSET
        + encode.STATE_KNOWN_HAND_OPP_DIM
    ]


# ---------------------------------------------------------------------------
# Stripe contents at N=2


def test_known_hand_opp_stripe_is_zero_when_nothing_is_known():
    """Freshly-created players start with an empty ``known_hand``, so the
    stripe is all zeros — pinned at N=2's frozen ``[1109, 1289)`` span."""
    eng, *_ = engine.Engine.create(seed=1)
    vec = encode.encode_state(eng.state, _main_action_decision())
    assert encode.STATE_KNOWN_HAND_OPP_OFFSET == 1109
    assert encode.STATE_KNOWN_HAND_OPP_DIM == 180
    stripe = vec[1109:1289]
    assert np.count_nonzero(stripe) == 0
    assert np.array_equal(stripe, _known_hand_stripe(vec))


def test_known_hand_opp_marks_exactly_the_known_bird():
    """A single known card sets exactly its own index and nothing else."""
    eng, birds, *_ = engine.Engine.create(seed=2)
    opponent = eng.state.players[1]
    bird = birds[5]
    opponent.known_hand = [bird]

    vec = encode.encode_state(eng.state, _main_action_decision(player_id=0))
    stripe = _known_hand_stripe(vec)
    assert np.flatnonzero(stripe).tolist() == [cards.bird_index(bird)]
    assert stripe[cards.bird_index(bird)] == 1.0


def test_known_hand_opp_marks_several_known_birds():
    """Several known cards set several bits, one per distinct bird."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    opponent = eng.state.players[1]
    known = [birds[1], birds[9], birds[17]]
    opponent.known_hand = list(known)

    vec = encode.encode_state(eng.state, _main_action_decision(player_id=0))
    stripe = _known_hand_stripe(vec)
    assert np.count_nonzero(stripe) == len(known)
    for bird in known:
        assert stripe[cards.bird_index(bird)] == 1.0


# ---------------------------------------------------------------------------
# POV rules: own hand never leaks; opponent's known_hand rotates with POV


def test_pov_players_own_known_hand_never_affects_the_vector():
    """The deciding player's own ``known_hand`` is not a real game concept —
    setting it must not change their own encoded state at all."""
    eng, birds, *_ = engine.Engine.create(seed=4)
    decision = _main_action_decision(player_id=0)
    baseline = encode.encode_state(eng.state, decision)

    eng.state.players[0].known_hand = [birds[2], birds[3]]
    with_own_known_hand = encode.encode_state(eng.state, decision)

    assert np.array_equal(baseline, with_own_known_hand)


def test_pov_symmetry_reads_the_other_seats_known_hand():
    """Encoding from either seat's POV reads the *other* seat's known_hand —
    never its own — so the stripe contents rotate with the decider."""
    eng, birds, *_ = engine.Engine.create(seed=5)
    p0, p1 = eng.state.players
    p0.known_hand = [birds[10]]
    p1.known_hand = [birds[20]]

    vec_from_p0 = encode.encode_state(eng.state, _main_action_decision(player_id=0))
    stripe_from_p0 = _known_hand_stripe(vec_from_p0)
    assert np.flatnonzero(stripe_from_p0).tolist() == [cards.bird_index(birds[20])]

    vec_from_p1 = encode.encode_state(eng.state, _main_action_decision(player_id=1))
    stripe_from_p1 = _known_hand_stripe(vec_from_p1)
    assert np.flatnonzero(stripe_from_p1).tolist() == [cards.bird_index(birds[10])]


# ---------------------------------------------------------------------------
# N=3: one block per opponent, adjacent, in opponents_clockwise order


def test_n3_known_hand_blocks_are_adjacent_and_clockwise():
    eng, birds, *_ = engine.Engine.create(seed=6, num_players=3)
    gs = eng.state
    p0, p1, p2 = gs.players
    p1.known_hand = [birds[30]]
    p2.known_hand = [birds[31]]

    spec3 = encode.EncodingSpec(num_players=3)
    cont = layout.state_cont_layout(spec3)
    off1 = cont.offset_of("known_hand_opp")
    size1 = cont.size_of("known_hand_opp")
    off2 = cont.offset_of("known_hand_opp2")
    size2 = cont.size_of("known_hand_opp2")
    assert off2 == off1 + size1  # immediately adjacent
    assert size1 == size2 == encode.HAND_MULTIHOT_DIM

    vec = encode.encode_state(gs, _main_action_decision(player_id=p0.id), spec=spec3)
    # opponents_clockwise(p0.id) == [p1, p2], so opp1's block (known_hand_opp)
    # carries p1's known hand and opp2's block (known_hand_opp2) carries p2's.
    assert gs.opponents_clockwise(p0.id) == [p1, p2]
    assert vec[off1 + cards.bird_index(birds[30])] == 1.0
    assert np.count_nonzero(vec[off1 : off1 + size1]) == 1
    assert vec[off2 + cards.bird_index(birds[31])] == 1.0
    assert np.count_nonzero(vec[off2 : off2 + size2]) == 1


# ---------------------------------------------------------------------------
# n_extra_hand_multihots: the playability pair plus one known-hand stripe
# per opponent


def test_n_extra_hand_multihots_grows_with_opponent_count():
    assert layout.n_extra_hand_multihots(layout.EncodingSpec(num_players=2)) == 3
    assert layout.n_extra_hand_multihots(layout.EncodingSpec(num_players=3)) == 4
    assert layout.n_extra_hand_multihots(layout.EncodingSpec(num_players=4)) == 5
