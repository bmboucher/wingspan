"""Collection-layer N-player tests: self-play records every seat, the
vs-random bootstrap phase records only the net's seat, and
``running_margin``/``running_own_score`` read the best OTHER seat.

Engine-level N-player coverage (game completion, turn order,
``opponents_clockwise``) lives in ``test_multiplayer_engine.py``; this file
covers the training-collection layer (``wingspan.training.collect``) added in
Stage 3 of the N-player plan.
"""

from __future__ import annotations

import random

import pytest

import helpers
from wingspan import agents, architecture, encode, model
from wingspan.training import collect

torch = pytest.importorskip("torch")

# A tiny 3-seat net, mirroring the small-arch convention used by
# ``test_mp_collect.py`` / ``test_bootstrap_opponent.py`` so every forward
# pass stays cheap.
_SMALL_ARCH = architecture.ModelArchitecture(
    num_players=3,
    trunk_layers=(32, 32),
    choice_layers=(32, 32),
    card_embed_dim=16,
    card_encoder_layers=(32,),
)
_SMALL_SPEC = encode.spec_for(True, 3)


def _net() -> model.PolicyValueNet:
    return model.PolicyValueNet(arch=_SMALL_ARCH, spec=_SMALL_SPEC)


def test_n3_self_play_records_every_seat():
    """Self-play (``opponent_agent`` omitted) records decisions for all three
    seats and carries a 3-length breakdown tuple + scores tuple."""
    net = _net()
    device = torch.device("cpu")
    rng = random.Random(0)
    record = collect.play_game(net, device, rng, seed=1, num_players=3)
    assert {step.player_id for step in record.steps} == {0, 1, 2}
    assert len(record.breakdowns) == 3
    assert len(record.scores) == 3
    assert record.winner in (-1, 0, 1, 2)


def test_n3_bootstrap_records_only_net_seat():
    """With an ``opponent_agent`` (the vs-random bootstrap phase), the net
    plays seat 0 and the opponent plays every other seat; only the net's
    decisions are recorded, even though every seat still finishes with a
    score breakdown."""
    net = _net()
    device = torch.device("cpu")
    rng = random.Random(0)
    opponent = agents.random_agent(random.Random(99))
    record = collect.play_game(
        net, device, rng, seed=2, opponent_agent=opponent, num_players=3
    )
    assert {step.player_id for step in record.steps} == {0}
    assert len(record.breakdowns) == 3


def test_running_margin_uses_best_other_seat():
    """``running_margin``'s "opponent" term is the BEST other seat, not a
    fixed neighbor — proven by making the strongest opponent the seat that is
    NOT the deciding seat's clockwise neighbor."""
    eng = helpers.make_engine(num_players=3, seed=0)
    game = eng.state
    game.players[0].round_goal_points = 10
    game.players[1].round_goal_points = 3  # clockwise-next seat -- weakest
    game.players[2].round_goal_points = 25  # not clockwise-next -- strongest
    margin = collect.running_margin(game, 0)
    assert margin == 10 - 25


def test_running_margin_at_2p_is_own_minus_lone_opponent():
    """At 2 players 'best other' collapses to the single opponent — the exact
    legacy formula."""
    eng = helpers.make_engine(num_players=2, seed=0)
    game = eng.state
    game.players[0].round_goal_points = 7
    game.players[1].round_goal_points = 4
    assert collect.running_margin(game, 0) == 7 - 4
    assert collect.running_margin(game, 1) == 4 - 7
