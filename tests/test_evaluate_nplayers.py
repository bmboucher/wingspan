"""Evaluation-layer N-player tests: seat rotation, best-other margin, and
fractional shared-victory ``win_credit``.

``play_eval_game``'s two "engineered outcome" tests stub
``engine.Engine.play_one_game`` to set each player's final score / supply
food directly rather than playing a real game to a specific result — the
margin/win_credit computation is exercised for real, only the game *outcome*
is controlled.

2-player evaluation coverage lives in ``test_evaluate_core.py``; this file
covers the N-player generalization added in Stage 3 of the N-player plan.
"""

from __future__ import annotations

import pytest
import torch

from wingspan import architecture, cards, encode, engine, model, state
from wingspan.training import evaluate, metrics

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


def _stub_final_scores_and_food(
    monkeypatch: pytest.MonkeyPatch, scores: list[int], supply_food: list[int]
) -> None:
    """Replace real gameplay with a stub that stamps each player's
    ``final_score`` / one-food-of-every-kind-times-``supply_food[i]`` supply
    directly onto the (already-dealt) ``GameState`` — so the eval margin /
    win-credit computation runs for real against a fully controlled outcome."""

    def fake_play_one_game(
        gs: state.GameState,
        seat_agents: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        for player, score, food_each in zip(gs.players, scores, supply_food):
            player.final_score = score
            for food in cards.ALL_FOODS:
                player.food[food] = food_each

    monkeypatch.setattr(
        engine.Engine, "play_one_game", staticmethod(fake_play_one_game)
    )


def test_evaluate_vs_opponent_n3_plays_every_seat(monkeypatch: pytest.MonkeyPatch):
    """``evaluate_vs_opponent`` at 3 players plays ``num_players * n_pairs``
    games, rotating the net through every seat of each deal."""
    net = _net()
    device = torch.device("cpu")
    seen_seats: list[int] = []
    real_play_eval_game = evaluate.play_eval_game

    def spy(
        net_: model.PolicyValueNet,
        opponent_net: model.PolicyValueNet | None,
        device_: torch.device,
        seed: int,
        net_seat: int,
        num_players: int = 2,
        split_setup_bonus: bool = False,
        split_setup_food: bool = False,
        combine_gain_food: bool = False,
    ) -> metrics.EvalGameOutcome:
        seen_seats.append(net_seat)
        return real_play_eval_game(
            net_,
            opponent_net,
            device_,
            seed,
            net_seat,
            num_players=num_players,
            split_setup_bonus=split_setup_bonus,
            split_setup_food=split_setup_food,
            combine_gain_food=combine_gain_food,
        )

    monkeypatch.setattr(evaluate, "play_eval_game", spy)
    result = evaluate.evaluate_vs_opponent(
        net, None, device, n_pairs=2, seed=10, num_players=3
    )
    assert result.n_games == 6
    assert seen_seats == [0, 1, 2, 0, 1, 2]


def test_play_eval_game_margin_is_net_minus_best_other(
    monkeypatch: pytest.MonkeyPatch,
):
    """``margin`` is the net seat's score minus the *best* other seat's score
    — proven with a deal where the best other is NOT the seat adjacent to the
    net (seat 1's 45, not seat 2's 10), so a "just check the neighbor" bug
    would fail this."""
    net = _net()
    device = torch.device("cpu")
    _stub_final_scores_and_food(monkeypatch, scores=[20, 45, 10], supply_food=[1, 1, 1])

    outcome = evaluate.play_eval_game(
        net, None, device, seed=0, net_seat=0, num_players=3
    )
    assert outcome.margin == 20 - 45
    assert outcome.win_credit == 0.0  # seat 1 is the sole winner, not seat 0


def test_play_eval_game_shared_tie_yields_fractional_win_credit(
    monkeypatch: pytest.MonkeyPatch,
):
    """A genuine 3-way shared victory — equal scores AND equal unused supply
    food, so ``scoring.winners`` finds no tiebreak — yields ``win_credit =
    1/3`` for the net's seat, not the legacy tie-counts-as-half convention
    (which only ever applied at 2 players anyway)."""
    net = _net()
    device = torch.device("cpu")
    _stub_final_scores_and_food(monkeypatch, scores=[40, 40, 40], supply_food=[2, 2, 2])

    outcome = evaluate.play_eval_game(
        net, None, device, seed=0, net_seat=0, num_players=3
    )
    assert outcome.margin == 0.0
    assert outcome.win_credit == pytest.approx(1.0 / 3.0)


def test_play_eval_game_score_tie_broken_by_food_is_not_fractional(
    monkeypatch: pytest.MonkeyPatch,
):
    """The accepted 2P-engine-drift tiebreak generalizes: a raw SCORE tie
    that is broken by unused supply food resolves to a sole winner, so
    ``margin == 0`` does not automatically mean fractional credit."""
    net = _net()
    device = torch.device("cpu")
    _stub_final_scores_and_food(
        monkeypatch, scores=[40, 40, 40], supply_food=[2, 5, 2]
    )  # seat 1 has strictly more unused supply food -> sole winner

    net_at_seat_1 = evaluate.play_eval_game(
        net, None, device, seed=0, net_seat=1, num_players=3
    )
    assert net_at_seat_1.margin == 0.0  # scores still tie
    assert net_at_seat_1.win_credit == 1.0  # but seat 1 alone wins the tiebreak

    net_at_seat_0 = evaluate.play_eval_game(
        net, None, device, seed=0, net_seat=0, num_players=3
    )
    assert net_at_seat_0.win_credit == 0.0


def test_summarize_eval_sums_fractional_credits():
    """``summarize_eval``'s win rate is the mean win_credit across games,
    fractional shared-victory credits included."""
    outcomes = [
        metrics.EvalGameOutcome(margin=5.0, win_credit=1.0),
        metrics.EvalGameOutcome(margin=-10.0, win_credit=0.0),
        metrics.EvalGameOutcome(margin=0.0, win_credit=1.0 / 3.0),
    ]
    result = evaluate.summarize_eval(outcomes, opponent_generation=2)
    assert result.n_games == 3
    assert result.win_rate == pytest.approx((1.0 + 0.0 + 1.0 / 3.0) / 3.0)
    assert result.mean_margin == pytest.approx((5.0 - 10.0 + 0.0) / 3.0)
    assert result.opponent_generation == 2
