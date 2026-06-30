"""Tests for the shared training-return kernel (``wingspan.training.returns``).

The same kernel feeds the in-game learner (per-step returns) and the setup
learner (the return at the seat's ``t=0`` setup decision). These tests pin
``terminal_values`` and ``setup_return``'s consistency with the in-game return
across reward mode / discount / basis / bonus — the property that makes the
setup critic ``V(s)`` train on the *same* target as the main learner.
"""

from __future__ import annotations

import pytest

from wingspan.training import config, returns, timestamps


def _training(**overrides: object) -> config.TrainingConfig:
    return config.TrainingConfig.model_validate(overrides)


# --- terminal_values -------------------------------------------------------


def test_terminal_values_margin_basis_with_bonus():
    values = returns.terminal_values(
        40.0, 30.0, winner=0, end_game_bonus=3.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (40.0 - 30.0 + 3.0, 30.0 - 40.0 - 3.0)


def test_terminal_values_own_score_basis_bonus_only_to_winner():
    values = returns.terminal_values(
        40.0, 30.0, winner=0, end_game_bonus=3.0, basis=config.RewardBasis.OWN_SCORE
    )
    assert values == (40.0 + 3.0, 30.0 + 0.0)


def test_terminal_values_tie_has_no_bonus():
    values = returns.terminal_values(
        30.0, 30.0, winner=-1, end_game_bonus=5.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (0.0, 0.0)


# --- setup_return: consistency with the in-game return at t=0 --------------


def test_setup_return_default_config_equals_margin_over_score_norm():
    """At the default config (TERMINAL_MARGIN, MARGIN, gamma=1, bonus=0) the
    setup target is exactly the legacy ``margin / score_norm`` — a no-op."""
    training = _training()
    own, opp = 42.0, 30.0
    target = returns.setup_return(
        own,
        opp,
        won=1,
        margin_checkpoints=[0.0, 5.0],
        score_checkpoints=[0.0, 20.0],
        decision_times=[1.0, 2.0],
        final_timestamp=10.0,
        training=training,
    )
    assert target == pytest.approx((own - opp) / training.score_norm)


def test_setup_return_folds_end_game_bonus():
    training = _training(end_game_bonus=4.0)
    target = returns.setup_return(
        40.0,
        30.0,
        won=1,
        margin_checkpoints=[],
        score_checkpoints=[],
        decision_times=[],
        final_timestamp=5.0,
        training=training,
    )
    assert target == pytest.approx((40.0 - 30.0 + 4.0) / training.score_norm)


def test_setup_return_honors_own_score_basis():
    training = _training(reward_basis=config.RewardBasis.OWN_SCORE, end_game_bonus=2.0)
    target = returns.setup_return(
        40.0,
        30.0,
        won=1,
        margin_checkpoints=[],
        score_checkpoints=[],
        decision_times=[],
        final_timestamp=5.0,
        training=training,
    )
    assert target == pytest.approx((40.0 + 2.0) / training.score_norm)


def test_setup_return_decision_delta_gamma1_telescopes_to_terminal():
    training = _training(
        reward_mode=config.RewardMode.DECISION_DELTA, reward_discount=1.0
    )
    own, opp = 42.0, 30.0
    target = returns.setup_return(
        own,
        opp,
        won=1,
        margin_checkpoints=[0.0, 5.0, 8.0],
        score_checkpoints=[0.0, 5.0, 8.0],
        decision_times=[1.0, 2.0, 3.0],
        final_timestamp=10.0,
        training=training,
    )
    # gamma=1 telescopes to (terminal - v0) / score_norm = (own - opp) / score_norm.
    assert target == pytest.approx((own - opp) / training.score_norm)


def test_setup_return_decision_delta_matches_in_game_kernel():
    """Under DECISION_DELTA with gamma<1 the setup target equals the in-game
    discounted-return kernel evaluated at the t=0 anchor — proving the two
    learners share one return definition."""
    training = _training(
        reward_mode=config.RewardMode.DECISION_DELTA, reward_discount=0.9
    )
    own, opp = 42.0, 30.0
    margin_checkpoints = [3.0, 7.0]
    decision_times = [1.0, 2.0]
    final_timestamp = 5.0
    terminal = own - opp  # MARGIN basis, no bonus

    target = returns.setup_return(
        own,
        opp,
        won=1,
        margin_checkpoints=margin_checkpoints,
        score_checkpoints=[],
        decision_times=decision_times,
        final_timestamp=final_timestamp,
        training=training,
    )
    expected = (
        timestamps.discounted_future_returns(
            [0.0, *margin_checkpoints, terminal],
            [timestamps.SETUP_KEEP_TIMESTAMP, *decision_times, final_timestamp],
            0.9,
        )[0]
        / training.score_norm
    )
    assert target == pytest.approx(expected)
