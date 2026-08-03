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
        (40.0, 30.0), winner=0, end_game_bonus=3.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (40.0 - 30.0 + 3.0, 30.0 - 40.0 - 3.0)


def test_terminal_values_own_score_basis_bonus_only_to_winner():
    values = returns.terminal_values(
        (40.0, 30.0), winner=0, end_game_bonus=3.0, basis=config.RewardBasis.OWN_SCORE
    )
    assert values == (40.0 + 3.0, 30.0 + 0.0)


def test_terminal_values_tie_has_no_bonus():
    values = returns.terminal_values(
        (30.0, 30.0), winner=-1, end_game_bonus=5.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (0.0, 0.0)


# --- terminal_values: N=2 legacy-equality grid ------------------------------


def _legacy_terminal_values(
    score_0: float,
    score_1: float,
    winner: int,
    end_game_bonus: float,
    basis: config.RewardBasis,
) -> tuple[float, float]:
    """The OLD (pre-N-player) seat-0-POV 2-seat terminal-value formula,
    inlined here so the new N-generic ``terminal_values`` can be pinned
    against it byte-for-byte, independent of the live implementation."""
    if basis is config.RewardBasis.OWN_SCORE:
        bonus_0 = end_game_bonus if winner == 0 else 0.0
        bonus_1 = end_game_bonus if winner == 1 else 0.0
        return (score_0 + bonus_0, score_1 + bonus_1)
    if winner == 0:
        bonus_0 = end_game_bonus
    elif winner == 1:
        bonus_0 = -end_game_bonus
    else:
        bonus_0 = 0.0
    return (score_0 - score_1 + bonus_0, score_1 - score_0 - bonus_0)


@pytest.mark.parametrize(
    "score_0, score_1",
    [(40.0, 30.0), (30.0, 40.0), (25.0, 25.0), (0.0, 0.0), (-5.0, 10.0)],
)
@pytest.mark.parametrize("winner", [0, 1, -1])
@pytest.mark.parametrize(
    "basis", [config.RewardBasis.MARGIN, config.RewardBasis.OWN_SCORE]
)
@pytest.mark.parametrize("end_game_bonus", [0.0, 3.5])
def test_terminal_values_n2_matches_legacy(
    score_0: float,
    score_1: float,
    winner: int,
    basis: config.RewardBasis,
    end_game_bonus: float,
):
    """At 2 players ``terminal_values`` reduces EXACTLY to the old seat-0-POV
    formula — same seats, same signs, same bytes — across a grid of score
    pairs (including ties), every winner value, and both reward bases."""
    got = returns.terminal_values((score_0, score_1), winner, end_game_bonus, basis)
    want = _legacy_terminal_values(score_0, score_1, winner, end_game_bonus, basis)
    assert got == want


# --- terminal_values: N=3 semantics -----------------------------------------


def test_terminal_values_n3_margin_vs_best_other():
    """MARGIN basis at 3 players: each seat's value is its own score minus
    the *best* other seat's score — not an average, and not a fixed
    neighbor seat."""
    scores = (50.0, 30.0, 45.0)  # seat 1 is weakest; must never be "the opponent"
    values = returns.terminal_values(
        scores, winner=0, end_game_bonus=0.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (
        50.0 - max(30.0, 45.0),
        30.0 - max(50.0, 45.0),
        45.0 - max(50.0, 30.0),
    )
    assert values == (5.0, -20.0, -5.0)


def test_terminal_values_n3_bonus_each_nonwinner():
    """A sole winner gets ``+bonus``; EVERY other seat gets ``-bonus`` in
    full (not a bonus split across the losers)."""
    scores = (10.0, 10.0, 10.0)  # every seat's own-minus-best-other term is 0
    values = returns.terminal_values(
        scores, winner=1, end_game_bonus=4.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (-4.0, 4.0, -4.0)


def test_terminal_values_shared_tie_zero_bonus():
    """``winner == -1`` (a genuine shared victory) adds no bonus to anyone,
    even though two seats are tied for the top score."""
    scores = (20.0, 20.0, 5.0)
    values = returns.terminal_values(
        scores, winner=-1, end_game_bonus=6.0, basis=config.RewardBasis.MARGIN
    )
    assert values == (0.0, 0.0, 5.0 - 20.0)


def test_terminal_values_own_score_basis_generalizes_to_n_seats():
    """OWN_SCORE basis is a per-seat function of that seat alone: every seat's
    own score, plus the bonus only for the sole winner."""
    scores = (12.0, 9.0, 15.0)
    values = returns.terminal_values(
        scores, winner=2, end_game_bonus=2.0, basis=config.RewardBasis.OWN_SCORE
    )
    assert values == (12.0, 9.0, 17.0)


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


def test_setup_return_n2_unchanged():
    """The ``opp_total`` -> ``best_other_total`` parameter rename (Stage 3) is
    pure signature rewording; behavior — and keyword-argument compatibility
    with the new name — is unchanged at 2 players."""
    training = _training()
    target = returns.setup_return(
        own_total=42.0,
        best_other_total=30.0,
        won=1,
        margin_checkpoints=[0.0, 5.0],
        score_checkpoints=[0.0, 20.0],
        decision_times=[1.0, 2.0],
        final_timestamp=10.0,
        training=training,
    )
    assert target == pytest.approx((42.0 - 30.0) / training.score_norm)


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
