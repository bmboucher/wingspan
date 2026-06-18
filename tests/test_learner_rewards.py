# pyright: reportPrivateUsage=false
"""Tests for the learner's REINFORCE reward computation (``learner._flatten``).

Two orthogonal axes determine the return for each step:

* ``reward_mode`` — *how* credit spreads: ``terminal_margin`` broadcasts the
  end-of-game value flat; ``decision_delta`` assigns per-decision changes
  discounted by γ^Δt of game-clock time.
* ``reward_basis`` — *what* value: ``margin`` = own − opponent (opposite signs
  per seat); ``own_score`` = player's absolute final score (both seats positive).

These tests pin the arithmetic for all four combinations against hand-computed
values so regressions in the discounting / telescoping logic are caught directly.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")  # learner imports torch at module load

from wingspan.training import collect, config, learner, metrics, steps


def _assert_close(actual: list[float], expected: list[float]) -> None:
    """Element-wise float comparison (pytest.approx is untyped under strict pyright)."""
    assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
    for got, want in zip(actual, expected):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9), f"{got} != {want}"


def _step(
    player_id: int,
    margin_before: float,
    score_before: float = 0.0,
    timestamp: float = 0.0,
) -> steps.Step:
    """A minimal recorded step — only ``player_id``, ``margin_before``,
    ``score_before``, and ``timestamp`` matter to return computation."""
    return steps.Step(
        state=np.zeros(1, dtype=np.float32),
        choices=np.zeros((1, 1), dtype=np.float32),
        chosen_idx=0,
        player_id=player_id,
        family_idx=0,
        margin_before=margin_before,
        score_before=score_before,
        timestamp=timestamp,
    )


def _breakdown(total: float) -> metrics.ScoreBreakdown:
    """A breakdown whose ``.total`` is ``total`` (parked entirely in ``birds``)."""
    return metrics.ScoreBreakdown(birds=total)


def _sample_record() -> collect.GameRecord:
    """A five-step interleaved game with known margins, own-scores, and clock times.

    Player 0 steps at times [1, 3, 5]; player 1 steps at times [2, 4].
    Finals: P0=12, P1=7; P0 wins. Per-seat own-score checkpoints:
      P0: [3, 6, 9] → terminal 12 (gaps Δv = 3, 3, 3)
      P1: [4, 5]    → terminal 7  (gaps Δv = 1, 2)
    Margin checkpoints (own − opp at each step): P0=[0, 3, 5]; P1=[0, -1]."""
    return collect.GameRecord(
        steps=[
            _step(player_id=0, margin_before=0.0, score_before=3.0, timestamp=1.0),
            _step(player_id=1, margin_before=0.0, score_before=4.0, timestamp=2.0),
            _step(player_id=0, margin_before=3.0, score_before=6.0, timestamp=3.0),
            _step(player_id=1, margin_before=-1.0, score_before=5.0, timestamp=4.0),
            _step(player_id=0, margin_before=5.0, score_before=9.0, timestamp=5.0),
        ],
        breakdowns=(_breakdown(12.0), _breakdown(7.0)),
        winner=0,
        seed=0,
        final_timestamp=6.0,
    )


def _cfg(
    reward_mode: config.RewardMode,
    reward_basis: config.RewardBasis = config.RewardBasis.MARGIN,
    discount: float = 1.0,
) -> config.RunConfig:
    """A config with ``score_norm=1`` so returns equal the raw (discounted) values."""
    return config.RunConfig(
        training=config.TrainingConfig(
            reward_mode=reward_mode,
            reward_basis=reward_basis,
            reward_discount=discount,
            score_norm=1.0,
        ),
    )


#### TERMINAL_MARGIN + MARGIN ####


def test_terminal_margin_broadcasts_pov_margin():
    """Every step gets its player's end-of-game margin (opposite signs)."""
    record = _sample_record()
    cfg = _cfg(config.RewardMode.TERMINAL_MARGIN)
    flat_steps, returns = learner._flatten([record], cfg)
    assert [step.player_id for step in flat_steps] == [0, 1, 0, 1, 0]
    # P0 margin = 12 - 7 = +5; P1 = -5; broadcast to every step of that seat.
    assert returns == [5.0, -5.0, 5.0, -5.0, 5.0]


#### TERMINAL_MARGIN + OWN_SCORE ####


def test_terminal_margin_own_score_broadcasts_own_score():
    """Every step gets its player's own absolute final score (both positive)."""
    record = _sample_record()  # P0=12, P1=7, P0 wins
    cfg = _cfg(config.RewardMode.TERMINAL_MARGIN, config.RewardBasis.OWN_SCORE)
    flat_steps, returns = learner._flatten([record], cfg)
    assert [step.player_id for step in flat_steps] == [0, 1, 0, 1, 0]
    assert returns == [12.0, 7.0, 12.0, 7.0, 12.0]


def test_terminal_margin_own_score_winner_bonus_only():
    """end_game_bonus adds to the winner's score only, not the loser's."""
    record = _sample_record()  # P0 wins
    cfg = config.RunConfig(
        training=config.TrainingConfig(
            reward_mode=config.RewardMode.TERMINAL_MARGIN,
            reward_basis=config.RewardBasis.OWN_SCORE,
            score_norm=1.0,
            end_game_bonus=5.0,
        )
    )
    _, returns = learner._flatten([record], cfg)
    # P0 wins: 12+5=17; P1 loses: 7 unchanged
    assert returns == [17.0, 7.0, 17.0, 7.0, 17.0]


#### DECISION_DELTA + MARGIN ####


@pytest.mark.parametrize(
    "discount, expected",
    [
        # gamma=0: immediate per-decision margin change only (every Δt > 0).
        (0.0, [3.0, -1.0, 2.0, -4.0, 0.0]),
        # gamma=0.5 over Δt=2 between checkpoints → factor 0.25 per link:
        # P0 opening = 3 + 0.25·2; P1 opening = -1 + 0.25·(-4).
        (0.5, [3.5, -2.0, 2.0, -4.0, 0.0]),
        # gamma=1: Δt-independent; telescopes to (terminal - margin_before).
        (1.0, [5.0, -5.0, 2.0, -4.0, 0.0]),
    ],
)
def test_decision_delta_discounted_returns(discount: float, expected: list[float]):
    """Per-decision γ^Δt-discounted margin returns match hand-computed values."""
    record = _sample_record()
    cfg = _cfg(config.RewardMode.DECISION_DELTA, discount=discount)
    _, returns = learner._flatten([record], cfg)
    _assert_close(returns, expected)


def test_decision_delta_gamma_one_telescopes_to_terminal_margin():
    """At gamma=1 the first decision of each player accrues its full terminal margin."""
    record = _sample_record()
    delta = learner._decision_delta_returns(
        record,
        discount=1.0,
        score_norm=1.0,
        end_game_bonus=0.0,
        basis=config.RewardBasis.MARGIN,
    )
    _assert_close([delta[0], delta[1]], [5.0, -5.0])


def test_decision_delta_handles_single_seat_record():
    """A record where only seat 0 recorded steps produces returns for that seat only."""
    record = collect.GameRecord(
        steps=[
            _step(player_id=0, margin_before=0.0, timestamp=1.0),
            _step(player_id=0, margin_before=4.0, timestamp=3.0),
        ],
        breakdowns=(_breakdown(10.0), _breakdown(6.0)),  # P0 terminal margin +4
        winner=0,
        seed=0,
        final_timestamp=5.0,
    )
    delta = learner._decision_delta_returns(
        record,
        discount=1.0,
        score_norm=1.0,
        end_game_bonus=0.0,
        basis=config.RewardBasis.MARGIN,
    )
    _assert_close(delta, [4.0, 0.0])


def test_decision_delta_zero_time_gap_applies_no_decay():
    """Two decisions at the same clock time pass credit through undecayed even at γ=0."""
    record = collect.GameRecord(
        steps=[
            _step(player_id=0, margin_before=0.0, timestamp=2.0 / 3.0),
            _step(player_id=0, margin_before=2.0, timestamp=2.0 / 3.0),
        ],
        breakdowns=(_breakdown(10.0), _breakdown(5.0)),  # P0 terminal margin +5
        winner=0,
        seed=0,
        final_timestamp=53.0,
    )
    delta = learner._decision_delta_returns(
        record,
        discount=0.0,
        score_norm=1.0,
        end_game_bonus=0.0,
        basis=config.RewardBasis.MARGIN,
    )
    _assert_close(delta, [5.0, 3.0])


def test_default_timestamps_degrade_to_telescoping():
    """A record without timestamps (all Δt=0) behaves like γ=1 regardless of γ."""
    record = collect.GameRecord(
        steps=[
            _step(player_id=0, margin_before=0.0),
            _step(player_id=0, margin_before=4.0),
        ],
        breakdowns=(_breakdown(10.0), _breakdown(6.0)),  # P0 terminal margin +4
        winner=0,
        seed=0,
    )
    delta = learner._decision_delta_returns(
        record,
        discount=0.0,
        score_norm=1.0,
        end_game_bonus=0.0,
        basis=config.RewardBasis.MARGIN,
    )
    _assert_close(delta, [4.0, 0.0])


def test_score_norm_scales_decision_delta_returns():
    """``score_norm`` divides the returns."""
    record = _sample_record()
    cfg = config.RunConfig(
        training=config.TrainingConfig(
            reward_mode=config.RewardMode.DECISION_DELTA,
            reward_discount=1.0,
            score_norm=5.0,
        )
    )
    _, returns = learner._flatten([record], cfg)
    _assert_close(returns, [1.0, -1.0, 0.4, -0.8, 0.0])


#### DECISION_DELTA + OWN_SCORE ####


@pytest.mark.parametrize(
    "discount, expected",
    [
        # gamma=0: immediate own-score change only (Δt > 0 between each pair).
        # P0: [3→6=3, 6→9=3, 9→12=3]; P1: [4→5=1, 5→7=2]; record order:
        (0.0, [3.0, 1.0, 3.0, 2.0, 3.0]),
        # gamma=1: telescopes to (terminal_own - score_before).
        # P0: [12-3=9, 12-6=6, 12-9=3]; P1: [7-4=3, 7-5=2]; record order:
        (1.0, [9.0, 3.0, 6.0, 2.0, 3.0]),
    ],
)
def test_decision_delta_own_score_returns(discount: float, expected: list[float]):
    """Per-decision own-score delta returns match hand-computed values."""
    record = _sample_record()
    cfg = _cfg(
        config.RewardMode.DECISION_DELTA,
        config.RewardBasis.OWN_SCORE,
        discount=discount,
    )
    _, returns = learner._flatten([record], cfg)
    _assert_close(returns, expected)


def test_decision_delta_own_score_winner_bonus():
    """end_game_bonus folds into the terminal own-score value and discounts back."""
    record = _sample_record()  # P0 wins, score_before=[3,6,9] for P0; [4,5] for P1
    cfg = config.RunConfig(
        training=config.TrainingConfig(
            reward_mode=config.RewardMode.DECISION_DELTA,
            reward_basis=config.RewardBasis.OWN_SCORE,
            reward_discount=1.0,
            score_norm=1.0,
            end_game_bonus=3.0,
        )
    )
    _, returns = learner._flatten([record], cfg)
    # P0 terminal = 12+3=15; P1 terminal = 7 (no bonus).
    # P0: [15-3=12, 15-6=9, 15-9=6]; P1: [7-4=3, 7-5=2]; record order:
    _assert_close(returns, [12.0, 3.0, 9.0, 2.0, 6.0])


def test_decision_delta_own_score_both_seats_positive():
    """With OWN_SCORE basis both seats always receive non-negative returns."""
    record = _sample_record()
    cfg = _cfg(config.RewardMode.DECISION_DELTA, config.RewardBasis.OWN_SCORE)
    _, returns = learner._flatten([record], cfg)
    assert all(ret >= 0.0 for ret in returns), f"Expected all ≥ 0, got {returns}"
