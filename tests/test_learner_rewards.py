# pyright: reportPrivateUsage=false
"""Tests for the learner's two REINFORCE reward modes (``learner._flatten``).

``terminal_margin`` broadcasts the end-of-game margin to every step;
``decision_delta`` credits each decision with the discounted sum of per-decision
margin changes. These pin the return arithmetic against hand-computed values so
a regression in the discounting / telescoping is caught directly.
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


def _step(player_id: int, margin_before: float) -> steps.Step:
    """A minimal recorded step — only ``player_id`` and ``margin_before`` matter
    to the return computation, so the feature arrays are dummies."""
    return steps.Step(
        state=np.zeros(1, dtype=np.float32),
        choices=np.zeros((1, 1), dtype=np.float32),
        chosen_idx=0,
        player_id=player_id,
        family_idx=0,
        margin_before=margin_before,
    )


def _breakdown(total: float) -> metrics.ScoreBreakdown:
    """A breakdown whose ``.total`` is ``total`` (parked entirely in ``birds``)."""
    return metrics.ScoreBreakdown(birds=total)


def _sample_record() -> collect.GameRecord:
    """A five-step interleaved game with known margins.

    Player 0 steps carry margin_before [0, 3, 5]; player 1 steps carry [0, -1].
    Finals are 12 vs 7, so the terminal margins are +5 (P0) and -5 (P1)."""
    return collect.GameRecord(
        steps=[
            _step(player_id=0, margin_before=0.0),
            _step(player_id=1, margin_before=0.0),
            _step(player_id=0, margin_before=3.0),
            _step(player_id=1, margin_before=-1.0),
            _step(player_id=0, margin_before=5.0),
        ],
        breakdowns=(_breakdown(12.0), _breakdown(7.0)),
        winner=0,
        seed=0,
    )


def _cfg(reward_mode: config.RewardMode, discount: float = 1.0) -> config.TrainConfig:
    """A config with ``score_norm=1`` so returns equal the raw (discounted) deltas."""
    return config.TrainConfig(
        reward_mode=reward_mode, reward_discount=discount, score_norm=1.0
    )


def test_terminal_margin_broadcasts_pov_margin():
    """Every step gets its player's end-of-game margin (opposite signs)."""
    record = _sample_record()
    cfg = _cfg(config.RewardMode.TERMINAL_MARGIN)
    flat_steps, returns = learner._flatten([record], cfg)
    assert [step.player_id for step in flat_steps] == [0, 1, 0, 1, 0]
    # P0 margin = 12 - 7 = +5; P1 = -5; broadcast to every step of that seat.
    assert returns == [5.0, -5.0, 5.0, -5.0, 5.0]


@pytest.mark.parametrize(
    "discount, expected",
    [
        # gamma=0: immediate per-decision margin change only.
        (0.0, [3.0, -1.0, 2.0, -4.0, 0.0]),
        # gamma=0.5: decaying future credit.
        (0.5, [4.0, -3.0, 2.0, -4.0, 0.0]),
        # gamma=1: telescopes to (terminal margin - margin_before) per step.
        (1.0, [5.0, -5.0, 2.0, -4.0, 0.0]),
    ],
)
def test_decision_delta_discounted_returns(discount: float, expected: list[float]):
    """Per-decision discounted returns match the hand-computed values, aligned
    to record order (P0 at 0/2/4, P1 at 1/3)."""
    record = _sample_record()
    cfg = _cfg(config.RewardMode.DECISION_DELTA, discount=discount)
    _, returns = learner._flatten([record], cfg)
    _assert_close(returns, expected)


def test_decision_delta_gamma_one_telescopes_to_terminal_margin():
    """At gamma=1 the first decision of each player accrues that player's full
    terminal margin (its margin_before is 0 here), i.e. the undiscounted return
    equals terminal_margin for the opening decision."""
    record = _sample_record()
    delta = learner._decision_delta_returns(record, discount=1.0, score_norm=1.0)
    # Opening decision of each seat (indices 0 and 1) recovers the seat margin.
    _assert_close([delta[0], delta[1]], [5.0, -5.0])


def test_decision_delta_handles_single_seat_record():
    """A record where only seat 0 recorded steps (the vs-random bootstrap) still
    produces returns for that seat and leaves no stray entries for the other."""
    record = collect.GameRecord(
        steps=[
            _step(player_id=0, margin_before=0.0),
            _step(player_id=0, margin_before=4.0),
        ],
        breakdowns=(_breakdown(10.0), _breakdown(6.0)),  # P0 terminal margin +4
        winner=0,
        seed=0,
    )
    delta = learner._decision_delta_returns(record, discount=1.0, score_norm=1.0)
    # checkpoints [0, 4, 4] -> rewards [4, 0] -> returns [4, 0] at gamma=1.
    _assert_close(delta, [4.0, 0.0])


def test_score_norm_scales_decision_delta_returns():
    """``score_norm`` divides the returns, exactly as in terminal mode."""
    record = _sample_record()
    cfg = config.TrainConfig(
        reward_mode=config.RewardMode.DECISION_DELTA,
        reward_discount=1.0,
        score_norm=5.0,
    )
    _, returns = learner._flatten([record], cfg)
    _assert_close(returns, [1.0, -1.0, 0.4, -0.8, 0.0])
