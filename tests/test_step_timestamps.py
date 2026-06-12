"""Tests for the game-clock timestamps (``wingspan.training.timestamps``).

Pins the clock the ``decision_delta`` reward mode discounts over: setup
decisions at 0 / 1/3 / 2/3, turn N's main action at exactly N, and mid-turn
decisions linearly interpolated into (N, N+1) once the game is complete.
Deliberately torch-free — the module under test never imports the training
stack's heavyweight dependencies.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions
from wingspan.training import steps, timestamps

_MAIN_FAMILY = decisions.family_index_for(decisions.MainActionDecision)
_FOOD_FAMILY = decisions.family_index_for(decisions.GainFoodDecision)


def _assert_close(actual: list[float], expected: list[float]) -> None:
    """Element-wise float comparison (pytest.approx is untyped under strict pyright)."""
    assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
    for got, want in zip(actual, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), f"{got} != {want}"


def _step(family_idx: int, timestamp: float, player_id: int = 0) -> steps.Step:
    """A minimal recorded step — only ``family_idx`` and the provisional
    ``timestamp`` matter to finalization, so the feature arrays are dummies."""
    return steps.Step(
        state=np.zeros(1, dtype=np.float32),
        choices=np.zeros((1, 1), dtype=np.float32),
        chosen_idx=0,
        player_id=player_id,
        family_idx=family_idx,
        timestamp=timestamp,
    )


def _main_action_decision() -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.MainActionChoice(label="g", action=decisions.MainAction.GAIN_FOOD)
        ],
    )


def _gain_food_decision() -> decisions.GainFoodDecision:
    return decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=[decisions.FoodChoice(label="seed", food=cards.Food.SEED)],
    )


def _bonus_pick_decision() -> decisions.BirdPowerPickBonusCardDecision:
    _, bonuses, _ = cards.load_all()
    return decisions.BirdPowerPickBonusCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BonusCardChoice(label=bonuses[0].name, bonus_card=bonuses[0])
        ],
    )


def _setup_decision() -> decisions.SetupDecision:
    birds, bonuses, _ = cards.load_all()
    return decisions.SetupDecision(
        player_id=0,
        prompt="x",
        choices=[decisions.SetupChoice(kept_cards=(), kept_foods=(), bonus_card=None)],
        dealt_cards=birds[:5],
        dealt_bonus=bonuses[:2],
    )


def test_provisional_in_turn_is_the_turn_counter():
    """Any decision asked during turn N provisionally lands on N — including a
    power-driven bonus pick, which must NOT fall back to the setup constant."""
    assert timestamps.provisional_timestamp(_main_action_decision(), 5) == 5.0
    assert timestamps.provisional_timestamp(_gain_food_decision(), 5) == 5.0
    assert timestamps.provisional_timestamp(_bonus_pick_decision(), 17) == 17.0


def test_provisional_setup_window_constants():
    """In the setup window (turn_counter 0) each decision kind has its fixed
    shared-clock time: keep at 0, deferred bonus at 1/3, deferred food at 2/3."""
    keep = timestamps.provisional_timestamp(_setup_decision(), 0)
    bonus = timestamps.provisional_timestamp(_bonus_pick_decision(), 0)
    food = timestamps.provisional_timestamp(_gain_food_decision(), 0)
    _assert_close(
        [keep, bonus, food],
        [
            timestamps.SETUP_KEEP_TIMESTAMP,
            timestamps.SETUP_BONUS_TIMESTAMP,
            timestamps.SETUP_FOOD_TIMESTAMP,
        ],
    )


def test_finalize_interpolates_mid_turn_after_main_action():
    """A turn recorded as [main, a, b] keeps the main action at T and spreads
    the k=2 followers to T + j/(k+1); a lone follower lands at T + 1/2."""
    recorded = [
        _step(_MAIN_FAMILY, 3.0),
        _step(_FOOD_FAMILY, 3.0),
        _step(_FOOD_FAMILY, 3.0),
        _step(_MAIN_FAMILY, 4.0),
        _step(_FOOD_FAMILY, 4.0),
    ]
    timestamps.finalize_timestamps(recorded)
    _assert_close(
        [step.timestamp for step in recorded],
        [3.0, 3.0 + 1.0 / 3.0, 3.0 + 2.0 / 3.0, 4.0, 4.5],
    )


def test_finalize_interpolates_group_without_main_action():
    """A turn window with no recorded main action (vs-random play: only the net
    seat's reactions during the opponent's turn are recorded) interpolates all
    of its steps — none is pinned to the integer."""
    recorded = [
        _step(_FOOD_FAMILY, 7.0),
        _step(_FOOD_FAMILY, 7.0),
    ]
    timestamps.finalize_timestamps(recorded)
    _assert_close(
        [step.timestamp for step in recorded], [7.0 + 1.0 / 3.0, 7.0 + 2.0 / 3.0]
    )


def test_finalize_leaves_setup_window_untouched():
    """Setup-window timestamps (below the first turn) are already final."""
    recorded = [
        _step(_FOOD_FAMILY, timestamps.SETUP_KEEP_TIMESTAMP),
        _step(_FOOD_FAMILY, timestamps.SETUP_BONUS_TIMESTAMP),
        _step(_FOOD_FAMILY, timestamps.SETUP_FOOD_TIMESTAMP),
        _step(_FOOD_FAMILY, timestamps.SETUP_FOOD_TIMESTAMP),
    ]
    timestamps.finalize_timestamps(recorded)
    _assert_close(
        [step.timestamp for step in recorded],
        [
            timestamps.SETUP_KEEP_TIMESTAMP,
            timestamps.SETUP_BONUS_TIMESTAMP,
            timestamps.SETUP_FOOD_TIMESTAMP,
            timestamps.SETUP_FOOD_TIMESTAMP,
        ],
    )


def test_finalize_keeps_each_player_monotone():
    """Over a realistic mixed sequence — both seats' setup windows recorded
    back-to-back, then alternating turns with an opponent reaction inside turn
    1 — each player's own timestamps come out non-decreasing and every mid-turn
    step lands strictly inside its turn's (T, T+1) window."""
    recorded = [
        # P0's setup window, then P1's (the global sequence dips here).
        _step(_FOOD_FAMILY, timestamps.SETUP_KEEP_TIMESTAMP, player_id=0),
        _step(_FOOD_FAMILY, timestamps.SETUP_FOOD_TIMESTAMP, player_id=0),
        _step(_FOOD_FAMILY, timestamps.SETUP_KEEP_TIMESTAMP, player_id=1),
        _step(_FOOD_FAMILY, timestamps.SETUP_FOOD_TIMESTAMP, player_id=1),
        # Turn 1 (P0): main action, a follow-up, and a P1 reaction.
        _step(_MAIN_FAMILY, 1.0, player_id=0),
        _step(_FOOD_FAMILY, 1.0, player_id=0),
        _step(_FOOD_FAMILY, 1.0, player_id=1),
        # Turn 2 (P1): main action only.
        _step(_MAIN_FAMILY, 2.0, player_id=1),
    ]
    timestamps.finalize_timestamps(recorded)

    for player_id in (0, 1):
        own = [step.timestamp for step in recorded if step.player_id == player_id]
        assert own == sorted(own), f"player {player_id} timestamps not monotone: {own}"

    mid_turn = [
        step
        for step in recorded
        if step.timestamp >= 1.0 and step.family_idx != _MAIN_FAMILY
    ]
    for step in mid_turn:
        turn = math.floor(step.timestamp)
        assert turn < step.timestamp < turn + 1.0
