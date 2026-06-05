# pyright: reportPrivateUsage=false
"""Tests for selfplay's regime resolution and setup-decision log annotation.

``wingspan-selfplay`` derives the ``split_setup_bonus`` regime from the
``TrainConfig`` stored in each loaded checkpoint — mirroring how the nets were
trained — rather than from a CLI flag, and exempts the ``SetupDecision`` from
the log's probability floor so its near-uniform opening distribution still
documents the policy's top picks.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wingspan import decisions, engine, selfplay, setup_model  # noqa: E402
from wingspan.training import config  # noqa: E402

#### Regime resolution ####


def test_no_ai_seats_defaults_to_combined():
    """Random-vs-random has no checkpoint config, so the engine's combined
    default applies."""
    assert selfplay._resolve_split_setup_bonus((None, None)) is False


def test_split_regime_checkpoint_activates_split():
    """A single split-regime checkpoint (mixed with a random seat) is enough to
    run the game in the split regime."""
    cfg = config.TrainConfig(use_setup_model=True, split_setup_bonus=True)
    assert selfplay._resolve_split_setup_bonus((cfg, None)) is True


def test_split_flag_is_gated_on_setup_model():
    """``split_setup_bonus`` is inert without the setup model (the
    ``split_setup_bonus_active`` gating), so such a checkpoint stays combined."""
    cfg = config.TrainConfig(use_setup_model=False, split_setup_bonus=True)
    assert selfplay._resolve_split_setup_bonus((cfg, None)) is False


def test_combined_regime_checkpoints_stay_combined():
    """Two combined-regime checkpoints agree on the combined default."""
    cfg = config.TrainConfig(use_setup_model=True, split_setup_bonus=False)
    assert selfplay._resolve_split_setup_bonus((cfg, cfg)) is False


def test_disagreeing_regimes_raise():
    """Checkpoints trained under different regimes cannot share a faithful game."""
    split_cfg = config.TrainConfig(use_setup_model=True, split_setup_bonus=True)
    combined_cfg = config.TrainConfig(use_setup_model=True, split_setup_bonus=False)
    with pytest.raises(ValueError, match="split_setup_bonus"):
        selfplay._resolve_split_setup_bonus((split_cfg, combined_cfg))


#### Setup-decision log annotation ####


def test_setup_decision_log_always_shows_top_options():
    """The ``SetupDecision`` is exempt from the probability floor: a near-uniform
    distribution over all 504 keeps (where nothing clears 1%) still logs the full
    top-``_MAX_LOGGED_OPTIONS`` ranked list."""
    eng, birds, bonuses, _goals = engine.Engine.create(seed=0)
    dealt_cards = birds[:5]
    dealt_bonus = bonuses[:2]
    choices = [
        candidate.to_setup_choice()
        for candidate in setup_model.enumerate_setup_candidates(
            dealt_cards, dealt_bonus
        )
    ]
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="test setup",
        choices=choices,
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
    )

    probs = np.full(len(choices), 1.0 / len(choices))
    selfplay._log_distribution(eng, decision, probs, greedy=False)

    player_name = eng.state.me().name
    header = f"[{player_name}: SetupDecision | {len(choices)} choices]"
    header_idx = eng.state.log.index(header)
    # Each shown option emits 2 lines (label line + prob/score line).
    ranked = eng.state.log[header_idx + 1 :]
    assert len(ranked) == selfplay._MAX_LOGGED_OPTIONS * 2


def test_non_setup_decision_keeps_probability_floor():
    """Other large decisions keep the floor: options under 1% stay suppressed."""
    eng, birds, _bonuses, _goals = engine.Engine.create(seed=0)
    choices = [decisions.BirdChoice(label=bird.name, bird=bird) for bird in birds[:10]]
    decision = decisions.BirdPowerPickBirdFromHandDecision(
        player_id=0,
        prompt="pick a bird",
        choices=choices,
    )

    # One dominant option; the other nine sit at 0.1%, below the 1% floor.
    probs = np.full(len(choices), 0.001)
    probs[0] = 1.0 - float(probs[1:].sum())
    selfplay._log_distribution(eng, decision, probs, greedy=False)

    player_name = eng.state.me().name
    header = (
        f"[{player_name}: BirdPowerPickBirdFromHandDecision | {len(choices)} choices]"
    )
    header_idx = eng.state.log.index(header)
    # Each shown option emits 2 lines (label line + prob/score line); only 1
    # option clears the 1% floor so we expect exactly 2 lines total.
    ranked = eng.state.log[header_idx + 1 :]
    assert len(ranked) == 2
