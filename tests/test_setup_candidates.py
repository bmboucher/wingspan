"""Tests for setup-candidate enumeration and selection.

The critical invariant: the candidates the engine offers at setup are exactly
``enumerate_setup_candidates(dealt_cards, dealt_bonus)`` — same 504 options, same
order — captured here from a real game through the public agent interface. Plus
the ``SetupCandidate`` round-trip and ``select_by_margins`` (argmax vs softmax).
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine  # noqa: E402
from wingspan.setup_model import candidates  # noqa: E402
from wingspan.training import collect  # noqa: E402


def _capture_setup_decisions(seed: int) -> list[decisions.SetupDecision]:
    """Play one random game, capturing the setup decisions the engine offers."""
    captured: list[decisions.SetupDecision] = []
    chooser = random.Random(seed)

    def agent[C: decisions.Choice](
        eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        choice = chooser.choice(decision.choices)
        if isinstance(decision, decisions.SetupDecision):
            captured.append(decision)
        return choice

    eng = collect.new_engine(seed)
    engine.Engine.play_one_game(eng.state, (agent, agent))
    return captured


def test_engine_offers_enumerated_candidates_in_order():
    setups = _capture_setup_decisions(3)
    assert len(setups) == 2  # one per player
    for decision in setups:
        model_candidates = candidates.enumerate_setup_candidates(
            decision.dealt_cards, decision.dealt_bonus
        )
        assert len(decision.choices) == 504
        assert len(model_candidates) == len(decision.choices)
        for choice, candidate in zip(decision.choices, model_candidates):
            assert choice.kept_cards == candidate.kept_cards
            assert choice.kept_foods == candidate.kept_foods
            assert choice.bonus_card == candidate.bonus_card


def test_enumerate_without_bonus_drops_the_bonus_axis():
    """``include_bonus=False`` (the split_setup_bonus regime) yields the distinct
    ``(kept_cards, kept_foods)`` keeps with no bonus — half the count, same order."""
    birds, bonuses, _ = cards.load_all()
    dealt_cards = list(birds[:5])
    dealt_bonus = list(bonuses[:2])
    with_bonus = candidates.enumerate_setup_candidates(dealt_cards, dealt_bonus)
    without_bonus = candidates.enumerate_setup_candidates(
        dealt_cards, dealt_bonus, include_bonus=False
    )
    assert len(with_bonus) == 504
    assert len(without_bonus) == 252  # the ×2 bonus axis is dropped
    assert all(candidate.bonus_card is None for candidate in without_bonus)

    # The bonus-free candidates are exactly the distinct keeps of the full set,
    # in first-seen order (each (mask, food_combo) emitted both bonuses adjacently).
    seen: set[tuple[tuple[cards.Bird, ...], tuple[cards.Food, ...]]] = set()
    distinct_keeps: list[tuple[tuple[cards.Bird, ...], tuple[cards.Food, ...]]] = []
    for candidate in with_bonus:
        key = (candidate.kept_cards, candidate.kept_foods)
        if key not in seen:
            seen.add(key)
            distinct_keeps.append(key)
    assert [(c.kept_cards, c.kept_foods) for c in without_bonus] == distinct_keeps


def test_setup_choice_round_trip():
    birds, bonuses, _ = cards.load_all()
    candidate = candidates.enumerate_setup_candidates(
        list(birds[:5]), list(bonuses[:2])
    )[200]
    restored = candidates.SetupCandidate.from_setup_choice(candidate.to_setup_choice())
    assert restored == candidate


def test_select_by_margins_argmax_and_softmax():
    margins = np.array([0.0, 5.0, 1.0, -2.0], dtype=np.float32)
    # rng=None -> deterministic argmax.
    assert candidates.select_by_margins(margins, temperature=1.0, rng=None) == 1
    # A low temperature concentrates the softmax mass on the argmax.
    rng = random.Random(0)
    picks = [
        candidates.select_by_margins(margins, temperature=0.05, rng=rng)
        for _ in range(50)
    ]
    assert picks.count(1) > 40
