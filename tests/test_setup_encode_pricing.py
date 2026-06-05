# pyright: reportPrivateUsage=false
# (reads the setup encoder's package-private block indices, per the
# setup_model/stripes.py convention)
"""Tests for the setup encoder's trailing candidate-pricing blocks.

The kept-bonus value block prices the kept bonus card against the keep itself
(kept qualifiers — every kept card for the hand-counting dynamic card — the
stepped / linear VP they would pay, tray potential), and the goal-affinity
block counts, per dealt goal, the kept cards that would advance its category
if played.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards  # noqa: E402
from wingspan.engine import scoring  # noqa: E402
from wingspan.setup_model import candidates  # noqa: E402
from wingspan.setup_model import encode as setup_encode  # noqa: E402
from wingspan.setup_model import stripes as setup_stripes  # noqa: E402

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}


class _Approx:
    """Tolerant float comparator (pytest.approx is untyped under strict pyright)."""

    def __init__(self, expected: float) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isclose(
            float(other), self.expected, rel_tol=1e-6, abs_tol=1e-9
        )


_KEPT_FOODS = (cards.Food.SEED, cards.Food.FISH, cards.Food.FRUIT)


def _context(
    goal_categories: tuple[str, ...],
    tray_birds: tuple[cards.Bird | None, ...] = (None, None, None),
) -> setup_encode.SetupContext:
    return setup_encode.SetupContext(
        tray_birds=tray_birds,
        birdfeeder_counts=(0, 0, 0, 0, 0, 0),
        round_goal_categories=goal_categories,
    )


def _kept_bonus_block(vec: np.ndarray) -> tuple[float, float, float, float]:
    base = setup_encode.OFF_KEPT_BONUS_VALUE
    return (
        float(vec[base + 0]),
        float(vec[base + 1]),
        float(vec[base + 2]),
        float(vec[base + 3]),
    )


def test_stripe_layout_matches_feature_dim():
    layout = setup_stripes.setup_stripe_layout()
    assert layout.total_size == setup_encode.SETUP_FEATURE_DIM == 308
    assert {stripe.name for stripe in layout.stripes} >= {
        "kept_bonus_value",
        "goal_affinity",
    }


def test_static_kept_bonus_is_priced_against_the_keep():
    """A type-counting bonus card prices the kept cards that pass its test and
    the tray birds that could still qualify it."""
    bird_counter = _BONUS_BY_NAME["Bird Counter"]  # 2 VP per qualifying bird
    tagged = [bird for bird in _BIRDS if bird_counter.name in bird.bonus_categories]
    untagged = next(
        bird for bird in _BIRDS if bird_counter.name not in bird.bonus_categories
    )
    candidate = candidates.SetupCandidate(
        kept_cards=(tagged[0], tagged[1]),
        kept_foods=_KEPT_FOODS,
        bonus_card=bird_counter,
    )
    context = _context(("birds_forest",) * 4, tray_birds=(tagged[2], untagged, None))
    vec = setup_encode.encode_setup_candidate(candidate, context)

    qual, stepped, linear, tray = _kept_bonus_block(vec)
    assert qual == _Approx(2 / 5)
    assert stepped == _Approx(scoring.bonus_score_for_count(bird_counter, 2) / 7)
    assert linear == _Approx(scoring.bonus_linear_value_for_count(bird_counter, 2) / 7)
    assert tray == _Approx(1 / 5)


def test_hand_counting_bonus_counts_every_kept_card():
    """Visionary Leader's keep value counts every kept card: 3 kept is below
    the first tier (stepped 0) but carries linear progress toward it."""
    visionary = _BONUS_BY_NAME["Visionary Leader"]
    candidate = candidates.SetupCandidate(
        kept_cards=tuple(_BIRDS[:3]),
        kept_foods=(cards.Food.SEED, cards.Food.FISH),
        bonus_card=visionary,
    )
    vec = setup_encode.encode_setup_candidate(
        candidate, _context(("birds_forest",) * 4)
    )

    qual, stepped, linear, _tray = _kept_bonus_block(vec)
    assert qual == _Approx(3 / 5)
    assert stepped == 0.0
    assert linear == _Approx(2.4 / 7)  # interpolating (0,0) -> (5,4) at 3


def test_no_kept_bonus_leaves_the_block_zero():
    candidate = candidates.SetupCandidate(
        kept_cards=(_BIRDS[0],),
        kept_foods=(
            cards.Food.SEED,
            cards.Food.FISH,
            cards.Food.FRUIT,
            cards.Food.RODENT,
        ),
        bonus_card=None,
    )
    vec = setup_encode.encode_setup_candidate(
        candidate, _context(("birds_forest",) * 4)
    )
    assert _kept_bonus_block(vec) == (0.0, 0.0, 0.0, 0.0)


def test_goal_affinity_counts_every_kept_card_for_birds_no_eggs():
    """Every kept card plays as an eggless bird, so each advances the
    anti-egg goal."""
    candidate = candidates.SetupCandidate(
        kept_cards=(_BIRDS[0], _BIRDS[1]),
        kept_foods=_KEPT_FOODS,
        bonus_card=None,
    )
    goal_categories = ("birds_no_eggs",) + ("birds_forest",) * 3
    vec = setup_encode.encode_setup_candidate(candidate, _context(goal_categories))
    assert float(vec[setup_encode.OFF_GOAL_AFFINITY + 0]) == _Approx(2 / 5)


def test_goal_affinity_counts_kept_cards_per_goal():
    """Two forest-only keeps: full affinity for a birds_forest goal and for
    total_birds, none for a wetland or egg goal."""
    forest_only = [bird for bird in _BIRDS if bird.habitats == (cards.Habitat.FOREST,)]
    candidate = candidates.SetupCandidate(
        kept_cards=(forest_only[0], forest_only[1]),
        kept_foods=_KEPT_FOODS,
        bonus_card=None,
    )
    goal_categories = ("birds_forest", "birds_wetland", "eggs_forest", "total_birds")
    vec = setup_encode.encode_setup_candidate(candidate, _context(goal_categories))

    base = setup_encode.OFF_GOAL_AFFINITY
    assert float(vec[base + 0]) == _Approx(2 / 5)
    assert float(vec[base + 1]) == 0.0
    assert float(vec[base + 2]) == 0.0  # egg goals can't be advanced at setup
    assert float(vec[base + 3]) == _Approx(2 / 5)
