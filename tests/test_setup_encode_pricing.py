# pyright: reportPrivateUsage=false
# (reads the setup encoder's package-private block indices, per the
# setup_model/stripes.py convention)
"""Tests for the setup encoder's trailing candidate-pricing blocks.

The kept-bonus value block prices the kept bonus card against the keep itself
(kept qualifiers — every kept card for the hand-counting dynamic card — the
stepped / linear VP they would pay, tray potential), and the goal-affinity
block prices, per dealt goal, the played-and-optimally-egg-populated affinity
of the kept cards (``scoring.goal_affinity_for_kept``) — nonzero for
egg-driven categories since v1.6, unlike the pre-1.6 play-instant pricing
(``wingspan.compat.v1_5.SetupNetV1_5`` freezes that behavior for old
artifacts; see ``tests/test_compat_v1_5.py``).
"""

from __future__ import annotations

import math

import numpy as np

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
    # Default encoding includes include_playable_kept_cards=True (488 dims).
    # Verify the layout sums to the encoding's own total_dim.
    from wingspan.setup_model import architecture as arch_module

    encoding = arch_module.SetupEncoding()
    layout = setup_stripes.setup_stripe_layout(encoding)
    assert layout.total_size == encoding.total_dim
    assert {stripe.name for stripe in layout.stripes} >= {
        "kept_bonus_value",
        "goal_affinity",
    }
    # Explicitly no-flags encoding still gives the SETUP_FEATURE_DIM base (308).
    enc_off = arch_module.SetupEncoding(include_playable_kept_cards=False)
    layout_off = setup_stripes.setup_stripe_layout(enc_off)
    assert layout_off.total_size == setup_encode.SETUP_FEATURE_DIM == 308


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
    total_birds, none for a wetland goal, and — since v1.6 — the *played and
    egg-populated* bound (not 0) for an egg goal: both birds are assumed
    played into forest and egg-filled, so eggs_forest prices their summed
    egg_limit."""
    forest_only = [bird for bird in _BIRDS if bird.habitats == (cards.Habitat.FOREST,)]
    bird_a, bird_b = forest_only[0], forest_only[1]
    assert (bird_a.name, bird_a.egg_limit) == ("Acorn Woodpecker", 4)
    assert (bird_b.name, bird_b.egg_limit) == ("American Redstart", 2)
    candidate = candidates.SetupCandidate(
        kept_cards=(bird_a, bird_b),
        kept_foods=_KEPT_FOODS,
        bonus_card=None,
    )
    goal_categories = ("birds_forest", "birds_wetland", "eggs_forest", "total_birds")
    vec = setup_encode.encode_setup_candidate(candidate, _context(goal_categories))

    base = setup_encode.OFF_GOAL_AFFINITY
    assert float(vec[base + 0]) == _Approx(2 / 5)
    assert float(vec[base + 1]) == 0.0
    # eggs_forest: both birds are single-habitat forest, so each is reachable
    # and prices its full egg_limit — (4 + 2) / 5.
    assert float(vec[base + 2]) == _Approx((bird_a.egg_limit + bird_b.egg_limit) / 5)
    assert float(vec[base + 3]) == _Approx(2 / 5)


def test_goal_affinity_bowl_cavity_star_keep_prices_egg_driven_categories():
    """The user's reported scenario: a bowl-nest, a cavity-nest, and a
    star-nest keep prices toward every egg-driven category it could reach —
    star nests are wild, so the star bird counts toward both the bowl and
    cavity birds-with-eggs categories (and is the sole contributor to a
    ground egg goal, having no ground-nest keep at all)."""
    bowl_bird = _BIRDS[0].model_copy(
        update={"nest": cards.NestType.BOWL, "egg_limit": 4}
    )
    cavity_bird = _BIRDS[1].model_copy(
        update={"nest": cards.NestType.CAVITY, "egg_limit": 3}
    )
    star_bird = _BIRDS[2].model_copy(
        update={"nest": cards.NestType.STAR, "egg_limit": 5}
    )
    candidate = candidates.SetupCandidate(
        kept_cards=(bowl_bird, cavity_bird, star_bird),
        kept_foods=_KEPT_FOODS,
        bonus_card=None,
    )
    goal_categories = (
        "bowl_birds_with_eggs",
        "cavity_birds_with_eggs",
        "eggs_ground",
        "total_birds",
    )
    vec = setup_encode.encode_setup_candidate(candidate, _context(goal_categories))

    base = setup_encode.OFF_GOAL_AFFINITY
    assert float(vec[base + 0]) == _Approx(2 / 5)  # bowl bird + wild star bird
    assert float(vec[base + 1]) == _Approx(2 / 5)  # cavity bird + wild star bird
    assert float(vec[base + 2]) == _Approx(star_bird.egg_limit / 5)  # star only
    assert float(vec[base + 3]) == _Approx(3 / 5)


def test_goal_affinity_egg_sets_3habitats_uses_best_kept_assignment():
    """Three single-habitat keeps covering all three habitats: each bird's
    only possible landing habitat is forced, so the egg-set affinity is the
    minimum of their egg limits (the hand-level assignment optimum), not a
    plain sum."""
    forest_bird = _BIRDS[0].model_copy(
        update={"habitats": (cards.Habitat.FOREST,), "egg_limit": 4}
    )
    grassland_bird = _BIRDS[1].model_copy(
        update={"habitats": (cards.Habitat.GRASSLAND,), "egg_limit": 6}
    )
    wetland_bird = _BIRDS[2].model_copy(
        update={"habitats": (cards.Habitat.WETLAND,), "egg_limit": 2}
    )
    candidate = candidates.SetupCandidate(
        kept_cards=(forest_bird, grassland_bird, wetland_bird),
        kept_foods=_KEPT_FOODS,
        bonus_card=None,
    )
    goal_categories = ("egg_sets_3habitats",) + ("birds_forest",) * 3
    vec = setup_encode.encode_setup_candidate(candidate, _context(goal_categories))

    assert float(vec[setup_encode.OFF_GOAL_AFFINITY + 0]) == _Approx(
        min(forest_bird.egg_limit, grassland_bird.egg_limit, wetland_bird.egg_limit) / 5
    )


def test_egg_bonus_keep_is_priced_by_egg_capacity():
    """Breeding Manager tags no bird, but kept cards whose egg capacity
    reaches 4 could come to qualify — the keep is priced at that optimistic
    count (v1.7): qual, the stepped/linear VP it pays, and the tray's
    egg-capable bird as potential."""
    breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
    big_nest = [bird for bird in _BIRDS if bird.egg_limit >= 4]
    small_nest = next(bird for bird in _BIRDS if bird.egg_limit < 4)
    candidate = candidates.SetupCandidate(
        kept_cards=(big_nest[0], big_nest[1], small_nest),
        kept_foods=(cards.Food.SEED, cards.Food.FISH),
        bonus_card=breeding_manager,
    )
    context = _context(
        ("birds_forest",) * 4, tray_birds=(big_nest[2], small_nest, None)
    )
    vec = setup_encode.encode_setup_candidate(candidate, context)

    qual, stepped, linear, tray = _kept_bonus_block(vec)
    assert qual == _Approx(2 / 5)
    assert stepped == _Approx(scoring.bonus_score_for_count(breeding_manager, 2) / 7)
    assert linear == _Approx(
        scoring.bonus_linear_value_for_count(breeding_manager, 2) / 7
    )
    assert tray == _Approx(1 / 5)


def test_bonus_card_affinity_min_max_over_dealt_cards():
    """Split-bonus mode: ``bonus_card_affinity`` is the min/max
    potential-qualifier count over the dealt bonus cards — the egg card priced
    by egg capacity (v1.7), the static card by its printed tag. First direct
    value test of the affinity stripe."""
    from wingspan.setup_model import architecture as arch_module

    breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
    bird_feeder = _BONUS_BY_NAME["Bird Feeder"]
    tagged = next(bird for bird in _BIRDS if bird_feeder.name in bird.bonus_categories)
    high_a = _BIRDS[0].model_copy(update={"egg_limit": 4, "bonus_categories": ()})
    high_b = _BIRDS[1].model_copy(update={"egg_limit": 5, "bonus_categories": ()})
    tagged_low = tagged.model_copy(update={"egg_limit": 2})
    candidate = candidates.SetupCandidate(
        kept_cards=(high_a, high_b, tagged_low),
        kept_foods=(cards.Food.SEED, cards.Food.FISH),
        bonus_card=None,
    )
    encoding = arch_module.SetupEncoding(split_bonus=True)
    context = setup_encode.SetupContext(
        tray_birds=(None, None, None),
        birdfeeder_counts=(0, 0, 0, 0, 0, 0),
        round_goal_categories=("birds_forest",) * 4,
        dealt_bonus_cards=(breeding_manager, bird_feeder),
    )
    vec = setup_encode.encode_setup_candidate(candidate, context, encoding)

    base = encoding.off_bonus_block + arch_module._BONUS_DIM
    # Bird Feeder counts its one tagged keep; Breeding Manager both
    # 4+-egg-capacity keeps (tagged_low's capacity of 2 misses the threshold).
    assert float(vec[base + 0]) == _Approx(1 / 5)
    assert float(vec[base + 1]) == _Approx(2 / 5)


def test_goal_affinity_can_exceed_one():
    """Two bowl-nest keeps whose egg limits sum past 5: the ÷5 normalization
    is a heuristic, not a hard cap, so the stripe value exceeds 1.0."""
    bowl_a = _BIRDS[0].model_copy(update={"nest": cards.NestType.BOWL, "egg_limit": 4})
    bowl_b = _BIRDS[1].model_copy(update={"nest": cards.NestType.BOWL, "egg_limit": 3})
    candidate = candidates.SetupCandidate(
        kept_cards=(bowl_a, bowl_b),
        kept_foods=_KEPT_FOODS,
        bonus_card=None,
    )
    goal_categories = ("eggs_bowl",) + ("birds_forest",) * 3
    vec = setup_encode.encode_setup_candidate(candidate, _context(goal_categories))

    affinity = float(vec[setup_encode.OFF_GOAL_AFFINITY + 0])
    assert affinity == _Approx((bowl_a.egg_limit + bowl_b.egg_limit) / 5)
    assert affinity > 1.0
