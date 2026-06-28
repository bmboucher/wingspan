"""Tests for the card parser, focused on food-cost parsing correctness.

Verifies that OR-cost birds (marked with the '/ (food cost)' column in
master.json) are parsed with ``is_or_cost=True`` and ``effective_total==1``,
and that AND-cost birds are unaffected."""

from __future__ import annotations

from wingspan import cards
from wingspan.agents import display

# Load all birds once for the whole module — fast enough since cards are cached.
_BIRDS, _, _ = cards.load_all()
_BY_NAME: dict[str, cards.Bird] = {bird.name: bird for bird in _BIRDS}

INV = cards.Food.INVERTEBRATE
SEED = cards.Food.SEED
FISH = cards.Food.FISH
FRUIT = cards.Food.FRUIT
RODENT = cards.Food.RODENT

# --- Parser: known OR-cost birds -----------------------------------------


def test_carolina_chickadee_is_or_cost() -> None:
    """Carolina Chickadee costs 1 inv OR 1 seed — canonical OR-cost example."""
    bird = _BY_NAME["Carolina Chickadee"]
    assert bird.food_cost.is_or_cost
    assert bird.food_cost.counts == (1, 1, 0, 0, 0, 0)
    assert bird.food_cost.effective_total == 1


def test_or_cost_wild_is_zero() -> None:
    """Core-set OR-cost birds have no wild slots in addition to the accepted mask."""
    bird = _BY_NAME["Carolina Chickadee"]
    assert bird.food_cost.wild == 0


def test_or_cost_total_reflects_mask_not_effective() -> None:
    """total() returns the sum of the mask (number of accepted foods),
    not the number of tokens required.  effective_total is what counts."""
    bird = _BY_NAME["Carolina Chickadee"]
    assert bird.food_cost.total == 2  # 2 accepted types in the mask
    assert bird.food_cost.effective_total == 1  # 1 token required


def test_american_robin_is_or_cost() -> None:
    """American Robin costs 1 inv OR 1 fruit."""
    bird = _BY_NAME["American Robin"]
    assert bird.food_cost.is_or_cost
    assert bird.food_cost.counts == (1, 0, 0, 1, 0, 0)
    assert bird.food_cost.effective_total == 1


# --- Parser: known AND-cost birds ----------------------------------------


def test_avocet_is_and_cost() -> None:
    """American Avocet: 2 inv + 1 seed, no slash marker."""
    bird = _BY_NAME["American Avocet"]
    assert not bird.food_cost.is_or_cost
    assert bird.food_cost.counts == (2, 1, 0, 0, 0, 0)
    assert bird.food_cost.effective_total == 3


def test_american_kestrel_is_and_cost() -> None:
    """American Kestrel: 1 inv + 1 rodent — two foods, AND cost."""
    bird = _BY_NAME["American Kestrel"]
    assert not bird.food_cost.is_or_cost
    assert bird.food_cost.counts == (1, 0, 0, 0, 1, 0)
    assert bird.food_cost.effective_total == 2


def test_free_bird_not_or_cost() -> None:
    """Birds with zero food cost must not be flagged as OR costs."""
    free_birds = [bird for bird in _BIRDS if bird.food_cost.is_free()]
    assert free_birds, "expected at least one free bird in the core set"
    for bird in free_birds:
        assert not bird.food_cost.is_or_cost, bird.name


# --- Parser: whole-corpus invariants -------------------------------------


def test_or_cost_bird_count() -> None:
    """The core set has exactly 31 OR-cost birds (80 total across all expansions,
    but load_all() returns only core-set birds)."""
    or_cost_birds = [bird for bird in _BIRDS if bird.food_cost.is_or_cost]
    assert (
        len(or_cost_birds) == 31
    ), f"Expected 31 core-set OR-cost birds, got {len(or_cost_birds)}"


def test_all_or_cost_birds_effective_total_one() -> None:
    """Every bird with is_or_cost=True must require exactly 1 token."""
    for bird in _BIRDS:
        if bird.food_cost.is_or_cost:
            assert (
                bird.food_cost.effective_total == 1
            ), f"{bird.name}: effective_total={bird.food_cost.effective_total}"


def test_non_or_cost_birds_effective_total_matches_total() -> None:
    """For AND-cost birds, effective_total == total (no difference)."""
    and_cost_birds = [bird for bird in _BIRDS if not bird.food_cost.is_or_cost]
    assert and_cost_birds
    for bird in and_cost_birds:
        assert bird.food_cost.effective_total == bird.food_cost.total, bird.name


def test_or_cost_birds_have_at_least_two_accepted_foods() -> None:
    """Every OR-cost bird must list at least 2 food types in its mask
    (otherwise the slash marker would be meaningless)."""
    for bird in _BIRDS:
        if bird.food_cost.is_or_cost:
            accepted = sum(1 for count in bird.food_cost.specific if count > 0)
            assert accepted >= 2, f"{bird.name}: only {accepted} accepted food type(s)"


# --- format_cost: separator differs by cost type -------------------------


def test_format_cost_or_uses_slash() -> None:
    cost = cards.BirdCost.from_specific({INV: 1, SEED: 1}, is_or_cost=True)
    formatted = display.format_cost(cost)
    assert "/" in formatted
    assert "+" not in formatted
    assert formatted == "invertebrate/seed"


def test_format_cost_and_uses_plus() -> None:
    cost = cards.BirdCost.from_specific({INV: 1, SEED: 1}, is_or_cost=False)
    formatted = display.format_cost(cost)
    assert "+" in formatted
    assert "/" not in formatted
    assert formatted == "invertebrate+seed"


def test_format_cost_free_is_unchanged() -> None:
    assert display.format_cost(cards.BirdCost()) == "free"


def test_format_cost_carolina_chickadee_uses_slash() -> None:
    bird = _BY_NAME["Carolina Chickadee"]
    formatted = display.format_cost(bird.food_cost)
    assert "/" in formatted
    assert "+" not in formatted


def test_format_cost_avocet_uses_plus() -> None:
    bird = _BY_NAME["American Avocet"]
    formatted = display.format_cost(bird.food_cost)
    assert "+" in formatted
