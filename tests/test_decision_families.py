"""Tests for the judgment-family taxonomy (``decisions.DecisionFamily``).

The RL model trains one scoring head per judgment family rather than one per
decision class (DECISIONS.md §0/§5). These tests pin the invariants the model
relies on:

1. The family map is *total* over ``ALL_DECISION_CLASSES`` — every decision
   routes somewhere, so a newly added ``Decision`` subclass that forgets to
   register a family fails loudly here.
2. Family indices are stable and in range for the model's ``ModuleList`` of
   per-family heads.
3. ``ALL_DECISION_FAMILIES`` covers every ``DecisionFamily`` member exactly
   once and every family is actually used (no orphan head).
4. The specific groupings the review argued for: acquiring a bird and giving
   one up are *distinct* families; gaining and spending food are distinct;
   choosing the turn's action type is split from choosing which bird to play;
   egg placement and removal are distinct; the rare/structural powers stay
   pooled.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import decisions


def test_family_map_is_total_over_all_decision_classes():
    """Every registered decision class resolves to a family (no KeyError)."""
    for decision_class in decisions.ALL_DECISION_CLASSES:
        family = decisions.family_for(decision_class)
        assert isinstance(family, decisions.DecisionFamily)


def test_family_indices_match_family_order_and_are_in_range():
    num_families = len(decisions.ALL_DECISION_FAMILIES)
    for decision_class in decisions.ALL_DECISION_CLASSES:
        idx = decisions.family_index_for(decision_class)
        assert 0 <= idx < num_families
        assert decisions.ALL_DECISION_FAMILIES[idx] == decisions.family_for(
            decision_class
        )


def test_all_decision_families_tuple_covers_enum_exactly_once():
    """``ALL_DECISION_FAMILIES`` is the stable head order; it must list every
    enum member exactly once (no dupes, no omissions)."""
    assert len(decisions.ALL_DECISION_FAMILIES) == len(decisions.DecisionFamily)
    assert set(decisions.ALL_DECISION_FAMILIES) == set(decisions.DecisionFamily)


def test_every_family_is_used_by_some_decision():
    """No orphan heads: each family in the stable order is the family of at
    least one as-built decision class."""
    used = {decisions.family_for(cls) for cls in decisions.ALL_DECISION_CLASSES}
    assert used == set(decisions.ALL_DECISION_FAMILIES)


def test_bird_acquisition_and_discard_are_distinct_families():
    """ "Which bird do I take?" (draw source, draft keep) and "which bird do I
    give up?" (tuck from hand, discard a card for food) are opposite judgments
    and must route to different heads (DECISIONS.md §3.3, review point 1)."""
    acquisition = {
        decisions.family_for(cls)
        for cls in (
            decisions.DrawCardsPickSourceDecision,
            decisions.BirdPowerPickBirdFromHandDecision,
        )
    }
    discard = {
        decisions.family_for(cls)
        for cls in (
            decisions.BirdPowerTuckFromHandDecision,
            decisions.GainExtraFoodDecision,
        )
    }
    assert acquisition == {decisions.DecisionFamily.BIRD_ACQUISITION}
    assert discard == {decisions.DecisionFamily.BIRD_DISCARD}
    assert acquisition.isdisjoint(discard)


def test_choosing_an_action_and_choosing_a_bird_to_play_are_distinct():
    """Picking the turn's action *type* (``MainActionDecision`` -> ``MACRO_ACTION``)
    and picking which bird to play (``PlayBirdDecision`` -> ``PLAY_BIRD``, used for
    both the main ``PLAY_BIRD`` branch and extra plays) are separate judgments on
    separate heads (the hierarchical macro split)."""
    assert (
        decisions.family_for(decisions.MainActionDecision)
        == decisions.DecisionFamily.MACRO_ACTION
    )
    assert (
        decisions.family_for(decisions.PlayBirdDecision)
        == decisions.DecisionFamily.PLAY_BIRD
    )
    assert decisions.DecisionFamily.MACRO_ACTION != decisions.DecisionFamily.PLAY_BIRD


def test_gain_food_and_spend_food_are_distinct_families():
    """Gaining a food and giving one up are inverse judgments (review points
    3 and 5)."""
    assert (
        decisions.family_for(decisions.GainFoodDecision)
        == decisions.DecisionFamily.GAIN_FOOD
    )
    assert (
        decisions.family_for(decisions.SpendFoodDecision)
        == decisions.DecisionFamily.SPEND_FOOD
    )
    assert decisions.DecisionFamily.GAIN_FOOD != decisions.DecisionFamily.SPEND_FOOD


def test_egg_placement_and_removal_are_distinct_families():
    """Placement and removal use the same ``BoardTargetChoice`` shape but
    opposite judgments, so they must route to different heads (review point 6
    renamed the removal decision)."""
    placement = decisions.family_for(decisions.LayEggDecision)
    removal = decisions.family_for(decisions.RemoveEggDecision)
    assert placement == decisions.DecisionFamily.EGG_PLACEMENT
    assert removal == decisions.DecisionFamily.EGG_REMOVAL
    assert placement != removal


def test_accept_exchange_is_commit_to_cost():
    """The unified yes/no "accept exchange?" decision routes to the
    commit-to-cost head (review point 7)."""
    assert (
        decisions.family_for(decisions.AcceptExchangeDecision)
        == decisions.DecisionFamily.COMMIT_TO_COST
    )


def test_setup_has_its_own_family():
    assert (
        decisions.family_for(decisions.SetupDecision) == decisions.DecisionFamily.SETUP
    )


def test_habitat_placement_head():
    assert (
        decisions.family_for(decisions.BirdPowerPickHabitatDecision)
        == decisions.DecisionFamily.HABITAT_PLACEMENT
    )


def test_rare_structural_powers_share_the_misc_head():
    """Repeat-a-power (#12) and pick-starting-player (#15) are too rare to give
    their own heads, so they stay pooled in the misc/rare head (review point
    8 / DECISIONS.md §3.11)."""
    assert (
        decisions.family_for(decisions.BirdPowerPickPlayedBirdDecision)
        == decisions.DecisionFamily.MISC_RARE
    )
    assert (
        decisions.family_for(decisions.BirdPowerPickStartingPlayerDecision)
        == decisions.DecisionFamily.MISC_RARE
    )
