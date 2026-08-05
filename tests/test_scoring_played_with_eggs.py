"""Tests for the "played and egg-populated" goal helpers in
``wingspan.engine.scoring``.

``goal_count_delta_for_bird`` prices only the play-instant delta, so every
egg-driven round-goal category (nest-egg totals, birds-with-eggs, per-habitat
egg totals, egg sets across the three habitats) returns 0 for it — a freshly
played bird starts with no eggs. ``goal_count_delta_for_bird_with_eggs`` is
the per-goal-optimistic counterpart: it assumes the bird is eventually played
and egg-populated to whatever level best advances the category being priced.
``goal_vp_delta_for_bird_with_eggs`` and ``goal_affinity_for_kept`` build on
it for placement-VP pricing and setup-time hand affinity, respectively.
"""

from __future__ import annotations

from wingspan import cards, state
from wingspan.engine import scoring


def _bird(
    name: str,
    *,
    nest: cards.NestType = cards.NestType.BOWL,
    egg_limit: int = 5,
    habitats: tuple[cards.Habitat, ...] = (cards.Habitat.FOREST,),
) -> cards.Bird:
    return cards.Bird(
        id=abs(hash(name)) % 100000,
        name=name,
        scientific_name="Testus birdus",
        color=cards.PowerColor.NONE,
        points=1,
        nest=nest,
        egg_limit=egg_limit,
        wingspan_cm=50,
        habitats=habitats,
        food_cost=cards.BirdCost(),
        flocking=False,
        predator=False,
        raw_power_text="",
        power=cards.Power(color=cards.PowerColor.NONE),
        bonus_categories=(),
    )


def _player_with(
    entries: list[tuple[cards.Bird, int, cards.Habitat]],
) -> state.Player:
    player = state.Player(id=0, name="P0")
    for bird, eggs, habitat in entries:
        player.board[habitat].append(state.PlayedBird(bird=bird, eggs=eggs))
    return player


def _goal(category: str) -> cards.EndRoundGoal:
    return cards.EndRoundGoal(id=1, description=category, category=category, tile_id=0)


# ---------------------------------------------------------------------------
# goal_count_delta_for_bird_with_eggs: nest-keyed categories


def test_star_nest_bird_advances_every_concrete_nest_goal():
    """A star nest is wild: it prices as a full match for a concrete-nest
    egg-total goal and a birds-with-eggs goal alike. The play-instant delta
    (``goal_count_delta_for_bird``) sees no eggs on a freshly played bird and
    returns 0 for the same inputs — the bug this feature fixes."""
    star_bird = _bird("Star", nest=cards.NestType.STAR, egg_limit=3)

    assert scoring.goal_count_delta_for_bird_with_eggs(star_bird, "eggs_bowl") == 3
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(star_bird, "bowl_birds_with_eggs")
        == 1
    )
    assert scoring.goal_count_delta_for_bird(star_bird, "eggs_bowl") == 0
    assert scoring.goal_count_delta_for_bird(star_bird, "bowl_birds_with_eggs") == 0


def test_concrete_nest_mismatch_and_no_nest_bird_score_zero():
    """A bowl-nesting bird cannot advance a cavity-nest goal, and a
    ``NestType.NONE`` bird (no nest icon) never matches any nest goal —
    unlike a star nest, ``NONE`` is not wild (see ``cards.nest_matches``)."""
    bowl_bird = _bird("Bowl", nest=cards.NestType.BOWL, egg_limit=4)
    assert scoring.goal_count_delta_for_bird_with_eggs(bowl_bird, "eggs_cavity") == 0

    no_nest_bird = _bird("Noner", nest=cards.NestType.NONE, egg_limit=4)
    assert scoring.goal_count_delta_for_bird_with_eggs(no_nest_bird, "eggs_bowl") == 0
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            no_nest_bird, "bowl_birds_with_eggs"
        )
        == 0
    )


def test_birds_with_eggs_requires_positive_egg_limit():
    """A matching-nest bird that can never hold an egg (``egg_limit == 0``)
    cannot advance a ``*_birds_with_eggs`` goal even optimistically."""
    zero_limit_bird = _bird("Zero", nest=cards.NestType.CAVITY, egg_limit=0)
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            zero_limit_bird, "cavity_birds_with_eggs"
        )
        == 0
    )


# ---------------------------------------------------------------------------
# goal_count_delta_for_bird_with_eggs: per-habitat egg totals


def test_eggs_in_habitat_committed_and_uncommitted():
    forest_bird = _bird("Forester", habitats=(cards.Habitat.FOREST,), egg_limit=4)
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            forest_bird, "eggs_forest", play_habitat=cards.Habitat.FOREST
        )
        == 4
    )
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            forest_bird, "eggs_forest", play_habitat=cards.Habitat.GRASSLAND
        )
        == 0
    )
    assert scoring.goal_count_delta_for_bird_with_eggs(forest_bird, "eggs_forest") == 4


# ---------------------------------------------------------------------------
# goal_count_delta_for_bird_with_eggs: the playability guard


def test_playability_guard_blocks_every_category_when_no_slot_open():
    """A player whose forest row is completely full cannot host an
    uncommitted forest-only candidate at all — the play-hypothesis is
    impossible, so every category (not just habitat-keyed ones) prices 0. A
    committed ``play_habitat`` is never guarded: the offered row already
    exists regardless of board occupancy elsewhere."""
    full_forest_player = _player_with(
        [
            (_bird(f"Filler{slot}", egg_limit=1), 0, cards.Habitat.FOREST)
            for slot in range(state.ROW_SLOTS)
        ]
    )
    blocked_bird = _bird(
        "Blocked",
        habitats=(cards.Habitat.FOREST,),
        nest=cards.NestType.BOWL,
        egg_limit=3,
    )

    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            blocked_bird, "eggs_forest", player=full_forest_player
        )
        == 0
    )
    for category in ("eggs_bowl", "total_birds"):
        assert (
            scoring.goal_count_delta_for_bird_with_eggs(
                blocked_bird, category, player=full_forest_player
            )
            == 0
        ), category

    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            blocked_bird,
            "eggs_forest",
            play_habitat=cards.Habitat.FOREST,
            player=full_forest_player,
        )
        == 3
    )


# ---------------------------------------------------------------------------
# goal_count_delta_for_bird_with_eggs: egg_sets_3habitats


def test_egg_sets_committed_uncommitted_and_no_player():
    """Committed: the exact raised-min delta from landing the bird's eggs on
    the offered row. Uncommitted: the best such delta over every reachable
    habitat. With no board context at all (setup pricing), the category is
    priced hand-level instead via ``goal_affinity_for_kept`` and returns 0
    here."""
    egg_sum_player = _player_with(
        [
            (_bird("F", habitats=(cards.Habitat.FOREST,)), 1, cards.Habitat.FOREST),
            (
                _bird("G", habitats=(cards.Habitat.GRASSLAND,)),
                3,
                cards.Habitat.GRASSLAND,
            ),
            (_bird("W", habitats=(cards.Habitat.WETLAND,)), 2, cards.Habitat.WETLAND),
        ]
    )
    # Forest sits at the current min (1). Landing a 2-egg-limit bird there
    # raises forest to 3, and the new min is wetland's 2 -> delta 1.
    raiser = _bird("Raiser", habitats=(cards.Habitat.FOREST,), egg_limit=2)
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            raiser,
            "egg_sets_3habitats",
            play_habitat=cards.Habitat.FOREST,
            player=egg_sum_player,
        )
        == 1
    )

    # Uncommitted: the bird could land in forest (the scarce habitat) or
    # grassland; the optimistic bound takes the best of the two, which is
    # the forest placement (same delta as committed above).
    dual_bird = _bird(
        "Dual",
        habitats=(cards.Habitat.FOREST, cards.Habitat.GRASSLAND),
        egg_limit=2,
    )
    assert (
        scoring.goal_count_delta_for_bird_with_eggs(
            dual_bird, "egg_sets_3habitats", player=egg_sum_player
        )
        == 1
    )

    assert (
        scoring.goal_count_delta_for_bird_with_eggs(dual_bird, "egg_sets_3habitats")
        == 0
    )


# ---------------------------------------------------------------------------
# Parity with the play-instant delta on categories the two agree on


def test_parity_with_play_instant_delta_on_bird_count_categories():
    """Bird-count categories (habitat spread, total, no-eggs) are identical
    by design between the two helpers: per-goal optimism means simply not
    laying eggs on the freshly played bird, which is exactly what the
    play-instant delta already prices."""
    bird = _bird(
        "Parity",
        habitats=(cards.Habitat.FOREST, cards.Habitat.WETLAND),
        egg_limit=2,
    )
    for category in ("total_birds", "birds_forest", "birds_no_eggs"):
        assert scoring.goal_count_delta_for_bird_with_eggs(
            bird, category
        ) == scoring.goal_count_delta_for_bird(bird, category), category


# ---------------------------------------------------------------------------
# goal_affinity_for_kept


def test_goal_affinity_for_kept_star_nest_counts_toward_every_concrete_nest():
    """The reported bug scenario: a kept star-nest bird should count toward
    every concrete-nest goal category it could be dealt, not just the one it
    happens to match exactly."""
    bowl_bird = _bird("BowlKeep", nest=cards.NestType.BOWL, egg_limit=4)
    cavity_bird = _bird("CavityKeep", nest=cards.NestType.CAVITY, egg_limit=3)
    star_bird = _bird("StarKeep", nest=cards.NestType.STAR, egg_limit=5)
    kept = [bowl_bird, cavity_bird, star_bird]

    assert scoring.goal_affinity_for_kept(kept, "bowl_birds_with_eggs") == 2
    assert scoring.goal_affinity_for_kept(kept, "cavity_birds_with_eggs") == 2
    assert scoring.goal_affinity_for_kept(kept, "eggs_ground") == star_bird.egg_limit
    assert scoring.goal_affinity_for_kept(kept, "total_birds") == 3


def test_best_kept_egg_sets_three_single_habitat_birds():
    """Each bird can only land in its one printed habitat, so the set count
    is forced to the minimum of the three egg limits."""
    forest_bird = _bird("F", habitats=(cards.Habitat.FOREST,), egg_limit=2)
    grassland_bird = _bird("G", habitats=(cards.Habitat.GRASSLAND,), egg_limit=4)
    wetland_bird = _bird("W", habitats=(cards.Habitat.WETLAND,), egg_limit=3)
    kept = [forest_bird, grassland_bird, wetland_bird]
    assert scoring.goal_affinity_for_kept(kept, "egg_sets_3habitats") == 2


def test_best_kept_egg_sets_two_habitats_only_scores_zero():
    """With only two of the three habitats coverable at all, the third stays
    at 0 eggs and the set count (the min across all three) is 0."""
    forest_bird = _bird("F2", habitats=(cards.Habitat.FOREST,), egg_limit=5)
    grassland_bird = _bird("G2", habitats=(cards.Habitat.GRASSLAND,), egg_limit=5)
    kept = [forest_bird, grassland_bird]
    assert scoring.goal_affinity_for_kept(kept, "egg_sets_3habitats") == 0


def test_best_kept_egg_sets_forces_multi_habitat_bird_into_scarce_habitat():
    """Two single-habitat birds already cover forest and grassland; a third,
    multi-habitat bird could also land in forest, but the optimal assignment
    routes it to wetland (the only habitat it alone can cover) to complete a
    set there instead."""
    forest_bird = _bird("F3", habitats=(cards.Habitat.FOREST,), egg_limit=5)
    grassland_bird = _bird("G3", habitats=(cards.Habitat.GRASSLAND,), egg_limit=5)
    flex_bird = _bird(
        "Flex", habitats=(cards.Habitat.FOREST, cards.Habitat.WETLAND), egg_limit=3
    )
    kept = [forest_bird, grassland_bird, flex_bird]
    assert scoring.goal_affinity_for_kept(kept, "egg_sets_3habitats") == 3


# ---------------------------------------------------------------------------
# goal_vp_delta_for_bird_with_eggs


def test_goal_vp_delta_for_bird_with_eggs_prices_count_and_vp():
    """Two players both starting at 0 on an ``eggs_bowl`` goal: a star/bowl
    bird's optimistic egg count both moves the category count and flips the
    placement VP from nothing (a count of 0 never places) to first."""
    goal = _goal("eggs_bowl")
    payouts = state.ROUND_GOAL_PAYOUTS[0]
    deciding_player = _player_with([])
    opponent = _player_with([])
    bird = _bird(
        "VPBowl",
        nest=cards.NestType.BOWL,
        egg_limit=4,
        habitats=(cards.Habitat.FOREST,),
    )

    count_delta, vp_delta = scoring.goal_vp_delta_for_bird_with_eggs(
        deciding_player, [opponent], goal, bird, payouts
    )
    assert count_delta == 4
    assert vp_delta > 0
