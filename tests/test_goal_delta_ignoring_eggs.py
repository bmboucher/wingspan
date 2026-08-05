# pyright: reportPrivateUsage=false
# (reads the private layout offsets to isolate the old goal_delta stripe,
# matching the test_encode.py convention)
"""Tests for the v1.6 ``goal_delta_ignoring_eggs`` choice stripe.

Unlike the play-instant ``goal_delta`` stripe (zero for every egg-driven
category on a freshly played bird), ``goal_delta_ignoring_eggs`` prices the
hypothesis that this row's bird is eventually played and egg-populated to
whatever level best advances the goal
(``scoring.goal_vp_delta_for_bird_with_eggs``). It is
filled on bird-card rows only — ``BirdChoice``, ``PlayBirdChoice``, and the
tray ``DrawSourceChoice`` row — mirroring the three call sites that already
fill ``goal_delta``.
"""

from __future__ import annotations

import numpy as np

from wingspan import cards, decisions, encode, engine, state
from wingspan.encode import layout

_NEW_STRIPE_SLICE = slice(
    encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET,
    encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
    + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM,
)
_OLD_SLOT0_COUNT = layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_COUNT
_NEW_SLOT0_COUNT = (
    encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET + layout._GOAL_DELTA_COUNT
)


def _filler_bird(name: str, habitat: cards.Habitat) -> cards.Bird:
    """A minimal synthetic bird for filling board slots — only its habitat and
    ``egg_limit`` matter for the playability guard (row-length only), so it
    need not be a real catalog card."""
    return cards.Bird(
        id=abs(hash(name)) % 100000,
        name=name,
        scientific_name="Testus fillerus",
        color=cards.PowerColor.NONE,
        points=1,
        nest=cards.NestType.BOWL,
        egg_limit=1,
        wingspan_cm=50,
        habitats=(habitat,),
        food_cost=cards.BirdCost(),
        flocking=False,
        predator=False,
        raw_power_text="",
        power=cards.Power(color=cards.PowerColor.NONE),
        bonus_categories=(),
    )


def _set_round0_goal(game_state: state.GameState, category: str) -> None:
    """Install a ``category`` goal as the (unscored) round-0 goal, keeping the
    other three rounds' goals untouched."""
    _, _, all_goals = cards.load_all()
    goal = next(g for g in all_goals if g.category == category)
    game_state.round_goals = [goal, *game_state.round_goals[1:]]


# ---------------------------------------------------------------------------
# (a) a committed PlayBirdChoice row for a bowl bird prices eggs_bowl's count
# slot at egg_limit/5, while goal_delta itself stays zero there


def test_bowl_bird_play_row_prices_eggs_bowl_while_goal_delta_stays_zero():
    eng, *_ = engine.Engine.create(seed=1)
    _set_round0_goal(eng.state, "eggs_bowl")

    all_birds, *_ = cards.load_all()
    bowl_bird = next(
        bird
        for bird in all_birds
        if bird.nest == cards.NestType.BOWL and bird.egg_limit > 0
    )
    habitat = next(iter(bowl_bird.habitats))
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.PlayBirdChoice(
                label=bowl_bird.name, bird=bowl_bird, habitat=habitat
            )
        ],
    )
    row = encode.encode_choices(decision, eng.state)[0]

    # goal_delta (play-instant): a freshly played bird has no eggs yet.
    assert row[_OLD_SLOT0_COUNT] == 0.0
    # goal_delta_ignoring_eggs: optimistically egg-populated to egg_limit.
    assert row[_NEW_SLOT0_COUNT] == bowl_bird.egg_limit / layout._GOAL_COUNT_SCALE


# ---------------------------------------------------------------------------
# (b) a star-nest bird's play row is nonzero for a concrete-nest goal (the
# wildcard rule)


def test_star_nest_bird_play_row_nonzero_for_concrete_nest_goal():
    eng, *_ = engine.Engine.create(seed=2)
    _set_round0_goal(eng.state, "eggs_cavity")

    all_birds, *_ = cards.load_all()
    star_bird = next(
        bird
        for bird in all_birds
        if bird.nest == cards.NestType.STAR and bird.egg_limit > 0
    )
    habitat = next(iter(star_bird.habitats))
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.PlayBirdChoice(
                label=star_bird.name, bird=star_bird, habitat=habitat
            )
        ],
    )
    row = encode.encode_choices(decision, eng.state)[0]

    assert row[_NEW_SLOT0_COUNT] > 0.0


# ---------------------------------------------------------------------------
# (c) an uncommitted BirdChoice row is slot-gated: a fully-occupied matching
# habitat zeroes the whole new stripe


def test_uncommitted_bird_choice_stripe_zero_when_no_slot_open():
    eng, *_ = engine.Engine.create(seed=3)
    _set_round0_goal(eng.state, "eggs_bowl")

    all_birds, *_ = cards.load_all()
    bowl_bird = next(
        bird
        for bird in all_birds
        if bird.nest == cards.NestType.BOWL
        and bird.egg_limit > 0
        and len(bird.habitats) == 1
    )
    habitat = bowl_bird.habitats[0]
    player = eng.state.players[0]
    player.board[habitat] = [
        state.PlayedBird(bird=_filler_bird(f"Filler{slot}", habitat))
        for slot in range(state.ROW_SLOTS)
    ]

    decision = decisions.BirdPowerTuckFromHandDecision(
        player_id=0,
        prompt="t",
        choices=[decisions.BirdChoice(label=bowl_bird.name, bird=bowl_bird)],
    )
    row = encode.encode_choices(decision, eng.state)[0]

    assert np.all(row[_NEW_STRIPE_SLICE] == 0.0)


# ---------------------------------------------------------------------------
# (d) a non-bird row (an egg-lay BoardTargetChoice) leaves the new stripe
# all-zero


def test_non_bird_board_target_row_has_zero_new_stripe():
    eng, birds, *_ = engine.Engine.create(seed=4)
    _set_round0_goal(eng.state, "eggs_bowl")
    eng.state.players[0].board[cards.Habitat.GRASSLAND] = [
        state.PlayedBird(bird=birds[0])
    ]
    target = decisions.BoardTargetChoice(
        label="x", habitat=cards.Habitat.GRASSLAND, slot=0
    )
    decision = decisions.LayEggDecision(player_id=0, prompt="x", choices=[target])
    row = encode.encode_choices(decision, eng.state)[0]

    assert np.all(row[_NEW_STRIPE_SLICE] == 0.0)


# ---------------------------------------------------------------------------
# (e) the new stripe is the last base choice stripe at the N=2 default spec


def test_new_stripe_is_the_last_base_choice_stripe():
    assert (
        encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
        + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        == encode.choice_feature_dim(encode.DEFAULT_SPEC)
    )


# ---------------------------------------------------------------------------
# Bonus: the third call site — a tray DrawSourceChoice row fills the new
# stripe; the blind deck row (no bird) stays zero


def test_tray_draw_source_bird_row_fills_new_stripe_deck_row_stays_zero():
    eng, *_ = engine.Engine.create(seed=6)
    _set_round0_goal(eng.state, "eggs_bowl")

    all_birds, *_ = cards.load_all()
    bowl_bird = next(
        bird
        for bird in all_birds
        if bird.nest == cards.NestType.BOWL and bird.egg_limit > 0
    )
    eng.state.tray[0] = bowl_bird

    decision = decisions.DrawCardsPickSourceDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.DrawSourceChoice(
                label="t", source="tray", tray_index=0, bird=bowl_bird
            ),
            decisions.DrawSourceChoice(label="d", source="deck"),
        ],
    )
    tray_row, deck_row = encode.encode_choices(decision, eng.state)

    assert tray_row[_NEW_SLOT0_COUNT] > 0.0
    assert np.all(deck_row[_NEW_STRIPE_SLICE] == 0.0)
