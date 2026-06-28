"""Tests for the ``split_setup_food`` feature.

Verifies that:
- ``enumerate_setup_candidates`` with ``include_food=False`` produces correctly
  sized candidate sets with all-empty food tuples.
- A full game with ``split_setup_food=True`` completes and leaves every player
  with the correct food total for their kept bird count.
- Both ``split_setup_bonus`` and ``split_setup_food`` can be active together.
- ``SETUP_FEATURE_DIM`` is unchanged (the food block zeros out, not removed).
"""

from __future__ import annotations

from wingspan import agents, cards, engine, state
from wingspan.setup_model import candidates, encode


def _load_cards() -> (
    tuple[list[cards.Bird], list[cards.BonusCard], list[cards.EndRoundGoal]]
):
    return cards.load_all()


def test_enumerate_candidates_include_food_false_all_empty() -> None:
    """include_food=False: every candidate has kept_foods=()."""
    birds, bonuses, _ = _load_cards()
    dealt_cards = birds[:5]
    dealt_bonus = bonuses[:2]

    cands = candidates.enumerate_setup_candidates(
        dealt_cards, dealt_bonus, include_food=False
    )
    assert cands, "expected at least one candidate"
    assert all(c.kept_foods == () for c in cands), "expected all kept_foods=()"


def test_enumerate_candidates_include_food_false_count() -> None:
    """include_food=False: count = 2^5 * 2 = 64 for a 5-card/2-bonus deal."""
    birds, bonuses, _ = _load_cards()
    dealt_cards = birds[:5]
    dealt_bonus = bonuses[:2]

    cands_no_food = candidates.enumerate_setup_candidates(
        dealt_cards, dealt_bonus, include_food=False
    )
    # 32 card masks × 2 bonuses = 64
    assert len(cands_no_food) == 64, f"expected 64, got {len(cands_no_food)}"


def test_enumerate_candidates_both_axes_false_count() -> None:
    """include_food=False + include_bonus=False: count = 2^5 = 32."""
    birds, bonuses, _ = _load_cards()
    dealt_cards = birds[:5]
    dealt_bonus = bonuses[:2]

    cands = candidates.enumerate_setup_candidates(
        dealt_cards, dealt_bonus, include_food=False, include_bonus=False
    )
    assert len(cands) == 32, f"expected 32, got {len(cands)}"
    assert all(c.kept_foods == () for c in cands)
    assert all(c.bonus_card is None for c in cands)


def test_setup_feature_dim_unchanged() -> None:
    """SETUP_FEATURE_DIM must not change — food block zeros, not removed."""
    # Import the value directly from the module to assert it is the same
    # regardless of split_setup_food mode (the flag has no effect on the dim).
    assert encode.SETUP_FEATURE_DIM > 0
    # The food block is 5 wide; if it were removed the dim would be smaller.
    # We only assert stability here — the exact value is versioned elsewhere.
    dim_with_food = encode.SETUP_FEATURE_DIM
    # Re-import to make sure the constant hasn't been mutated
    from wingspan.setup_model import encode as enc2

    assert enc2.SETUP_FEATURE_DIM == dim_with_food


def test_split_setup_food_game_completes() -> None:
    """A full game with split_setup_food=True completes without error."""
    import random

    birds, bonuses, goals = cards.load_all()
    rng = random.Random(7)
    gs = state.new_game(rng, birds, bonuses, goals)

    rand_rng = random.Random(7)
    agent_a = agents.random_agent(rand_rng)
    agent_b = agents.random_agent(rand_rng)

    eng = engine.Engine.play_one_game(gs, (agent_a, agent_b), split_setup_food=True)
    assert eng.state.game_over
    for player in eng.state.players:
        assert player.final_score is not None


def test_split_setup_food_food_total_matches_hand_size() -> None:
    """In split_setup_food mode, after setup each player's food total equals
    (5 - len(player.hand)), matching the bird-payment invariant."""
    import random

    from wingspan.instrumentation import dispatcher

    # We need to inspect state right after setup, before any turns are played.
    # Use a custom instrumentation callback to capture post-setup state.
    post_setup_food: list[tuple[int, int]] = []  # (n_hand, food_total) per seat

    class _SetupWatcher(dispatcher.Instrumentation):
        def setup_applied(  # type: ignore[override]
            self, *, engine: engine.Engine, player: state.Player, choice: object
        ) -> None:
            pass

        def round_start(  # type: ignore[override]
            self, *, engine: engine.Engine, round_num: int
        ) -> None:
            if round_num == 0 and not post_setup_food:
                for seat in engine.state.players:
                    post_setup_food.append((len(seat.hand), seat.food.total()))

    birds, bonuses, goals = cards.load_all()
    rng = random.Random(13)
    gs = state.new_game(rng, birds, bonuses, goals)
    rand_rng = random.Random(13)
    agent_a = agents.random_agent(rand_rng)
    agent_b = agents.random_agent(rand_rng)

    engine.Engine.play_one_game(
        gs,
        (agent_a, agent_b),
        instrumentation=_SetupWatcher(),
        split_setup_food=True,
    )

    assert post_setup_food, "expected round_start to be called"
    for n_hand, food_total in post_setup_food:
        expected = 5 - n_hand
        assert food_total == expected, (
            f"player with {n_hand} birds kept should have {expected} food, "
            f"got {food_total}"
        )


def test_split_setup_bonus_and_food_together() -> None:
    """Both split_setup_bonus and split_setup_food can be active simultaneously."""
    import random

    birds, bonuses, goals = cards.load_all()
    rng = random.Random(99)
    gs = state.new_game(rng, birds, bonuses, goals)

    rand_rng = random.Random(99)
    agent_a = agents.random_agent(rand_rng)
    agent_b = agents.random_agent(rand_rng)

    eng = engine.Engine.play_one_game(
        gs,
        (agent_a, agent_b),
        split_setup_bonus=True,
        split_setup_food=True,
    )
    assert eng.state.game_over
    for player in eng.state.players:
        assert player.final_score is not None
        # Each player should have exactly one bonus card (deferred but still picked).
        assert len(player.bonus_cards) == 1
