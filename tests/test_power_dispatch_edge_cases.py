"""Tests for the dispatch module's null-handler paths and lay_one_egg_on_nest
optional skip — edge cases that don't surface through any real bird's activation."""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state
from wingspan.engine.powers import dispatch


def _no_agent[C: decisions.Choice](
    _eng: engine.Engine, _decision: decisions.Decision[C]
) -> C:
    raise AssertionError(
        f"agent should not be consulted (got {type(_decision).__name__})"
    )


def _minimal_engine() -> tuple[engine.Engine, state.Player, state.PlayedBird]:
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(0), birds, bonuses, goals)
    gs.current_player = 0
    eng = engine.Engine(gs)
    player = gs.me()
    template = next(bird for bird in birds if bird.color == cards.PowerColor.BROWN)
    pb = state.PlayedBird(bird=template)
    return eng, player, pb


# ---------------------------------------------------------------------------
# Null-handler paths: pink effects and UNIMPLEMENTED

_PINK_KINDS = (
    cards.EffectKind.PINK_LAY_EGG_ON_NEST,
    cards.EffectKind.PINK_PREDATOR_FEEDER,
    cards.EffectKind.PINK_PLAY_BIRD_GAIN,
    cards.EffectKind.PINK_PLAY_BIRD_TUCK,
    cards.EffectKind.PINK_GAIN_FOOD_CACHE,
)


def test_apply_effect_pink_kinds_are_silent_no_ops():
    """All five pink EffectKinds pass through apply_effect as a silent no-op;
    they fire from the reactor hooks, not from here."""
    eng, player, pb = _minimal_engine()
    eggs_before = {
        p.id: sum(slot.eggs for row in p.board.values() for slot in row)
        for p in eng.state.players
    }
    for pink_kind in _PINK_KINDS:
        eff = cards.Effect(kind=pink_kind, amount=0)
        dispatch.apply_effect(
            eng, _no_agent, player, pb, cards.Habitat.WETLAND, eff, "activate"
        )
    eggs_after = {
        p.id: sum(slot.eggs for row in p.board.values() for slot in row)
        for p in eng.state.players
    }
    assert eggs_before == eggs_after


def test_apply_effect_unimplemented_logs_skip_message():
    """An UNIMPLEMENTED effect logs a 'not modeled' line — surfacing unparsed
    powers so they can be identified and added."""
    eng, player, pb = _minimal_engine()
    raw = "Gain 999 mystery cubes."
    bird_with_raw = pb.bird.model_copy(update={"raw_power_text": raw})
    pb_raw = state.PlayedBird(bird=bird_with_raw)
    eff = cards.Effect(kind=cards.EffectKind.UNIMPLEMENTED, amount=0, raw_text=raw)

    log_lines: list[str] = []
    eng.log = lambda msg, player_id=None: log_lines.append(msg)  # type: ignore[method-assign]
    dispatch.apply_effect(
        eng, _no_agent, player, pb_raw, cards.Habitat.WETLAND, eff, "activate"
    )
    assert any(
        "not modeled" in line for line in log_lines
    ), f"Expected 'not modeled' in log; got: {log_lines}"


# ---------------------------------------------------------------------------
# lay_one_egg_on_nest anti-egg-goal skip


def test_lay_one_egg_on_nest_birds_no_eggs_skip_returns_none():
    """When the birds_no_eggs round goal is active, lay_one_egg_on_nest offers
    an AcceptExchangeDecision veto gate before the mandatory pick.  Answering
    with SkipChoice returns None and leaves the bird's egg count unchanged."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(0), birds, bonuses, goals)
    gs.current_player = 0

    # Inject the anti-egg goal into the active round slot.
    anti_egg_goal = cards.EndRoundGoal(
        id=0, description="[bird] with no [egg]", category="birds_no_eggs", tile_id=0
    )
    gs.round_goals = [anti_egg_goal] * 4

    eng = engine.Engine(gs)
    player = gs.me()

    # Stage a bowl-nest bird with room for at least one egg.
    bowl_bird = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.BOWL and bird.egg_limit > 0
    )
    bowl_pb = state.PlayedBird(bird=bowl_bird, eggs=0)
    player.board[cards.Habitat.WETLAND] = [bowl_pb]

    def skip_agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        return typing.cast(
            C,
            next(ch for ch in decision.choices if isinstance(ch, decisions.SkipChoice)),
        )

    eng.agents = [skip_agent, skip_agent]
    result: state.PlayedBird | None = dispatch.lay_one_egg_on_nest(
        eng, player, cards.NestType.BOWL, "test-label"
    )
    assert result is None
    assert bowl_pb.eggs == 0
