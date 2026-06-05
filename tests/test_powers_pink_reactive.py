"""Pink reactive powers must fire on *another* player's action.

These four core birds were previously dead: their "when another player ..."
triggers had no parser pattern, so the generic matchers mis-modelled them as the
bird's own when-played effect, which never fires for a pink bird. This pins the
fix (parser routing + the engine reactor hooks):

* Belted Kingfisher / Eastern Kingbird — gain a food when an opponent plays a
  bird in the matching habitat;
* Horned Lark — tuck a card from hand on the same trigger;
* Loggerhead Shrike — cache a rodent when an opponent gains one.
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import typing  # noqa: E402

from wingspan import agents, cards, decisions, engine, state  # noqa: E402
from wingspan.engine import core as engine_core  # noqa: E402
from wingspan.engine import reactors


def _engine() -> engine.Engine:
    eng, *_ = engine.Engine.create(seed=1)
    eng.agents = [
        agents.random_agent(random.Random(1)),
        agents.random_agent(random.Random(2)),
    ]
    return eng


def _always_activate_agent() -> engine_core.Agent:
    """Scripted agent that always picks the first non-skip choice."""

    def _agent[C: decisions.Choice](
        eng: engine_core.Engine, decision: decisions.Decision[C]
    ) -> C:
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    return typing.cast(engine_core.Agent, _agent)


def _always_skip_agent() -> engine_core.Agent:
    """Scripted agent that always picks skip when available, else first choice."""

    def _agent[C: decisions.Choice](
        eng: engine_core.Engine, decision: decisions.Decision[C]
    ) -> C:
        for choice in decision.choices:
            if isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    return typing.cast(engine_core.Agent, _agent)


def _bird_named(name: str) -> cards.Bird:
    birds, _, _ = cards.load_all()
    return next(bird for bird in birds if bird.name == name)


def test_belted_kingfisher_gains_fish_when_opponent_plays_in_wetland() -> None:
    eng = _engine()
    eng.state.players[1].board[cards.Habitat.WETLAND] = [
        state.PlayedBird(bird=_bird_named("Belted Kingfisher"))
    ]
    before = eng.state.players[1].food[cards.Food.FISH]
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.WETLAND
    )
    assert eng.state.players[1].food[cards.Food.FISH] == before + 1


def test_kingfisher_silent_on_wrong_habitat() -> None:
    eng = _engine()
    eng.state.players[1].board[cards.Habitat.WETLAND] = [
        state.PlayedBird(bird=_bird_named("Belted Kingfisher"))
    ]
    before = eng.state.players[1].food[cards.Food.FISH]
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.FOREST
    )
    assert eng.state.players[1].food[cards.Food.FISH] == before


def test_kingfisher_does_not_fire_on_its_own_owners_play() -> None:
    # "another player" — the active player's own reactor must not fire.
    eng = _engine()
    eng.state.players[0].board[cards.Habitat.WETLAND] = [
        state.PlayedBird(bird=_bird_named("Belted Kingfisher"))
    ]
    before = eng.state.players[0].food[cards.Food.FISH]
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.WETLAND
    )
    assert eng.state.players[0].food[cards.Food.FISH] == before


def test_horned_lark_tucks_a_card_when_opponent_plays_in_grassland() -> None:
    eng = _engine()
    eng.agents[1] = _always_activate_agent()
    lark = state.PlayedBird(bird=_bird_named("Horned Lark"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [lark]
    eng.state.players[1].hand = [_bird_named("Belted Kingfisher")]
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )
    assert lark.tucked_cards == 1
    assert eng.state.players[1].hand == []


def test_horned_lark_skip_leaves_hand_unchanged() -> None:
    eng = _engine()
    eng.agents[1] = _always_skip_agent()
    lark = state.PlayedBird(bird=_bird_named("Horned Lark"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [lark]
    eng.state.players[1].hand = [_bird_named("Belted Kingfisher")]
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )
    assert lark.tucked_cards == 0
    assert len(eng.state.players[1].hand) == 1


def test_horned_lark_silent_when_hand_empty() -> None:
    eng = _engine()
    eng.agents[1] = _always_activate_agent()
    lark = state.PlayedBird(bird=_bird_named("Horned Lark"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [lark]
    eng.state.players[1].hand = []
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )
    assert lark.tucked_cards == 0


def test_loggerhead_shrike_caches_rodent_when_opponent_gains_rodent() -> None:
    eng = _engine()
    shrike = state.PlayedBird(bird=_bird_named("Loggerhead Shrike"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [shrike]
    reactors.trigger_pink_gain_food_reactors(
        eng, eng.state.players[0], {cards.Food.RODENT}
    )
    assert shrike.cached_food[cards.Food.RODENT] == 1


def test_loggerhead_shrike_silent_when_no_rodent_gained() -> None:
    eng = _engine()
    shrike = state.PlayedBird(bird=_bird_named("Loggerhead Shrike"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [shrike]
    reactors.trigger_pink_gain_food_reactors(
        eng, eng.state.players[0], {cards.Food.SEED}
    )
    assert shrike.cached_food[cards.Food.RODENT] == 0


# ---------------------------------------------------------------------------
# Gap #10 — once-between-turns cap


def test_pink_fired_cap_fires_only_once_per_window() -> None:
    """A pink bird fires at most once per between-turns window; the second
    trigger within the same window is silently skipped (gap #10)."""
    eng = _engine()
    shrike = state.PlayedBird(bird=_bird_named("Loggerhead Shrike"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [shrike]

    reactors.trigger_pink_gain_food_reactors(
        eng, eng.state.players[0], {cards.Food.RODENT}
    )
    reactors.trigger_pink_gain_food_reactors(
        eng, eng.state.players[0], {cards.Food.RODENT}
    )
    assert (
        shrike.cached_food[cards.Food.RODENT] == 1
    ), "cap: second trigger in the same window must be silently skipped"


def test_pink_fired_reset_allows_fire_again() -> None:
    """After ``pink_fired`` is cleared (simulating a new turn), the bird can
    fire again in the next between-turns window (gap #10)."""
    eng = _engine()
    shrike = state.PlayedBird(bird=_bird_named("Loggerhead Shrike"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [shrike]

    reactors.trigger_pink_gain_food_reactors(
        eng, eng.state.players[0], {cards.Food.RODENT}
    )
    assert shrike.pink_fired is True

    shrike.pink_fired = False  # simulates _take_turn clearing the flag
    reactors.trigger_pink_gain_food_reactors(
        eng, eng.state.players[0], {cards.Food.RODENT}
    )
    assert (
        shrike.cached_food[cards.Food.RODENT] == 2
    ), "after reset the bird must be able to fire again"


def test_pink_fired_decline_does_not_consume_use() -> None:
    """Declining a reactive offer does NOT set ``pink_fired``; the bird can
    still fire if the same trigger re-occurs in the window (gap #10)."""
    eng = _engine()
    eng.agents[1] = _always_skip_agent()
    lark = state.PlayedBird(bird=_bird_named("Horned Lark"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [lark]
    eng.state.players[1].hand = [_bird_named("Belted Kingfisher")]

    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )
    assert lark.tucked_cards == 0
    assert lark.pink_fired is False, "a declined fire must NOT set pink_fired"

    # Second trigger: switch to an activating agent — bird should still fire.
    eng.agents[1] = _always_activate_agent()
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )
    assert lark.tucked_cards == 1
    assert lark.pink_fired is True


# ---------------------------------------------------------------------------
# Gap #21 — exclude_self wording


def test_pink_lay_egg_another_bird_sets_exclude_self_true() -> None:
    """'lay 1 egg on another bird' should parse with exclude_self=True (gap #21)."""
    power = cards.parse_power(
        cards.PowerColor.PINK,
        "When another player takes the [lay eggs] action, "
        "lay 1 [egg] on another bird with a [bowl] nest.",
    )
    eff = next(
        e for e in power.effects if e.kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST
    )
    assert eff.exclude_self is True


def test_pink_lay_egg_a_bird_sets_exclude_self_false() -> None:
    """'lay 1 egg on a bird' should parse with exclude_self=False (gap #21)."""
    power = cards.parse_power(
        cards.PowerColor.PINK,
        "When another player takes the [lay eggs] action, "
        "lay 1 [egg] on a bird with a [bowl] nest.",
    )
    eff = next(
        e for e in power.effects if e.kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST
    )
    assert eff.exclude_self is False


def test_fire_pink_lay_egg_exclude_self_true_skips_reactor_as_sole_target() -> None:
    """When exclude_self=True and the reactor is the only eligible target, no
    egg is laid and ``fire_pink_lay_egg`` returns False (gap #21)."""
    eng = _engine()
    eng.agents[1] = _always_activate_agent()

    # Find any bird whose pink power uses PINK_LAY_EGG_ON_NEST with exclude_self=True.
    reactor_bird = next(
        bird
        for bird in (cards.load_all()[0])
        if any(
            e.kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST and e.exclude_self
            for e in bird.power.effects
        )
    )
    eff = next(
        e
        for e in reactor_bird.power.effects
        if e.kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST
    )
    nest = eff.nest
    assert nest is not None

    # Place the reactor bird as the ONLY eligible bird on P1's board.
    pb = state.PlayedBird(bird=reactor_bird, eggs=0)
    for h in cards.ALL_HABITATS:
        eng.state.players[1].board[h].clear()
    eng.state.players[1].board[reactor_bird.habitats[0]] = [pb]

    committed = reactors.fire_pink_lay_egg(
        eng, eng.state.players[1], pb, reactor_bird.habitats[0], eff
    )
    assert not committed
    assert pb.eggs == 0


def test_fire_pink_lay_egg_exclude_self_false_allows_self_targeting() -> None:
    """When exclude_self=False the reactor bird IS a legal target for its own egg (gap #21)."""
    eng = _engine()

    # Synthesise an effect with exclude_self=False for a bowl nest.
    eff = cards.Effect(
        kind=cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        nest=cards.NestType.BOWL,
        exclude_self=False,
        raw_text="test",
    )
    # Find any bowl bird we can use as the reactor.
    bowl_bird = next(
        bird
        for bird in (cards.load_all()[0])
        if bird.nest == cards.NestType.BOWL and bird.egg_limit >= 1
    )
    pb = state.PlayedBird(bird=bowl_bird, eggs=0)
    for h in cards.ALL_HABITATS:
        eng.state.players[1].board[h].clear()
    eng.state.players[1].board[bowl_bird.habitats[0]] = [pb]

    # Use an always-activate agent — it will pick the first BoardTargetChoice.
    eng.agents[1] = _always_activate_agent()
    committed = reactors.fire_pink_lay_egg(
        eng, eng.state.players[1], pb, bowl_bird.habitats[0], eff
    )
    assert committed
    assert pb.eggs == 1  # self-targeted successfully
