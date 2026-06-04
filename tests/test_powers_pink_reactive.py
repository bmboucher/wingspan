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
