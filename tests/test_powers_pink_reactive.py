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

from wingspan import agents, cards, engine, state  # noqa: E402
from wingspan.engine import reactors  # noqa: E402


def _engine() -> engine.Engine:
    eng, *_ = engine.Engine.create(seed=1)
    eng.agents = [
        agents.random_agent(random.Random(1)),
        agents.random_agent(random.Random(2)),
    ]
    return eng


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
    lark = state.PlayedBird(bird=_bird_named("Horned Lark"))
    eng.state.players[1].board[cards.Habitat.GRASSLAND] = [lark]
    eng.state.players[1].hand = [_bird_named("Belted Kingfisher")]
    reactors.trigger_pink_play_bird_reactors(
        eng, eng.state.players[0], cards.Habitat.GRASSLAND
    )
    assert lark.tucked_cards == 1
    assert eng.state.players[1].hand == []


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
