"""The instrumentation dispatcher fires the right events over a real game.

A single recording handler subscribed to every event asserts the expected
sequence (game start first, game end last, four rounds, two setups/finals,
paired making/made decisions) and that one shared instance accumulates state
across all of them.
"""

from __future__ import annotations

import collections
import os
import pathlib
import random
import sys
import typing

import pydantic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import agents, cards, decisions, engine, state
from wingspan.engine import core
from wingspan.instrumentation import config, dispatcher, events


class _RecordingHandler(
    events.GameStartHandler,
    events.GameEndHandler,
    events.RoundStartHandler,
    events.RoundEndHandler,
    events.TurnStartHandler,
    events.TurnEndHandler,
    events.MakingDecisionHandler,
    events.MadeDecisionHandler,
    events.BirdPlacedHandler,
    events.FoodGainedHandler,
    events.EggsLaidHandler,
    events.CardsDrawnHandler,
    events.RoundGoalScoredHandler,
    events.PlayerFinalScoredHandler,
    events.SetupAppliedHandler,
):
    """Appends each event it sees to a shared ``seen`` list (one instance is
    assigned to every event, so the list demonstrates cross-event state)."""

    _seen: list[str] = pydantic.PrivateAttr(default_factory=list[str])

    @property
    def seen(self) -> list[str]:
        return self._seen

    def game_start(self, *, engine: core.Engine) -> None:
        self._seen.append("game_start")

    def game_end(self, *, engine: core.Engine) -> None:
        self._seen.append("game_end")

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        self._seen.append("round_start")

    def round_end(self, *, engine: core.Engine, round_num: int) -> None:
        self._seen.append("round_end")

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        self._seen.append("turn_start")

    def turn_end(self, *, engine: core.Engine, player: state.Player) -> None:
        self._seen.append("turn_end")

    def making_decision(
        self, *, engine: core.Engine, decision: decisions.Decision[typing.Any]
    ) -> None:
        self._seen.append("making")

    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
        self._seen.append("made")

    def bird_placed(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        bird: cards.Bird,
        habitat: cards.Habitat,
        played_bird: state.PlayedBird,
    ) -> None:
        self._seen.append("bird_placed")

    def food_gained(
        self, *, engine: core.Engine, player: state.Player, gained: set[cards.Food]
    ) -> None:
        self._seen.append("food_gained")

    def eggs_laid(
        self, *, engine: core.Engine, player: state.Player, count: int
    ) -> None:
        self._seen.append("eggs_laid")

    def cards_drawn(
        self, *, engine: core.Engine, player: state.Player, count: int
    ) -> None:
        self._seen.append("cards_drawn")

    def round_goal_scored(
        self,
        *,
        engine: core.Engine,
        round_num: int,
        goal: cards.EndRoundGoal,
        counts: list[int],
    ) -> None:
        self._seen.append("round_goal_scored")

    def player_final_scored(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        total: int,
        bird_pts: int,
        bonus_pts: int,
        eggs: int,
        tucked: int,
        cached: int,
        round_goal: int,
    ) -> None:
        self._seen.append("player_final_scored")

    def setup_applied(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        choice: decisions.SetupChoice,
    ) -> None:
        self._seen.append("setup_applied")


def _play_recorded(seed: int) -> _RecordingHandler:
    recorder = _RecordingHandler()
    router = dispatcher.Instrumentation(
        by_event={event: [recorder] for event in events.EventName}
    )
    eng, *_ = engine.Engine.create(seed=seed)
    rng = random.Random(seed)
    agent = agents.random_agent(rng)
    engine.Engine.play_one_game(eng.state, (agent, agent), instrumentation=router)
    return recorder


def test_event_sequence_brackets_the_game():
    recorder = _play_recorded(seed=7)
    seen = recorder.seen
    assert seen[0] == "game_start"
    assert seen[-1] == "game_end"
    counts = collections.Counter(seen)
    assert counts["setup_applied"] == 2
    assert counts["round_start"] == 4
    assert counts["round_end"] == 4
    assert counts["round_goal_scored"] == 4
    assert counts["player_final_scored"] == 2
    assert counts["turn_start"] == counts["turn_end"] > 0
    assert counts["bird_placed"] >= 1


def test_made_decision_only_for_genuine_decisions():
    # making/made fire as a pair around every *genuine* decision and never for a
    # forced single-option one (the engine returns those without asking), so the
    # two counts match and are positive.
    counts = collections.Counter(_play_recorded(seed=11).seen)
    assert counts["making"] == counts["made"] > 0


def test_uninstrumented_game_still_runs():
    # The default EMPTY router means no instrumentation argument is required.
    eng, *_ = engine.Engine.create(seed=3)
    rng = random.Random(3)
    agent = agents.random_agent(rng)
    engine.Engine.play_one_game(eng.state, (agent, agent))
    assert eng.state.game_over


def test_open_close_visit_each_handler_once():
    # A handler on two events is opened/closed once, not per-event.
    opens: list[int] = []

    class _Counter(events.TurnStartHandler, events.GameEndHandler):
        def open(self, context: config.RunContext) -> None:
            opens.append(1)

        def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
            pass

        def game_end(self, *, engine: core.Engine) -> None:
            pass

    handler = _Counter()
    router = dispatcher.Instrumentation(
        by_event={
            events.EventName.TURN_START: [handler],
            events.EventName.GAME_END: [handler],
        }
    )
    router.open(config.RunContext(output_dir=pathlib.Path("."), run_name="t", seed=0))
    router.close()
    assert sum(opens) == 1
