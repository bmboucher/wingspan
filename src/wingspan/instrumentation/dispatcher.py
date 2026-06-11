"""The live event router an ``Engine`` holds.

``Instrumentation`` maps each ``EventName`` to the ordered list of handlers
assigned to it and exposes one typed ``fire`` method per event. The engine calls
those methods directly (``engine.instrumentation.round_start(...)``); each
iterates the event's handlers and invokes the matching method. An engine with no
instrumentation holds the shared ``EMPTY`` router, so firing an event costs one
dict lookup that misses — no per-call work.

This is a plain class, not a Pydantic model: it is a behavioral object wiring
live handler instances together, not a serializable record. The *config*
(``instrumentation.config.InstrumentationConfig``) is the Pydantic record that
round-trips; ``InstrumentationConfig.build`` produces one of these.
"""

from __future__ import annotations

import typing

from wingspan.instrumentation import events

if typing.TYPE_CHECKING:
    from wingspan import cards, decisions, state
    from wingspan.engine import core
    from wingspan.instrumentation import config

# Shared empty handler list returned for any unassigned event, so the no-handler
# fast path allocates nothing.
_NO_HANDLERS: tuple[events.CallbackHandler, ...] = ()


class Instrumentation:
    """Routes game events to the handlers assigned to each.

    ``by_event`` is the resolved assignment (one shared handler instance may
    appear under several events). ``open`` / ``close`` fan out across the
    *unique* handler set so a multi-event handler's resources are acquired and
    released exactly once.
    """

    def __init__(
        self,
        by_event: dict[events.EventName, list[events.CallbackHandler]] | None = None,
    ) -> None:
        self.by_event: dict[events.EventName, list[events.CallbackHandler]] = (
            by_event if by_event is not None else {}
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, context: config.RunContext) -> None:
        """Open every assigned handler once, before the first game."""
        for handler in self._unique_handlers():
            handler.open(context)

    def close(self) -> None:
        """Close every assigned handler once, after the last game."""
        for handler in self._unique_handlers():
            handler.close()

    # ------------------------------------------------------------------
    # Per-event fire methods (one per EventName)
    # ------------------------------------------------------------------

    def game_start(self, *, engine: core.Engine) -> None:
        for handler in self._for(events.EventName.GAME_START):
            typing.cast(events.GameStartHandler, handler).game_start(engine=engine)

    def game_end(self, *, engine: core.Engine) -> None:
        for handler in self._for(events.EventName.GAME_END):
            typing.cast(events.GameEndHandler, handler).game_end(engine=engine)

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        for handler in self._for(events.EventName.ROUND_START):
            typing.cast(events.RoundStartHandler, handler).round_start(
                engine=engine, round_num=round_num
            )

    def round_end(self, *, engine: core.Engine, round_num: int) -> None:
        for handler in self._for(events.EventName.ROUND_END):
            typing.cast(events.RoundEndHandler, handler).round_end(
                engine=engine, round_num=round_num
            )

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        for handler in self._for(events.EventName.TURN_START):
            typing.cast(events.TurnStartHandler, handler).turn_start(
                engine=engine, player=player
            )

    def turn_end(self, *, engine: core.Engine, player: state.Player) -> None:
        for handler in self._for(events.EventName.TURN_END):
            typing.cast(events.TurnEndHandler, handler).turn_end(
                engine=engine, player=player
            )

    def making_decision(
        self, *, engine: core.Engine, decision: decisions.Decision[typing.Any]
    ) -> None:
        for handler in self._for(events.EventName.MAKING_DECISION):
            typing.cast(events.MakingDecisionHandler, handler).making_decision(
                engine=engine, decision=decision
            )

    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
        for handler in self._for(events.EventName.MADE_DECISION):
            typing.cast(events.MadeDecisionHandler, handler).made_decision(
                engine=engine, decision=decision, choice=choice
            )

    def bird_placed(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        bird: cards.Bird,
        habitat: cards.Habitat,
        played_bird: state.PlayedBird,
    ) -> None:
        for handler in self._for(events.EventName.BIRD_PLACED):
            typing.cast(events.BirdPlacedHandler, handler).bird_placed(
                engine=engine,
                player=player,
                bird=bird,
                habitat=habitat,
                played_bird=played_bird,
            )

    def food_gained(
        self, *, engine: core.Engine, player: state.Player, gained: set[cards.Food]
    ) -> None:
        for handler in self._for(events.EventName.FOOD_GAINED):
            typing.cast(events.FoodGainedHandler, handler).food_gained(
                engine=engine, player=player, gained=gained
            )

    def eggs_laid(
        self, *, engine: core.Engine, player: state.Player, count: int
    ) -> None:
        for handler in self._for(events.EventName.EGGS_LAID):
            typing.cast(events.EggsLaidHandler, handler).eggs_laid(
                engine=engine, player=player, count=count
            )

    def cards_drawn(
        self, *, engine: core.Engine, player: state.Player, count: int
    ) -> None:
        for handler in self._for(events.EventName.CARDS_DRAWN):
            typing.cast(events.CardsDrawnHandler, handler).cards_drawn(
                engine=engine, player=player, count=count
            )

    def round_goal_scored(
        self,
        *,
        engine: core.Engine,
        round_num: int,
        goal: cards.EndRoundGoal,
        counts: list[int],
    ) -> None:
        for handler in self._for(events.EventName.ROUND_GOAL_SCORED):
            typing.cast(events.RoundGoalScoredHandler, handler).round_goal_scored(
                engine=engine, round_num=round_num, goal=goal, counts=counts
            )

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
        for handler in self._for(events.EventName.PLAYER_FINAL_SCORED):
            typing.cast(events.PlayerFinalScoredHandler, handler).player_final_scored(
                engine=engine,
                player=player,
                total=total,
                bird_pts=bird_pts,
                bonus_pts=bonus_pts,
                eggs=eggs,
                tucked=tucked,
                cached=cached,
                round_goal=round_goal,
            )

    def setup_applied(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        choice: decisions.SetupChoice,
    ) -> None:
        for handler in self._for(events.EventName.SETUP_APPLIED):
            typing.cast(events.SetupAppliedHandler, handler).setup_applied(
                engine=engine, player=player, choice=choice
            )

    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        for handler in self._for(events.EventName.SETUP_START):
            typing.cast(events.SetupStartHandler, handler).setup_start(
                engine=engine, player=player, dealt_bonus=dealt_bonus
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _for(self, event: events.EventName) -> typing.Sequence[events.CallbackHandler]:
        """The handlers assigned to ``event`` (empty when none)."""
        return self.by_event.get(event, _NO_HANDLERS)

    def _unique_handlers(self) -> list[events.CallbackHandler]:
        """Each distinct handler instance once, preserving first-seen order, so a
        handler assigned to several events is opened / closed only once."""
        seen: dict[int, events.CallbackHandler] = {}
        for handlers in self.by_event.values():
            for handler in handlers:
                seen.setdefault(id(handler), handler)
        return list(seen.values())


# The router an uninstrumented engine holds: every event misses the empty map.
EMPTY = Instrumentation()
