"""Event taxonomy and the per-shape abstract callback-handler base classes.

The instrumentation framework exposes a fixed set of named game *events*
(``EventName``). Each event has its own abstract ``CallbackHandler`` subclass
carrying a single keyword-only method whose signature *is* that event's shape
(e.g. ``RoundStartHandler.round_start(*, engine, round_num)``). A concrete
recorder subclasses one or more of these bases — multiply-inheriting several to
observe several events with shared instance state — and implements the matching
methods.

All game-object parameters (``Engine``, ``Player``, ``Bird`` …) are referenced
only inside ``if typing.TYPE_CHECKING`` so this package imports no ``engine`` /
``decisions`` / ``state`` / ``cards`` at runtime — keeping it free of the import
cycle the engine would otherwise create (``engine.core`` imports the dispatcher).
Because the type names appear only in *method* annotations (never as Pydantic
*fields*), and every module uses ``from __future__ import annotations``, Pydantic
never tries to resolve them at model-build time.
"""

from __future__ import annotations

import abc
import enum
import typing

import pydantic

if typing.TYPE_CHECKING:
    from wingspan import cards, decisions, state
    from wingspan.engine import core
    from wingspan.instrumentation import config


class EventName(enum.StrEnum):
    """The fixed set of instrumented game events.

    Each member's value is also the method name the matching handler base
    declares (``ROUND_START`` -> ``RoundStartHandler.round_start``), so the
    dispatcher and the handler bases stay in lockstep.
    """

    GAME_START = "game_start"
    GAME_END = "game_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MAKING_DECISION = "making_decision"
    MADE_DECISION = "made_decision"
    BIRD_PLACED = "bird_placed"
    FOOD_GAINED = "food_gained"
    EGGS_LAID = "eggs_laid"
    CARDS_DRAWN = "cards_drawn"
    ROUND_GOAL_SCORED = "round_goal_scored"
    PLAYER_FINAL_SCORED = "player_final_scored"
    SETUP_APPLIED = "setup_applied"
    SETUP_START = "setup_start"


class CallbackHandler(pydantic.BaseModel):
    """Abstract base of every instrumentation handler.

    A handler's declared Pydantic *fields* are its plain, serializable
    configuration (the constructor kwargs named in the run config). All
    *runtime* state — open files, accumulators — must live in
    ``pydantic.PrivateAttr`` so it is excluded from ``model_dump`` and a
    handler reconstructed from config (e.g. on a checkpoint reload) starts with
    fresh state. ``open`` / ``close`` bracket a run so a handler can acquire and
    release those resources.
    """

    def open(self, context: config.RunContext) -> None:
        """Called once per run, before any event fires. Override to open output
        files or otherwise acquire run-scoped resources."""

    def close(self) -> None:
        """Called once per run, after the last event. Override to flush and
        release whatever ``open`` acquired."""


class GameStartHandler(CallbackHandler):
    """Handler invoked once when a game begins."""

    @abc.abstractmethod
    def game_start(self, *, engine: core.Engine) -> None:
        """Fired at the start of each game, before the setup phase."""


class GameEndHandler(CallbackHandler):
    """Handler invoked once when a game ends (after final scoring)."""

    @abc.abstractmethod
    def game_end(self, *, engine: core.Engine) -> None:
        """Fired after final scoring, with every ``Player.final_score`` set."""


class RoundStartHandler(CallbackHandler):
    """Handler invoked at the start of each of the four rounds."""

    @abc.abstractmethod
    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        """Fired after per-round state reset, before the first turn.
        ``round_num`` is the 0-based round index."""


class RoundEndHandler(CallbackHandler):
    """Handler invoked at the end of each round (after goal scoring)."""

    @abc.abstractmethod
    def round_end(self, *, engine: core.Engine, round_num: int) -> None:
        """Fired after the round goal is scored and the tray reset."""


class TurnStartHandler(CallbackHandler):
    """Handler invoked at the start of each player turn."""

    @abc.abstractmethod
    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        """Fired after the turn's scratch state is reset, before the main
        action decision."""


class TurnEndHandler(CallbackHandler):
    """Handler invoked at the end of each player turn."""

    @abc.abstractmethod
    def turn_end(self, *, engine: core.Engine, player: state.Player) -> None:
        """Fired after extra plays are consumed and the tray refilled."""


class MakingDecisionHandler(CallbackHandler):
    """Handler invoked just before an agent resolves a genuine decision."""

    @abc.abstractmethod
    def making_decision(
        self, *, engine: core.Engine, decision: decisions.Decision[typing.Any]
    ) -> None:
        """Fired immediately before the agent is asked. Forced single-option
        decisions never reach here (they are not real decisions)."""


class MadeDecisionHandler(CallbackHandler):
    """Handler invoked right after an agent resolves a genuine decision."""

    @abc.abstractmethod
    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
        """Fired after the chosen option is validated against the offered
        choices. The universal decision choke point — every decision type
        flows through here."""


class BirdPlacedHandler(CallbackHandler):
    """Handler invoked when a bird is placed onto a player's board."""

    @abc.abstractmethod
    def bird_placed(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        bird: cards.Bird,
        habitat: cards.Habitat,
        played_bird: state.PlayedBird,
    ) -> None:
        """Fired the moment the bird lands in its habitat row, before its
        WHITE 'when played' power and any pink reactors resolve."""


class FoodGainedHandler(CallbackHandler):
    """Handler invoked when a Gain Food action completes."""

    @abc.abstractmethod
    def food_gained(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        gained: set[cards.Food],
    ) -> None:
        """Fired after all dice are taken, conversions offered, row powers
        activated, and pink reactors resolved. ``gained`` is the set of food
        types whose supply rose during the action."""


class EggsLaidHandler(CallbackHandler):
    """Handler invoked when a Lay Eggs action completes."""

    @abc.abstractmethod
    def eggs_laid(
        self, *, engine: core.Engine, player: state.Player, count: int
    ) -> None:
        """Fired after the lay action fully resolves; ``count`` is the number
        of eggs the base action laid (excluding the optional conversion)."""


class CardsDrawnHandler(CallbackHandler):
    """Handler invoked when a Draw Cards action completes."""

    @abc.abstractmethod
    def cards_drawn(
        self, *, engine: core.Engine, player: state.Player, count: int
    ) -> None:
        """Fired after the draw action fully resolves; ``count`` is the number
        of cards the base action drew (excluding the optional conversion)."""


class RoundGoalScoredHandler(CallbackHandler):
    """Handler invoked when a round goal is scored."""

    @abc.abstractmethod
    def round_goal_scored(
        self,
        *,
        engine: core.Engine,
        round_num: int,
        goal: cards.EndRoundGoal,
        counts: list[int],
    ) -> None:
        """Fired after round-goal VP is awarded. ``counts`` is the per-seat
        category count (indexed by ``Player.id``)."""


class PlayerFinalScoredHandler(CallbackHandler):
    """Handler invoked once per player during final scoring."""

    @abc.abstractmethod
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
        """Fired after a player's ``final_score`` is computed, with the six
        score components broken out."""


class SetupAppliedHandler(CallbackHandler):
    """Handler invoked after a player's setup keep is applied."""

    @abc.abstractmethod
    def setup_applied(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        choice: decisions.SetupChoice,
    ) -> None:
        """Fired once per seat after the chosen starting hand / food / bonus is
        applied to the player."""


class SetupStartHandler(CallbackHandler):
    """Handler invoked just before a player makes their setup choices."""

    @abc.abstractmethod
    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        """Fired after cards are dealt to ``player`` but before they choose
        their starting hand, food, and bonus card. ``dealt_bonus`` is the list
        of offered bonus cards (not yet in ``player.bonus_cards``)."""


# The event -> required-base-class table. The config validator uses it to reject
# assigning a handler to an event whose method it does not implement, and it is
# the single source of truth pairing each ``EventName`` with its handler base.
EVENT_BASE: dict[EventName, type[CallbackHandler]] = {
    EventName.GAME_START: GameStartHandler,
    EventName.GAME_END: GameEndHandler,
    EventName.ROUND_START: RoundStartHandler,
    EventName.ROUND_END: RoundEndHandler,
    EventName.TURN_START: TurnStartHandler,
    EventName.TURN_END: TurnEndHandler,
    EventName.MAKING_DECISION: MakingDecisionHandler,
    EventName.MADE_DECISION: MadeDecisionHandler,
    EventName.BIRD_PLACED: BirdPlacedHandler,
    EventName.FOOD_GAINED: FoodGainedHandler,
    EventName.EGGS_LAID: EggsLaidHandler,
    EventName.CARDS_DRAWN: CardsDrawnHandler,
    EventName.ROUND_GOAL_SCORED: RoundGoalScoredHandler,
    EventName.PLAYER_FINAL_SCORED: PlayerFinalScoredHandler,
    EventName.SETUP_APPLIED: SetupAppliedHandler,
    EventName.SETUP_START: SetupStartHandler,
}
