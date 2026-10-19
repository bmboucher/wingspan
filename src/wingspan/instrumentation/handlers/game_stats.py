"""Reusable data-capture handler: per-game score series, actions, and outcome.

:class:`GameStatsCapture` is a per-game recorder for offline studies (the home
``docs/RESEARCH.md`` names for such handlers). One instance observes exactly one
game — build a fresh one per game, there is no cross-game state — and
accumulates, for every seat:

* ``score_proj`` / ``score_raw`` — the seat's score after each of its own turns
  (index 0 is the pre-game baseline, index ``t`` the score after the seat's
  ``t``-th turn; ``TURNS_PER_GAME + 1`` entries each). The *projected*
  convention counts unscored round goals at their live standing
  (``game_log_capture.score_breakdown``), so it has no jumps at round ends; the
  *raw* convention is ``scoring.running_score`` (round-goal VP appear only once
  a round's goal is scored). At every ``round_end`` both seats are
  re-snapshotted and the round's boundary turn is overwritten in both series,
  which folds the freshly awarded round-goal VP into the boundary turn.
* ``actions`` — the main action (a :class:`decisions.MainAction` value) chosen on
  each of the seat's turns, ``TURNS_PER_GAME`` entries.
* the final score and its six-way breakdown, as fired by
  ``scoring.final_scoring``, and whether the seat won outright.

**Turn indexing.** One mechanism only: an explicit per-seat counter, incremented
at ``turn_start``, is the seat's cumulative turn number (1-based). ``turn_end``
cross-checks it against the engine's own cube bookkeeping
(``ROUND_CUBES[:round_idx]`` summed, plus ``ROUND_CUBES[round_idx]`` minus the
cubes left), so a drift between the two raises instead of silently mislabelling
a series.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import decisions, state
from wingspan.engine import scoring
from wingspan.instrumentation import dispatcher, events
from wingspan.reporting import game_log_capture

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# Action cubes each seat spends over a whole game: one main action per cube.
TURNS_PER_GAME = sum(state.ROUND_CUBES)


class SeatCapture(pydantic.BaseModel):
    """Everything :class:`GameStatsCapture` records about one seat of one game."""

    seat: int
    # Score after each of the seat's turns; index 0 is the pre-game baseline.
    score_proj: list[int] = pydantic.Field(
        default_factory=lambda: [0] * (TURNS_PER_GAME + 1)
    )
    score_raw: list[int] = pydantic.Field(
        default_factory=lambda: [0] * (TURNS_PER_GAME + 1)
    )
    # ``MainAction`` values by cumulative turn; entry ``t - 1`` is turn ``t``.
    actions: list[str] = pydantic.Field(default_factory=lambda: [""] * TURNS_PER_GAME)
    final_score: int = 0
    bird_pts: int = 0
    bonus_pts: int = 0
    eggs: int = 0
    tucked: int = 0
    cached: int = 0
    round_goal: int = 0
    # True only for a sole winner; a shared victory leaves every seat False.
    won: bool = False


class GameStatsCapture(
    events.TurnStartHandler,
    events.TurnEndHandler,
    events.RoundEndHandler,
    events.MadeDecisionHandler,
    events.PlayerFinalScoredHandler,
    events.GameEndHandler,
):
    """Capture one game's per-turn scores, main actions, and outcome.

    Wire it with :meth:`instrumentation` and pass the result to
    ``Engine.play_one_game``; read :attr:`seats` and :attr:`tie` afterwards."""

    _seats: dict[int, SeatCapture] = pydantic.PrivateAttr(
        default_factory=dict[int, SeatCapture]
    )
    _turn_of_seat: dict[int, int] = pydantic.PrivateAttr(default_factory=dict[int, int])
    _baseline_taken: bool = pydantic.PrivateAttr(default=False)
    _tie: bool = pydantic.PrivateAttr(default=False)

    @property
    def seats(self) -> list[SeatCapture]:
        """The per-seat captures in seat order."""
        return [self._seats[seat] for seat in sorted(self._seats)]

    @property
    def tie(self) -> bool:
        """Whether the game ended in a shared victory (set at ``game_end``)."""
        return self._tie

    def instrumentation(self) -> dispatcher.Instrumentation:
        """The event router subscribing this handler to every event it uses."""
        return dispatcher.Instrumentation(
            by_event={
                events.EventName.TURN_START: [self],
                events.EventName.TURN_END: [self],
                events.EventName.ROUND_END: [self],
                events.EventName.MADE_DECISION: [self],
                events.EventName.PLAYER_FINAL_SCORED: [self],
                events.EventName.GAME_END: [self],
            }
        )

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        if not self._baseline_taken:
            self._baseline_taken = True
            for seated in engine.state.players:
                self._seats[seated.id] = SeatCapture(seat=seated.id)
                self._turn_of_seat[seated.id] = 0
                self._snapshot(engine, seated, 0)
        self._turn_of_seat[player.id] += 1

    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
        if isinstance(decision, decisions.MainActionDecision) and isinstance(
            choice, decisions.MainActionChoice
        ):
            turn = self._turn_of_seat[decision.player_id]
            self._seats[decision.player_id].actions[turn - 1] = choice.action.value

    def turn_end(self, *, engine: core.Engine, player: state.Player) -> None:
        turn = self._turn_of_seat[player.id]
        round_idx = engine.state.round_idx
        engine_turn = (
            sum(state.ROUND_CUBES[:round_idx])
            + state.ROUND_CUBES[round_idx]
            - player.action_cubes_left
        )
        if turn != engine_turn:
            raise AssertionError(
                f"seat {player.id} turn counter {turn} disagrees with the "
                f"engine's cube bookkeeping {engine_turn}"
            )
        self._snapshot(engine, player, turn)

    def round_end(self, *, engine: core.Engine, round_num: int) -> None:
        boundary = sum(state.ROUND_CUBES[: round_num + 1])
        for player in engine.state.players:
            self._snapshot(engine, player, boundary)

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
        seat = self._seats[player.id]
        seat.final_score = total
        seat.bird_pts = bird_pts
        seat.bonus_pts = bonus_pts
        seat.eggs = eggs
        seat.tucked = tucked
        seat.cached = cached
        seat.round_goal = round_goal

    def game_end(self, *, engine: core.Engine) -> None:
        winner_ids = scoring.winners(engine.state.players)
        self._tie = len(winner_ids) > 1
        if not self._tie:
            self._seats[winner_ids[0]].won = True

    ###### PRIVATE #######

    def _snapshot(self, engine: core.Engine, player: state.Player, turn: int) -> None:
        """Record both score conventions for ``player`` at cumulative ``turn``."""
        seat = self._seats[player.id]
        seat.score_proj[turn] = game_log_capture.score_breakdown(
            player, engine.state
        ).total
        seat.score_raw[turn] = scoring.running_score(player)
