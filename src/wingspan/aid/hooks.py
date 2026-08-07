"""``AidHandler`` -- instrumentation hooks wiring the physical-table dialogs
into the engine's turn loop: log flushing on every event, the opponent's
pre-turn "what did they play?" dialog (with hand surgery so the turn's
follow-up decisions carry the real bird), and the final-round opponent-bonus
entry for exact scoring.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import state
from wingspan.aid import console as console_module
from wingspan.aid import entry, models
from wingspan.aid import oracle as oracle_module
from wingspan.aid import placeholders
from wingspan.instrumentation import dispatcher, events

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# The aid feature is 2-player only: seat 0 is always the user, seat 1 is
# always the opponent.
_OPPONENT_SEAT = 1


class AidHandler(
    events.GameStartHandler,
    events.GameEndHandler,
    events.RoundStartHandler,
    events.RoundEndHandler,
    events.TurnStartHandler,
    events.TurnEndHandler,
):
    """Bridges the engine's instrumentation events to the physical-table
    dialogs: flushes the log narration before every event, runs the
    opponent's pre-turn play report (swapping placeholders for the real
    birds before the turn's decisions are built), and offers the opponent's
    bonus card for exact scoring at the end of the final round."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    con: console_module.Console
    echo: console_module.LogEcho
    oracle: oracle_module.SessionOracle
    notes: models.TurnNotes
    registry: placeholders.PlaceholderRegistry
    opponent_bonus_entered: bool = False

    def game_start(self, *, engine: core.Engine) -> None:
        """Flush the log; there is nothing else to do before setup starts."""
        self.echo.flush()

    def game_end(self, *, engine: core.Engine) -> None:
        """Flush the log after final scoring."""
        self.echo.flush()

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        """Flush the log; no per-round dialog runs at round start."""
        self.echo.flush()

    def turn_end(self, *, engine: core.Engine, player: state.Player) -> None:
        """Flush the log; end-of-turn cleanup has no dialog of its own."""
        self.echo.flush()

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        """For the opponent's turn: reset the per-turn notes, then loop
        asking whether they played (another) bird, identifying it and
        swapping the placeholder in their tracked hand before the turn's
        main-action decision is built. Our own turn needs no dialog here --
        the advisor sweeps our hand per-decision instead."""
        self.echo.flush()
        if player.id != _OPPONENT_SEAT:
            return
        self.notes.clear()
        prompt = "Did the opponent play a bird this turn?"
        while self.con.confirm(prompt):
            bird = entry.identify_bird(self.con, "Which bird did they play? ")
            habitat = entry.pick_habitat(self.con, bird)
            hand = engine.state.players[_OPPONENT_SEAT].hand
            if self.registry.first_bird_in(hand) is not None:
                self.registry.swap_bird(hand, bird)
            else:
                self.con.say(
                    "No face-down card left in the opponent's tracked hand — "
                    "skipping the swap."
                )
            self.notes.plays.append(models.OpponentPlayNote(bird=bird, habitat=habitat))
            prompt = "Did they play ANOTHER bird?"

    def round_end(self, *, engine: core.Engine, round_num: int) -> None:
        """Flush the log; on the final round, offer to enter the opponent's
        bonus card(s) for exact end-game scoring instead of leaving them as
        unscored placeholders."""
        self.echo.flush()
        if round_num != len(state.ROUND_CUBES) - 1:
            return
        opponent = engine.state.players[_OPPONENT_SEAT]
        for bonus_card in opponent.bonus_cards:
            if not self.registry.is_placeholder(bonus_card):
                continue
            if not self.con.confirm(
                "Game over — enter the opponent's bonus card for exact scoring?"
            ):
                self.opponent_bonus_entered = False
                break
            real = entry.identify_bonus(self.con, "Which bonus card was it? ")
            self.registry.swap_bonus(opponent.bonus_cards, real)
            self.opponent_bonus_entered = True


def build_instrumentation(handler: AidHandler) -> dispatcher.Instrumentation:
    """Wire ``handler`` into a fresh ``Instrumentation`` covering all six
    events it implements."""
    by_event: dict[events.EventName, list[events.CallbackHandler]] = {
        events.EventName.GAME_START: [handler],
        events.EventName.GAME_END: [handler],
        events.EventName.ROUND_START: [handler],
        events.EventName.ROUND_END: [handler],
        events.EventName.TURN_START: [handler],
        events.EventName.TURN_END: [handler],
    }
    return dispatcher.Instrumentation(by_event=by_event)
