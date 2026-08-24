"""``AidHandler`` -- instrumentation hooks wiring the physical-table dialogs
into the engine's turn loop: log flushing on every event, the opponent's
pre-turn "what did they play?" dialog (with hand surgery so the turn's
follow-up decisions carry the real bird), and the final-round opponent-bonus
entry for exact scoring.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import decisions, state
from wingspan.aid import console as console_module
from wingspan.aid import entry, models
from wingspan.aid import oracle as oracle_module
from wingspan.aid import placeholders
from wingspan.instrumentation import dispatcher, events

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# The aid feature is 2-player only: seat 0 is always the user, seat 1 is
# always the opponent.
_OUR_SEAT = 0
_OPPONENT_SEAT = 1

# Opponent-turn-start menu labels for the 4 real main actions, matching
# ``engine.core.Engine._main_action_decision``'s own wording exactly so the
# pre-turn menu never drifts out of sync with what the engine itself offers.
_MAIN_ACTION_LABELS: dict[decisions.MainAction, str] = {
    decisions.MainAction.GAIN_FOOD: "gain food (forest)",
    decisions.MainAction.LAY_EGGS: "lay eggs (grassland)",
    decisions.MainAction.DRAW_CARDS: "draw cards (wetland)",
    decisions.MainAction.PLAY_BIRD: "play a bird",
}


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
        """Flush the log; after our own seat's turn (including any trailing
        tray-refill identity prompts) fully resolves, pause and wait for the
        user to signal they're ready before play moves on."""
        self.echo.flush()
        if player.id == _OUR_SEAT:
            self.con.ask("Press Enter when ready to continue> ")

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        """For the opponent's turn: reset the per-turn notes, then offer a
        4-option main-action menu (labeled exactly like the engine's own
        ``MainActionDecision``). Picking ``PLAY_BIRD`` drives the
        identify-bird + pick-habitat sub-flow, swapping the placeholder in
        their tracked hand before the turn's decisions are built, and
        re-asking "another bird?" only when the board could plausibly
        support one (see ``_opponent_could_play_another_bird``). Any other
        action needs no further prompt here -- ``notes.main_action`` alone
        lets ``relay.py`` auto-answer the upcoming ``MainActionDecision``,
        and that action's own specifics are still asked reactively by the
        generic fallback when the engine reaches them. Our own turn needs no
        dialog here -- the advisor sweeps our hand per-decision instead."""
        self.echo.flush()
        if player.id != _OPPONENT_SEAT:
            return
        self.notes.clear()
        main_actions = list(decisions.MainAction)
        menu_lines = [
            decisions.MainActionChoice(
                action=action, label=_MAIN_ACTION_LABELS[action]
            ).display_label()
            for action in main_actions
        ]
        chosen_idx = self.con.menu("What did the opponent do this turn?", menu_lines)
        action = main_actions[chosen_idx]
        self.notes.main_action = action
        if action != decisions.MainAction.PLAY_BIRD:
            return
        _record_opponent_bird_plays(self.con, self.registry, self.notes, engine)

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
                continue
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


###### PRIVATE #######


def _record_opponent_bird_plays(
    con: console_module.Console,
    registry: placeholders.PlaceholderRegistry,
    notes: models.TurnNotes,
    engine: core.Engine,
) -> None:
    """Identify each bird the opponent played this turn -- entered via
    ``entry.identify_bird``/``entry.pick_habitat`` and swapped into the
    placeholder sitting in their tracked hand -- appending one
    ``OpponentPlayNote`` per play. Loops back to ask "another bird?" only
    when ``_opponent_could_play_another_bird`` says the board could
    plausibly support one; otherwise stops silently after the first play."""
    while True:
        bird = entry.identify_bird(con, "Which bird did they play? ")
        habitat = entry.pick_habitat(con, bird)
        hand = engine.state.players[_OPPONENT_SEAT].hand
        if registry.first_bird_in(hand) is not None:
            registry.swap_bird(hand, bird)
        else:
            con.say(
                "No face-down card left in the opponent's tracked hand — "
                "skipping the swap."
            )
        notes.plays.append(models.OpponentPlayNote(bird=bird, habitat=habitat))
        if not _opponent_could_play_another_bird(engine, registry):
            break
        if not con.confirm("Did they play ANOTHER bird?"):
            break


def _opponent_could_play_another_bird(
    engine: core.Engine, registry: placeholders.PlaceholderRegistry
) -> bool:
    """Whether the opponent's board holds any known (non-placeholder) bird
    whose power grants an extra bird play (``schema.Bird.plays_another_bird``,
    true exactly when the power includes ``PLAY_ADDITIONAL_BIRD`` or
    ``PLAY_ADDITIONAL_BIRD_HERE``). Placeholder birds are skipped -- their
    real power is unknown, so it cannot be checked.

    Scope limitation: this is a deliberately simple heuristic. It only checks
    for a *standing* extra-play power already on the board; it does not
    attempt to determine whether a *non*-play main action's row-power trigger
    could also grant an extra play -- that would require simulating which
    specific board birds actually activate for a given row action, not just
    whether one is present."""
    board = engine.state.players[_OPPONENT_SEAT].board
    return any(
        played_bird.bird.plays_another_bird
        for row in board.values()
        for played_bird in row
        if not registry.is_placeholder(played_bird.bird)
    )
