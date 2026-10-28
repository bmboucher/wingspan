"""Reusable data-capture handler: per-seat cards held, round by round.

:class:`CardStatsCapture` is a per-game recorder for the ``card-stats`` study
(``docs/RESEARCH.md``, "General card stats"). One instance observes exactly one
game — build a fresh one per game — and records, for every seat:

* the dealt birds and dealt bonus cards (``setup_start``), and the opener after
  setup: the kept birds and the *original* bonus card. Both are read at the
  first ``round_start`` (round 0), not at ``setup_applied``, because the
  deferred-bonus regime picks the bonus card after setup is applied.
* per round (1-4), the set of birds "in hand during the round" and the set of
  *acquired* bonus cards (held bonus cards other than the original) "held during
  the round".
* the outcome: sole winner / shared victory (``scoring.winners``) and the final
  score.

**"Held during the round".** A card counts for round ``r`` if the seat held it at
any point of the round, *unless it arrived on the seat's final turn of that
round*, since there is then no chance to play it that round. Concretely, for each
turn of the seat:

* at ``turn_start`` the seat's hand and acquired bonus cards join the round's
  sets and are remembered as the turn's *start hand* (so the opener, the cards
  carried over from the previous round, and cards drawn between turns all count);
* birds placed during the turn are tracked through ``bird_placed``;
* at ``turn_end``, if the seat still has action cubes left (the cube is spent
  before the action runs, so ``action_cubes_left > 0`` at ``turn_end`` means
  another turn follows this round), its current hand, the birds it placed this
  turn, and its acquired bonus cards join the round's sets — this is how a bird
  drawn and played in the same turn is credited. On the final turn nothing is
  added beyond the start hand; cards drawn on it appear in the next round's
  ``turn_start`` snapshot. Extra plays granted by powers happen inside the turn
  and are part of it.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import cards, state
from wingspan.engine import scoring
from wingspan.instrumentation import dispatcher, events

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# Index of the opening round, whose ``round_start`` freezes the post-setup opener.
_OPENING_ROUND = 0


class CardSeatCapture(pydantic.BaseModel):
    """Everything :class:`CardStatsCapture` records about one seat of one game."""

    seat: int
    dealt_birds: list[str] = pydantic.Field(default_factory=list[str])
    kept_birds: list[str] = pydantic.Field(default_factory=list[str])
    dealt_bonus: list[str] = pydantic.Field(default_factory=list[str])
    # The bonus card held at the start of round 1; blank if the seat holds none.
    original_bonus: str = ""
    # Names of the birds held during each round (index = 0-based round).
    round_birds: list[set[str]] = pydantic.Field(
        default_factory=lambda: [set[str]() for _ in state.ROUND_CUBES]
    )
    # Names of the acquired (non-original) bonus cards held during each round.
    round_acquired_bonus: list[set[str]] = pydantic.Field(
        default_factory=lambda: [set[str]() for _ in state.ROUND_CUBES]
    )
    # Birds placed on the board during the turn in progress.
    placed_this_turn: set[str] = pydantic.Field(default_factory=set[str])
    final_score: int = 0
    # True only for a sole winner; a shared victory leaves every seat False.
    won: bool = False


class CardStatsCapture(
    events.SetupStartHandler,
    events.RoundStartHandler,
    events.TurnStartHandler,
    events.BirdPlacedHandler,
    events.TurnEndHandler,
    events.GameEndHandler,
):
    """Capture one game's openers, per-round card holdings, and outcome.

    Wire it with :meth:`instrumentation` and pass the result to
    ``Engine.play_one_game``; read :attr:`seats` and :attr:`tie` afterwards."""

    _seats: dict[int, CardSeatCapture] = pydantic.PrivateAttr(
        default_factory=dict[int, CardSeatCapture]
    )
    _tie: bool = pydantic.PrivateAttr(default=False)

    @property
    def seats(self) -> list[CardSeatCapture]:
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
                events.EventName.SETUP_START: [self],
                events.EventName.ROUND_START: [self],
                events.EventName.TURN_START: [self],
                events.EventName.BIRD_PLACED: [self],
                events.EventName.TURN_END: [self],
                events.EventName.GAME_END: [self],
            }
        )

    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        seat = CardSeatCapture(seat=player.id)
        seat.dealt_birds = [bird.name for bird in player.hand]
        seat.dealt_bonus = [bonus.name for bonus in dealt_bonus]
        self._seats[player.id] = seat

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        if round_num != _OPENING_ROUND:
            return
        for player in engine.state.players:
            seat = self._seats[player.id]
            seat.kept_birds = [bird.name for bird in player.hand]
            seat.original_bonus = (
                player.bonus_cards[0].name if player.bonus_cards else ""
            )

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        seat = self._seats[player.id]
        seat.placed_this_turn = set()
        self._record_held(engine, player, include_placed=False)

    def bird_placed(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        bird: cards.Bird,
        habitat: cards.Habitat,
        played_bird: state.PlayedBird,
    ) -> None:
        self._seats[player.id].placed_this_turn.add(bird.name)

    def turn_end(self, *, engine: core.Engine, player: state.Player) -> None:
        # The cube is spent before the action runs, so cubes left here means
        # another turn of this round follows; on the final turn only the start
        # hand counts.
        if player.action_cubes_left > 0:
            self._record_held(engine, player, include_placed=True)

    def game_end(self, *, engine: core.Engine) -> None:
        winner_ids = scoring.winners(engine.state.players)
        self._tie = len(winner_ids) > 1
        for player in engine.state.players:
            seat = self._seats[player.id]
            if player.final_score is None:
                raise AssertionError(f"seat {player.id} has no final score at game end")
            seat.final_score = player.final_score
            seat.won = not self._tie and player.id in winner_ids

    ###### PRIVATE #######

    def _record_held(
        self, engine: core.Engine, player: state.Player, *, include_placed: bool
    ) -> None:
        """Add the seat's current hand (and, at turn end, the birds it placed this
        turn) and its acquired bonus cards to the current round's sets."""
        seat = self._seats[player.id]
        round_idx = engine.state.round_idx
        seat.round_birds[round_idx].update(bird.name for bird in player.hand)
        if include_placed:
            seat.round_birds[round_idx].update(seat.placed_this_turn)
        seat.round_acquired_bonus[round_idx].update(
            bonus.name
            for bonus in player.bonus_cards
            if bonus.name != seat.original_bonus
        )
