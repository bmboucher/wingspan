"""Pydantic ``GameState``/``Birdfeeder`` subclasses that route every hidden
or random outcome through a ``SessionOracle`` instead of ``rng`` -- plus
``build_state``, the aid-package mirror of ``state.new_game`` that seeds a
game from user-entered setup facts instead of a shuffle.
"""

from __future__ import annotations

import random

import pydantic

from wingspan import cards, state
from wingspan.aid import models
from wingspan.aid import oracle as oracle_module
from wingspan.cards.parse import catalog


class OracleBirdfeeder(state.Birdfeeder):
    """Birdfeeder whose reroll is supplied by the physical dice, entered
    through ``oracle.feeder_roll`` -- never actually rolled."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    oracle: oracle_module.SessionOracle

    def reroll(self, rng: random.Random) -> None:
        """Overrides the base reroll: set the dice from the oracle's
        entered feeder reading. ``rng`` is accepted only to match the base
        signature -- the physical dice have already been rolled at the
        table, so no randomness is consulted here."""
        entry = self.oracle.feeder_roll()
        self.counts = entry.to_food_pool()
        self.choice_dice = entry.choice_dice

    def roll_out_of_feeder(
        self, rng: random.Random, n: int
    ) -> tuple[state.FoodPool, int]:
        """Overrides the base roll: dice-predator rolls outside the feeder
        are also read from the physical table via the oracle, not
        ``rng``."""
        return self.oracle.dice_roll(n)


class OracleGameState(state.GameState):
    """``GameState`` whose card reveals are supplied by the oracle instead
    of being drawn blind from the deck. ``arbitrary_types_allowed`` is
    already set on the base ``GameState``, so no extra config is needed
    here for the ``oracle`` field."""

    oracle: oracle_module.SessionOracle

    def draw_bird(self) -> cards.Bird | None:
        """Mirrors the base reshuffle bookkeeping exactly, then substitutes
        the oracle-revealed bird for the popped filler card so
        ``len(bird_deck)`` stays truthful (the encoder reads only the
        length) while the returned identity is the real, oracle-supplied
        card."""
        if not self.bird_deck:
            if not self.bird_discard:
                return None
            self.bird_deck = self.bird_discard
            self.bird_discard = []
            self.rng.shuffle(self.bird_deck)
        self.bird_deck.pop()
        return self.oracle.reveal_bird()

    def draw_bonus(self) -> cards.BonusCard | None:
        """Mirrors :meth:`state.GameState.draw_bonus`'s empty-deck check,
        then substitutes the oracle-revealed bonus card for the popped
        filler, mirroring :meth:`draw_bird`."""
        if not self.bonus_deck:
            return None
        self.bonus_deck.pop()
        return self.oracle.reveal_bonus()


def build_state(
    rng: random.Random,
    setup: models.SetupEntry,
    session_oracle: oracle_module.SessionOracle,
) -> OracleGameState:
    """Mirror :func:`wingspan.state.new_game`, replacing every random/hidden
    outcome with the physical-table facts recorded in ``setup``.

    1. Build the fixed two-seat player list (seat 0 is the user, seat 1 the
       opponent); turn order starts at ``setup.start_player``.
    2. Fill the bird/bonus decks with the full catalog as filler -- only
       their *lengths* matter downstream, since every reveal routes through
       the oracle -- take the four entered round goals verbatim, and build
       a fresh oracle-backed birdfeeder.
    3. Construct the ``OracleGameState`` and attach the oracle's two
       backrefs (``session_oracle.game_state`` and
       ``session_oracle.echo.game_state``) so later prompts can flush the
       log and reference the live state.
    4. Queue the entered tray as the next three bird reveals, then call
       ``refill_tray`` to consume them in slot order.
    5. Queue the entered feeder roll, then reroll the birdfeeder to consume
       it.
    6. Pre-seed the engine's own setup deal, in the engine's fixed dealing
       order (seat 0's five birds, then seat 0's two bonus cards, then
       seat 1's five/two as silent placeholders): the entered hand and
       bonus pair are queued first, followed by ``None`` placeholders for
       the opponent's unseen deal.
    """
    # Step 1: seats and turn order. The aid feature is 2-player only, so
    # the seat ids/names are fixed; the physical starting seat is whichever
    # the user entered.
    players = [state.Player(id=0, name="You"), state.Player(id=1, name="Opponent")]
    current_player = start_player = setup.start_player
    round_idx = 0

    # Step 2: filler decks (only lengths matter -- every reveal is
    # oracle-fed), the entered round goals, and a fresh birdfeeder.
    bird_deck = list(catalog.birds_ordered())
    bonus_deck = list(catalog.bonus_cards_ordered())
    round_goals = list(setup.goals)
    birdfeeder = OracleBirdfeeder(oracle=session_oracle)

    # Step 3: construct the state and attach the oracle's backrefs.
    gs = OracleGameState(
        oracle=session_oracle,
        rng=rng,
        players=players,
        current_player=current_player,
        start_player=start_player,
        round_idx=round_idx,
        bird_deck=bird_deck,
        bonus_deck=bonus_deck,
        round_goals=round_goals,
        birdfeeder=birdfeeder,
    )
    session_oracle.game_state = gs
    session_oracle.echo.game_state = gs

    # Step 4: reveal the entered tray by queuing it ahead of refill_tray.
    session_oracle.queue_bird_reveals(setup.tray)
    gs.refill_tray()

    # Step 5: reveal the entered feeder roll by queuing it ahead of reroll.
    session_oracle.queue_feeder_roll(setup.feeder)
    gs.birdfeeder.reroll(gs.rng)

    # Step 6: pre-seed the engine's own setup deal (seat 0 dealt+decides
    # before seat 1): our real hand/bonus, then silent placeholders for the
    # opponent's unseen deal, in the engine's fixed per-seat dealing order.
    session_oracle.queue_bird_reveals(
        list(setup.hand) + [None] * state.STARTING_HAND_SIZE
    )
    session_oracle.queue_bonus_reveals(
        list(setup.bonus_pair) + [None] * state.STARTING_BONUS_CARDS_DEAL
    )
    return gs
