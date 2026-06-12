"""The game clock: a fractional timestamp for every recorded decision.

The ``decision_delta`` reward mode discounts between a player's consecutive
decisions with ``λ^Δt`` of *game time*, not λ per decision step — a play-bird
turn with chained powers should not decay the future faster than a bare
lay-eggs turn. This module defines that clock:

* Setup-window decisions (``GameState.turn_counter == 0``) sit at fixed times
  shared by both seats (modeled as simultaneous): the hand keep at 0, the
  deferred bonus pick at 1/3, the deferred food picks at 2/3.
* Turn ``N``'s main-action decision sits at exactly ``N`` (turns are counted
  globally across both seats, so consecutive integers alternate players).
* Every other decision recorded during turn ``N``'s window — follow-ups,
  power resolutions, and even the *opponent's* reaction decisions — is
  linearly interpolated into ``(N, N+1)``: the j-th of k gets ``N + j/(k+1)``.

Collection writes a provisional value per step (:func:`provisional_timestamp`)
and resolves the interpolation once the game is complete
(:func:`finalize_timestamps`), since a turn's mid-turn decision count is only
known after the turn ends. Deliberately torch-free so its unit tests import
without the heavyweight training stack.
"""

from __future__ import annotations

import itertools
import typing

from wingspan import decisions
from wingspan.training import steps

# The shared-clock times of the setup window's decisions (both seats).
SETUP_KEEP_TIMESTAMP = 0.0
SETUP_BONUS_TIMESTAMP = 1.0 / 3.0
SETUP_FOOD_TIMESTAMP = 2.0 / 3.0

# ``GameState.turn_counter`` value of the game's first turn; anything below it
# (and any timestamp below it as a float) belongs to the setup window.
_FIRST_TURN = 1

_MAIN_ACTION_FAMILY_INDEX = decisions.family_index_for(decisions.MainActionDecision)


def provisional_timestamp(
    decision: decisions.Decision[typing.Any], turn_counter: int
) -> float:
    """The timestamp recorded at decision time, before interpolation.

    During a turn this is the turn counter itself (the main action's final
    value; mid-turn decisions are spread into the turn's window later by
    :func:`finalize_timestamps`). In the setup window (``turn_counter == 0``)
    it is already final, keyed on the decision type: the combined keep, the
    deferred bonus pick, or the deferred food asks. Keying the window on the
    counter rather than the type matters because ``BirdPowerPickBonusCardDecision``
    also fires mid-game from bird powers.
    """
    if turn_counter >= _FIRST_TURN:
        return float(turn_counter)
    if decisions.is_setup_decision(decision):
        return SETUP_KEEP_TIMESTAMP
    if isinstance(decision, decisions.BirdPowerPickBonusCardDecision):
        return SETUP_BONUS_TIMESTAMP
    return SETUP_FOOD_TIMESTAMP


def finalize_timestamps(recorded: list[steps.Step]) -> None:
    """Spread each turn's mid-turn decisions into the turn's window, in place.

    ``recorded`` is a finished game's steps in decision order, carrying
    provisional timestamps. Setup-window steps (timestamp below the first
    turn) are already final and untouched. In-turn steps share their turn
    counter ``T`` as a provisional value and arrive contiguously (the counter
    strictly increases between turns), so each run of equal timestamps is one
    turn's recorded window: its main action (identified by scoring-head
    family) stays at ``T`` and the k others get ``T + j/(k+1)``. A window with
    no recorded main action (vs-random play: only the net seat's reactions
    during the opponent's turn are recorded) interpolates all its steps.
    """
    for _, group_steps in itertools.groupby(recorded, key=lambda step: step.timestamp):
        turn_window = list(group_steps)
        turn_start = turn_window[0].timestamp
        if turn_start < float(_FIRST_TURN):
            continue
        if turn_window[0].family_idx == _MAIN_ACTION_FAMILY_INDEX:
            turn_window = turn_window[1:]
        for position, step in enumerate(turn_window, start=1):
            step.timestamp = turn_start + position / (len(turn_window) + 1)


def final_timestamp(turn_counter: int) -> float:
    """The terminal checkpoint's time: the end of the final turn's window
    (one full turn after that turn's main action), shared by both seats."""
    return float(turn_counter) + 1.0
