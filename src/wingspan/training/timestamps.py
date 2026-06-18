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

:func:`discounted_future_returns` is the shared math kernel: the backward
discounted-sum inner loop extracted from ``learner._decision_delta_returns`` so
the timeline chart builder can reuse it without importing torch.
:func:`finalize_provisional_timestamps` mirrors :func:`finalize_timestamps` for
non-Step data (parallel float/int lists), used by the chart builder.
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


def discounted_future_returns(
    checkpoints: list[float], times: list[float], discount: float
) -> list[float]:
    """Backward discounted-sum returns for one player's decision sequence.

    ``checkpoints`` has N+1 entries — N per-decision margin snapshots plus one
    terminal value — and ``times`` has N+1 matching game-clock timestamps (the
    N decision timestamps plus the final terminal timestamp). Returns N raw
    return values (not divided by ``score_norm``), one per decision, computed
    as the backward discounted sum:

        G[k] = (v[k+1] − v[k]) + discount^(t[k+1] − t[k]) · G[k+1]

    With ``discount == 1`` this telescopes to ``terminal − v[k]`` for every k.
    With ``discount == 0`` it reduces to the single-step reward ``v[k+1] − v[k]``.
    The ``0.0 ** 0 == 1.0`` Python identity means two simultaneous decisions
    (``Δt == 0``) correctly apply no decay between them regardless of ``discount``.

    This is the kernel formerly inline in ``learner._decision_delta_returns``,
    extracted so the HTML timeline chart builder can reuse it without torch.
    """
    assert len(times) == len(
        checkpoints
    ), "checkpoints and times must be the same length"
    n = len(checkpoints) - 1
    out = [0.0] * n
    running = 0.0
    for position in reversed(range(n)):
        reward = checkpoints[position + 1] - checkpoints[position]
        running = reward + discount ** (times[position + 1] - times[position]) * running
        out[position] = running
    return out


def gae_advantages(
    checkpoints: list[float],
    times: list[float],
    values: list[float],
    score_norm: float,
    discount: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """GAE advantages and value targets for one player's decision sequence.

    ``checkpoints`` has N+1 entries (N per-decision margin/score snapshots plus
    the terminal value); ``times`` has N+1 matching game-clock timestamps; and
    ``values`` has N per-decision critic estimates V(s_t) in normalized-return
    units.  Returns ``(advantages, value_targets)`` — both length N — computed
    via the backward TD sweep:

        reward  = (checkpoint[k+1] − checkpoint[k]) / score_norm
        next_v  = values[k+1] if k+1 < N else 0.0  (terminal: V = 0)
        δ[k]    = reward + discount^Δt · next_v − values[k]
        A[k]    = δ[k] + (discount·λ)^Δt · A[k+1]
        target[k] = A[k] + values[k]

    With λ=1, γ=1 this reduces exactly to the ``decision_delta`` advantage
    ``G/score_norm − V`` and target ``G/score_norm`` — the correctness check.
    With λ=0 it reduces to one-step TD.

    Deliberately torch-free (parallel to :func:`discounted_future_returns`) so
    unit tests import without the training stack.
    """
    n = len(values)
    assert len(checkpoints) == n + 1, "checkpoints must be length N+1"
    assert len(times) == n + 1, "times must be length N+1"
    advantages = [0.0] * n
    value_targets = [0.0] * n
    running = 0.0
    for position in reversed(range(n)):
        reward = (checkpoints[position + 1] - checkpoints[position]) / score_norm
        dt = times[position + 1] - times[position]
        next_v = values[position + 1] if position + 1 < n else 0.0
        delta = reward + discount**dt * next_v - values[position]
        running = delta + (discount * lam) ** dt * running
        advantages[position] = running
        value_targets[position] = running + values[position]
    return advantages, value_targets


def finalize_provisional_timestamps(
    provisional: list[float], family_indices: list[int]
) -> list[float]:
    """Finalize provisional timestamps for a non-Step sequence.

    Companion to :func:`finalize_timestamps` for callers that have parallel
    float/int lists instead of :class:`~wingspan.training.steps.Step` objects.
    ``provisional`` and ``family_indices`` are parallel lists (recording order)
    of provisional timestamps and decision-family indices. Returns a new list
    with setup-window items unchanged and in-turn items spread into each turn's
    ``(N, N+1)`` window exactly as :func:`finalize_timestamps` does."""
    result = list(provisional)
    for _, group_iter in itertools.groupby(range(len(result)), key=lambda i: result[i]):
        group_indices = list(group_iter)
        turn_start = result[group_indices[0]]
        if turn_start < float(_FIRST_TURN):
            continue
        # Skip the main action (stays at T); spread the remaining items.
        if family_indices[group_indices[0]] == _MAIN_ACTION_FAMILY_INDEX:
            group_indices = group_indices[1:]
        n_mid = len(group_indices)
        for position, idx in enumerate(group_indices, start=1):
            result[idx] = turn_start + position / (n_mid + 1)
    return result
