"""The normalized training return of a finished game, shared by both learners.

Single source of truth for "the return of a seat's trajectory", so the in-game
learner (:mod:`wingspan.training.learner`) and the separate setup learner
(:mod:`wingspan.training.setup_learner`) compute the *same* return under the
*same* ``reward_mode`` / ``reward_basis`` / ``reward_discount`` /
``end_game_bonus``.  The setup keep is the ``t=0`` decision of the same game
whose in-game steps are ``t>0`` (``timestamps.SETUP_KEEP_TIMESTAMP``), so its
target is this kernel evaluated at that anchor — consistent by construction
rather than a separately-defined raw margin.

Two orthogonal config axes select the signal:

* ``reward_basis`` — ``MARGIN`` (own score minus the *best* other seat's
  score — margin vs the strongest opponent; reduces to own − the single
  opponent at 2 players) vs ``OWN_SCORE`` (a seat's absolute score; every seat
  positive).
* ``reward_mode`` — ``TERMINAL_MARGIN`` broadcasts the end-of-game value to
  every step; ``DECISION_DELTA`` / ``GAE`` credit each decision with the change
  from that decision onward, discounted by ``reward_discount`` per game-clock
  unit.

Deliberately torch-free (stdlib + config enums + the torch-free
:func:`timestamps.discounted_future_returns` kernel) so both learners and unit
tests import it without the heavyweight training stack.
"""

from __future__ import annotations

import typing

from wingspan.training import config, timestamps

# Denominator floor for per-batch advantage whitening ``(A − mean) / (std + ε)``,
# shared by the in-game learner and the setup learner so both normalize the
# policy-gradient advantage identically (TRAINING.md §3.3 / §6.5).
ADV_STD_EPS = 1e-6


def seat_winner_bonus(seat: int, winner: int, end_game_bonus: float) -> float:
    """``seat``'s end-game bonus delta: ``+bonus`` when ``seat`` is the sole
    winner, ``-bonus`` when any other seat is the sole winner, ``0`` on a
    shared victory (``winner == -1``, no sole winner)."""
    if winner < 0:
        return 0.0
    return end_game_bonus if seat == winner else -end_game_bonus


def terminal_values(
    scores: typing.Sequence[float],
    winner: int,
    end_game_bonus: float,
    basis: config.RewardBasis,
) -> tuple[float, ...]:
    """Every seat's terminal (final) value, in seat order, before ``score_norm``.

    With ``MARGIN`` basis a seat's value is its own score minus the *best*
    other seat's score (margin vs the strongest opponent), with
    ``end_game_bonus`` added for the sole winner and subtracted for *every*
    other seat (:func:`seat_winner_bonus`) — or left at 0 for everyone on a
    shared victory (``winner == -1``).  With ``OWN_SCORE`` basis a seat's
    value is its own absolute score plus ``end_game_bonus`` only when it is
    the sole winner.

    At 2 players ``MARGIN`` reduces exactly to the legacy
    ``(s0 − s1 + b0, s1 − s0 − b0)`` formula: "best other" collapses to the
    single opponent, and ``seat_winner_bonus`` matches the old seat-0-POV
    ``winner_bonus`` branch for branch.  This is the exact terminal
    computation the in-game learner uses at every reward-mode call site."""
    if basis is config.RewardBasis.OWN_SCORE:
        return tuple(
            score + (end_game_bonus if seat == winner else 0.0)
            for seat, score in enumerate(scores)
        )
    return tuple(
        score
        - max(other for seat_j, other in enumerate(scores) if seat_j != seat)
        + seat_winner_bonus(seat, winner, end_game_bonus)
        for seat, score in enumerate(scores)
    )


def setup_return(
    own_total: float,
    best_other_total: float,
    won: int,
    margin_checkpoints: list[float],
    score_checkpoints: list[float],
    decision_times: list[float],
    final_timestamp: float,
    training: config.TrainingConfig,
) -> float:
    """The normalized in-game return evaluated at the seat's ``t=0`` setup keep.

    ``own_total`` is the seat's final score; ``best_other_total`` is the
    *best* other seat's final score (``max`` over every opponent — the
    caller's job to compute — reducing to the single opponent's score at 2
    players).  ``won`` is the seat-relative outcome (``+1`` win, ``-1`` loss,
    ``0`` tie / shared victory).  The three parallel lists are the seat's
    in-game decision checkpoints (``margin_before`` / ``score_before``) and
    their game-clock timestamps, and ``final_timestamp`` is the terminal time.

    Under ``TERMINAL_MARGIN`` this is the seat's terminal value / ``score_norm``
    (so at the default config it equals the legacy ``margin / score_norm``).
    Under ``DECISION_DELTA`` / ``GAE`` it is the discounted Monte-Carlo return at
    the ``t=0`` anchor (``v=0`` at ``SETUP_KEEP_TIMESTAMP``): a single t=0 bandit
    decision cannot bootstrap, so GAE also uses the MC discounted return."""
    # Relabel the deciding seat as "seat 0" of a 2-entry scores sequence so the
    # shared terminal computation is reused exactly (no parallel formula to
    # drift from the in-game path) — margin-vs-best-other is inherently a
    # 2-party relationship once "best other" has already been resolved.
    seat_as_winner = 0 if won == 1 else (1 if won == -1 else -1)
    terminal_seat = terminal_values(
        (own_total, best_other_total),
        seat_as_winner,
        training.end_game_bonus,
        training.reward_basis,
    )[0]

    if training.reward_mode is config.RewardMode.TERMINAL_MARGIN:
        return terminal_seat / training.score_norm

    checkpoints = (
        score_checkpoints
        if training.reward_basis is config.RewardBasis.OWN_SCORE
        else margin_checkpoints
    )
    sequence = [0.0, *checkpoints, terminal_seat]
    sequence_times = [timestamps.SETUP_KEEP_TIMESTAMP, *decision_times, final_timestamp]
    raw = timestamps.discounted_future_returns(
        sequence, sequence_times, training.reward_discount
    )
    return raw[0] / training.score_norm
