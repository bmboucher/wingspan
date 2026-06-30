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

* ``reward_basis`` — ``MARGIN`` (own − opponent; seats opposite-signed) vs
  ``OWN_SCORE`` (a seat's absolute score; both positive).
* ``reward_mode`` — ``TERMINAL_MARGIN`` broadcasts the end-of-game value to
  every step; ``DECISION_DELTA`` / ``GAE`` credit each decision with the change
  from that decision onward, discounted by ``reward_discount`` per game-clock
  unit.

Deliberately torch-free (stdlib + config enums + the torch-free
:func:`timestamps.discounted_future_returns` kernel) so both learners and unit
tests import it without the heavyweight training stack.
"""

from __future__ import annotations

from wingspan.training import config, timestamps

# Denominator floor for per-batch advantage whitening ``(A − mean) / (std + ε)``,
# shared by the in-game learner and the setup learner so both normalize the
# policy-gradient advantage identically (TRAINING.md §3.3 / §6.5).
ADV_STD_EPS = 1e-6


def winner_bonus(winner: int, end_game_bonus: float) -> float:
    """Seat-0-POV bonus delta: ``+bonus`` when seat 0 wins, ``-bonus`` when seat 1
    wins, ``0`` on a tie (``winner == -1``)."""
    if winner == 0:
        return end_game_bonus
    if winner == 1:
        return -end_game_bonus
    return 0.0


def terminal_values(
    score_0: float,
    score_1: float,
    winner: int,
    end_game_bonus: float,
    basis: config.RewardBasis,
) -> tuple[float, float]:
    """Both seats' terminal (final) values ``(seat_0, seat_1)`` before ``score_norm``.

    With ``MARGIN`` basis a seat's value is own − opponent score with
    ``end_game_bonus`` added/subtracted symmetrically (``winner_bonus``); with
    ``OWN_SCORE`` it is the seat's absolute score plus ``end_game_bonus`` only
    for the winner.  This is the exact terminal computation the in-game learner
    uses at every reward-mode call site."""
    if basis is config.RewardBasis.OWN_SCORE:
        bonus_0 = end_game_bonus if winner == 0 else 0.0
        bonus_1 = end_game_bonus if winner == 1 else 0.0
        return (score_0 + bonus_0, score_1 + bonus_1)
    bonus_0 = winner_bonus(winner, end_game_bonus)
    return (score_0 - score_1 + bonus_0, score_1 - score_0 - bonus_0)


def setup_return(
    own_total: float,
    opp_total: float,
    won: int,
    margin_checkpoints: list[float],
    score_checkpoints: list[float],
    decision_times: list[float],
    final_timestamp: float,
    training: config.TrainingConfig,
) -> float:
    """The normalized in-game return evaluated at the seat's ``t=0`` setup keep.

    ``own_total`` / ``opp_total`` are the seat's and opponent's final scores;
    ``won`` is the seat-relative outcome (``+1`` win, ``-1`` loss, ``0`` tie).
    The three parallel lists are the seat's in-game decision checkpoints
    (``margin_before`` / ``score_before``) and their game-clock timestamps, and
    ``final_timestamp`` is the terminal time.

    Under ``TERMINAL_MARGIN`` this is the seat's terminal value / ``score_norm``
    (so at the default config it equals the legacy ``margin / score_norm``).
    Under ``DECISION_DELTA`` / ``GAE`` it is the discounted Monte-Carlo return at
    the ``t=0`` anchor (``v=0`` at ``SETUP_KEEP_TIMESTAMP``): a single t=0 bandit
    decision cannot bootstrap, so GAE also uses the MC discounted return."""
    # Relabel the deciding seat as "seat 0" so the shared terminal computation
    # is reused exactly (no parallel formula to drift from the in-game path).
    seat_as_winner = 0 if won == 1 else (1 if won == -1 else -1)
    terminal_seat = terminal_values(
        own_total,
        opp_total,
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
