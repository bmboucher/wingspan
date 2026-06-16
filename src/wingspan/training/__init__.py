"""Training + live-monitoring pipeline ("FLIGHT PLAN").

A ``top``-style terminal application that runs the TRAINING.md Phase-1 program
— synchronous self-play collection, a length-bucketed REINFORCE update with a
value baseline and advantage normalization, periodic paired-game evaluation
against the random agent, and resumable checkpoints — while repainting a fixed
full-screen dashboard of the run: iteration/game/throughput status with live
wall-clocks, the average score split into its six sources, the live judgment
histogram, and a braille convergence chart.

Run it with ``wingspan dashboard`` (or ``python -m wingspan.training``).
See ``TRAINING.md`` for the full training program and ``DECISIONS.md`` for the
decision taxonomy the policy network is trained over.

The package is split by concern:

- ``config``    — ``RunConfig`` (self-describing hyperparameters, TRAINING.md §5.1)
- ``metrics``   — ``ScoreBreakdown`` / ``FamilyCounts`` / ``IterationMetrics``
- ``policy``    — single-decision sampling (collect) + greedy (eval)
- ``collect``   — self-play game → recorded steps + score breakdown
- ``learner``   — the length-bucketed REINFORCE update (TRAINING.md §3.3, §4.2a)
- ``evaluate``  — paired-game strength vs the random agent (TRAINING.md §7)
- ``loop``      — the orchestrator (worker thread)
- ``runstate``  — the shared live state the dashboard reads
- ``theme`` / ``charts`` / ``dashboard`` — the rich-based display
- ``app``       — argparse entry point + the Live render loop
"""

from wingspan.training.app import main
from wingspan.training.config import RunConfig
from wingspan.training.loop import TrainingLoop

__all__ = [
    "RunConfig",
    "TrainingLoop",
    "main",
]
