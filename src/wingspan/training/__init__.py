"""Training + live-monitoring pipeline (the "FLYWAY CONTROL" dashboard).

A ``top``-style terminal application that runs the TRAINING.md Phase-1 program
— synchronous self-play collection, a length-bucketed REINFORCE update with a
value baseline and advantage normalization, periodic paired-game evaluation
against the random agent, and resumable checkpoints — while repainting a fixed
full-screen dashboard of the run: iteration/game/throughput status with live
wall-clocks, the average score split into its six sources, the live judgment
histogram, and a braille convergence chart.

Run it with ``python -m wingspan.training`` (or the ``wingspan-dashboard``
console script). The package is split by concern:

- ``config``    — ``TrainConfig`` (self-describing hyperparameters, §5.1)
- ``metrics``   — ``ScoreBreakdown`` / ``FamilyCounts`` / ``IterationMetrics``
- ``policy``    — single-decision sampling (collect) + greedy (eval)
- ``collect``   — self-play game → recorded steps + score breakdown
- ``learner``   — the length-bucketed REINFORCE update (§3.3, §4.2a)
- ``evaluate``  — paired-game strength vs the random agent (§7)
- ``loop``      — the orchestrator (worker thread)
- ``runstate``  — the shared live state the dashboard reads
- ``theme`` / ``charts`` / ``dashboard`` — the rich-based display
- ``app``       — argparse entry point + the Live render loop

For a single, minimal REINFORCE cycle without the dashboard or scaled
collection, see the standalone :mod:`wingspan.train` module — it is the
readable reference; this package is the production pipeline.
"""

from wingspan.training.app import main
from wingspan.training.config import TrainConfig
from wingspan.training.loop import TrainingLoop

__all__ = [
    "TrainConfig",
    "TrainingLoop",
    "main",
]
