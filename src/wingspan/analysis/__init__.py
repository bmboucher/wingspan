"""Offline architecture-importance diagnostics for a trained ``PolicyValueNet``.

Loads a checkpoint, self-plays a sample of on-distribution decisions, and
measures how much each part of the network matters: board-attention
contribution/entropy and ablation sensitivity (zero / uniform / per-head
knockout), trunk and choice-encoder layer capacity (effective rank, linear
predictability, dead units), and the input groups the trunk actually reads
from. Exposed as the ``wingspan analysis probe`` CLI command
(:mod:`wingspan.analysis.cli`). See ``docs/analysis/INDEX.md`` for the
module map and ``docs/TRAINING.md`` "Representation diagnostics" for how to
read the numbers.

This package intentionally holds no training-loop integration (Stage 2):
it only reads a checkpoint and reports on it.
"""
