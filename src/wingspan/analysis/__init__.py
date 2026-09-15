"""Architecture-importance diagnostics for a ``PolicyValueNet``.

Measures how much each part of the network matters: board-attention
contribution/entropy and ablation sensitivity (zero / uniform / per-head
knockout), trunk and choice-encoder layer capacity (effective rank, linear
predictability, dead units), and the input groups the trunk actually reads
from. Two callers feed this package a :class:`probe_set.ProbeSet` of
on-distribution decisions and run :func:`representation.measure` over it:

* The offline ``wingspan analysis probe`` CLI (:mod:`wingspan.analysis.cli`)
  loads a checkpoint and self-plays a fresh sample of games
  (:func:`probe_set.from_self_play`).
* The live training loop (``wingspan.training.loop_probe.maybe_probe``) runs
  periodically against the in-memory net, over a subsample of the current
  iteration's already-collected steps (:func:`probe_set.from_steps`) — no
  checkpoint load, no extra self-play.

See ``docs/analysis/INDEX.md`` for the module map and ``docs/TRAINING.md``
"Representation diagnostics" (§6.5) for how to read the numbers and how the
live cadence is configured.
"""
