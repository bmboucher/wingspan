"""Torch CPU thread policy for the training process.

The ``mp_collect`` worker processes pin themselves to one intra-op thread each
(``mp_collect._init_worker``), so the *main* process's thread pool only matters
for the work it runs itself: the batched update phase, which wants every core,
and the rare in-process per-decision inference paths. Per-decision
(batch-of-one) inference is the exception: on CPU those tiny forward passes
run *slower* when many threads contend over them — measured ~5.7 games/s at
torch's default 12 threads vs ~7.5 games/s at 1-2 threads (+33%). Until
2026-09 ``TrainingLoop`` applied that cap process-wide, which throttled the
update once collection had moved into the worker pool; the cap now applies
only around the remaining in-process per-decision CPU path (the
target-milestone self-play eval, ``loop_target``).
"""

from __future__ import annotations

import contextlib
import typing

import torch

INFERENCE_INTRAOP_THREADS = 2
"""The measured sweet spot for batch-of-one CPU inference (module docstring)."""


@contextlib.contextmanager
def inference_thread_cap(device: torch.device) -> typing.Generator[None, None, None]:
    """Cap torch's intra-op threads at :data:`INFERENCE_INTRAOP_THREADS` around
    a per-decision CPU inference block, restoring the previous count after.
    A no-op for non-CPU devices (the GPU does the work) and when the count is
    already at or below the cap."""
    if device.type != "cpu":
        yield
        return
    previous = torch.get_num_threads()
    if previous <= INFERENCE_INTRAOP_THREADS:
        yield
        return
    torch.set_num_threads(INFERENCE_INTRAOP_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
