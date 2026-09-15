"""Periodic architecture-probe measurement for ``TrainingLoop``.

A free function whose first argument is a ``TrainingLoop`` instance re-runs
the offline architecture-probe machinery (``analysis.probe_set``,
``analysis.representation`` — the same code ``wingspan analysis probe`` calls,
``docs/TRAINING.md`` §6.5) against the live checkpoint on a configurable
cadence, mirroring how ``loop_eval.maybe_evaluate`` runs the periodic
paired-game eval. Pure plumbing: no new diagnostic logic lives here, only the
cadence gate, the on-distribution subsample, and the call into Stage 1's
``measure`` / ``summarize_for_loop``.
"""

from __future__ import annotations

import random
import time
import typing

from wingspan.analysis import models as analysis_models
from wingspan.analysis import probe_set, representation
from wingspan.training import runstate

if typing.TYPE_CHECKING:
    from wingspan.training import collect, loop


def maybe_probe(
    training_loop: "loop.TrainingLoop",
    iteration: int,
    records: list[collect.GameRecord],
) -> tuple[analysis_models.RepresentationMetrics | None, float]:
    """Run the periodic architecture probe; return ``(result, elapsed_seconds)``.

    Cadence-gated by ``config.run.probe_every`` (0 disables). On a probed
    iteration, a reproducible random subsample of up to
    ``config.run.probe_decisions`` of this iteration's just-collected steps is
    measured through the same ``analysis.representation.measure`` pass the
    offline CLI runs, then projected down via ``summarize_for_loop``. Returns
    ``(None, 0.0)`` in every skip case, exactly like ``loop_eval.maybe_evaluate``.

    Any exception raised while building or measuring the probe set is caught,
    pushed as an ALARM event, and reported as a skip (``(None, elapsed)``)
    rather than propagating — unlike the offline CLI, this runs mid-training,
    so a probe failure (a CUDA OOM on the extra full passes, a linalg error in
    the ridge fit, ...) must never be allowed to end the run.
    """
    if (
        training_loop.config.run.probe_every <= 0
        or iteration % training_loop.config.run.probe_every != 0
    ):
        return None, 0.0

    start = time.monotonic()
    # A different multiplier/offset than maybe_evaluate's eval_seed (*101+1)
    # so the two seed streams never correlate.
    rng = random.Random(training_loop.config.misc.seed * 7919 + iteration * 131 + 3)
    try:
        steps = probe_set.subsample_steps(
            records, training_loop.config.run.probe_decisions, rng
        )
        sample = probe_set.from_steps(training_loop.net, steps, n_games=len(records))
        report = representation.measure(
            training_loop.net,
            sample,
            device=training_loop.train_device,
            score_norm=training_loop.config.training.score_norm,
        )
        metrics_result = representation.summarize_for_loop(report)
    except Exception as error:  # noqa: BLE001 — a probe failure must never end a run
        elapsed_seconds = time.monotonic() - start
        with training_loop.lock:
            training_loop.state.push_event(
                runstate.EventKind.ALARM,
                f"PROBE failed at iter {iteration} after {elapsed_seconds:.1f}s "
                f"— {error}",
            )
        return None, elapsed_seconds
    elapsed_seconds = time.monotonic() - start

    uniform_kl = (
        f"{metrics_result.uniform_kl:.3f}"
        if metrics_result.uniform_kl is not None
        else "n/a"
    )
    with training_loop.lock:
        training_loop.state.push_event(
            runstate.EventKind.INFO,
            f"PROBE {metrics_result.probe_decisions} decisions in "
            f"{elapsed_seconds:.1f}s · uniform KL {uniform_kl}",
        )
    return metrics_result, elapsed_seconds
