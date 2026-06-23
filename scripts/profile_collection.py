"""Throughput profiler for self-play collection (measure-first baseline).

Five measurements, printed in order:

1. **Sequential split** — ``collect.play_game`` looped on one thread, with the
   per-decision wall-clock broken into five buckets: engine game-logic + scoring
   (the residual), ``encode_state``, ``encode_choices``, the inference forward,
   and step bookkeeping. The engine-vs-inference ratio is the headline for the
   "rewrite collection in C++?" question: the engine residual is GIL-bound (only
   C++/nogil threads parallelize it), while the inference forward already releases
   the GIL (batching, not threading, is its lever).

2. **Batched g/s** — the CUDA-style threaded path (``batched_collect``). The
   speedup over sequential reveals the GIL ceiling for threads that share one
   forward but serialize the per-decision engine + encode work.

3. **mp_collect g/s** — the real CPU training path (``ProcessCollector``):
   process-parallel, batch-of-one forward per worker. Parallel efficiency vs the
   sequential x workers ideal is the headroom a perfect in-process threaded
   collector could recover — the realistic ceiling on a C++ rewrite's win.

4. **IPC payload** — per-game pickled trajectory size before/after fp16
   compaction, plus compaction time. The concrete cost processes pay that
   in-process threads would not.

5. **Attribution** — a single-threaded cProfile, sorted by self time.

All timing uses ``perf_counter`` (``process_time`` is unreliable on this box).

Run: ``python scripts/profile_collection.py --seq-games 12 --games 32 --mp-games 32``
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pathlib
import pickle
import pstats
import random
import sys
import tempfile
import threading
import time
import types
import typing

# Tests + scripts prepend src/ themselves so this runs without an install.
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from wingspan import encode, model  # noqa: E402
from wingspan.training import (  # noqa: E402
    batched_collect,
    collect,
    config,
    mp_collect,
    policy,
)

# Match the training loop's CPU thread pinning so the baseline reflects reality.
_CPU_INTRAOP_THREADS = 2

# The wide-decision soft-threshold warning fires per decision and floods output;
# silence it so the profile is readable (it is itself hot-path noise — see notes).
logging.getLogger("wingspan.encode").setLevel(logging.ERROR)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(_CPU_INTRAOP_THREADS)
    device = torch.device("cpu")
    # Build the net from a default RunConfig so one arch drives every measurement
    # — including the mp_collect workers, which rebuild their net from a config.
    cfg = config.RunConfig()
    net = model.PolicyValueNet(arch=cfg.arch, spec=cfg.encoding_spec)
    net.eval()

    seq_gps = _measure_sequential(net, device, args.seq_games)
    _measure_batched(net, device, args.games, seq_gps)
    if not args.skip_mp:
        _measure_mp_collect(net, device, args.mp_games, seq_gps)
    _measure_ipc_payload(net, device)
    if not args.skip_attribution:
        _measure_attribution(net, device, args.profile_games, args.sort)


def _measure_sequential(
    net: model.PolicyValueNet, device: torch.device, games: int
) -> float:
    """Loop ``collect.play_game`` single-threaded and print the five-bucket
    per-decision split. Returns sequential games/sec (the parallel baseline)."""
    rng = random.Random(0)
    collect.play_game(net, device, rng, 1)  # warm-up

    # Time each hot-path function on the single thread (no GIL contention), so a
    # per-call perf_counter delta is real cost — unlike under concurrency, where
    # numpy releases the GIL and a thread's timer absorbs other threads' work.
    timer = _BucketTimer(_sequential_patch_specs())
    total_steps = 0
    with timer:
        start = time.perf_counter()
        for i in range(games):
            total_steps += len(collect.play_game(net, device, rng, 5_000 + i).steps)
        wall = time.perf_counter() - start

    # The residual (wall minus every measured bucket) is the engine turn-loop,
    # bird-power handlers, setup, and final scoring — the C++ port target.
    measured = sum(timer.seconds.values())
    print("=" * 72)
    print("1. SEQUENTIAL - collect.play_game loop, single thread, per-decision split")
    print("=" * 72)
    print(f"games / wall    : {games} / {wall:.2f}s")
    print(f"throughput      : {games / wall:.2f} games/sec")
    print(f"decisions/game  : {total_steps / max(games, 1):.1f}")
    print("-" * 72)
    _print_bucket("engine + scoring", wall - measured, wall, total_steps)
    _print_bucket("encode_state", timer.seconds.get("encode_state", 0.0), wall, total_steps)
    _print_bucket("encode_choices", timer.seconds.get("encode_choices", 0.0), wall, total_steps)
    _print_bucket("inference (fwd)", timer.seconds.get("inference", 0.0), wall, total_steps)
    _print_bucket("bookkeeping", timer.seconds.get("bookkeeping", 0.0), wall, total_steps)
    print()
    return games / wall


def _measure_batched(
    net: model.PolicyValueNet, device: torch.device, games: int, seq_gps: float
) -> None:
    """Time the threaded shared-forward path and report its speedup over
    sequential (the GIL ceiling for in-process threads on CPU)."""
    seeds = [1_000 + i for i in range(games)]
    batched_collect.collect_games(net, device, seeds[:1])  # warm-up

    start = time.perf_counter()
    records = batched_collect.collect_games(net, device, seeds)
    elapsed = time.perf_counter() - start

    total_steps = sum(len(rec.steps) for rec in records)
    gps = len(records) / elapsed
    print("=" * 72)
    print("2. BATCHED - batched_collect.collect_games (threads share one forward)")
    print("=" * 72)
    print(f"games / wall    : {len(records)} / {elapsed:.2f}s")
    print(f"throughput      : {gps:.2f} games/sec")
    print(f"decisions/game  : {total_steps / max(len(records), 1):.1f}")
    print(f"speedup vs seq  : {gps / seq_gps:.2f}x  ({_CPU_INTRAOP_THREADS}-thread torch)")
    print()


def _measure_mp_collect(
    net: model.PolicyValueNet, device: torch.device, games: int, seq_gps: float
) -> None:
    """Time the production process-parallel path and report parallel efficiency
    vs the sequential x workers ideal (the headroom a threaded collector recovers)."""
    seeds = [4_000 + i for i in range(games)]
    # A temp checkpoint dir holds the broadcast weights file; the default arch
    # matches ``net`` so each worker's rebuilt net strict-loads the broadcast.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config.RunConfig(run=config.RunSettings(checkpoint_dir=tmp))
        collector = mp_collect.ProcessCollector(cfg)
        workers = collector.num_workers
        try:
            collector.collect_games(net, device, seeds[:workers])  # warm-up: spawn + load
            start = time.perf_counter()
            records = collector.collect_games(net, device, seeds)
            elapsed = time.perf_counter() - start
        finally:
            collector.close()

    total_steps = sum(len(rec.steps) for rec in records)
    gps = len(records) / elapsed
    efficiency = gps / (seq_gps * workers) if seq_gps > 0 else 0.0
    print("=" * 72)
    print("3. MP_COLLECT - ProcessCollector (process-parallel, the real CPU path)")
    print("=" * 72)
    print(f"workers         : {workers}")
    print(f"games / wall    : {len(records)} / {elapsed:.2f}s")
    print(f"throughput      : {gps:.2f} games/sec")
    print(f"decisions/game  : {total_steps / max(len(records), 1):.1f}")
    print(f"speedup vs seq  : {gps / seq_gps:.2f}x")
    print(f"parallel eff.   : {efficiency * 100:.0f}%  (g/s / (seq g/s x {workers} workers))")
    print()


def _measure_ipc_payload(net: model.PolicyValueNet, device: torch.device) -> None:
    """Measure the per-game pickled trajectory size before/after fp16 compaction
    and the compaction time — the IPC tax processes pay that threads avoid."""
    rng = random.Random(0)
    record = collect.play_game(net, device, rng, 7_000)
    n_steps = len(record.steps)

    before = len(pickle.dumps(record))
    # Reach into the real private compaction path to time it and size its output —
    # there is no public entry point, and a profiler must measure the actual path
    # workers use. ``_compact`` mutates in place, so the fp32 ``before`` pickle is
    # captured first.
    start = time.perf_counter()
    mp_collect._compact(record)  # type: ignore[reportPrivateUsage]
    compact_ms = (time.perf_counter() - start) * 1e3
    after = len(pickle.dumps(record))

    print("=" * 72)
    print("4. IPC PAYLOAD - pickled GameRecord, pre/post fp16 compaction (one game)")
    print("=" * 72)
    print(f"decisions       : {n_steps}")
    print(f"pickled (fp32)  : {before / 1024:.0f} KB  ({before / max(n_steps, 1):.0f} B/decision)")
    print(f"pickled (fp16)  : {after / 1024:.0f} KB  ({after / max(before, 1) * 100:.0f}% of fp32)")
    print(f"compaction      : {compact_ms:.2f} ms/game")
    print()


def _measure_attribution(
    net: model.PolicyValueNet, device: torch.device, games: int, sort: str
) -> None:
    """Single-threaded cProfile attribution for the per-function breakdown."""
    rng = random.Random(0)
    seeds = [9_000 + i for i in range(games)]
    collect.play_game(net, device, rng, seeds[0])  # warm-up

    profiler = cProfile.Profile()
    profiler.enable()
    for seed in seeds:
        collect.play_game(net, device, rng, seed)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).strip_dirs()
    stats.sort_stats(sort)
    stats.print_stats(30)
    print("=" * 72)
    print(f"5. ATTRIBUTION - single-threaded cProfile ({games} games, sort={sort})")
    print("=" * 72)
    print(buffer.getvalue())


###### PRIVATE #######


class _PatchSpec(typing.NamedTuple):
    """One monkeypatch target: ``module.attr`` is wrapped with a timer that
    accumulates into ``bucket`` (multiple specs may share one bucket)."""

    module: types.ModuleType
    attr: str
    bucket: str


class _BucketTimer:
    """Accumulates wall-time into named buckets by monkeypatching a set of
    module-level functions for the duration of a ``with`` block.

    Single-threaded use gives faithful per-call cost. The lock keeps accumulation
    correct if a wrapped function is ever called from multiple threads, but under
    concurrency numpy/torch release the GIL, so a bucket's seconds then over-count
    (a thread's timer absorbs others' overlapping work) — read the split from the
    single-threaded sequential run, not a concurrent one."""

    def __init__(self, specs: typing.Sequence[_PatchSpec]) -> None:
        self.seconds: dict[str, float] = {}
        self._specs = specs
        self._lock = threading.Lock()
        self._originals: list[
            tuple[types.ModuleType, str, typing.Callable[..., typing.Any]]
        ] = []

    def __enter__(self) -> _BucketTimer:
        for spec in self._specs:
            original = getattr(spec.module, spec.attr)
            self._originals.append((spec.module, spec.attr, original))
            setattr(spec.module, spec.attr, self._wrap(original, spec.bucket))
        return self

    def __exit__(self, *exc: object) -> None:
        for module, attr, original in self._originals:
            setattr(module, attr, original)
        self._originals.clear()

    def _wrap(
        self, original: typing.Callable[..., typing.Any], bucket: str
    ) -> typing.Callable[..., typing.Any]:
        def timed(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                with self._lock:
                    self.seconds[bucket] = (
                        self.seconds.get(bucket, 0.0) + time.perf_counter() - start
                    )

        return timed


def _sequential_patch_specs() -> list[_PatchSpec]:
    """The per-decision hot-path functions ``collect.play_game`` routes through.

    All are accessed module-qualified at their call sites (the project's import
    rule), so patching the module attribute is seen by the caller: ``model.core``
    calls ``encode.encode_state`` / ``encode_choices``; ``collect._recording_agent``
    calls ``policy.policy_value_and_probs`` and the bare module-level
    ``running_margin`` / ``running_own_score``."""
    return [
        _PatchSpec(encode, "encode_state", "encode_state"),
        _PatchSpec(encode, "encode_choices", "encode_choices"),
        _PatchSpec(policy, "policy_value_and_probs", "inference"),
        _PatchSpec(collect, "running_margin", "bookkeeping"),
        _PatchSpec(collect, "running_own_score", "bookkeeping"),
    ]


def _print_bucket(label: str, seconds: float, wall: float, steps: int) -> None:
    """Print one bucket as seconds, percent of wall, and microseconds/decision."""
    pct = seconds / wall * 100 if wall > 0 else 0.0
    micros = seconds / steps * 1e6 if steps > 0 else 0.0
    print(f"  {label:<16}: {seconds:6.2f}s  {pct:4.0f}% wall  {micros:8.1f} us/decision")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64, help="batched game count")
    parser.add_argument(
        "--seq-games", type=int, default=12, help="sequential game count"
    )
    parser.add_argument("--mp-games", type=int, default=64, help="mp_collect game count")
    parser.add_argument(
        "--profile-games", type=int, default=12, help="games to cProfile single-threaded"
    )
    parser.add_argument(
        "--skip-mp", action="store_true", help="skip the mp_collect section"
    )
    parser.add_argument(
        "--skip-attribution", action="store_true", help="skip the cProfile section"
    )
    parser.add_argument(
        "--sort",
        default="tottime",
        choices=["tottime", "cumtime", "ncalls"],
        help="pstats sort key",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
