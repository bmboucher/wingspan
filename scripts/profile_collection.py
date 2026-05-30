"""Throughput profiler for self-play collection (measure-first baseline).

Four measurements, printed in order:

1. **Sequential g/s** — ``collect.play_game`` in a loop on one thread. The
   floor: no batching, batch-of-one forward per decision.

2. **Batched g/s** — the real training path (``batched_collect.collect_games``)
   at the configured ``--games``. The number to beat. The *speedup ratio* over
   sequential reveals the GIL ceiling: the per-decision engine + encode work is
   pure-Python/numpy-scalar and holds the GIL, so threads cannot overlap it —
   only the forward pass (server thread, GIL released) overlaps.

3. **Encode share** — total wall-time spent inside ``encode_state`` +
   ``encode_choices`` during the batched run (accumulated across all game
   threads), as a fraction of batched wall-clock. Because encoding is
   GIL-serialized, this sum *is* (roughly) the wall-clock it costs.

4. **Attribution** — a single-threaded cProfile, sorted by self time, for the
   per-function breakdown.

Run: ``python scripts/profile_collection.py --games 64 --profile-games 12``
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pathlib
import pstats
import random
import sys
import threading
import time

# Tests + scripts prepend src/ themselves so this runs without an install.
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from wingspan import encode, model  # noqa: E402
from wingspan.training import batched_collect, collect  # noqa: E402

# Match the training loop's CPU thread pinning so the baseline reflects reality.
_CPU_INTRAOP_THREADS = 2

# The wide-decision soft-threshold warning fires per decision and floods output;
# silence it so the profile is readable (it is itself hot-path noise — see notes).
logging.getLogger("wingspan.encode").setLevel(logging.ERROR)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(_CPU_INTRAOP_THREADS)
    device = torch.device("cpu")
    net = model.PolicyValueNet()
    net.eval()

    seq_gps = _measure_sequential(net, device, args.seq_games)
    _measure_batched(net, device, args.games, seq_gps)
    _measure_attribution(net, device, args.profile_games, args.sort)


def _measure_sequential(
    net: model.PolicyValueNet, device: torch.device, games: int
) -> float:
    rng = random.Random(0)
    collect.play_game(net, device, rng, 1)  # warm-up

    # Time encode CPU on the *single* thread (no contention), so the per-call
    # perf_counter delta is real cost — unlike under concurrency, where numpy
    # releases the GIL and a thread's timer absorbs other threads' work.
    encode_seconds = _EncodeTimer()
    total_steps = 0
    with encode_seconds.patch():
        start = time.perf_counter()
        for i in range(games):
            total_steps += len(collect.play_game(net, device, rng, 5_000 + i).steps)
        elapsed = time.perf_counter() - start

    gps = games / elapsed
    print("=" * 70)
    print("1. SEQUENTIAL — collect.play_game loop, single thread")
    print("=" * 70)
    print(f"games / wall    : {games} / {elapsed:.2f}s")
    print(f"throughput      : {gps:.2f} games/sec")
    print(f"decisions/game  : {total_steps / max(games, 1):.1f}")
    print(f"encode_state    : {encode_seconds.state:.2f}s ({encode_seconds.state / elapsed * 100:.0f}% of wall)")
    print(f"encode_choices  : {encode_seconds.choices:.2f}s ({encode_seconds.choices / elapsed * 100:.0f}% of wall)")
    print(f"encode total    : {encode_seconds.total:.2f}s ({encode_seconds.total / elapsed * 100:.0f}% of wall)")
    print(f"encode/decision : {encode_seconds.total / max(total_steps, 1) * 1e3:.3f} ms")
    print()
    return gps


def _measure_batched(
    net: model.PolicyValueNet, device: torch.device, games: int, seq_gps: float
) -> None:
    seeds = [1_000 + i for i in range(games)]
    batched_collect.collect_games(net, device, seeds[:1])  # warm-up

    start = time.perf_counter()
    records = batched_collect.collect_games(net, device, seeds)
    elapsed = time.perf_counter() - start

    total_steps = sum(len(rec.steps) for rec in records)
    gps = len(records) / elapsed
    print("=" * 70)
    print("2. BATCHED — batched_collect.collect_games (the real training path)")
    print("=" * 70)
    print(f"games / wall    : {len(records)} / {elapsed:.2f}s")
    print(f"throughput      : {gps:.2f} games/sec")
    print(f"decisions/game  : {total_steps / max(len(records), 1):.1f}")
    print(f"speedup vs seq  : {gps / seq_gps:.2f}x  (over {_CPU_INTRAOP_THREADS}-thread torch, up to 64 game threads)")
    print()


def _measure_attribution(
    net: model.PolicyValueNet, device: torch.device, games: int, sort: str
) -> None:
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
    print("=" * 70)
    print(f"4. ATTRIBUTION — single-threaded cProfile ({games} games, sort={sort})")
    print("=" * 70)
    print(buffer.getvalue())


class _EncodeTimer:
    """Accumulates wall-time spent in the two encode entry points across all
    threads by monkeypatching them for the duration of a ``with`` block."""

    def __init__(self) -> None:
        self.state = 0.0
        self.choices = 0.0
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        return self.state + self.choices

    def patch(self) -> _EncodeTimer:
        self._orig_state = encode.encode_state
        self._orig_choices = encode.encode_choices
        return self

    def __enter__(self) -> _EncodeTimer:
        orig_state = self._orig_state
        orig_choices = self._orig_choices

        def timed_state(*args, **kwargs):  # type: ignore[no-untyped-def]
            start = time.perf_counter()
            try:
                return orig_state(*args, **kwargs)
            finally:
                with self._lock:
                    self.state += time.perf_counter() - start

        def timed_choices(*args, **kwargs):  # type: ignore[no-untyped-def]
            start = time.perf_counter()
            try:
                return orig_choices(*args, **kwargs)
            finally:
                with self._lock:
                    self.choices += time.perf_counter() - start

        encode.encode_state = timed_state  # type: ignore[assignment]
        encode.encode_choices = timed_choices  # type: ignore[assignment]
        # batched_collect imported the names directly into its module namespace.
        batched_collect.encode.encode_state = timed_state  # type: ignore[attr-defined]
        batched_collect.encode.encode_choices = timed_choices  # type: ignore[attr-defined]
        return self

    def __exit__(self, *exc: object) -> None:
        encode.encode_state = self._orig_state  # type: ignore[assignment]
        encode.encode_choices = self._orig_choices  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64, help="batched game count")
    parser.add_argument("--seq-games", type=int, default=12, help="sequential game count")
    parser.add_argument(
        "--profile-games", type=int, default=12, help="games to cProfile single-threaded"
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
