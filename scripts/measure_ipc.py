"""Size the IPC payload of process-parallel collection (measure-first).

The mp collector ships every finished game's ``GameRecord`` — including each
recorded step's full ``(n_choices, choice_dim)`` feature matrix — back to the
main process through a pipe (pickle). This script quantifies that payload so we
know whether IPC is the next limiter before optimizing it:

* total pickled bytes per iteration (64 games)
* pickle (worker side) + unpickle (main side) wall-time
* the choices-matrix sparsity (the compression headroom)
* the serialize+deserialize time as a fraction of the measured ~1.4s
  collection wall, i.e. how much of collection is pure marshalling

Run: ``python scripts/measure_ipc.py --games 64``
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import random
import sys
import time

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wingspan import model  # noqa: E402
from wingspan.training import collect  # noqa: E402

logging.getLogger("wingspan.encode").setLevel(logging.ERROR)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    net = model.PolicyValueNet()
    net.eval()

    records = _play(net, device, args.games)
    blob = _measure_pickle(records, args.games)
    _measure_sparsity(records)


def _play(
    net: model.PolicyValueNet, device: torch.device, games: int
) -> list[collect.GameRecord]:
    rng = random.Random(0)
    return [collect.play_game(net, device, rng, 6_000 + i) for i in range(games)]


def _measure_pickle(records: list[collect.GameRecord], games: int) -> bytes:
    # Pickle each record separately — that's how the pool ships them (one result
    # per future), so per-record protocol overhead is counted realistically.
    start = time.perf_counter()
    blobs = [pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL) for record in records]
    pickle_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for blob in blobs:
        pickle.loads(blob)
    unpickle_seconds = time.perf_counter() - start

    total_bytes = sum(len(blob) for blob in blobs)
    total_steps = sum(len(record.steps) for record in records)
    print("=" * 66)
    print(f"IPC PAYLOAD — {games} games (one iteration)")
    print("=" * 66)
    print(f"total pickled    : {total_bytes / 1e6:.1f} MB")
    print(f"per game         : {total_bytes / games / 1e6:.2f} MB")
    print(f"per step         : {total_bytes / max(total_steps, 1) / 1e3:.2f} KB")
    print(f"pickle (worker)  : {pickle_seconds * 1e3:.0f} ms  ({pickle_seconds / games * 1e3:.2f} ms/game)")
    print(f"unpickle (main)  : {unpickle_seconds * 1e3:.0f} ms  (serial, main thread)")
    print(f"marshalling total: {(pickle_seconds + unpickle_seconds) * 1e3:.0f} ms")
    print(f"  vs ~1.4s collect: {(pickle_seconds + unpickle_seconds) / 1.4 * 100:.0f}% of wall")
    print()
    return b"".join(blobs)


def _measure_sparsity(records: list[collect.GameRecord]) -> None:
    total_elems = 0
    nonzero_elems = 0
    max_rows = 0
    for record in records:
        for step in record.steps:
            total_elems += step.choices.size
            nonzero_elems += int(np.count_nonzero(step.choices))
            max_rows = max(max_rows, step.choices.shape[0])
    density = nonzero_elems / max(total_elems, 1)
    print("=" * 66)
    print("CHOICES SPARSITY (the compression headroom)")
    print("=" * 66)
    print(f"choice elems     : {total_elems / 1e6:.1f} M float32 ({total_elems * 4 / 1e6:.1f} MB raw)")
    print(f"nonzero          : {nonzero_elems / 1e6:.2f} M ({density * 100:.1f}% dense)")
    print(f"widest decision  : {max_rows} candidates")
    print(f"float16 would be : {total_elems * 2 / 1e6:.1f} MB (-50%)")
    print(f"CSR-ish nonzero  : ~{nonzero_elems * 8 / 1e6:.1f} MB (idx+val, if recomputed in learner)")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main()
