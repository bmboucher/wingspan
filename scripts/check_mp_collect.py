"""Smoke + throughput check for the process-parallel collector.

Run directly: ``python scripts/check_mp_collect.py``. Guarded under __main__ so
Windows spawn re-imports this module cleanly without re-running the check.
"""

from __future__ import annotations

import pathlib
import sys
import time

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from wingspan import model  # noqa: E402
from wingspan.training import collect, config, mp_collect  # noqa: E402


def main() -> None:
    torch.set_num_threads(2)
    device = torch.device("cpu")
    net = model.PolicyValueNet()
    net.eval()
    cfg = config.TrainConfig(device="cpu", checkpoint_dir="checkpoints")

    games = 64
    seeds = [4_000 + i for i in range(games)]
    collector = mp_collect.ProcessCollector(cfg)
    print(f"workers: {collector.num_workers}")

    done: list[int] = []
    # Warm-up: pays the pool spawn + first weight load so the timed run is steady.
    collector.collect_games(net, device, seeds[:collector.num_workers])

    start = time.perf_counter()
    records = collector.collect_games(
        net, device, seeds, on_game_done=lambda r: done.append(len(r.steps))
    )
    elapsed = time.perf_counter() - start
    collector.close()

    assert len(records) == games, f"got {len(records)} records, expected {games}"
    assert all(len(r.steps) > 0 for r in records), "a game recorded no steps"
    assert len(done) == games, f"on_game_done fired {len(done)} times"
    total_steps = sum(len(r.steps) for r in records)

    print(f"games        : {len(records)}")
    print(f"wall-clock   : {elapsed:.2f}s")
    print(f"throughput   : {len(records) / elapsed:.2f} games/sec")
    print(f"decisions/gm : {total_steps / games:.1f}")
    print(f"sample score : {records[0].scores}")
    print("OK")


if __name__ == "__main__":
    main()
