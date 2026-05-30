"""Tests for the process-parallel self-play collector (``mp_collect``).

These spawn a small worker pool, so they exercise the real Windows-spawn path:
picklable worker entry points, on-disk weight broadcast, and GameRecords pickled
back. Kept tiny (2 workers, a few games) so the spawn cost stays modest.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from wingspan import model
from wingspan.training import collect, config, mp_collect


def test_process_collector_plays_games(tmp_path: pathlib.Path) -> None:
    device = torch.device("cpu")
    net = model.PolicyValueNet()
    net.eval()
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)

    seeds = [101, 102, 103, 104]
    done: list[collect.GameRecord] = []
    try:
        records = collector.collect_games(net, device, seeds, on_game_done=done.append)
    finally:
        collector.close()

    assert len(records) == len(seeds)
    assert len(done) == len(seeds)
    assert all(len(record.steps) > 0 for record in records)
    assert all(record.winner in (-1, 0, 1) for record in records)
    # Each recorded step must carry the encoder outputs the learner consumes.
    sample = records[0].steps[0]
    assert sample.state.ndim == 1
    assert sample.choices.ndim == 2


def test_process_collector_same_seed_is_deterministic(tmp_path: pathlib.Path) -> None:
    """The same seed yields the same game (identical decision sequence) no matter
    which worker plays it — collection stays reproducible per seed."""
    device = torch.device("cpu")
    net = model.PolicyValueNet()
    net.eval()
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        records = collector.collect_games(net, device, [7, 7, 7])
    finally:
        collector.close()

    decision_sequences = {tuple(step.chosen_idx for step in r.steps) for r in records}
    assert len(decision_sequences) == 1, "same seed produced diverging games"


def test_process_collector_empty_seeds(tmp_path: pathlib.Path) -> None:
    """No seeds returns immediately and never spawns the pool."""
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        assert (
            collector.collect_games(model.PolicyValueNet(), torch.device("cpu"), [])
            == []
        )
    finally:
        collector.close()
