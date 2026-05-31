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
from wingspan.training import collect, config, evaluate, mp_collect


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


def test_eval_matches_sequential_vs_random(tmp_path: pathlib.Path) -> None:
    """Process-parallel eval vs the random agent yields the same summary as the
    sequential ``evaluate_vs_opponent`` — identical games, order-independent
    stats. ``set_num_threads(1)`` makes the main process match the single-thread
    workers so the greedy argmax cannot diverge on a float-reduction tie."""
    torch.set_num_threads(1)
    net = model.PolicyValueNet()
    net.eval()
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    sequential = evaluate.evaluate_vs_opponent(
        net, None, torch.device("cpu"), n_pairs=4, seed=123, opponent_generation=0
    )
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        parallel = collector.evaluate_games(
            net, None, torch.device("cpu"), n_pairs=4, seed=123, opponent_generation=0
        )
    finally:
        collector.close()
    assert parallel.n_games == sequential.n_games == 8
    assert parallel.win_rate == sequential.win_rate
    assert parallel.mean_margin == sequential.mean_margin
    assert parallel.ci95 == sequential.ci95


def test_eval_matches_sequential_vs_frozen_opponent(tmp_path: pathlib.Path) -> None:
    """Process-parallel eval vs a frozen opponent net matches the sequential
    path; the opponent weights are broadcast to the workers."""
    torch.set_num_threads(1)
    net = model.PolicyValueNet()
    net.eval()
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    opponent = model.PolicyValueNet()
    opponent.eval()
    sequential = evaluate.evaluate_vs_opponent(
        net, opponent, torch.device("cpu"), n_pairs=4, seed=7, opponent_generation=1
    )
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        parallel = collector.evaluate_games(
            net, opponent, torch.device("cpu"), n_pairs=4, seed=7, opponent_generation=1
        )
    finally:
        collector.close()
    assert parallel.n_games == sequential.n_games == 8
    assert parallel.win_rate == sequential.win_rate
    assert parallel.mean_margin == sequential.mean_margin


def test_eval_empty_is_noop(tmp_path: pathlib.Path) -> None:
    """Zero pairs returns an empty result and never spawns the pool."""
    net = model.PolicyValueNet()
    net.eval()
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        result = collector.evaluate_games(
            net, None, torch.device("cpu"), n_pairs=0, seed=1
        )
    finally:
        collector.close()
    assert result.n_games == 0
    assert result.win_rate == 0.0
