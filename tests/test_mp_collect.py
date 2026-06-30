# pyright: reportPrivateUsage=false
# (test accesses ProcessCollector._pool to poll for pool startup — deliberate)
"""Tests for the process-parallel self-play collector (``mp_collect``).

These spawn a small worker pool, so they exercise the real Windows-spawn path:
picklable worker entry points, on-disk weight broadcast, and GameRecords pickled
back. Kept tiny (2 workers, a few games, a small net) so the spawn cost stays
modest — the tests assert spawn/broadcast/parity mechanics, not model capacity,
and the small net keeps every per-decision forward pass cheap.
"""

from __future__ import annotations

import pathlib
import threading
import time

import torch

from wingspan import model
from wingspan.training import collect, config, evaluate, mp_collect

# The workers rebuild their local net from ``cfg.arch`` and strict-load the
# broadcast weights, so the main-process net must be built from the same config
# (``_small_net``) for the shapes to agree.
_SMALL_LAYERS = (32, 32)
_SMALL_CARD_EMBED_DIM = 16
_SMALL_CARD_ENCODER_LAYERS = (32,)


def _small_config(tmp_path: pathlib.Path) -> config.TrainConfig:
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=_SMALL_LAYERS,
                choice_layers=_SMALL_LAYERS,
                card_embed_dim=_SMALL_CARD_EMBED_DIM,
                card_encoder_layers=_SMALL_CARD_ENCODER_LAYERS,
            ),
        ),
    )


def _small_net(cfg: config.TrainConfig) -> model.PolicyValueNet:
    net = model.PolicyValueNet(arch=cfg.arch, spec=cfg.encoding_spec)
    net.eval()
    return net


def test_process_collector_plays_games(tmp_path: pathlib.Path) -> None:
    device = torch.device("cpu")
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)
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
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        records = collector.collect_games(net, device, [7, 7, 7])
    finally:
        collector.close()

    decision_sequences = {tuple(step.chosen_idx for step in r.steps) for r in records}
    assert len(decision_sequences) == 1, "same seed produced diverging games"


def test_process_collector_empty_seeds(tmp_path: pathlib.Path) -> None:
    """No seeds returns immediately and never spawns the pool."""
    cfg = _small_config(tmp_path)
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        assert collector.collect_games(_small_net(cfg), torch.device("cpu"), []) == []
    finally:
        collector.close()


def test_eval_matches_sequential_vs_random(tmp_path: pathlib.Path) -> None:
    """Process-parallel eval vs the random agent yields the same summary as the
    sequential ``evaluate_vs_opponent`` — identical games, order-independent
    stats. ``set_num_threads(1)`` makes the main process match the single-thread
    workers so the greedy argmax cannot diverge on a float-reduction tie."""
    torch.set_num_threads(1)
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)
    sequential = evaluate.evaluate_vs_opponent(
        net,
        None,
        torch.device("cpu"),
        n_pairs=4,
        seed=123,
        opponent_generation=0,
        split_setup_bonus=cfg.split_setup_bonus_active,
        split_setup_food=cfg.split_setup_food_active,
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
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)
    opponent = _small_net(cfg)
    sequential = evaluate.evaluate_vs_opponent(
        net,
        opponent,
        torch.device("cpu"),
        n_pairs=4,
        seed=7,
        opponent_generation=1,
        split_setup_bonus=cfg.split_setup_bonus_active,
        split_setup_food=cfg.split_setup_food_active,
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
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    try:
        result = collector.evaluate_games(
            net, None, torch.device("cpu"), n_pairs=0, seed=1
        )
    finally:
        collector.close()
    assert result.n_games == 0
    assert result.win_rate == 0.0


def test_terminate_before_pool_is_noop(tmp_path: pathlib.Path) -> None:
    """terminate() on a never-used collector does not raise and leaves no _mp_* files."""
    cfg = _small_config(tmp_path)
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)
    collector.terminate()  # pool never created — must not raise
    assert not list(tmp_path.glob("_mp_*.pt"))
    assert not list(tmp_path.glob("_mp_*.pt.tmp"))


def test_terminate_kills_workers_and_is_fast(tmp_path: pathlib.Path) -> None:
    """terminate() kills in-flight workers immediately and returns in under a second.

    A background thread starts a collect (which spawns workers and submits games);
    we wait for the pool to exist, then call terminate() from the main thread and
    assert it returns quickly and that a subsequent collect_games raises.
    """
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)
    device = torch.device("cpu")
    collector = mp_collect.ProcessCollector(cfg, num_workers=2)

    collect_error: list[Exception] = []

    def _run_collect() -> None:
        try:
            # A large seed list keeps workers busy long enough for terminate() to land.
            seeds = list(range(200, 232))
            collector.collect_games(net, device, seeds)
        except Exception as error:  # noqa: BLE001
            collect_error.append(error)

    bg = threading.Thread(target=_run_collect, daemon=True)
    bg.start()

    # Wait until the pool is up (workers spawned and accepting tasks).
    deadline = time.monotonic() + 30.0
    while collector._pool is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert collector._pool is not None, "pool never started within 30s"

    # terminate() must return quickly.
    terminate_start = time.monotonic()
    collector.terminate()
    elapsed = time.monotonic() - terminate_start
    assert elapsed < 2.0, f"terminate() took {elapsed:.2f}s — expected < 2s"

    bg.join(timeout=10.0)
    assert not bg.is_alive(), "background collect thread did not finish within 10s"

    # The background collect should have raised (BrokenProcessPool or RuntimeError).
    assert collect_error, "expected collect_games to raise after terminate()"

    # A subsequent call must raise RuntimeError (not silently spawn a new pool).
    try:
        collector.collect_games(net, device, [1, 2])
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "collect_games after terminate() should raise RuntimeError"
