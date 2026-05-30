"""Process-parallel self-play collection (sidesteps the GIL).

``batched_collect`` threads share one forward pass, but the per-decision engine
and encoding work is GIL-bound, so up to 64 game threads give only ~1.35x on CPU
(see the ``training-throughput-bottleneck`` analysis). This module fans games
across worker *processes* instead — one GIL per core — so collection scales with
physical cores. Each worker rebuilds a local :class:`model.PolicyValueNet` from
the broadcast weights and plays its games with the ordinary synchronous engine +
batch-of-one inference (:func:`collect.play_game`), returning the same
:class:`collect.GameRecord` objects the other collectors produce.

Windows uses *spawn*, which shapes the design:

* Workers live in a **persistent pool** (a :class:`ProcessCollector` the
  TrainingLoop holds across iterations), so the per-worker spawn + ``import
  torch`` cost is paid once rather than every iteration.
* The worker entry points are module-level and picklable.
* Each worker pins ``torch.set_num_threads(1)`` so N workers don't oversubscribe
  the cores against each other.
* The per-iteration weights are broadcast through a single **versioned on-disk
  ``state_dict`` file** rather than re-pickled through the task pipe for every
  game; a worker reloads only when the version it holds is stale.

The public surface is :class:`ProcessCollector`. Its :meth:`~ProcessCollector.
collect_games` mirrors :func:`batched_collect.collect_games` (same arguments,
same return), so ``loop._collect`` can choose either path by device. Results
come back in completion order, not seed order — every downstream aggregate is
order-independent.
"""

from __future__ import annotations

import os
import pathlib
import random
import typing
from concurrent import futures

import pydantic
import torch

from wingspan import model
from wingspan.training import collect, config

# Match batched_collect's per-game sampling salt so a seed maps to the same
# sampling stream regardless of which collector plays it.
_SAMPLE_RNG_SALT = 0x9E3779B9

# Broadcast weights filename, written under the run's checkpoint dir each
# iteration and read by workers when their cached version is stale.
_WEIGHTS_FILENAME = "_mp_weights.pt"

# Leave a couple of cores for the main thread, the OS, and the dashboard render.
_RESERVED_CORES = 2
# Spawning + importing torch per worker is not free; past this the marginal
# parallelism rarely beats the startup + per-iteration weight-reload cost.
_MAX_WORKERS = 16


class _WorkerArch(pydantic.BaseModel):
    """The network shape a worker needs to build its local net before any
    weights arrive. Passed once to each worker as pool ``initargs``."""

    model_config = pydantic.ConfigDict(frozen=True)

    state_dim: int
    choice_dim: int
    hidden: int


class _GameTask(pydantic.BaseModel):
    """One unit of work shipped to a worker: which weights to play under (by
    on-disk path + version) and which game seed to play."""

    model_config = pydantic.ConfigDict(frozen=True)

    weights_path: str
    weights_version: int
    seed: int


class ProcessCollector:
    """A persistent pool of self-play worker processes.

    Construct once per run and reuse across iterations; call :meth:`close` at
    shutdown. The pool is created lazily on the first :meth:`collect_games` so
    merely constructing a TrainingLoop (e.g. in the configurator or a test)
    spawns nothing.
    """

    def __init__(self, cfg: config.TrainConfig, num_workers: int | None = None):
        self._arch = _WorkerArch(
            state_dim=cfg.state_dim, choice_dim=cfg.choice_dim, hidden=cfg.hidden
        )
        self._weights_path = pathlib.Path(cfg.checkpoint_dir) / _WEIGHTS_FILENAME
        self._num_workers = num_workers or _default_worker_count(cfg.games_per_iter)
        self._pool: futures.ProcessPoolExecutor | None = None
        self._weights_version = 0

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def collect_games(
        self,
        net: model.PolicyValueNet,
        device: torch.device,
        seeds: typing.Sequence[int],
        on_game_done: typing.Callable[[collect.GameRecord], None] | None = None,
        should_stop: typing.Callable[[], bool] | None = None,
    ) -> list[collect.GameRecord]:
        """Play ``len(seeds)`` self-play games across the worker pool.

        Broadcasts ``net``'s current weights once, submits one task per seed,
        and fires ``on_game_done`` as each game completes (from this thread, so
        the callback may safely touch shared state under the caller's lock).
        ``should_stop`` is polled as games finish; once set, pending games are
        cancelled and games already in flight are awaited and kept."""
        if not seeds:
            return []
        pool = self._ensure_pool()
        self._weights_version += 1
        self._broadcast_weights(net)
        tasks = [
            _GameTask(
                weights_path=str(self._weights_path),
                weights_version=self._weights_version,
                seed=seed,
            )
            for seed in seeds
        ]

        results: list[collect.GameRecord] = []
        pending = {pool.submit(_worker_play, task): task for task in tasks}
        for future in futures.as_completed(pending):
            record = future.result()
            results.append(record)
            if on_game_done is not None:
                on_game_done(record)
            if should_stop is not None and should_stop():
                for other in pending:
                    other.cancel()
                break
        return results

    def close(self) -> None:
        """Shut the pool down (waiting for in-flight games) and remove the
        broadcast weights file. Idempotent."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._weights_path.unlink(missing_ok=True)

    ###### PRIVATE #######

    def _ensure_pool(self) -> futures.ProcessPoolExecutor:
        if self._pool is None:
            self._pool = futures.ProcessPoolExecutor(
                max_workers=self._num_workers,
                initializer=_worker_init,
                initargs=(self._arch,),
            )
        return self._pool

    def _broadcast_weights(self, net: model.PolicyValueNet) -> None:
        """Write the current weights to the versioned file atomically so a
        worker mid-read never sees a half-written ``state_dict``."""
        self._weights_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = {
            name: tensor.detach().to("cpu") for name, tensor in net.state_dict().items()
        }
        tmp = self._weights_path.with_suffix(self._weights_path.suffix + ".tmp")
        torch.save(cpu_state, tmp)
        os.replace(tmp, self._weights_path)


def _default_worker_count(games_per_iter: int) -> int:
    """Workers default to (cores - reserved), capped, and never more than the
    games on offer."""
    cores = os.cpu_count() or 4
    usable = max(1, min(cores - _RESERVED_CORES, _MAX_WORKERS))
    return max(1, min(usable, games_per_iter))


# ---------------------------------------------------------------------------
# Worker-process state. One set of these per worker process, populated by
# ``_worker_init`` (run once when the pool spawns the worker) and refreshed by
# ``_maybe_reload_weights`` when the broadcast version advances.

_worker_net: model.PolicyValueNet | None = None
_worker_device: torch.device | None = None
_worker_weights_version: int = -1


def _worker_init(arch: _WorkerArch) -> None:
    """Build this worker's local net once, before any games. Pins torch to a
    single thread so the workers parallelize across cores rather than fighting
    over them."""
    global _worker_net, _worker_device, _worker_weights_version
    torch.set_num_threads(1)
    _worker_device = torch.device("cpu")
    _worker_net = model.PolicyValueNet(
        state_dim=arch.state_dim, choice_dim=arch.choice_dim, hidden=arch.hidden
    ).to(_worker_device)
    _worker_net.eval()
    _worker_weights_version = -1


def _worker_play(task: _GameTask) -> collect.GameRecord:
    """Play one seeded self-play game under the task's weights."""
    net = _worker_net
    device = _worker_device
    assert net is not None and device is not None, "worker net not initialized"
    _maybe_reload_weights(net, device, task.weights_path, task.weights_version)
    rng = random.Random(task.seed ^ _SAMPLE_RNG_SALT)
    return collect.play_game(net, device, rng, task.seed)


def _maybe_reload_weights(
    net: model.PolicyValueNet, device: torch.device, path: str, version: int
) -> None:
    """Load ``path`` into ``net`` if this worker's cached weights are older than
    ``version`` — at most once per iteration per worker."""
    global _worker_weights_version
    if version == _worker_weights_version:
        return
    # torch.load's stubs return a partially-unknown type; the file is our own
    # plain-tensor state_dict, so narrow it for the strict checker.
    state_dict = typing.cast(
        "dict[str, torch.Tensor]",
        torch.load(path, map_location=device, weights_only=True),
    )
    net.load_state_dict(state_dict)
    net.eval()
    _worker_weights_version = version
