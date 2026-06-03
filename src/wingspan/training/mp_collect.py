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

Selection: ``loop._collect`` routes **CPU** collection here and CUDA collection
to ``batched_collect``. See ``training/COLLECTORS.md`` for the side-by-side.
"""

from __future__ import annotations

import logging
import os
import pathlib
import random
import typing
from concurrent import futures

import numpy as np
import pydantic
import torch

from wingspan import agents, architecture, encode, model, setup_model
from wingspan.training import collect, config, evaluate, metrics, setup_net

# Match batched_collect's per-game sampling salt so a seed maps to the same
# sampling stream regardless of which collector plays it.
_SAMPLE_RNG_SALT = 0x9E3779B9

# Distinct salt for the bootstrap phase's random opponent, kept separate from
# the policy-sampling stream so a seed reproduces both the same game and the
# same opponent without the two RNGs sharing a sequence.
_OPPONENT_RNG_SALT = 0x85EBCA6B

# Broadcast weights filename, written under the run's checkpoint dir each
# iteration and read by workers when their cached version is stale.
_WEIGHTS_FILENAME = "_mp_weights.pt"

# Broadcast reference-opponent weights filename, written only when the opponent
# generation advances (the random-agent opponent at generation 0 needs no file).
_OPPONENT_FILENAME = "_mp_opponent.pt"

# Broadcast setup-net weights filename, written each model-driven iteration (the
# setup net is unused — and unwritten — in the random setup phases).
_SETUP_WEIGHTS_FILENAME = "_mp_setup_weights.pt"

# Leave a couple of cores for the main thread, the OS, and the dashboard render.
_RESERVED_CORES = 2
# Spawning + importing torch per worker is not free; past this the marginal
# parallelism rarely beats the startup + per-iteration weight-reload cost.
_MAX_WORKERS = 16


class _WorkerArch(pydantic.BaseModel):
    """The network shape a worker needs to build its local net before any
    weights arrive. Passed once to each worker as pool ``initargs``; the full
    topology travels in ``arch`` so the worker rebuilds a byte-identical net."""

    model_config = pydantic.ConfigDict(frozen=True)

    state_dim: int
    choice_dim: int
    arch: architecture.ModelArchitecture
    # Whether the main net carries setup (``encode.EncodingSpec.include_setup``);
    # the worker rebuilds its net with a matching spec so its encoded vectors and
    # head count line up with the broadcast weights.
    include_setup: bool = True
    # Setup-model shape + generation knobs a worker needs to build its local
    # setup net and random generator. Absent (``setup_enabled=False``) when the
    # run does not use the setup model.
    setup_enabled: bool = False
    setup_arch: setup_model.SetupArchitecture | None = None
    setup_feature_dim: int = 0
    setup_hand_combos: int = 1
    setup_food_sets: int = 1
    setup_tuples_per_batch: int = 1
    setup_temperature: float = 1.0
    # Whether the opening bonus pick is deferred to the in-game ``CHOOSE_BONUS``
    # head (the ``split_setup_bonus`` regime); applied in both the setup-collection
    # and the eval game paths so workers match the main process.
    split_setup_bonus: bool = False


class _GameTask(pydantic.BaseModel):
    """One unit of work shipped to a worker: which weights to play under (by
    on-disk path + version) and which game seed to play. ``vs_random`` selects
    the bootstrap phase, where the net plays seat 0 against the random agent
    instead of self-play."""

    model_config = pydantic.ConfigDict(frozen=True)

    weights_path: str
    weights_version: int
    seed: int
    vs_random: bool = False


class _SetupGameTask(pydantic.BaseModel):
    """One setup-model game: the policy weights to play under, the setup-net
    weights (used only in the model-driven phase), and the per-game setup
    directive. ``vs_random`` selects the bootstrap phase (net at seat 0)."""

    model_config = pydantic.ConfigDict(frozen=True)

    weights_path: str
    weights_version: int
    setup_weights_path: str
    setup_weights_version: int
    spec: collect.SetupGameSpec
    vs_random: bool = False


class _EvalTask(pydantic.BaseModel):
    """One held-out eval game: the policy weights and reference-opponent weights
    to play under (by path + version), the paired-deal seed, and which seat the
    policy takes. ``opponent_is_random`` selects the random agent (generation 0),
    in which case the opponent path/version are unused."""

    model_config = pydantic.ConfigDict(frozen=True)

    weights_path: str
    weights_version: int
    opponent_path: str
    opponent_version: int
    opponent_is_random: bool
    pair_seed: int
    net_seat: int


class ProcessCollector:
    """A persistent pool of self-play worker processes.

    Construct once per run and reuse across iterations; call :meth:`close` at
    shutdown. The pool is created lazily on the first :meth:`collect_games` so
    merely constructing a TrainingLoop (e.g. in the configurator or a test)
    spawns nothing.
    """

    def __init__(self, cfg: config.TrainConfig, num_workers: int | None = None):
        self._arch = _WorkerArch(
            state_dim=cfg.state_dim,
            choice_dim=cfg.choice_dim,
            arch=cfg.arch,
            include_setup=cfg.encoding_spec.include_setup,
            setup_enabled=cfg.use_setup_model,
            setup_arch=cfg.setup_arch if cfg.use_setup_model else None,
            setup_feature_dim=setup_model.SETUP_FEATURE_DIM,
            setup_hand_combos=cfg.setup_hand_combos,
            setup_food_sets=cfg.setup_food_sets,
            setup_tuples_per_batch=cfg.setup_tuples_per_batch,
            setup_temperature=cfg.setup_policy_temperature,
            split_setup_bonus=cfg.split_setup_bonus_active,
        )
        self._weights_path = pathlib.Path(cfg.checkpoint_dir) / _WEIGHTS_FILENAME
        self._opponent_path = pathlib.Path(cfg.checkpoint_dir) / _OPPONENT_FILENAME
        self._setup_weights_path = (
            pathlib.Path(cfg.checkpoint_dir) / _SETUP_WEIGHTS_FILENAME
        )
        self._num_workers = num_workers or _default_worker_count(cfg.games_per_iter)
        self._pool: futures.ProcessPoolExecutor | None = None
        self._weights_version = 0
        self._setup_weights_version = 0
        # The opponent file is rewritten only when the generation advances; -1
        # forces the first non-random eval to broadcast it.
        self._opponent_broadcast_generation = -1

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
        vs_random: bool = False,
    ) -> list[collect.GameRecord]:
        """Play ``len(seeds)`` games across the worker pool — self-play, or (when
        ``vs_random``) the net at seat 0 against the random agent.

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
                vs_random=vs_random,
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

    def collect_games_with_setup(
        self,
        net: model.PolicyValueNet,
        setup_policy_net: setup_net.SetupNet | None,
        device: torch.device,
        specs: typing.Sequence[collect.SetupGameSpec],
        on_game_done: typing.Callable[[collect.GameRecord], None] | None = None,
        should_stop: typing.Callable[[], bool] | None = None,
        vs_random: bool = False,
    ) -> list[collect.GameRecord]:
        """Play one game per ``SetupGameSpec`` across the worker pool with setups
        chosen by the random generator / setup net (mirrors :meth:`collect_games`
        but for the setup-model collection path). Broadcasts the policy weights
        every call and the setup-net weights when any spec is model-driven."""
        if not specs:
            return []
        pool = self._ensure_pool()
        self._weights_version += 1
        self._broadcast_weights(net)
        model_driven = any(
            spec.phase is collect.SetupPhase.MODEL_DRIVEN for spec in specs
        )
        if model_driven and setup_policy_net is not None:
            self._setup_weights_version += 1
            self._broadcast_setup_weights(setup_policy_net)
        tasks = [
            _SetupGameTask(
                weights_path=str(self._weights_path),
                weights_version=self._weights_version,
                setup_weights_path=str(self._setup_weights_path),
                setup_weights_version=self._setup_weights_version,
                spec=spec,
                vs_random=vs_random,
            )
            for spec in specs
        ]

        results: list[collect.GameRecord] = []
        pending = {pool.submit(_worker_play_setup, task): task for task in tasks}
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

    def evaluate_games(
        self,
        net: model.PolicyValueNet,
        opponent_net: model.PolicyValueNet | None,
        device: torch.device,
        n_pairs: int,
        seed: int,
        opponent_generation: int = 0,
        on_progress: evaluate.EvalProgress | None = None,
    ) -> metrics.EvalResult:
        """Play ``n_pairs`` mirrored eval deals across the worker pool and
        summarize, mirroring :func:`evaluate.evaluate_vs_opponent` (same args,
        same :class:`metrics.EvalResult`). ``opponent_net=None`` plays the random
        agent. Workers run the *same* :func:`evaluate.play_eval_game`, so the
        result is identical to the sequential path game-for-game; only the
        completion order differs, and every summary statistic is
        order-independent. ``on_progress`` fires from this thread as games land."""
        if n_pairs <= 0:
            return evaluate.summarize_eval([], opponent_generation)
        pool = self._ensure_pool()
        self._weights_version += 1
        self._broadcast_weights(net)
        opponent_is_random = opponent_net is None
        if opponent_net is not None:
            self._broadcast_opponent(opponent_net, opponent_generation)
        tasks = [
            _EvalTask(
                weights_path=str(self._weights_path),
                weights_version=self._weights_version,
                opponent_path=str(self._opponent_path),
                opponent_version=opponent_generation,
                opponent_is_random=opponent_is_random,
                pair_seed=seed + pair * 2,
                net_seat=net_seat,
            )
            for pair in range(n_pairs)
            for net_seat in (0, 1)
        ]

        n_games = 2 * n_pairs
        margins: list[int] = []
        for future in futures.as_completed(
            [pool.submit(_worker_eval, task) for task in tasks]
        ):
            margins.append(future.result())
            if on_progress is not None:
                on_progress(len(margins), n_games)
        return evaluate.summarize_eval(margins, opponent_generation)

    def close(self) -> None:
        """Shut the pool down (waiting for in-flight games) and remove the
        broadcast weights files. Idempotent."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._weights_path.unlink(missing_ok=True)
        self._opponent_path.unlink(missing_ok=True)
        self._setup_weights_path.unlink(missing_ok=True)

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
        """Write the current policy weights to the versioned file (every
        iteration / eval) so workers pick them up on the next version bump."""
        _save_state_atomic(self._weights_path, net)

    def _broadcast_opponent(
        self, opponent_net: model.PolicyValueNet, generation: int
    ) -> None:
        """Write the reference-opponent weights, but only when the generation
        has advanced since the last broadcast — the opponent is frozen between
        advances, so re-writing its (unchanged) weights every eval is waste."""
        if generation == self._opponent_broadcast_generation:
            return
        _save_state_atomic(self._opponent_path, opponent_net)
        self._opponent_broadcast_generation = generation

    def _broadcast_setup_weights(self, setup_policy_net: setup_net.SetupNet) -> None:
        """Write the current setup-net weights to the versioned file so workers
        pick them up on the next model-driven version bump."""
        _save_module_state_atomic(self._setup_weights_path, setup_policy_net)


def _default_worker_count(games_per_iter: int) -> int:
    """Workers default to (cores - reserved), capped, and never more than the
    games on offer."""
    cores = os.cpu_count() or 4
    usable = max(1, min(cores - _RESERVED_CORES, _MAX_WORKERS))
    return max(1, min(usable, games_per_iter))


def _save_state_atomic(path: pathlib.Path, net: model.PolicyValueNet) -> None:
    """Write ``net``'s CPU ``state_dict`` to ``path`` via a temp file + rename so
    a worker mid-read never sees a half-written file."""
    _save_module_state_atomic(path, net)


def _save_module_state_atomic(path: pathlib.Path, module: torch.nn.Module) -> None:
    """Atomically write any module's CPU ``state_dict`` to ``path`` (the policy
    net and the setup net share this write-then-rename path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        name: tensor.detach().to("cpu") for name, tensor in module.state_dict().items()
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_state, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Worker-process state. One set of these per worker process, populated by
# ``_worker_init`` (run once when the pool spawns the worker) and refreshed by
# ``_maybe_reload_weights`` when the broadcast version advances.

_worker_arch: _WorkerArch | None = None
_worker_net: model.PolicyValueNet | None = None
_worker_device: torch.device | None = None
_worker_weights_version: int = -1
# Lazily built on the first non-random eval (the random-agent opponent needs no
# net); refreshed when the broadcast opponent generation advances.
_worker_opponent_net: model.PolicyValueNet | None = None
_worker_opponent_version: int = -1
# Setup-model worker state: the local setup net + its broadcast version, the
# random generator, and the softmax temperature. Built in ``_worker_init`` only
# when the run uses the setup model.
_worker_setup_net: setup_net.SetupNet | None = None
_worker_setup_version: int = -1
_worker_generator: setup_model.RandomSetupGenerator | None = None
_worker_setup_temperature: float = 1.0


def _worker_init(arch: _WorkerArch) -> None:
    """Build this worker's local net once, before any games. Pins torch to a
    single thread so the workers parallelize across cores rather than fighting
    over them."""
    global _worker_arch, _worker_net, _worker_device, _worker_weights_version
    global _worker_setup_net, _worker_setup_version, _worker_generator
    global _worker_setup_temperature
    torch.set_num_threads(1)
    _worker_arch = arch
    _worker_device = torch.device("cpu")
    _worker_net = model.PolicyValueNet(
        state_dim=arch.state_dim,
        choice_dim=arch.choice_dim,
        arch=arch.arch,
        spec=encode.EncodingSpec(include_setup=arch.include_setup),
    ).to(_worker_device)
    _worker_net.eval()
    _worker_weights_version = -1
    # Build the setup net + random generator once (only when the run uses the
    # setup model), so each game just reloads weights and plays.
    if arch.setup_enabled and arch.setup_arch is not None:
        _worker_setup_net = setup_net.SetupNet(
            feature_dim=arch.setup_feature_dim, arch=arch.setup_arch
        ).to(_worker_device)
        _worker_setup_net.eval()
        _worker_setup_version = -1
        _worker_setup_temperature = arch.setup_temperature
        _worker_generator = setup_model.RandomSetupGenerator(
            hand_combos=arch.setup_hand_combos,
            food_sets=arch.setup_food_sets,
            tuples_per_batch=arch.setup_tuples_per_batch,
        )
    # This process inherits no logging handlers — the dashboard configures file
    # logging in the main process only. Without a handler, a WARNING+ record
    # emitted here (e.g. the encoder's wide-decision notices) falls through to
    # logging.lastResort, which writes to stderr and flickers the live FLYWAY
    # CONTROL canvas. A NullHandler on the worker's root logger keeps any such
    # record from ever reaching the terminal.
    logging.getLogger().addHandler(logging.NullHandler())


def _worker_play(task: _GameTask) -> collect.GameRecord:
    """Play one seeded game under the task's weights — self-play, or the net
    (seat 0) vs the random agent in the bootstrap phase."""
    net = _worker_net
    device = _worker_device
    assert net is not None and device is not None, "worker net not initialized"
    _maybe_reload_weights(net, device, task.weights_path, task.weights_version)
    rng = random.Random(task.seed ^ _SAMPLE_RNG_SALT)
    opponent = (
        agents.random_agent(random.Random(task.seed ^ _OPPONENT_RNG_SALT))
        if task.vs_random
        else None
    )
    return _compact(collect.play_game(net, device, rng, task.seed, opponent))


def _worker_play_setup(task: _SetupGameTask) -> collect.GameRecord:
    """Play one setup-model game: reload the policy (and, in the model-driven
    phase, the setup-net) weights, then resolve setups via the worker's generator
    / setup net and play. The opponent is the random agent in the bootstrap
    phase, seeded off the continuation seed so a seed reproduces the matchup."""
    net = _worker_net
    device = _worker_device
    generator = _worker_generator
    arch = _worker_arch
    assert net is not None and device is not None, "worker net not initialized"
    assert generator is not None, "worker setup generator not initialized"
    assert arch is not None, "worker arch not initialized"
    _maybe_reload_weights(net, device, task.weights_path, task.weights_version)

    setup_policy_net: setup_net.SetupNet | None = None
    if task.spec.phase is collect.SetupPhase.MODEL_DRIVEN:
        setup_policy_net = _worker_setup_net
        assert setup_policy_net is not None, "worker setup net not initialized"
        _maybe_reload_setup_weights(
            setup_policy_net,
            device,
            task.setup_weights_path,
            task.setup_weights_version,
        )
    opponent = (
        agents.random_agent(
            random.Random(task.spec.continuation_seed ^ _OPPONENT_RNG_SALT)
        )
        if task.vs_random
        else None
    )
    return _compact(
        collect.play_game_with_setup(
            net,
            device,
            task.spec,
            generator,
            setup_policy_net,
            _worker_setup_temperature,
            opponent,
            split_setup_bonus=arch.split_setup_bonus,
        )
    )


def _worker_eval(task: _EvalTask) -> int:
    """Play one greedy held-out eval game and return the policy's score margin.
    Reuses :func:`evaluate.play_eval_game`, so the worker produces the same
    margin the sequential path would for this ``(pair_seed, net_seat)``."""
    net = _worker_net
    device = _worker_device
    arch = _worker_arch
    assert net is not None and device is not None, "worker net not initialized"
    assert arch is not None, "worker arch not initialized"
    _maybe_reload_weights(net, device, task.weights_path, task.weights_version)
    opponent = _ensure_worker_opponent(task, device)
    return evaluate.play_eval_game(
        net,
        opponent,
        device,
        task.pair_seed,
        task.net_seat,
        split_setup_bonus=arch.split_setup_bonus,
    )


def _compact(record: collect.GameRecord) -> collect.GameRecord:
    """Downcast each recorded step's feature arrays to float16 in place before
    the record is pickled back to the main process.

    The per-candidate ``choices`` matrices dominate the IPC payload (measured
    ~1.8 MB/game, ~90% of it choice features) and are ~96% zeros. The features
    are normalized to roughly [0, 1.5], so float16 (~3-4 significant digits)
    preserves them far below the policy-gradient noise floor — and the learner
    re-tensorizes to float32 on the way into the network regardless
    (``learner._forward_bucket``). Halving the element size roughly halves both
    the pickled payload crossing the pipe and the trajectory buffer's resident
    size, at no measurable cost to the update."""
    for step in record.steps:
        step.state = step.state.astype(np.float16)
        step.choices = step.choices.astype(np.float16)
    # The setup samples' multi-hot/one-hot features are exact in float16 (0/1)
    # and the small count/goal stripes lose nothing meaningful; the margin stays
    # a float. Halving the feature payload mirrors the step compaction above.
    for sample in record.setup_samples:
        sample.features = sample.features.astype(np.float16)
    return record


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


def _maybe_reload_setup_weights(
    net: setup_net.SetupNet, device: torch.device, path: str, version: int
) -> None:
    """Load ``path`` into the worker's setup net if its cached weights are older
    than ``version`` — at most once per model-driven iteration per worker."""
    global _worker_setup_version
    if version == _worker_setup_version:
        return
    state_dict = typing.cast(
        "dict[str, torch.Tensor]",
        torch.load(path, map_location=device, weights_only=True),
    )
    net.load_state_dict(state_dict)
    net.eval()
    _worker_setup_version = version


def _ensure_worker_opponent(
    task: _EvalTask, device: torch.device
) -> model.PolicyValueNet | None:
    """Return the reference-opponent net for an eval task, or ``None`` for the
    random agent. Builds the opponent net once (from the worker's architecture)
    and reloads its weights only when the broadcast generation advances."""
    global _worker_opponent_net, _worker_opponent_version
    if task.opponent_is_random:
        return None
    assert _worker_arch is not None, "worker arch not initialized"
    if _worker_opponent_net is None:
        _worker_opponent_net = model.PolicyValueNet(
            state_dim=_worker_arch.state_dim,
            choice_dim=_worker_arch.choice_dim,
            arch=_worker_arch.arch,
            spec=encode.EncodingSpec(include_setup=_worker_arch.include_setup),
        ).to(device)
        _worker_opponent_net.eval()
    if task.opponent_version != _worker_opponent_version:
        state_dict = typing.cast(
            "dict[str, torch.Tensor]",
            torch.load(task.opponent_path, map_location=device, weights_only=True),
        )
        _worker_opponent_net.load_state_dict(state_dict)
        _worker_opponent_net.eval()
        _worker_opponent_version = task.opponent_version
    return _worker_opponent_net
