"""Shared shard runner for the self-play studies (``game-stats``, ``card-stats``).

A study plans consecutive-game shards (:class:`~wingspan.research.models.GameStatsShard`)
and hands each to a pure ``study_shard(spec, shard, nets)`` function.
:func:`stream_study` runs every shard — in-process for one worker, otherwise on a
spawn-context process pool whose workers each load their nets once — and writes
the shards' rows to a CSV in shard order, so the in-process path and the pool
produce byte-identical files.
"""

from __future__ import annotations

import csv
import functools
import multiprocessing
import pathlib
import typing
from concurrent import futures

import pydantic
import torch

from wingspan.research import models, progress

# Per-process state of the pool workers, filled once by ``_worker_init``.
_worker_nets: object | None = None

type ShardFn[Nets, Record] = typing.Callable[
    [models.GameStatsSpec, models.GameStatsShard, Nets], list[Record]
]
type NetsLoader[Nets] = typing.Callable[[pathlib.Path, torch.device], Nets]


def stream_study[Nets, Record: pydantic.BaseModel](
    *,
    spec: models.GameStatsSpec,
    shards: typing.Sequence[models.GameStatsShard],
    nets: Nets,
    load_nets: NetsLoader[Nets],
    study_shard: ShardFn[Nets, Record],
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    out_path: pathlib.Path,
    columns: list[str],
    to_row: typing.Callable[[Record], dict[str, object]],
    progress_stream: typing.TextIO | None = None,
) -> int:
    """Run every shard and stream the records' CSV rows to ``out_path`` in shard
    order, returning the number of rows written.

    ``load_nets`` must be a module-level function (the pool workers import it by
    reference to load their own copy of the nets). When ``progress_stream`` is
    given, a throttled progress line is written to it as shards complete."""
    reporter = (
        None
        if progress_stream is None
        else progress.StudyProgress(spec.games, progress_stream)
    )
    rows = 0
    games_done = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        results = _shard_results(
            spec, shards, nets, load_nets, study_shard, checkpoint_dir, device
        )
        for shard, records in zip(shards, results):
            for record in records:
                writer.writerow(to_row(record))
            rows += len(records)
            games_done += shard.num_games
            if reporter is not None:
                reporter.update(games_done)
    if reporter is not None:
        reporter.finish(games_done)
    return rows


###### PRIVATE #######


def _shard_results[Nets, Record](
    spec: models.GameStatsSpec,
    shards: typing.Sequence[models.GameStatsShard],
    nets: Nets,
    load_nets: NetsLoader[Nets],
    study_shard: ShardFn[Nets, Record],
    checkpoint_dir: pathlib.Path,
    device: torch.device,
) -> typing.Iterator[list[Record]]:
    """Yield each shard's records in shard order — played in-process through
    ``nets`` for a single worker, otherwise by a spawn-context process pool whose
    workers each load their own copy of the nets from ``checkpoint_dir`` once."""
    if spec.workers == 1:
        for shard in shards:
            yield study_shard(spec, shard, nets)
        return

    tasks = [models.GameStatsTask(spec=spec, shard=shard) for shard in shards]
    with futures.ProcessPoolExecutor(
        max_workers=spec.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(load_nets, str(checkpoint_dir), str(device)),
    ) as pool:
        yield from pool.map(functools.partial(_worker_study, study_shard), tasks)


def _worker_init(
    load_nets: NetsLoader[object], checkpoint_dir: str, device: str
) -> None:
    """Load this worker's nets once, pinned to a single torch thread so the
    workers parallelize across cores rather than fighting over them."""
    global _worker_nets
    torch.set_num_threads(1)
    _worker_nets = load_nets(pathlib.Path(checkpoint_dir), torch.device(device))


def _worker_study[Nets, Record](
    study_shard: ShardFn[Nets, Record], task: models.GameStatsTask
) -> list[Record]:
    assert _worker_nets is not None, "pool worker used before _worker_init"
    return study_shard(task.spec, task.shard, typing.cast(Nets, _worker_nets))
