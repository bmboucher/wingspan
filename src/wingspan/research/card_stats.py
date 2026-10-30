"""The general card-stats study (``docs/RESEARCH.md``, "General card stats").

Plays N games of greedy self-play — the same nets, engine regime, and per-game
seeds as :mod:`wingspan.research.game_stats`, so with equal ``--seed`` and
``--games`` the games are identical to a game-stats run — and records, per game
and per seat, the dealt and kept opener, the birds and acquired bonus cards held
through each round, and the outcome. Capture is the reusable
:class:`~wingspan.instrumentation.handlers.card_stats.CardStatsCapture` handler.
:mod:`wingspan.research.card_stats_agg` turns the raw table into per-card
"games in hand" win rates.

Net loading, shard planning, and the in-process / process-pool runner are shared
with game-stats (:mod:`wingspan.research.game_stats`,
:mod:`wingspan.research.shard_runner`), so the pool writes a CSV byte-identical
to the in-process path.
"""

from __future__ import annotations

import pathlib
import time
import typing

import torch

from wingspan import engine
from wingspan.instrumentation.handlers import card_stats
from wingspan.research import constants, game_stats, models, shard_runner
from wingspan.training import collect, policy


def run_card_stats_study(
    spec: models.GameStatsSpec,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    out_path: pathlib.Path,
    *,
    progress_stream: typing.TextIO | None = None,
) -> models.GameStatsSummary:
    """Run the whole study against the run in ``checkpoint_dir`` and write the
    raw per-seat table to ``out_path`` (rows stream to disk shard by shard).

    Raises before the output file is created when the run cannot be loaded."""
    started = time.monotonic()
    nets = game_stats.load_game_nets(checkpoint_dir, device)
    rows = shard_runner.stream_study(
        spec=spec,
        shards=game_stats.plan_shards(spec, constants.CARD_STATS_GAMES_PER_SHARD),
        nets=nets,
        load_nets=game_stats.load_game_nets,
        study_shard=study_shard,
        checkpoint_dir=checkpoint_dir,
        device=device,
        out_path=out_path,
        columns=models.card_stats_csv_columns(),
        to_row=models.CardSeatRecord.to_csv_row,
        progress_stream=progress_stream,
    )
    return models.GameStatsSummary(
        out_path=str(out_path),
        games=spec.games,
        rows=rows,
        workers=spec.workers,
        elapsed_seconds=time.monotonic() - started,
    )


def study_shard(
    spec: models.GameStatsSpec,
    shard: models.GameStatsShard,
    nets: game_stats.GameNets,
) -> list[models.CardSeatRecord]:
    """Play every game of ``shard`` and return its per-seat records in game
    order, then seat order."""
    records: list[models.CardSeatRecord] = []
    for game_index in shard.game_indices:
        records.extend(play_game(spec, game_index, nets))
    return records


def play_game(
    spec: models.GameStatsSpec, game_index: int, nets: game_stats.GameNets
) -> list[models.CardSeatRecord]:
    """Play one seeded greedy self-play game and return one record per seat."""
    seed = spec.game_seed(game_index)
    eng = collect.new_engine(seed, nets.num_players)
    agent = policy.greedy_agent(nets.policy_net, nets.device, setup_net=nets.setup_net)
    capture = card_stats.CardStatsCapture()
    engine.Engine.play_one_game(
        eng.state,
        [agent] * nets.num_players,
        instrumentation=capture.instrumentation(),
        verbose_turn_log=False,
        split_setup_bonus=nets.split_setup_bonus,
        split_setup_food=nets.split_setup_food,
        combine_gain_food=nets.combine_gain_food,
    )
    return [
        models.CardSeatRecord(
            game_index=game_index,
            seed=seed,
            seat=seat.seat,
            won=seat.won,
            tie=capture.tie,
            final_score=seat.final_score,
            dealt_birds=seat.dealt_birds,
            kept_birds=seat.kept_birds,
            dealt_bonus=seat.dealt_bonus,
            original_bonus=seat.original_bonus,
            played_birds=sorted(seat.played_birds),
            birds_by_round=[sorted(names) for names in seat.round_birds],
            acquired_bonus_by_round=[
                sorted(names) for names in seat.round_acquired_bonus
            ],
        )
        for seat in capture.seats
    ]
