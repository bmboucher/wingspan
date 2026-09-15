"""The setup keep-rate experience study (``docs/RESEARCH.md``, "Setup card
stats", Q1).

Samples random deals exactly as the engine deals them — a fresh seeded game at
the study's seat count, then each seat's five birds, two bonus cards, and one
of each food through ``Engine.deal_setup_inputs`` — enumerates every keep the
setup model would be offered (``setup_model.enumerate_setup_candidates`` at the
net's own split regime), scores all of them in one batched forward pass, and
records one *exposure* row per dealt bird with a 0/1 ``kept`` observation flag.
The table is an actuarial experience study: a card's keep rate is
``sum(kept) / count(rows)`` over its exposures, and every row carries the rest
of the setup (tray, birdfeeder, goals, bonus offer) plus the card's printed
attributes so the rate can be pivoted by any of them.

Work is planned as shards of consecutive games (``constants.GAMES_PER_SHARD``).
:func:`study_shard` is a pure function of the spec, the shard, and the net, so
the in-process path and the process pool produce byte-identical CSVs, and the
pool streams shard results to disk in shard order.
"""

from __future__ import annotations

import csv
import multiprocessing
import pathlib
import random
import time
import typing
from concurrent import futures

import numpy as np
import torch

from wingspan import cards, engine, setup_model, state
from wingspan.players import loaders
from wingspan.research import constants, models
from wingspan.training import setup_net as setup_net_module

# Width of the per-seat sampling seed drawn from the game RNG after the deal.
_SAMPLE_SEED_BITS = 32

# Per-process state of the pool workers, filled once by ``_worker_init``.
_worker_net: setup_net_module.SetupNet | None = None
_worker_device: torch.device | None = None

# Bird metadata is identical for every exposure of a card; built once per id.
_bird_metadata_cache: dict[int, models.BirdMetadata] = {}


def run_setup_keep_study(
    spec: models.SetupKeepStudySpec,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    out_path: pathlib.Path,
) -> models.SetupKeepSummary:
    """Run the whole study against the run in ``checkpoint_dir`` and write the
    exposure table to ``out_path`` (rows stream to disk shard by shard).

    Raises ``FileNotFoundError`` when the run has no setup model and
    ``ValueError`` when its setup net cannot rank keeps — both before the
    output file is created."""
    started = time.monotonic()
    net = load_setup_net(checkpoint_dir, device)
    shards = plan_shards(spec)
    rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=models.csv_columns())
        writer.writeheader()
        for records in _shard_results(spec, shards, net, checkpoint_dir, device):
            for record in records:
                writer.writerow(record.to_csv_row())
            rows += len(records)
    return models.SetupKeepSummary(
        out_path=str(out_path),
        setups=spec.setups,
        games=spec.games,
        rows=rows,
        workers=spec.workers,
        elapsed_seconds=time.monotonic() - started,
    )


def plan_shards(spec: models.SetupKeepStudySpec) -> list[models.SetupKeepShard]:
    """Split the study's games into consecutive shards of at most
    ``constants.GAMES_PER_SHARD`` games."""
    return [
        models.SetupKeepShard(
            first_game_index=first,
            num_games=min(constants.GAMES_PER_SHARD, spec.games - first),
        )
        for first in range(0, spec.games, constants.GAMES_PER_SHARD)
    ]


def load_setup_net(
    checkpoint_dir: pathlib.Path, device: torch.device
) -> setup_net_module.SetupNet:
    """The run's setup net, in eval mode on ``device``.

    Unlike ``players.loaders.load_setup_net`` this never degrades to ``None``:
    a run without a setup model raises ``FileNotFoundError``, and a value-only
    net (no policy head, so no ranking) raises ``ValueError``."""
    net = loaders.load_setup_net(checkpoint_dir, device)
    if net is None:
        raise FileNotFoundError(
            f"no setup model in {checkpoint_dir}: the keep-rate study needs a run "
            "trained with use_setup_model=True (setup.pt + run_config_*.json)"
        )
    if not net.arch.use_policy_head:
        raise ValueError(
            f"the setup net in {checkpoint_dir} has no policy head "
            "(use_policy_head=False), so it cannot rank keeps"
        )
    return net


def study_shard(
    spec: models.SetupKeepStudySpec,
    shard: models.SetupKeepShard,
    net: setup_net_module.SetupNet,
    device: torch.device,
) -> list[models.SetupKeepRecord]:
    """Deal every game of ``shard``, score every recorded seat's candidate keeps
    in one forward pass, and return the seats' exposure rows in deal order."""
    deals = deal_shard(spec, shard, net.encoding)
    if not deals:
        return []
    features, blocks = _encode_deals(deals, net)
    logits, values = _score(net, device, features)
    records: list[models.SetupKeepRecord] = []
    for seat_deal, (start, stop) in zip(deals, blocks):
        records.extend(
            _seat_records(spec, seat_deal, logits[start:stop], float(values[start]))
        )
    return records


def deal_shard(
    spec: models.SetupKeepStudySpec,
    shard: models.SetupKeepShard,
    encoding: setup_model.SetupEncoding,
) -> list[models.SeatDeal]:
    """Deal the shard's games and enumerate each recorded seat's candidate keeps
    at the ``encoding``'s split regime."""
    birds, bonuses, goals = cards.load_all()
    deals: list[models.SeatDeal] = []
    for game_index in shard.game_indices:
        deals.extend(_deal_game(spec, game_index, encoding, birds, bonuses, goals))
    return deals


###### PRIVATE #######

#### Sampling ####


def _deal_game(
    spec: models.SetupKeepStudySpec,
    game_index: int,
    encoding: setup_model.SetupEncoding,
    birds: list[cards.Bird],
    bonuses: list[cards.BonusCard],
    goals: list[cards.EndRoundGoal],
) -> list[models.SeatDeal]:
    """Deal one seeded game's setup inputs for every seat, through the engine's
    own deal path, keeping the seats whose ``setup_id`` falls inside the study."""
    game_seed = spec.game_seed(game_index)
    rng = random.Random(game_seed)
    game_state = state.new_game(
        rng, birds, bonuses, goals, num_players=spec.num_players
    )
    dealer = engine.Engine(game_state)

    deals: list[models.SeatDeal] = []
    for player in game_state.players:
        # Every seat is dealt (so the deck is consumed exactly as in a real
        # game) even when the last game's surplus seats are not recorded.
        dealt_cards, dealt_bonus = dealer.deal_setup_inputs(player)
        setup_id = spec.setup_id(game_index, player.id)
        if setup_id >= spec.setups:
            continue
        deals.append(
            models.SeatDeal(
                setup_id=setup_id,
                game_seed=game_seed,
                seat=player.id,
                sample_seed=rng.getrandbits(_SAMPLE_SEED_BITS),
                dealt_cards=tuple(dealt_cards),
                dealt_bonus=tuple(dealt_bonus),
                context=setup_model.SetupContext.from_state(game_state, dealt_bonus),
                deal=models.DealContext.from_state(
                    game_state, dealt_cards, dealt_bonus
                ),
                candidates=tuple(
                    setup_model.enumerate_setup_candidates(
                        dealt_cards,
                        dealt_bonus,
                        include_bonus=not encoding.split_bonus,
                        include_food=not encoding.split_food,
                    )
                ),
            )
        )
    return deals


#### Scoring ####


def _encode_deals(
    deals: typing.Sequence[models.SeatDeal], net: setup_net_module.SetupNet
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Encode every seat's candidates through the net's own ``encode_candidate``
    seam into one ``(total_candidates, feature_dim)`` matrix, returning it with
    each seat's ``(start, stop)`` row block."""
    matrices: list[np.ndarray] = []
    blocks: list[tuple[int, int]] = []
    start = 0
    for seat_deal in deals:
        matrix = np.stack(
            [
                net.encode_candidate(candidate, seat_deal.context)
                for candidate in seat_deal.candidates
            ]
        )
        matrices.append(matrix)
        blocks.append((start, start + len(matrix)))
        start += len(matrix)
    return np.concatenate(matrices), blocks


def _score(
    net: setup_net_module.SetupNet, device: torch.device, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass over the whole shard: per-candidate policy logits and
    the per-row critic value (constant within a seat's block)."""
    with torch.no_grad():
        feats = torch.tensor(features, dtype=torch.float32, device=device)
        logits, values = net.policy_and_value(feats)
    return logits.cpu().numpy(), values.cpu().numpy()


def _seat_records(
    spec: models.SetupKeepStudySpec,
    seat_deal: models.SeatDeal,
    logits: np.ndarray,
    deal_value: float,
) -> list[models.SetupKeepRecord]:
    """The seat's five exposure rows: pick the keep (argmax, or a softmax sample
    at the study temperature), then flag each dealt card as kept or not."""
    # The policy distribution at the study temperature; the pick follows it.
    temperature = spec.temperature if spec.temperature is not None else 1.0
    probs = _softmax(logits / temperature)
    sample_rng = (
        random.Random(seat_deal.sample_seed) if spec.temperature is not None else None
    )
    chosen_idx = setup_model.select_by_margins(logits, temperature, sample_rng)
    chosen = seat_deal.candidates[chosen_idx]

    # Marginal keep probability per dealt card: the summed probability of every
    # candidate that contains it.
    contains = np.array(
        [
            [_contains(candidate, bird) for bird in seat_deal.dealt_cards]
            for candidate in seat_deal.candidates
        ],
        dtype=np.float64,
    )
    keep_probs = probs @ contains

    kept_ids = {bird.id for bird in chosen.kept_cards}
    bonus_kept = chosen.bonus_card.name if chosen.bonus_card is not None else ""
    records: list[models.SetupKeepRecord] = []
    for card_slot, bird in enumerate(seat_deal.dealt_cards):
        records.append(
            models.SetupKeepRecord(
                key=models.ExposureKey(
                    setup_id=seat_deal.setup_id,
                    game_seed=seat_deal.game_seed,
                    seat=seat_deal.seat,
                    num_players=spec.num_players,
                    card_slot=card_slot,
                ),
                observation=models.KeepObservation(
                    kept=int(bird.id in kept_ids),
                    keep_prob=float(keep_probs[card_slot]),
                    n_kept=len(chosen.kept_cards),
                    deal_value=deal_value,
                    bonus_kept=bonus_kept,
                ),
                bird=_bird_metadata(bird),
                deal=seat_deal.deal,
            )
        )
    return records


def _contains(candidate: setup_model.SetupCandidate, bird: cards.Bird) -> bool:
    return any(kept.id == bird.id for kept in candidate.kept_cards)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores.astype(np.float64) - scores.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def _bird_metadata(bird: cards.Bird) -> models.BirdMetadata:
    if bird.id not in _bird_metadata_cache:
        _bird_metadata_cache[bird.id] = models.BirdMetadata.from_bird(bird)
    return _bird_metadata_cache[bird.id]


#### Work distribution ####


def _shard_results(
    spec: models.SetupKeepStudySpec,
    shards: typing.Sequence[models.SetupKeepShard],
    net: setup_net_module.SetupNet,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
) -> typing.Iterator[list[models.SetupKeepRecord]]:
    """Yield each shard's records in shard order — scored in-process through
    ``net`` for a single worker, otherwise by a spawn-context process pool whose
    workers each load their own copy of the net from ``checkpoint_dir`` once."""
    if spec.workers == 1:
        for shard in shards:
            yield study_shard(spec, shard, net, device)
        return

    tasks = [models.ShardTask(spec=spec, shard=shard) for shard in shards]
    with futures.ProcessPoolExecutor(
        max_workers=spec.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(str(checkpoint_dir), str(device)),
    ) as pool:
        yield from pool.map(_worker_study, tasks)


def _worker_init(checkpoint_dir: str, device: str) -> None:
    """Load this worker's setup net once, pinned to a single torch thread so
    the workers parallelize across cores rather than fighting over them."""
    global _worker_net, _worker_device
    torch.set_num_threads(1)
    _worker_device = torch.device(device)
    _worker_net = load_setup_net(pathlib.Path(checkpoint_dir), _worker_device)


def _worker_study(task: models.ShardTask) -> list[models.SetupKeepRecord]:
    assert (
        _worker_net is not None and _worker_device is not None
    ), "pool worker used before _worker_init"
    return study_shard(task.spec, task.shard, _worker_net, _worker_device)
