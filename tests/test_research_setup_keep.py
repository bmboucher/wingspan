"""The setup keep-rate experience study (``wingspan.research.setup_keep``).

Black-box over the study's public seams: the exposure rows a shard produces
(five per setup, ``kept`` consistent with the chosen keep, ``keep_prob`` the
true marginal of the candidate distribution), determinism and shard-plan
invariance, the CSV header/row contract, the bird-metadata projection, and the
end-to-end run-directory path through both the in-process runner and the
``wingspan research setup-keep`` CLI (including the process-pool ``--workers``
path, which must match the in-process CSV byte for byte).
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import cards, setup_model, version  # noqa: E402
from wingspan.research import app, constants, models, setup_keep  # noqa: E402
from wingspan.training import setup_runmeta  # noqa: E402
from wingspan.training import (  # noqa: E402
    artifacts,
    config,
    loop_checkpoint,
    runmeta,
)
from wingspan.training import setup_net as setup_net_module  # noqa: E402

_DEVICE = torch.device("cpu")
# Exposures per setup: one per dealt bird.
_HAND_SIZE = 5


def _tiny_net(
    *, split_bonus: bool = True, split_food: bool = True
) -> setup_net_module.SetupNet:
    """An untrained live-era setup net at the requested split regime."""
    net = setup_net_module.SetupNet(
        encoding=setup_model.SetupEncoding(
            split_bonus=split_bonus, split_food=split_food
        ),
        arch=setup_model.SetupArchitecture(head_layers=(16,)),
    )
    net.eval()
    return net


def _spec(
    *,
    setups: int = 4,
    seed: int = 7,
    num_players: int = 2,
    temperature: float | None = None,
    workers: int = 1,
) -> models.SetupKeepStudySpec:
    return models.SetupKeepStudySpec(
        setups=setups,
        seed=seed,
        num_players=num_players,
        temperature=temperature,
        workers=workers,
    )


def _records(
    spec: models.SetupKeepStudySpec, net: setup_net_module.SetupNet
) -> list[models.SetupKeepRecord]:
    """Every exposure row of the study, across all its shards."""
    rows: list[models.SetupKeepRecord] = []
    for shard in setup_keep.plan_shards(spec):
        rows.extend(setup_keep.study_shard(spec, shard, net, _DEVICE))
    return rows


def _by_setup(
    records: list[models.SetupKeepRecord],
) -> dict[int, list[models.SetupKeepRecord]]:
    groups: dict[int, list[models.SetupKeepRecord]] = {}
    for record in records:
        groups.setdefault(record.key.setup_id, []).append(record)
    return groups


def _write_run_dir(checkpoint_dir: pathlib.Path, *, with_setup: bool) -> None:
    """A minimal run directory the setup-net loader accepts: the unified
    ``run_config_<stamp>.json`` plus (optionally) a ``setup.pt`` holding a
    fresh tiny net's weights."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(
            run_name="research-test", checkpoint_dir=str(checkpoint_dir)
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(8, 8),
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
            setup=config.SetupNetArchitecture(head_layers=(8,)),
        ),
    )
    runmeta.write_run_config(
        str(checkpoint_dir),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
    if with_setup:
        descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
        payload: dict[str, object] = {
            "setup_model": setup_net_module.SetupNet.from_setup_config(
                descriptor
            ).state_dict(),
            "version": version.MODEL_VERSION,
        }
        loop_checkpoint.atomic_save(payload, checkpoint_dir / artifacts.SETUP_CKPT)


def _read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), rows


###### Exposure rows ######


def test_five_exposures_per_setup_with_kept_matching_the_keep():
    """Every setup yields one row per dealt card (slots 0..4); the ``kept``
    flags sum to the shared ``n_kept``; the critic value is shared across the
    seat; kept cards carry positive keep probability."""
    spec = _spec(setups=4)
    records = _records(spec, _tiny_net())
    assert len(records) == spec.setups * _HAND_SIZE

    groups = _by_setup(records)
    assert sorted(groups) == list(range(spec.setups))
    for rows in groups.values():
        assert [row.key.card_slot for row in rows] == list(range(_HAND_SIZE))
        assert len({row.observation.n_kept for row in rows}) == 1
        assert len({row.observation.deal_value for row in rows}) == 1
        assert sum(row.observation.kept for row in rows) == rows[0].observation.n_kept
        assert len({row.bird.bird_id for row in rows}) == _HAND_SIZE
        for row in rows:
            assert 0.0 <= row.observation.keep_prob <= 1.0
            assert row.key.num_players == spec.num_players
            if row.observation.kept:
                assert row.observation.keep_prob > 0.0


def test_keep_prob_is_the_marginal_of_the_candidate_distribution():
    """Per seat, the five ``keep_prob`` values sum to the expected keep count
    under the policy's softmax over its candidate keeps — recomputed here from
    the public deal + encode + score seams."""
    net = _tiny_net()
    spec = _spec(setups=2)
    shard = setup_keep.plan_shards(spec)[0]
    records = _by_setup(setup_keep.study_shard(spec, shard, net, _DEVICE))
    for seat_deal in setup_keep.deal_shard(spec, shard, net.encoding):
        feats = torch.tensor(
            np.stack(
                [
                    net.encode_candidate(candidate, seat_deal.context)
                    for candidate in seat_deal.candidates
                ]
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = net.policy_logits(feats).numpy().astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        expected_kept = sum(
            prob * len(candidate.kept_cards)
            for prob, candidate in zip(probs, seat_deal.candidates)
        )
        observed = sum(row.observation.keep_prob for row in records[seat_deal.setup_id])
        assert observed == pytest.approx(expected_kept)


def test_bonus_kept_blank_when_split_and_named_when_folded():
    """Under the split-bonus regime the setup model never picks a bonus, so
    ``bonus_kept`` is blank; a folded-bonus net keeps one of the two on offer."""
    spec = _spec(setups=2)
    split_rows = _records(spec, _tiny_net(split_bonus=True))
    assert all(row.observation.bonus_kept == "" for row in split_rows)

    folded_rows = _records(spec, _tiny_net(split_bonus=False, split_food=False))
    assert len(folded_rows) == spec.setups * _HAND_SIZE
    for row in folded_rows:
        assert row.observation.bonus_kept in {row.deal.bonus_1, row.deal.bonus_2}
        assert row.observation.bonus_kept != ""


def test_deal_context_records_the_shared_table():
    """Both seats of a game see the same tray / feeder / goals, the hand cell
    lists the seat's five dealt birds, and the feeder holds all five dice."""
    spec = _spec(setups=2)
    groups = _by_setup(_records(spec, _tiny_net()))
    seat0, seat1 = groups[0][0].deal, groups[1][0].deal
    assert (seat0.tray_1, seat0.tray_2, seat0.tray_3) == (
        seat1.tray_1,
        seat1.tray_2,
        seat1.tray_3,
    )
    assert (seat0.goal_1, seat0.goal_2, seat0.goal_3, seat0.goal_4) == (
        seat1.goal_1,
        seat1.goal_2,
        seat1.goal_3,
        seat1.goal_4,
    )
    assert len({seat0.goal_1, seat0.goal_2, seat0.goal_3, seat0.goal_4}) == 4
    dice = (
        seat0.feeder_invertebrate
        + seat0.feeder_seed
        + seat0.feeder_fish
        + seat0.feeder_fruit
        + seat0.feeder_rodent
        + seat0.feeder_choice
    )
    assert dice == 5
    hand_names = seat0.hand.split(constants.NAME_SEPARATOR)
    assert hand_names == [row.bird.bird_name for row in groups[0]]
    assert seat0.bonus_1 and seat0.bonus_2 and seat0.bonus_1 != seat0.bonus_2


###### Determinism, sharding, surplus seats ######


def test_rows_are_deterministic_and_shard_plan_invariant(
    monkeypatch: pytest.MonkeyPatch,
):
    """The same spec reproduces the same rows, and splitting the games into
    single-game shards changes nothing — the pool path relies on this."""
    net = _tiny_net()
    spec = _spec(setups=5, temperature=0.7)
    first = _records(spec, net)
    assert first == _records(spec, net)

    monkeypatch.setattr(constants, "GAMES_PER_SHARD", 1)
    assert len(setup_keep.plan_shards(spec)) == spec.games
    assert _records(spec, net) == first


def test_plan_shards_covers_every_game_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(constants, "GAMES_PER_SHARD", 3)
    spec = _spec(setups=14, num_players=2)  # 7 games
    shards = setup_keep.plan_shards(spec)
    assert [(shard.first_game_index, shard.num_games) for shard in shards] == [
        (0, 3),
        (3, 3),
        (6, 1),
    ]
    covered = [index for shard in shards for index in shard.game_indices]
    assert covered == list(range(spec.games))


def test_surplus_seats_of_the_last_game_are_not_recorded():
    """An odd setup count at a 2-seat table deals a final game whose second
    seat falls outside the study: dealt, but not written."""
    spec = _spec(setups=3, num_players=2)
    assert spec.games == 2
    records = _records(spec, _tiny_net())
    assert sorted(_by_setup(records)) == [0, 1, 2]
    assert len(records) == 3 * _HAND_SIZE
    assert {row.key.game_seed for row in records} == {spec.seed, spec.seed + 1}


###### CSV contract and bird metadata ######


def test_csv_columns_are_unique_and_match_the_row():
    columns = models.csv_columns()
    assert len(columns) == len(set(columns))
    record = _records(_spec(setups=1), _tiny_net())[0]
    assert list(record.to_csv_row()) == columns


def test_bird_metadata_projects_cost_habitats_and_power():
    birds, _, _ = cards.load_all()
    or_cost_bird = next(bird for bird in birds if bird.food_cost.is_or_cost)
    metadata = models.BirdMetadata.from_bird(or_cost_bird)
    assert metadata.cost_is_or == 1
    assert metadata.cost_total == or_cost_bird.food_cost.total
    assert metadata.points == or_cost_bird.points

    for bird in birds:
        metadata = models.BirdMetadata.from_bird(bird)
        flags = (
            metadata.habitat_forest
            + metadata.habitat_grassland
            + metadata.habitat_wetland
        )
        assert flags == metadata.n_habitats == len(bird.habitats)
        specific = (
            metadata.cost_invertebrate
            + metadata.cost_seed
            + metadata.cost_fish
            + metadata.cost_fruit
            + metadata.cost_rodent
        )
        assert specific + metadata.cost_wild == metadata.cost_total
        assert metadata.power_color == bird.color
        kinds = metadata.effect_kinds.split(constants.NAME_SEPARATOR)
        assert len([kind for kind in kinds if kind]) == len(bird.power.effects)


###### Run directory + CLI ######


def test_run_study_writes_csv_from_a_run_dir(tmp_path: pathlib.Path):
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, with_setup=True)
    out_path = tmp_path / "keep.csv"
    spec = _spec(setups=3)
    summary = setup_keep.run_setup_keep_study(spec, run_dir, _DEVICE, out_path)
    assert summary.rows == 3 * _HAND_SIZE
    assert summary.games == 2
    assert summary.setups == 3
    header, rows = _read_csv(out_path)
    assert header == models.csv_columns()
    assert len(rows) == summary.rows
    assert {row["kept"] for row in rows} <= {"0", "1"}


def test_run_study_refuses_a_run_without_a_setup_model(tmp_path: pathlib.Path):
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, with_setup=False)
    out_path = tmp_path / "keep.csv"
    with pytest.raises(FileNotFoundError, match="no setup model"):
        setup_keep.run_setup_keep_study(_spec(setups=1), run_dir, _DEVICE, out_path)
    assert not out_path.exists()


def test_cli_setup_keep_end_to_end(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    """The CLI resolves the seat count from the run config, writes the CSV,
    and prints the row count."""
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, with_setup=True)
    out_path = tmp_path / "keep.csv"
    code = app.main(
        [
            "setup-keep",
            "--setups",
            "2",
            "--checkpoint-dir",
            str(run_dir),
            "--out",
            str(out_path),
            "--seed",
            "3",
        ]
    )
    assert code == 0
    _, rows = _read_csv(out_path)
    assert len(rows) == 2 * _HAND_SIZE
    assert {row["num_players"] for row in rows} == {"2"}
    assert {row["game_seed"] for row in rows} == {"3"}
    assert "wrote 10 rows (2 setups)" in capsys.readouterr().out


def test_cli_reports_a_missing_setup_model(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, with_setup=False)
    out_path = tmp_path / "keep.csv"
    code = app.main(
        [
            "setup-keep",
            "--setups",
            "1",
            "--checkpoint-dir",
            str(run_dir),
            "--out",
            str(out_path),
        ]
    )
    assert code == 1
    assert "no setup model" in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_rejects_an_unknown_study():
    with pytest.raises(SystemExit):
        app.main(["no-such-study"])


def test_worker_pool_matches_the_in_process_csv(tmp_path: pathlib.Path):
    """``--workers 2`` streams shard results in plan order, so the pooled CSV
    is byte-identical to the single-process one."""
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, with_setup=True)
    serial_path = tmp_path / "serial.csv"
    pooled_path = tmp_path / "pooled.csv"
    setup_keep.run_setup_keep_study(_spec(setups=4), run_dir, _DEVICE, serial_path)
    summary = setup_keep.run_setup_keep_study(
        _spec(setups=4, workers=2), run_dir, _DEVICE, pooled_path
    )
    assert summary.workers == 2
    assert pooled_path.read_bytes() == serial_path.read_bytes()
