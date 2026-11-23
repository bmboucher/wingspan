"""The forced-card bias study (``wingspan.research.forced_card``).

Black-box over the study's seams: the pure reweighting rule, the decision-matching
helpers on hand-built decisions against a real engine state, the sampling agent's
bias with the net's probabilities stubbed, the per-game seat alternation and seed
pairing on a real tiny net, the summary aggregation on a hand-built fixture, and
the ``wingspan research forced-card`` CLI.
"""

from __future__ import annotations

import csv
import pathlib
import random

import numpy as np
import pydantic
import pytest

torch = pytest.importorskip("torch")

from wingspan import cards, decisions, engine, model, version  # noqa: E402
from wingspan.research import (  # noqa: E402
    app,
    constants,
    forced_card,
    forced_card_agg,
    game_stats,
    models,
)
from wingspan.training import setup_runmeta  # noqa: E402
from wingspan.training import (  # noqa: E402
    artifacts,
    collect,
    config,
    loop_checkpoint,
    policy,
    runmeta,
    runstate,
)
from wingspan.training import setup_net as setup_net_module  # noqa: E402

_DEVICE = torch.device("cpu")
_SEAT = 0


def _write_run_dir(checkpoint_dir: pathlib.Path) -> None:
    """A minimal run directory the study loads: the unified run config, a
    ``last.pt`` with a fresh tiny policy net, and ``setup.pt``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(
            run_name="forced-card-test", checkpoint_dir=str(checkpoint_dir)
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
    net = model.PolicyValueNet(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )
    main_payload: dict[str, object] = {
        "config": cfg.model_dump(),
        "model": net.state_dict(),
        "progress": runstate.RunProgress(iteration=1, total_games=2).model_dump(),
        "git_sha": None,
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(main_payload, checkpoint_dir / artifacts.LAST_CKPT)
    descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
    setup_payload: dict[str, object] = {
        "setup_model": setup_net_module.SetupNet.from_setup_config(
            descriptor
        ).state_dict(),
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(setup_payload, checkpoint_dir / artifacts.SETUP_CKPT)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("forced_card_run")
    _write_run_dir(directory)
    return directory


###### bias_probs ######


def test_bias_x_zero_is_identity():
    probs = np.array([0.2, 0.3, 0.5])
    biased = forced_card.bias_probs(probs, np.array([True, False, False]), 0.0)
    assert biased == pytest.approx(probs)


def test_bias_intermediate_value_is_exact():
    biased = forced_card.bias_probs(np.array([0.5, 0.5]), np.array([True, False]), 0.5)
    assert biased == pytest.approx([0.75, 0.25])


def test_bias_x_one_zeroes_non_conformant_and_keeps_conformant_ratios():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    mask = np.array([True, False, True, False])
    biased = forced_card.bias_probs(probs, mask, 1.0)
    assert biased[~mask] == pytest.approx([0.0, 0.0])
    assert biased[0] / biased[2] == pytest.approx(0.1 / 0.3)
    assert biased.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_bias_sums_to_one(factor: float):
    probs = np.array([0.15, 0.25, 0.6])
    biased = forced_card.bias_probs(probs, np.array([False, True, True]), factor)
    assert biased.sum() == pytest.approx(1.0)


def test_bias_underflow_falls_back_to_uniform_over_conformant():
    probs = np.array([1.0, 0.0, 0.0])
    mask = np.array([False, True, True])
    biased = forced_card.bias_probs(probs, mask, 1.0)
    assert biased == pytest.approx([0.0, 0.5, 0.5])


def test_bias_without_conformant_option_is_unchanged():
    probs = np.array([0.4, 0.6])
    biased = forced_card.bias_probs(probs, np.array([False, False]), 0.7)
    assert biased == pytest.approx(probs)


def test_bias_x_minus_one_zeroes_conformant_and_keeps_non_conformant_ratios():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    mask = np.array([True, False, True, False])
    biased = forced_card.bias_probs(probs, mask, -1.0)
    assert biased[mask] == pytest.approx([0.0, 0.0])
    assert biased[1] / biased[3] == pytest.approx(0.2 / 0.4)
    assert biased.sum() == pytest.approx(1.0)


def test_bias_negative_intermediate_value_is_exact():
    biased = forced_card.bias_probs(np.array([0.5, 0.5]), np.array([True, False]), -0.5)
    assert biased == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize("factor", [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 1.0])
def test_bias_sums_to_one_across_the_full_range(factor: float):
    probs = np.array([0.15, 0.25, 0.6])
    biased = forced_card.bias_probs(probs, np.array([False, True, True]), factor)
    assert biased.sum() == pytest.approx(1.0)


def test_bias_negative_underflow_falls_back_to_uniform_over_non_conformant():
    probs = np.array([0.0, 0.5, 0.5])
    mask = np.array([False, True, True])
    biased = forced_card.bias_probs(probs, mask, -1.0)
    assert biased == pytest.approx([1.0, 0.0, 0.0])
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    mask = np.array([True, False, False, False])
    biased = forced_card.bias_probs(probs, mask, -1.0)
    assert biased == pytest.approx([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_bias_all_conformant_at_minus_one_returns_input_unchanged():
    probs = np.array([0.3, 0.7])
    biased = forced_card.bias_probs(probs, np.array([True, True]), -1.0)
    assert biased == pytest.approx(probs)


@pytest.mark.parametrize("factor", [-1.1, 1.1])
def test_bias_rejects_factor_outside_the_range(factor: float):
    with pytest.raises(ValueError):
        forced_card.bias_probs(np.array([1.0]), np.array([True]), factor)


def test_forced_card_spec_validates_the_factor():
    with pytest.raises(pydantic.ValidationError):
        models.ForcedCardSpec(force_card_factor=1.5, forced_card_id=1)
    with pytest.raises(pydantic.ValidationError):
        models.ForcedCardSpec(force_card_factor=-1.5, forced_card_id=1)
    spec = models.ForcedCardSpec(force_card_factor=-1.0, forced_card_id=1)
    assert spec.force_card_factor == -1.0


###### Forced card resolution ######


def test_default_forced_card_is_cedar_waxwing_by_name():
    forced_id = forced_card.resolve_forced_card_id(None)
    assert forced_card.bird_name(forced_id) == constants.DEFAULT_FORCED_CARD_NAME


def test_explicit_id_is_validated_against_the_catalog():
    first = cards.birds_ordered()[0]
    assert forced_card.resolve_forced_card_id(first.id) == first.id
    unused = max(bird.id for bird in cards.birds_ordered()) + 1
    with pytest.raises(ValueError, match="not a bird in the card database"):
        forced_card.resolve_forced_card_id(unused)


###### Decision matching ######


def _cheap_birds() -> list[cards.Bird]:
    """Distinct birds costing exactly one specific food token."""
    return [
        bird
        for bird in cards.birds_ordered()
        if bird.food_cost.wild == 0 and bird.food_cost.total == 1 and bird.habitats
    ]


def _engine_holding(birds: list[cards.Bird], *, fed: bool) -> engine.Engine:
    """A fresh engine whose seat 0 holds ``birds`` and, when ``fed``, one token
    of every food so every cheap bird is affordable."""
    eng = collect.new_engine(1, 2)
    player = eng.state.players[_SEAT]
    player.hand = list(birds)
    for food in cards.ALL_FOODS:
        player.food[food] = 1 if fed else 0
    return eng


def _main_action_decision(*, with_play: bool) -> decisions.MainActionDecision:
    actions = [
        decisions.MainAction.GAIN_FOOD,
        decisions.MainAction.LAY_EGGS,
        decisions.MainAction.DRAW_CARDS,
    ]
    if with_play:
        actions.append(decisions.MainAction.PLAY_BIRD)
    return decisions.MainActionDecision(
        player_id=_SEAT,
        prompt="main action",
        choices=[
            decisions.MainActionChoice(label=action.value, action=action)
            for action in actions
        ],
    )


def _play_menu(
    plays: list[tuple[cards.Bird, cards.Habitat]],
) -> decisions.PlayBirdDecision:
    return decisions.PlayBirdDecision(
        player_id=_SEAT,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(label=bird.name, bird=bird, habitat=habitat)
            for bird, habitat in plays
        ],
    )


def test_main_action_matches_only_when_the_forced_bird_is_playable():
    forced, other = _cheap_birds()[:2]
    decision = _main_action_decision(with_play=True)

    playable = _engine_holding([forced, other], fed=True)
    mask = forced_card.conformant_mask(playable, decision, forced.id, 0.5)
    assert mask is not None
    assert mask.tolist() == [False, False, False, True]

    unaffordable = _engine_holding([forced, other], fed=False)
    assert forced_card.conformant_mask(unaffordable, decision, forced.id, 0.5) is None

    not_held = _engine_holding([other], fed=True)
    assert forced_card.conformant_mask(not_held, decision, forced.id, 0.5) is None


def test_main_action_negative_factor_matches_only_a_sole_playable_forced_bird():
    forced, other = _cheap_birds()[:2]
    decision = _main_action_decision(with_play=True)

    sole = _engine_holding([forced], fed=True)
    for factor in (-1.0, -0.3):
        mask = forced_card.conformant_mask(sole, decision, forced.id, factor)
        assert mask is not None
        assert mask.tolist() == [False, False, False, True]

    both = _engine_holding([forced, other], fed=True)
    assert forced_card.conformant_mask(both, decision, forced.id, -0.3) is None
    # Non-negative factors match in both situations.
    for factor in (0.0, 0.3):
        for eng in (sole, both):
            matched = forced_card.conformant_mask(eng, decision, forced.id, factor)
            assert matched is not None

    not_held = _engine_holding([other], fed=True)
    assert forced_card.conformant_mask(not_held, decision, forced.id, -0.3) is None


def test_main_action_without_a_play_option_is_untouched():
    forced = _cheap_birds()[0]
    eng = _engine_holding([forced], fed=True)
    decision = _main_action_decision(with_play=False)
    assert forced_card.conformant_mask(eng, decision, forced.id, 0.5) is None


def test_play_bird_matches_every_habitat_of_the_forced_bird():
    forced = next(bird for bird in _cheap_birds() if len(bird.habitats) >= 2)
    other = next(bird for bird in _cheap_birds() if bird.id != forced.id)
    eng = _engine_holding([], fed=True)
    plays = [(other, other.habitats[0])] + [
        (forced, habitat) for habitat in forced.habitats
    ]
    for factor in (0.5, -0.5):
        mask = forced_card.conformant_mask(eng, _play_menu(plays), forced.id, factor)
        assert mask is not None
        assert mask.tolist() == [False] + [True] * len(forced.habitats)


def test_play_bird_without_the_forced_bird_is_untouched():
    forced, other = _cheap_birds()[:2]
    eng = _engine_holding([], fed=True)
    menu = _play_menu([(other, other.habitats[0])])
    assert forced_card.conformant_mask(eng, menu, forced.id, 0.5) is None


def test_other_decisions_are_untouched():
    forced = _cheap_birds()[0]
    eng = _engine_holding([forced], fed=True)
    skip = decisions.RemoveEggDecision(
        player_id=_SEAT,
        prompt="skip",
        choices=[decisions.SkipChoice(label="skip")],
    )
    assert forced_card.conformant_mask(eng, skip, forced.id, 0.5) is None


###### Sampling agent ######


def _uniform_probs(
    _net: model.PolicyValueNet,
    _device: object,
    _state_vec: np.ndarray,
    choice_feats: np.ndarray,
    _family_idx: int,
) -> np.ndarray:
    count = choice_feats.shape[0]
    return np.array([1.0 / count] * count, dtype=np.float64)


def _draw_actions(
    run_dir: pathlib.Path, factor: float, monkeypatch: pytest.MonkeyPatch
) -> list[decisions.MainAction]:
    monkeypatch.setattr(policy, "policy_probs", _uniform_probs)
    nets = game_stats.load_game_nets(run_dir, _DEVICE)
    forced_bird = _cheap_birds()[0]
    eng = _engine_holding([forced_bird], fed=True)
    spec = models.ForcedCardSpec(
        force_card_factor=factor, forced_card_id=forced_bird.id
    )
    agent = forced_card.sampling_agent(
        nets.policy_net,
        _DEVICE,
        random.Random(0),
        setup_net=nets.setup_net,
        forced=spec,
    )
    decision = _main_action_decision(with_play=True)
    return [agent(eng, decision).action for _ in range(30)]


def test_agent_at_factor_one_always_plays_the_forced_bird(
    run_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    drawn = _draw_actions(run_dir, 1.0, monkeypatch)
    assert set(drawn) == {decisions.MainAction.PLAY_BIRD}


def test_agent_at_factor_minus_one_never_plays_the_sole_forced_bird(
    run_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    drawn = _draw_actions(run_dir, -1.0, monkeypatch)
    assert decisions.MainAction.PLAY_BIRD not in drawn
    assert len(set(drawn)) > 1


def test_agent_at_factor_zero_samples_every_action(
    run_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    drawn = _draw_actions(run_dir, 0.0, monkeypatch)
    assert len(set(drawn)) > 1


###### Games: seat alternation and seed pairing ######


def _forced(factor: float, bird_id: int | None = None) -> models.ForcedCardSpec:
    return models.ForcedCardSpec(
        force_card_factor=factor,
        forced_card_id=forced_card.resolve_forced_card_id(bird_id),
    )


def test_biased_seat_alternates_by_game_parity():
    assert [forced_card.biased_seat(index, 2) for index in range(4)] == [0, 1, 0, 1]
    assert [forced_card.biased_seat(index, 3) for index in range(4)] == [0, 1, 2, 0]


def test_games_mark_one_alternating_biased_seat_with_paired_seeds(
    run_dir: pathlib.Path,
):
    nets = game_stats.load_game_nets(run_dir, _DEVICE)
    spec = models.GameStatsSpec(games=2, seed=11)
    forced = _forced(0.5)
    first = forced_card.play_game(spec, 0, nets, forced)
    second = forced_card.play_game(spec, 1, nets, forced)
    assert [record.biased for record in first] == [True, False]
    assert [record.biased for record in second] == [False, True]
    assert {record.seed for record in first} == {spec.game_seed(0)}
    assert {record.seed for record in second} == {spec.game_seed(1)}
    for record in first + second:
        assert not record.forced_card_played or record.forced_card_in_hand


def test_games_are_reproducible_and_paired_across_factors(run_dir: pathlib.Path):
    nets = game_stats.load_game_nets(run_dir, _DEVICE)
    spec = models.GameStatsSpec(games=1, seed=5)
    again = forced_card.play_game(spec, 0, nets, _forced(0.5))
    assert again == forced_card.play_game(spec, 0, nets, _forced(0.5))

    # At factor 0 the bias is off: the forced bird cannot change the game.
    other_id = _cheap_birds()[0].id
    assert [
        (record.won, record.final_score)
        for record in forced_card.play_game(spec, 0, nets, _forced(0.0))
    ] == [
        (record.won, record.final_score)
        for record in forced_card.play_game(spec, 0, nets, _forced(0.0, other_id))
    ]


###### Summary aggregation (hand-built fixture) ######


def _seat_record(
    game_index: int, *, biased: bool, won: bool, in_hand: bool, played: bool
) -> models.ForcedCardSeatRecord:
    return models.ForcedCardSeatRecord(
        game_index=game_index,
        seed=100 + game_index,
        seat=0 if biased else 1,
        biased=biased,
        won=won,
        tie=False,
        final_score=50,
        forced_card_in_hand=in_hand,
        forced_card_played=played,
    )


def _fixture_records() -> list[models.ForcedCardSeatRecord]:
    return [
        # Biased: win with the bird played; loss with it held unplayed; win and
        # loss without ever holding it.
        _seat_record(0, biased=True, won=True, in_hand=True, played=True),
        _seat_record(1, biased=True, won=False, in_hand=True, played=False),
        _seat_record(2, biased=True, won=True, in_hand=False, played=False),
        _seat_record(3, biased=True, won=False, in_hand=False, played=False),
        # Unbiased seats (the forced bird is not tracked for them in the summary).
        _seat_record(0, biased=False, won=False, in_hand=False, played=False),
        _seat_record(1, biased=False, won=True, in_hand=False, played=False),
    ]


def test_summary_row_matches_hand_computation():
    row = forced_card_agg.summary_row(0.5, 7, _fixture_records())
    assert (row.biased.n, row.biased.wins) == (4, 2)
    assert row.biased.win_rate == pytest.approx(0.5)
    assert (row.opponent.n, row.opponent.wins) == (2, 1)
    assert row.in_hand_games == 2
    assert row.play_rate == pytest.approx(0.5)
    assert (row.played.n, row.played.wins) == (1, 1)
    assert (row.held_unplayed.n, row.held_unplayed.wins) == (1, 0)
    # Wilson 95% interval for 1 success in 2 trials.
    assert row.play_rate_ci95_lo == pytest.approx(0.09453, abs=1e-4)
    assert row.play_rate_ci95_hi == pytest.approx(0.90547, abs=1e-4)


def test_summary_row_blank_rates_when_the_bird_is_never_held():
    records = [_seat_record(0, biased=True, won=True, in_hand=False, played=False)]
    row = forced_card_agg.summary_row(1.0, 7, records)
    assert row.in_hand_games == 0
    assert row.play_rate is None
    assert row.played.win_rate is None
    assert row.held_unplayed.win_rate is None


def test_raw_record_round_trips_through_csv_cells():
    record = _seat_record(3, biased=True, won=True, in_hand=True, played=False)
    row = {key: str(value) for key, value in record.to_csv_row().items()}
    assert models.ForcedCardSeatRecord.from_csv_row(row) == record
    assert models.ForcedCardSeatRecord.csv_columns() == [
        "game_index",
        "seed",
        "seat",
        "biased",
        "won",
        "tie",
        "final_score",
        "forced_card_in_hand",
        "forced_card_played",
    ]


def test_aggregate_writes_the_combined_summary(tmp_path: pathlib.Path):
    forced_id = forced_card.resolve_forced_card_id(None)
    runs: list[forced_card_agg.FactorRun] = []
    for factor in (1.0, -0.5, 0.0):
        raw = tmp_path / f"raw_{factor}.csv"
        with raw.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=models.ForcedCardSeatRecord.csv_columns()
            )
            writer.writeheader()
            for record in _fixture_records():
                writer.writerow(record.to_csv_row())
        runs.append(forced_card_agg.FactorRun(factor, forced_id, raw))
    written = forced_card_agg.aggregate(runs, tmp_path / "out")
    assert {path.name for path in written} == {
        constants.FORCED_CARD_SUMMARY_FILENAME,
        constants.FORCED_CARD_SUMMARY_MARKDOWN_FILENAME,
    }
    with (tmp_path / "out" / constants.FORCED_CARD_SUMMARY_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == models.ForcedCardSummaryRow.csv_columns()
        rows = list(reader)
    assert [row["force_card_factor"] for row in rows] == ["-0.5", "0.0", "1.0"]
    assert rows[1]["biased_win_rate"] == "0.5"
    markdown = tmp_path / "out" / constants.FORCED_CARD_SUMMARY_MARKDOWN_FILENAME
    assert markdown.read_text(encoding="utf-8").startswith("Forced card: Cedar Waxwing")


###### CLI ######


def test_cli_plays_every_factor_and_writes_the_summary(
    run_dir: pathlib.Path, tmp_path: pathlib.Path
):
    out_dir = tmp_path / "study"
    code = app.main(
        [
            "forced-card",
            "--games",
            "2",
            "--seed",
            "9",
            "--force-card-factor",
            "0,1",
            "--out",
            str(out_dir),
            "--checkpoint-dir",
            str(run_dir),
        ]
    )
    assert code == 0
    for factor in ("0.000", "1.000"):
        raw = out_dir / f"factor_{factor}" / constants.FORCED_CARD_RAW_FILENAME
        with raw.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert [row["biased"] for row in rows] == ["True", "False", "False", "True"]
    assert (out_dir / constants.FORCED_CARD_SUMMARY_FILENAME).stat().st_size > 0


def test_cli_plays_negative_factors_in_the_equals_form(
    run_dir: pathlib.Path, tmp_path: pathlib.Path
):
    out_dir = tmp_path / "study"
    code = app.main(
        [
            "forced-card",
            "--games",
            "2",
            "--seed",
            "9",
            "--force-card-factor=-1,-0.5,0",
            "--out",
            str(out_dir),
            "--checkpoint-dir",
            str(run_dir),
        ]
    )
    assert code == 0
    for factor in ("-1.000", "-0.500", "0.000"):
        assert (
            out_dir / f"factor_{factor}" / constants.FORCED_CARD_RAW_FILENAME
        ).is_file()
    with (out_dir / constants.FORCED_CARD_SUMMARY_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["force_card_factor"] for row in rows] == ["-1.0", "-0.5", "0.0"]


@pytest.mark.parametrize("text", ["0,1.5", "-1.5,0"])
def test_cli_rejects_a_factor_outside_the_range(tmp_path: pathlib.Path, text: str):
    with pytest.raises(SystemExit) as raised:
        app.main(["forced-card", "--out", str(tmp_path), f"--force-card-factor={text}"])
    assert raised.value.code == 2


def test_cli_unknown_bird_id_exits_one(
    run_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
    unused = max(bird.id for bird in cards.birds_ordered()) + 1
    code = app.main(
        [
            "forced-card",
            "--games",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--checkpoint-dir",
            str(run_dir),
            "--forced-card-id",
            str(unused),
        ]
    )
    assert code == 1
    assert "not a bird in the card database" in capsys.readouterr().err
