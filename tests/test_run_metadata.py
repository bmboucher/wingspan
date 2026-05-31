"""Tests for the per-run JSON metadata sidecars (``wingspan.training.runmeta``)
and the persisted per-game outcome row.

* ``model_config.json`` carries the architecture descriptor, round-trips, and is
  rewritten (not appended) on each startup.
* ``process_<stamp>.json`` carries the session's full config + runtime context,
  derives the resumed flag from the resume iteration, and never overwrites a
  prior same-second session record.
* ``metrics.GameOutcome`` rows round-trip (the per-game ``games.jsonl`` shape).

The per-game write path itself — seeds and counts flowing through the real
collector into ``games.jsonl`` — is covered end-to-end in
``test_training_dashboard.test_training_loop_one_iteration``.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")

from wingspan.training import artifacts, config, metrics, runmeta


def test_write_model_config_round_trips(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", run_name="alpha", hidden=64)
    path = runmeta.write_model_config(str(tmp_path), cfg)
    assert path.name == artifacts.MODEL_CONFIG_JSON
    descriptor = runmeta.ModelConfig.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    assert descriptor.run_name == "alpha" and descriptor.hidden == 64
    # The shape fields are exactly the weight-compatibility key.
    assert (
        descriptor.state_dim,
        descriptor.choice_dim,
        descriptor.family_order,
        descriptor.hidden,
        descriptor.card_embed_dim,
    ) == cfg.architecture_key


def test_write_model_config_overwrites(tmp_path: pathlib.Path):
    runmeta.write_model_config(
        str(tmp_path), config.TrainConfig(device="cpu", hidden=64)
    )
    runmeta.write_model_config(
        str(tmp_path), config.TrainConfig(device="cpu", hidden=128)
    )
    descriptor = runmeta.ModelConfig.model_validate_json(
        (tmp_path / artifacts.MODEL_CONFIG_JSON).read_text(encoding="utf-8")
    )
    assert descriptor.hidden == 128  # rewritten in place, not appended


def test_write_session_record_captures_context(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu", run_name="beta", games_per_iter=256)
    path = runmeta.write_session_record(
        str(tmp_path),
        cfg,
        stamp="20260530-191500",
        started_at="2026-05-30T19:15:00",
        git_sha="abc1234",
        resumed_from_iteration=0,
    )
    assert path.name == "process_20260530-191500.json"
    record = runmeta.SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.run_name == "beta" and record.git_sha == "abc1234"
    assert record.resumed is False and record.resumed_from_iteration == 0
    # The full config is embedded — the batch size and every other knob.
    assert record.config.games_per_iter == 256


def test_write_session_record_resumed_flag(tmp_path: pathlib.Path):
    path = runmeta.write_session_record(
        str(tmp_path),
        config.TrainConfig(device="cpu"),
        stamp="20260530-200000",
        started_at="2026-05-30T20:00:00",
        git_sha=None,
        resumed_from_iteration=42,
    )
    record = runmeta.SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.resumed is True and record.resumed_from_iteration == 42


def test_write_session_record_unique_on_collision(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(device="cpu")
    first = runmeta.write_session_record(
        str(tmp_path),
        cfg,
        stamp="S",
        started_at="t",
        git_sha=None,
        resumed_from_iteration=0,
    )
    second = runmeta.write_session_record(
        str(tmp_path),
        cfg,
        stamp="S",
        started_at="t",
        git_sha=None,
        resumed_from_iteration=0,
    )
    assert first.name == "process_S.json"
    assert second.name == "process_S-1.json"  # a same-second restart keeps both
    assert len(list(tmp_path.glob(artifacts.PROCESS_GLOB))) == 2


def test_game_outcome_round_trips():
    family = metrics.FamilyCounts()
    family.bump(0)
    family.bump(2)
    breakdown = metrics.ScoreBreakdown(birds=20, eggs=5)
    outcome = metrics.GameOutcome(
        iteration=3,
        seed=123,
        winner=0,
        decisions=2,
        breakdowns=(breakdown, breakdown),
        family_counts=family,
    )
    reloaded = metrics.GameOutcome.model_validate_json(outcome.model_dump_json())
    assert reloaded.seed == 123 and reloaded.iteration == 3
    assert reloaded.breakdowns[0].birds == 20
    assert reloaded.family_counts.total() == 2
