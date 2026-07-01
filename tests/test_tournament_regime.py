"""The tournament runs its games under the competitors' trained regime.

Covers :func:`participants.resolve_regime_flags` (config-free fields default to
the engine's off-regime, agreeing model configs are honored, disagreeing ones
are refused) and confirms :func:`runner.run_tournament` forwards the resolved
flags into ``Engine.play_one_game`` so each net plays under the regime it was
trained on.
"""

from __future__ import annotations

import typing

import pytest

from wingspan import engine
from wingspan.tournament import models, participants, runner
from wingspan.training import config
from wingspan.training.configure import runs


def _model_spec(spec_id: str, checkpoint_dir: str) -> models.ParticipantSpec:
    return models.ParticipantSpec(
        id=spec_id,
        display_name=spec_id,
        kind=models.ParticipantKind.MODEL,
        checkpoint_dir=checkpoint_dir,
    )


def _random_spec(spec_id: str) -> models.ParticipantSpec:
    return models.ParticipantSpec(
        id=spec_id, display_name=spec_id, kind=models.ParticipantKind.RANDOM
    )


def _patch_configs(
    monkeypatch: pytest.MonkeyPatch, by_dir: dict[str, config.RunConfig]
) -> None:
    """Make ``runs.inspect_run`` return each checkpoint dir's chosen config, so
    the resolver reads regimes without any checkpoint on disk."""

    def fake_inspect_run(checkpoint_dir: str) -> runs.RunSummary:
        return runs.RunSummary(
            checkpoint_dir=checkpoint_dir, train_config=by_dir.get(checkpoint_dir)
        )

    monkeypatch.setattr(participants.runs, "inspect_run", fake_inspect_run)


def test_all_random_field_resolves_to_off_regime() -> None:
    """A config-free (random-only) field expresses no preference and resolves to
    the engine's default all-off regime — no checkpoint reads needed."""
    regime = participants.resolve_regime_flags([_random_spec("r0"), _random_spec("r1")])
    assert regime == models.RegimeFlags()


def test_agreeing_model_configs_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two agreeing model configs resolve to exactly the regime they were trained
    under (mirroring each config's own derived flags — here the default arch's
    split regimes plus an enabled ``combine_gain_food``); a config-free random
    competitor in the field does not veto it."""
    combined = config.RunConfig(engine=config.EngineConfig(combine_gain_food=True))
    _patch_configs(monkeypatch, {"a": combined, "b": combined})

    regime = participants.resolve_regime_flags(
        [_model_spec("a", "a"), _model_spec("b", "b"), _random_spec("r")]
    )

    assert regime == models.RegimeFlags(
        split_setup_bonus=combined.split_setup_bonus_active,
        split_setup_food=combined.split_setup_food_active,
        combine_gain_food=combined.engine.combine_gain_food,
    )
    assert regime.combine_gain_food is True


def test_disagreeing_model_configs_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two models trained under different food-gain regimes cannot share a
    faithful game, so the tournament refuses up front."""
    combined = config.RunConfig(engine=config.EngineConfig(combine_gain_food=True))
    sequential = config.RunConfig(engine=config.EngineConfig(combine_gain_food=False))
    _patch_configs(monkeypatch, {"a": combined, "b": sequential})

    with pytest.raises(ValueError, match="combine_gain_food"):
        participants.resolve_regime_flags(
            [_model_spec("a", "a"), _model_spec("b", "b")]
        )


def test_run_tournament_forwards_regime_to_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved regime reaches ``Engine.play_one_game`` for every game, so
    the games actually run under the trained variant instead of engine defaults."""
    captured: list[dict[str, typing.Any]] = []
    real_play_one_game = engine.Engine.play_one_game

    def spy_play_one_game(
        gs: typing.Any, agents: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        captured.append(kwargs)
        return real_play_one_game(gs, agents, *args, **kwargs)

    monkeypatch.setattr(engine.Engine, "play_one_game", spy_play_one_game)

    cfg = models.TournamentConfig(
        participants=[_random_spec("r0"), _random_spec("r1")],
        games_per_pair=2,
        base_seed=0,
    )
    regime = models.RegimeFlags(combine_gain_food=True, split_setup_food=True)
    runner.run_tournament(cfg, in_process=True, regime=regime)

    assert captured, "expected at least one game to be played"
    assert all(kwargs["combine_gain_food"] is True for kwargs in captured)
    assert all(kwargs["split_setup_food"] is True for kwargs in captured)
    assert all(kwargs["split_setup_bonus"] is False for kwargs in captured)
