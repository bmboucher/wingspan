"""The unified ``RunConfig`` artifact: flat→nested migration and round-trip.

``RunConfig`` (v0.5+) groups every hyperparameter into six nested sections and
is written as one dated ``run_config_<stamp>.json`` per session. Two seams are
covered here:

* :func:`config.run_config_from_artifact` — validates a checkpoint's embedded
  config at the artifact's own era. A ≤0.4 payload is *flat* (every field at the
  top level); it is reshaped into the six sections, preserving the legacy
  ``bootstrap_opponent`` migration. A ≥0.5 payload is already nested and passes
  through. Either way the era is adopted from the payload stamp when the config
  does not carry its own.
* :func:`runmeta.write_run_config` / :func:`runmeta.read_run_config` — the dated
  file round-trips a ``RunConfig`` plus its session context byte-for-byte.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")

from wingspan import architecture, version  # noqa: E402
from wingspan.training import config, runmeta  # noqa: E402

#### Flat (≤0.4) → nested migration ####


def test_flat_fields_group_into_their_sections():
    """Every flat ≤0.4 key lands in its nested section — across architecture,
    run, training (incl. setup), opponent, and misc."""
    flat = {
        "lr": 7e-4,
        "games_per_iter": 64,
        "device": "cpu",
        "seed": 11,
        "trunk_layers": (32, 32),
        "card_embed_dim": 8,
        "setup_lr": 5e-4,
        "bootstrap_opponent": "random",
        "eval_ewma_alpha": 0.4,
    }
    cfg = config.run_config_from_artifact(flat, version.MODEL_VERSION)

    assert cfg.training.lr == 7e-4
    assert cfg.run.games_per_iter == 64
    assert cfg.misc.device == "cpu"
    assert cfg.misc.seed == 11
    assert cfg.architecture.main.trunk_layers == (32, 32)
    assert cfg.architecture.main.card_embed_dim == 8
    assert cfg.training.setup.lr == 5e-4
    assert cfg.opponent.bootstrap_opponent == "random"
    assert cfg.opponent.eval_ewma_alpha == 0.4


def test_flat_unknown_keys_are_dropped():
    """A retired flat field from another era is silently discarded, not carried
    as an extra — the migration only keeps keys it can place."""
    cfg = config.run_config_from_artifact(
        {"lr": 1e-3, "some_retired_field": 123}, version.MODEL_VERSION
    )
    assert cfg.training.lr == 1e-3


def test_legacy_bootstrap_opponent_migration():
    """The pre-``bootstrap_opponent`` pair migrates to the single field:
    ``initial_vs_random`` False → ``"none"``; a checkpoint path → that path;
    neither → the ``"random"`` default."""
    disabled = config.run_config_from_artifact(
        {"initial_vs_random": False}, version.MODEL_VERSION
    )
    assert disabled.opponent.bootstrap_opponent == "none"

    pathed = config.run_config_from_artifact(
        {"initial_vs_random": True, "bootstrap_opponent_checkpoint": "opp.pt"},
        version.MODEL_VERSION,
    )
    assert pathed.opponent.bootstrap_opponent == "opp.pt"

    default = config.run_config_from_artifact({}, version.MODEL_VERSION)
    assert default.opponent.bootstrap_opponent == "random"


#### Era stamp adoption ####


def test_flat_config_adopts_the_payload_stamp():
    """A pre-field flat config (no ``encoding_version``) adopts the artifact's
    own era stamp; one that carries the field keeps it. With no pre-1.0 shims the
    only loadable stamp is the live era, so adoption is validated against it."""
    adopted = config.run_config_from_artifact({"lr": 1e-3}, version.MODEL_VERSION)
    assert adopted.encoding_version == version.MODEL_VERSION

    kept = config.run_config_from_artifact(
        {"lr": 1e-3, "encoding_version": version.MODEL_VERSION}, version.MODEL_VERSION
    )
    assert kept.encoding_version == version.MODEL_VERSION


#### Nested (≥0.5) passthrough ####


def test_nested_config_passes_through_unchanged():
    """A ≥0.5 nested dump validates directly, preserving every section value."""
    original = config.RunConfig(
        misc=config.MiscConfig(device="cpu", seed=3),
        training=config.TrainingConfig(lr=2e-4),
        run=config.RunSettings(games_per_iter=8),
    )
    restored = config.run_config_from_artifact(
        original.model_dump(), version.MODEL_VERSION
    )
    assert restored == original


def test_nested_config_without_era_adopts_the_stamp():
    """A nested dump whose architecture omits ``encoding_version`` adopts the
    passed payload stamp (the live era, the only loadable one at 1.0)."""
    raw = config.RunConfig(misc=config.MiscConfig(device="cpu")).model_dump()
    raw["architecture"].pop("encoding_version")
    adopted = config.run_config_from_artifact(raw, version.MODEL_VERSION)
    assert adopted.encoding_version == version.MODEL_VERSION


#### Dated file round-trip ####


def test_run_config_file_round_trips(tmp_path: pathlib.Path):
    """``write_run_config`` then ``read_run_config`` reconstitutes the config and
    its session context; the file is named with the ``run_config_`` prefix."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        training=config.TrainingConfig(lr=4e-4),
        run=config.RunSettings(checkpoint_dir=str(tmp_path), run_name="round-trip"),
    )
    path = runmeta.write_run_config(
        str(tmp_path),
        cfg,
        stamp="20260615-120000",
        started_at="2026-06-15T12:00:00",
        git_sha=None,
        resumed_from_iteration=0,
    )
    assert path.name.startswith("run_config_")

    loaded = runmeta.read_run_config(str(tmp_path))
    assert loaded.config == cfg
    assert loaded.version == version.MODEL_VERSION
    assert loaded.resumed is False
    assert loaded.started_at == "2026-06-15T12:00:00"


def test_read_run_config_missing_raises(tmp_path: pathlib.Path):
    """An empty directory has no unified file — the reader refuses rather than
    inventing a default (callers needing legacy support check first)."""
    with pytest.raises(FileNotFoundError):
        runmeta.read_run_config(str(tmp_path))


#### legacy activation-field migration (≤0.8 → between/final) ####


def test_legacy_activation_fields_migrate_to_between_final():
    """A ≤0.8 architecture dump (flat ``activation`` + ``encoder_final_activation``
    + per-block ``*_activation``) rehydrates into the between/final scheme without
    a compat shim — the same REGIME migration on both the runtime
    ``architecture.ModelArchitecture`` and the config ``MainNetArchitecture``."""
    legacy = {
        "activation": "gelu",
        "encoder_final_activation": True,
        "card_activation": "relu",
        "trunk_activation": "tanh",
        "value_activation": "silu",
    }
    for arch_cls in (architecture.ModelArchitecture, config.MainNetArchitecture):
        arch = arch_cls.model_validate(dict(legacy))
        assert arch.between_activation == architecture.ActivationName.GELU
        assert arch.final_activation == architecture.ActivationName.NONE
        # A per-block override survives; encoder_final=True carries it to final.
        assert arch.card_between_activation == architecture.ActivationName.RELU
        assert arch.card_final_activation == architecture.ActivationName.RELU
        # No hand override -> between inherits (None); final resolves to the global.
        assert arch.hand_between_activation is None
        assert arch.hand_final_activation == architecture.ActivationName.GELU
        assert arch.trunk_between_activation == architecture.ActivationName.TANH
        # Readout blocks take between only; final is always NONE on migrated runs.
        assert arch.value_between_activation == architecture.ActivationName.SILU
        assert arch.value_final_activation == architecture.ActivationName.NONE

    # encoder_final omitted (False) -> migrated encoder finals are NONE.
    no_final = config.MainNetArchitecture.model_validate(
        {"activation": "relu", "card_activation": "gelu"}
    )
    assert no_final.card_between_activation == architecture.ActivationName.GELU
    assert no_final.card_final_activation == architecture.ActivationName.NONE

    # The setup net carries the simpler single-field migration.
    setup = config.SetupNetArchitecture.model_validate({"activation": "tanh"})
    assert setup.between_activation == architecture.ActivationName.TANH
    assert setup.final_activation == architecture.ActivationName.NONE

    # A config that already uses the new scheme is passed through untouched.
    modern = config.MainNetArchitecture.model_validate(
        {"between_activation": "gelu", "final_activation": "none"}
    )
    assert modern.between_activation == architecture.ActivationName.GELU


#### validate_launchable ####


def test_validate_launchable_clean_config_returns_empty():
    """A factory-default config has no launchable problems."""
    assert config.validate_launchable(config.RunConfig()) == []


def test_validate_launchable_checkpoint_bootstrap_on_cuda_flagged():
    """A checkpoint bootstrap opponent with device='cuda' is flagged."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cuda"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
    )
    problems = config.validate_launchable(cfg)
    assert any("cpu" in problem for problem in problems)


def test_validate_launchable_random_bootstrap_on_cuda_ok():
    """'random' or 'none' bootstrap on cuda is allowed."""
    for bootstrap in ("random", "none"):
        cfg = config.RunConfig(
            misc=config.MiscConfig(device="cuda"),
            opponent=config.OpponentConfig(bootstrap_opponent=bootstrap),
        )
        assert config.validate_launchable(cfg) == []


def test_validate_launchable_target_exceeds_max_flagged():
    """target_iterations > max_iterations when both nonzero is flagged."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=500, max_iterations=100),
    )
    problems = config.validate_launchable(cfg)
    assert any("target_iterations" in problem for problem in problems)


def test_validate_launchable_clone_iters_with_checkpoint_validates():
    """clone_iters > 0 with a checkpoint bootstrap_opponent validates cleanly.

    This is the original bug: prior to Workstream E this combination raised a
    ValidationError because the cross-field validator rejected it. Now the config
    is valid and validate_launchable returns no problems (device='cpu' satisfied).
    """
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(
            bootstrap_opponent="checkpoints/archive/run_iter500/last.pt"
        ),
        dagger=config.DaggerConfig(clone_iters=5),
    )
    # Must not raise; dagger_active_at correctly returns True before clone_iters.
    assert cfg.dagger_active_at(0) is True
    assert config.validate_launchable(cfg) == []
