"""Loading a full :class:`~wingspan.training.config.RunConfig` from a standalone
file, for ``wingspan dashboard --config FILE``.

Four on-disk shapes are accepted, detected from the parsed top level's keys, in
priority order:

* a dated ``run_config_<stamp>.json`` artifact (:class:`~wingspan.training.config.RunConfigFile`)
  — has ``"config"`` — the embedded ``config`` mapping is used;
* a ``configurator_defaults.json``-style envelope
  (:class:`~wingspan.training.configure.user_defaults.DefaultsFile`) — has
  ``"settings"`` — the ``settings`` mapping is used;
* a cloud run-file's ``train:`` block — has ``"train"`` — the ``train`` mapping
  is used, with a top-level ``run_name`` copied into ``train.run.run_name``
  (mirroring ``CloudRunFile._align_train_to_run``); the ``s3`` / ``sync``
  blocks are ignored, since they configure S3 persistence, not the run itself;
* otherwise the mapping is treated as a bare ``RunConfig`` dump.

Every shape is reduced to a raw ``RunConfig`` settings mapping and validated at
the live ``MODEL_VERSION`` — the same era-stripping
:func:`~wingspan.training.configure.user_defaults.strip_derived_descriptor_fields`
applies before ``configurator_defaults.json`` loads — so an older-era file
still loads cleanly. The era backstop for an actual resumable directory
remains ``configure.runs.align_era`` (opening the config screen) and
``loop_resume.adopt_checkpoint_era`` (a headless launch); this module only
decides what the *file itself* means, never what a checkpoint directory allows.

This module is deliberately torch-free (only ``json`` / ``yaml`` / ``pydantic``
plus the training config models) so ``wingspan dashboard --config FILE``'s
error paths (a missing file, bad JSON/YAML, a failed validation) never pay for
importing torch. It parses YAML directly rather than importing
:mod:`wingspan.cloud.runfile`: that module lives in the ``wingspan.cloud``
package, whose ``__init__.py`` also imports sibling modules (``runner``,
``s3sync``) that pull in ``boto3`` — a dependency this module must not require.
"""

from __future__ import annotations

import json
import pathlib
import typing

import pydantic
import yaml

from wingspan.training import config
from wingspan.training.configure import user_defaults

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


class ConfigFileError(ValueError):
    """Raised when ``--config FILE`` cannot be loaded into a ``RunConfig``.

    Carries the offending ``path`` and a human-readable ``reason`` — a missing
    file, unparseable JSON/YAML, a non-mapping top level, or a pydantic
    validation failure — so ``app.main`` can print one clear line and exit.
    """

    def __init__(self, path: pathlib.Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def load_run_config(path: pathlib.Path) -> config.RunConfig:
    """Load a full :class:`~wingspan.training.config.RunConfig` from ``path``.

    Detects the on-disk shape (artifact / defaults envelope / cloud run-file /
    bare dump) from its top-level keys and validates the reduced settings at
    the live ``MODEL_VERSION``. Raises :class:`ConfigFileError` — never a raw
    ``OSError``, ``yaml.YAMLError``, ``json.JSONDecodeError``, or
    ``pydantic.ValidationError`` — for every failure mode.
    """
    raw = _read_mapping(path)
    try:
        settings = _settings_from_raw(raw)
    except ValueError as error:
        raise ConfigFileError(path, str(error)) from error
    user_defaults.strip_derived_descriptor_fields(settings)
    try:
        return config.RunConfig.model_validate(settings)
    except pydantic.ValidationError as error:
        raise ConfigFileError(path, f"invalid config — {error}") from error


###### PRIVATE #######


def _read_mapping(path: pathlib.Path) -> dict[str, typing.Any]:
    """Read ``path`` and parse it as JSON, or YAML for a ``.yaml`` / ``.yml``
    suffix, into a top-level mapping."""
    if not path.exists():
        raise ConfigFileError(path, "file not found")
    text = path.read_text(encoding="utf-8")
    try:
        parsed = (
            yaml.safe_load(text)
            if path.suffix.lower() in _YAML_SUFFIXES
            else json.loads(text)
        )
    except (yaml.YAMLError, json.JSONDecodeError) as error:
        raise ConfigFileError(path, f"could not parse — {error}") from error
    if not isinstance(parsed, dict):
        raise ConfigFileError(path, "top level must be a mapping")
    return typing.cast("dict[str, typing.Any]", parsed)


def _settings_from_raw(raw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Reduce a parsed top-level mapping to a raw ``RunConfig`` settings dict,
    detecting the shape from its top-level keys (artifact, defaults envelope,
    cloud run-file, or a bare dump), in that priority order."""
    if "config" in raw:
        return _as_mapping(raw["config"])
    if "settings" in raw:
        return _as_mapping(raw["settings"])
    if "train" in raw:
        return _settings_from_cloud_shape(raw)
    return raw


def _settings_from_cloud_shape(raw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Reduce a cloud run-file's ``train:`` block, mirroring
    ``CloudRunFile._align_train_to_run``: a top-level ``run_name`` is
    authoritative over ``train.run.run_name``. The ``s3`` / ``sync`` blocks are
    ignored — they configure S3 persistence, not the training run itself."""
    settings = _as_mapping(raw["train"])
    run_name = raw.get("run_name")
    if isinstance(run_name, str):
        run_section = settings.get("run")
        if not isinstance(run_section, dict):
            run_section = {}
            settings["run"] = run_section
        typing.cast("dict[str, typing.Any]", run_section)["run_name"] = run_name
    return settings


def _as_mapping(value: typing.Any) -> dict[str, typing.Any]:
    """``value`` as a settings mapping, or a ``ValueError`` naming the shape
    that was expected but not found."""
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return typing.cast("dict[str, typing.Any]", value)
