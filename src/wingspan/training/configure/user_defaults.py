"""Persisting the user's preferred configurator settings as a defaults file.

``[D] save defaults`` in the FLIGHT PLAN configurator writes the working
config's hyperparameters to ``./configurator_defaults.json`` (the project
working directory — checked into git, so the preferred setup travels with the
checkout). When the configurator later opens on a directory with no saved run,
it seeds the editor from this file instead of the factory defaults; ``[R]``
reset offers it alongside the factory option.

Run-identity and derived-descriptor fields never ride along: the artifact era
and its dims are properties of a run directory, ``resume`` is a launch-time
decision, and ``checkpoint_dir`` / ``run_name`` / ``device`` are per-run
choices rather than reusable hyperparameters. Because the era is excluded,
loading always re-validates at the live ``MODEL_VERSION`` — a file written
under older code degrades gracefully (renamed fields are ignored, missing
fields take factory values) and an unreadable one falls back to factory
defaults with a warning instead of raising.
"""

from __future__ import annotations

import datetime
import pathlib
import typing

import pydantic

from wingspan import version
from wingspan.training import config

DEFAULTS_FILENAME = "configurator_defaults.json"
"""The defaults file name, anchored at the process working directory."""

EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "encoding_version",
        "state_dim",
        "choice_dim",
        "family_order",
        "resume",
        "checkpoint_dir",
        "run_name",
        "device",
    }
)
"""``TrainConfig`` fields that are never persisted (and are stripped on load,
so a hand-edited file cannot smuggle them back in)."""


class DefaultsFile(pydantic.BaseModel):
    """The on-disk envelope around the persisted settings."""

    saved_with_version: str  # MODEL_VERSION at save time (diagnostics only)
    saved_at: str  # ISO timestamp of the save
    # The raw TrainConfig dump minus EXCLUDED_FIELDS. Deliberately untyped: it
    # is a cross-version payload validated against the *current* TrainConfig on
    # load, the same pattern as a checkpoint's embedded raw config.
    settings: dict[str, typing.Any]


class LoadedDefaults(pydantic.BaseModel):
    """The outcome of :func:`load_defaults`: a config, a warning, or neither
    (no file saved yet). The config field is named ``train_config`` so it does
    not shadow the ``config`` module in its own annotation (the same pattern
    as ``runs.RunSummary``)."""

    train_config: config.TrainConfig | None = None
    warning: str | None = None


def save_defaults(
    cfg: config.TrainConfig, directory: pathlib.Path | None = None
) -> pathlib.Path:
    """Write ``cfg``'s reusable settings to the defaults file and return its
    path. ``directory`` overrides the working directory (tests only)."""
    settings = {
        key: value
        for key, value in cfg.model_dump(mode="json").items()
        if key not in EXCLUDED_FIELDS
    }
    envelope = DefaultsFile(
        saved_with_version=version.MODEL_VERSION,
        saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
        settings=settings,
    )
    path = _defaults_path(directory)
    path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_defaults(
    current: config.TrainConfig, directory: pathlib.Path | None = None
) -> LoadedDefaults:
    """Read the defaults file and validate it into a config that keeps
    ``current``'s run-identity fields (checkpoint dir, run name, device,
    resume). Never raises: a missing file returns an empty result, an
    unreadable or invalid one returns a warning so the caller can fall back to
    factory defaults."""
    path = _defaults_path(directory)
    if not path.exists():
        return LoadedDefaults()
    saved_with = "unknown"
    try:
        envelope = DefaultsFile.model_validate_json(path.read_text(encoding="utf-8"))
        saved_with = envelope.saved_with_version
        settings = {
            key: value
            for key, value in envelope.settings.items()
            if key not in EXCLUDED_FIELDS
        }
        loaded = config.TrainConfig.model_validate(
            {
                **settings,
                "checkpoint_dir": current.checkpoint_dir,
                "run_name": current.run_name,
                "device": current.device,
                "resume": current.resume,
            }
        )
    except (OSError, ValueError, pydantic.ValidationError):
        # ValueError covers json decode errors raised by model_validate_json.
        return LoadedDefaults(
            warning=(
                f"{DEFAULTS_FILENAME} (saved with {saved_with}) could not be "
                "applied — using factory defaults"
            )
        )
    return LoadedDefaults(train_config=loaded)


###### PRIVATE #######


def _defaults_path(directory: pathlib.Path | None) -> pathlib.Path:
    return (directory or pathlib.Path.cwd()) / DEFAULTS_FILENAME
