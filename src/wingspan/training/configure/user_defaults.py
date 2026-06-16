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

EXCLUDED_FIELDS: list[str] = [
    "encoding_version",
    "state_dim",
    "choice_dim",
    "family_order",
    "checkpoint_dir",
    "run_name",
    "resume",
    "device",
]
"""Flat field names stripped from the saved settings by :func:`save_defaults`.

These are per-run identity, era, and launch-time fields that do not travel with
the defaults file. They are removed from their respective nested sections
(``architecture``, ``run``, ``misc``) by :func:`_strip_identity_fields` before
the envelope is written."""


class DefaultsFile(pydantic.BaseModel):
    """The on-disk envelope around the persisted settings."""

    saved_with_version: str  # MODEL_VERSION at save time (diagnostics only)
    saved_at: str  # ISO timestamp of the save
    # The raw RunConfig dump minus identity/era fields. Deliberately untyped: it
    # is a cross-version payload validated against the *current* RunConfig on
    # load, the same pattern as a checkpoint's embedded raw config.
    settings: dict[str, typing.Any]


class LoadedDefaults(pydantic.BaseModel):
    """The outcome of :func:`load_defaults`: a config, a warning, or neither
    (no file saved yet). The config field is named ``train_config`` so it does
    not shadow the ``config`` module in its own annotation (the same pattern
    as ``runs.RunSummary``)."""

    train_config: config.RunConfig | None = None
    warning: str | None = None


def save_defaults(
    cfg: config.RunConfig, directory: pathlib.Path | None = None
) -> pathlib.Path:
    """Write ``cfg``'s reusable settings to the defaults file and return its
    path. ``directory`` overrides the working directory (tests only)."""
    # Dump the full nested config then strip the per-run identity fields so
    # the saved file contains only transferable hyperparameters.
    settings = cfg.model_dump(mode="json")
    _strip_identity_fields(settings)
    envelope = DefaultsFile(
        saved_with_version=version.MODEL_VERSION,
        saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
        settings=settings,
    )
    path = _defaults_path(directory)
    path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_defaults(
    current: config.RunConfig, directory: pathlib.Path | None = None
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
        settings = envelope.settings
        # Strip any stale identity fields that might have been saved by older code
        # or hand-edited in; then inject the current run's identity fields.
        _strip_identity_fields(settings)
        _inject_identity_fields(settings, current)
        loaded = config.RunConfig.model_validate(settings)
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


def _strip_identity_fields(settings: dict[str, typing.Any]) -> None:
    """Remove per-run identity and era fields from a nested ``RunConfig`` dump
    in place. These are never persisted to the defaults file because they are
    properties of a specific run directory or a launch-time decision."""
    # Remove the architecture era + synced dims (re-derived on load from live code).
    arch = settings.get("architecture")
    if isinstance(arch, dict):
        arch_typed = typing.cast("dict[str, typing.Any]", arch)
        for key in ("encoding_version", "state_dim", "choice_dim", "family_order"):
            arch_typed.pop(key, None)

    # Remove run-identity fields from the ``run`` section.
    run = settings.get("run")
    if isinstance(run, dict):
        run_typed = typing.cast("dict[str, typing.Any]", run)
        for key in ("checkpoint_dir", "run_name", "resume"):
            run_typed.pop(key, None)

    # Remove the device from the ``misc`` section.
    misc = settings.get("misc")
    if isinstance(misc, dict):
        misc_typed = typing.cast("dict[str, typing.Any]", misc)
        misc_typed.pop("device", None)


def _inject_identity_fields(
    settings: dict[str, typing.Any], current: config.RunConfig
) -> None:
    """Write the current run's identity fields into the nested settings dict.
    Called before validating so the loaded config targets the correct directory
    and uses the correct device."""
    if "run" not in settings or not isinstance(settings["run"], dict):
        settings["run"] = {}
    settings["run"]["checkpoint_dir"] = current.run.checkpoint_dir
    settings["run"]["run_name"] = current.run.run_name
    settings["run"]["resume"] = current.run.resume

    if "misc" not in settings or not isinstance(settings["misc"], dict):
        settings["misc"] = {}
    settings["misc"]["device"] = current.misc.device
