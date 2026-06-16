"""Inspecting and managing the training runs stored under a checkpoint dir.

:func:`inspect_run` reads the metadata a configurator screen shows (iteration,
games, best win-rate, opponent generation, the saved hyperparameters) straight
out of ``last.pt``, which embeds the full ``TrainConfig`` and a ``RunProgress``
snapshot. :func:`architecture_compatible` is the single resume gate, shared with
``loop`` through ``TrainConfig.architecture_key``. :func:`archive_run` relocates
a finished run's artifacts into ``<checkpoint_dir>/archive/<label>/`` (preserving
them) and :func:`clear_run` deletes them (the destructive overwrite path); both
move/remove the same canonical artifact set in a crash-survivable order, leaving
unrelated files (e.g. the test suite's ``_test.pt``) untouched.
"""

from __future__ import annotations

import enum
import pathlib
import shutil
import typing

import pydantic
import torch

from wingspan import version
from wingspan.training import artifacts, config, runstate

# Order matters: ``last.pt`` is relocated/removed LAST so an interruption
# mid-archive never strands the directory in a state that resumes a run whose
# best / opponent snapshots have already moved. Logs, stale temp files, and the
# dated process records go first (swept by glob); ``last.pt`` is appended by
# :func:`_archive_sources`.
_SWEEP_BEFORE_LAST = (
    artifacts.METRICS_LOG,
    artifacts.GAMES_LOG,
    artifacts.MODEL_CONFIG_JSON,
    artifacts.BEST_CKPT,
    artifacts.OPPONENT_CKPT,
    # Setup-model artifacts (present only when the run used the setup model);
    # missing ones are dropped by the presence filter in ``_archive_sources``.
    artifacts.SETUP_CKPT,
    artifacts.SETUP_CONFIG_JSON,
    artifacts.SETUP_DATA_LOG,
)
_MAX_LABEL_SUFFIX = 1000  # give up de-duplicating a label dir after this many


class RunStatus(enum.StrEnum):
    """What [S] Start will do against the inspected directory, given the working
    config — drives the run-management panel's verdict and color."""

    EMPTY = "empty"  # no checkpoint — Start launches a fresh run here
    RESUMABLE = "resumable"  # compatible checkpoint — Start resumes it
    INCOMPATIBLE = "incompatible"  # checkpoint differs — needs a fresh run
    UNREADABLE = "unreadable"  # checkpoint present but could not be read


class ArchiveEntry(pydantic.BaseModel):
    """One previously-archived run under ``<checkpoint_dir>/archive/``."""

    label: str
    modified: float  # directory mtime (epoch seconds)
    has_checkpoint: bool  # whether it still holds a last.pt
    # Metadata loaded from last.pt when the checkpoint is present; None when the
    # checkpoint is absent or unreadable.
    model_version: str | None = None  # payload["version"]
    total_games: int | None = None  # RunProgress.total_games
    # Stamp of the earliest ``process_<stamp>.json`` session record found in the
    # archive directory — approximates when training first started for this run.
    # Derived from the filename only (no file read needed).
    first_session_stamp: str | None = None


def _empty_names() -> list[str]:
    return []


def _empty_archives() -> list[ArchiveEntry]:
    return []


class ArchiveResult(pydantic.BaseModel):
    """The outcome of an :func:`archive_run` call."""

    destination: str
    moved: list[str] = pydantic.Field(default_factory=_empty_names)
    errors: list[str] = pydantic.Field(default_factory=_empty_names)

    @property
    def ok(self) -> bool:
        return not self.errors


class RunSummary(pydantic.BaseModel):
    """A read-only snapshot of the run currently stored in a checkpoint dir."""

    checkpoint_dir: str
    exists: bool = False  # a last.pt is present
    readable: bool = True  # last.pt parsed (only meaningful when exists)
    # The saved run's hyperparameters (named ``train_config`` rather than
    # ``config`` so the field does not shadow the ``config`` module in its own
    # annotation at class-definition time).
    train_config: config.RunConfig | None = None
    # The checkpoint carried no embedded config, or one that no longer validates
    # (e.g. a value since constrained out of bounds). The payload still loaded,
    # but the run is treated as not-resumable so Start routes through the
    # fresh-run prompt instead of handing the loop a config it would reject
    # mid-resume.
    config_invalid: bool = False
    iteration: int | None = None
    total_games: int | None = None
    best_win_rate: float | None = None
    opponent_generation: int = 0
    has_best: bool = False
    has_opponent: bool = False
    has_metrics: bool = False
    has_games: bool = False
    git_sha: str | None = None
    modified: float | None = None  # last.pt mtime (epoch seconds)
    archives: list[ArchiveEntry] = pydantic.Field(default_factory=_empty_archives)
    note: str = ""


def inspect_run(checkpoint_dir: str) -> RunSummary:
    """Read the run stored in ``checkpoint_dir`` (and list its archives). Never
    raises: an unreadable checkpoint is reported via ``readable=False``."""
    path = pathlib.Path(checkpoint_dir)
    summary = RunSummary(
        checkpoint_dir=checkpoint_dir, archives=list_archives(checkpoint_dir)
    )
    last = path / artifacts.LAST_CKPT
    if not last.exists():
        return summary

    summary.exists = True
    summary.has_best = (path / artifacts.BEST_CKPT).exists()
    summary.has_opponent = (path / artifacts.OPPONENT_CKPT).exists()
    summary.has_metrics = (path / artifacts.METRICS_LOG).exists()
    summary.has_games = (path / artifacts.GAMES_LOG).exists()
    summary.modified = last.stat().st_mtime

    payload = _load_payload(last)
    if payload is None:
        summary.readable = False
        summary.note = f"{artifacts.LAST_CKPT} could not be read"
        return summary
    _fill_from_payload(summary, payload)
    return summary


def architecture_compatible(
    saved: config.RunConfig | None, current: config.RunConfig
) -> bool:
    """Whether ``current`` can resume a run saved with ``saved``. Checkpoints are
    self-describing: one with no readable embedded config (``saved is None``) is
    never resumable, matching ``loop._architecture_matches``."""
    return saved is not None and saved.architecture_key == current.architecture_key


def resolve_status(summary: RunSummary, working: config.RunConfig) -> RunStatus:
    """Classify what Start will do against ``summary`` for the working config."""
    if not summary.exists:
        return RunStatus.EMPTY
    if not summary.readable or summary.config_invalid:
        return RunStatus.UNREADABLE
    if architecture_compatible(summary.train_config, working):
        return RunStatus.RESUMABLE
    return RunStatus.INCOMPATIBLE


def align_era(summary: RunSummary, working: config.RunConfig) -> config.RunConfig:
    """Re-pin ``working``'s era after any mutation: the saved run's era while
    the re-keyed working config stays architecture-compatible with it (so Start
    resumes at the run's own frozen geometry), the live MODEL_VERSION otherwise
    (so a fresh launch never inherits a stale era). Keys off
    ``architecture_key`` — never ``ChangeImpact`` labels — so the recompute
    cannot drift from :func:`resolve_status`; the editor-side mirror of
    ``loop_resume.adopt_checkpoint_era``."""
    saved = summary.train_config
    if (
        summary.exists
        and summary.readable
        and not summary.config_invalid
        and saved is not None
    ):
        candidate = _with_era(working, saved.encoding_version)
        if candidate.architecture_key == saved.architecture_key:
            return candidate
    return _with_era(working, version.MODEL_VERSION)


def list_archives(checkpoint_dir: str) -> list[ArchiveEntry]:
    """Existing archived runs under ``<checkpoint_dir>/archive/``, oldest first.

    Metadata (model version, total games, first session stamp) is loaded from
    each archive's checkpoint and process records where available. Failures are
    suppressed so a corrupt archive never blocks the configurator from opening."""
    root = pathlib.Path(checkpoint_dir) / artifacts.ARCHIVE_SUBDIR
    if not root.is_dir():
        return []
    entries = [
        _build_archive_entry(child)
        for child in sorted(root.iterdir())
        if child.is_dir()
    ]
    return sorted(entries, key=lambda entry: entry.modified)


def _build_archive_entry(directory: pathlib.Path) -> ArchiveEntry:
    """Build an :class:`ArchiveEntry` for one archive directory, loading metadata
    from the checkpoint and process records where available."""
    last_pt = directory / artifacts.LAST_CKPT
    has_checkpoint = last_pt.exists()

    # Load model_version and total_games from the checkpoint payload.
    model_version: str | None = None
    total_games: int | None = None
    if has_checkpoint:
        payload = _load_payload(last_pt)
        if payload is not None:
            raw_version = payload.get("version")
            model_version = str(raw_version) if raw_version is not None else None
            progress = _progress_from_payload(payload)
            total_games = progress.total_games

    # Derive the earliest session stamp from process_*.json filenames (no reads).
    process_files = sorted(directory.glob(artifacts.PROCESS_GLOB))
    first_session_stamp: str | None = None
    if process_files:
        # Filename is ``process_<stamp>.json``; strip prefix and suffix for stamp.
        stem = process_files[0].stem  # e.g. "process_20240611-142030"
        prefix = artifacts.PROCESS_PREFIX
        if stem.startswith(prefix):
            first_session_stamp = stem[len(prefix) :]

    return ArchiveEntry(
        label=directory.name,
        modified=directory.stat().st_mtime,
        has_checkpoint=has_checkpoint,
        model_version=model_version,
        total_games=total_games,
        first_session_stamp=first_session_stamp,
    )


def default_archive_label(summary: RunSummary, timestamp: str) -> str:
    """A descriptive, filesystem-safe archive label: ``<run>_iterNNNN_<stamp>``.

    ``timestamp`` is supplied by the caller (so this stays pure / testable)."""
    run_name = (
        summary.train_config.run.run_name if summary.train_config is not None else "run"
    )
    iteration = summary.iteration if summary.iteration is not None else 0
    return f"{_sanitize(run_name)}_iter{iteration:04d}_{timestamp}"


def archive_run(checkpoint_dir: str, label: str) -> ArchiveResult:
    """Move the run's artifacts into ``<checkpoint_dir>/archive/<label>/``.

    The label directory is made unique if it already exists, so repeated
    archives never silently merge. Each move is independent — a failure on one
    file (e.g. a Windows lock) is collected into ``errors`` rather than aborting
    the rest, and ``last.pt`` is moved last so a partial archive stays
    resume-loadable.
    """
    path = pathlib.Path(checkpoint_dir)
    destination = _unique_dir(path / artifacts.ARCHIVE_SUBDIR / label)
    destination.mkdir(parents=True, exist_ok=True)
    result = ArchiveResult(destination=str(destination))
    for source in _archive_sources(path):
        try:
            shutil.move(str(source), str(destination / source.name))
            result.moved.append(source.name)
        except OSError as error:
            result.errors.append(f"{source.name}: {_os_error_text(error)}")
    return result


def clear_run(checkpoint_dir: str) -> list[str]:
    """Delete the run's artifacts (the destructive overwrite path). Returns the
    names removed; unrelated files are left untouched."""
    removed: list[str] = []
    for source in _archive_sources(pathlib.Path(checkpoint_dir)):
        try:
            source.unlink()
            removed.append(source.name)
        except OSError:
            pass
    return removed


###### PRIVATE #######


def _with_era(cfg: config.RunConfig, era: str) -> config.RunConfig:
    """``cfg`` re-keyed at ``era`` — ``cfg`` itself when already there, so the
    common no-op path skips a full re-validation."""
    if cfg.encoding_version == era:
        return cfg
    return config.with_encoding_version(cfg, era)


def _load_payload(path: pathlib.Path) -> dict[str, typing.Any] | None:
    """Deserialize a checkpoint to CPU, or ``None`` if it is unreadable. Our own
    checkpoints carry a config + progress dict, not just tensors, so the full
    (non weights-only) unpickler is required."""
    try:
        return typing.cast(
            "dict[str, typing.Any]",
            torch.load(path, map_location="cpu", weights_only=False),
        )
    except Exception:  # noqa: BLE001 — a corrupt/unreadable checkpoint is reported
        return None


def _fill_from_payload(summary: RunSummary, payload: dict[str, typing.Any]) -> None:
    """Populate ``summary`` from a parsed checkpoint payload.

    The saved config is rehydrated at the payload's own artifact era
    (``train_config_from_artifact``), so an era-pinned run's summary carries
    the era it actually trained at — the configurator then seeds the working
    config from it and the run reads RESUMABLE under newer code."""
    raw_config = payload.get("config")
    if raw_config is None:
        summary.config_invalid = True
        summary.note = "checkpoint has no embedded config"
    else:
        artifact_version = str(payload.get("version", version.PRE_VERSIONING_VERSION))
        try:
            summary.train_config = config.run_config_from_artifact(
                raw_config, artifact_version
            )
        except pydantic.ValidationError:
            summary.config_invalid = True
            summary.note = "saved config could not be parsed"
    progress = _progress_from_payload(payload)
    summary.iteration = progress.iteration
    summary.total_games = progress.total_games
    summary.best_win_rate = progress.best_win_rate
    summary.opponent_generation = progress.opponent_generation
    sha = payload.get("git_sha")
    summary.git_sha = sha if isinstance(sha, str) else None


def _progress_from_payload(payload: dict[str, typing.Any]) -> runstate.RunProgress:
    """The resumable progress snapshot in a checkpoint; zeroed counters when the
    snapshot is missing or malformed (``inspect_run`` never raises)."""
    raw = payload.get("progress")
    if raw is None:
        return runstate.RunProgress()
    try:
        return runstate.RunProgress.model_validate(raw)
    except pydantic.ValidationError:
        return runstate.RunProgress()


def _archive_sources(path: pathlib.Path) -> list[pathlib.Path]:
    """The run's artifacts in crash-survivable relocation order (``last.pt``
    last). Logs, stale temp files, and the dated process records are swept first,
    then the history logs, model descriptor, and best / opponent snapshots, then
    the resumable head."""
    sources: list[pathlib.Path] = []
    sources.extend(sorted(path.glob(artifacts.LOG_GLOB)))
    sources.extend(sorted(path.glob(artifacts.TMP_GLOB)))
    sources.extend(sorted(path.glob(artifacts.PROCESS_GLOB)))
    sources.extend(sorted(path.glob(artifacts.RUN_CONFIG_GLOB)))
    sources.extend(path / name for name in _SWEEP_BEFORE_LAST)
    sources.append(path / artifacts.LAST_CKPT)
    # De-dup (a glob could re-list a named artifact) while preserving order, and
    # drop anything not actually present.
    seen: set[pathlib.Path] = set()
    ordered: list[pathlib.Path] = []
    for source in sources:
        if source not in seen and source.exists():
            seen.add(source)
            ordered.append(source)
    return ordered


def _unique_dir(base: pathlib.Path) -> pathlib.Path:
    """``base`` if free, else ``base-1`` / ``base-2`` / … so labels never merge."""
    if not base.exists():
        return base
    for index in range(1, _MAX_LABEL_SUFFIX):
        candidate = base.parent / f"{base.name}-{index}"
        if not candidate.exists():
            return candidate
    return base


def _sanitize(name: str) -> str:
    """Reduce a run name to a filesystem-safe archive-label fragment."""
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return cleaned.strip("_") or "run"


def _os_error_text(error: OSError) -> str:
    return error.strerror or str(error)
