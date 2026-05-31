"""Per-run JSON metadata sidecars: the model descriptor and the session log.

These complement the binary checkpoints with the two human-readable JSON
artifacts a long run leaves behind in its checkpoint dir, alongside ``last.pt``
and the ``metrics.jsonl`` / ``games.jsonl`` history logs:

* ``model_config.json`` — the weight-compatibility descriptor (network shape +
  judgment-family head order). One per run, rewritten on every startup. If any
  field here changes the previously-saved weights can no longer be trusted, and
  the resume gate (``loop._architecture_matches``) refuses to load them.

* ``process_<stamp>.json`` — one *session's* meta-configuration (the batch size,
  learning rate, eval cadence, device, … — the full :class:`config.TrainConfig`
  in force) plus the runtime context it started in (git SHA, the iteration it
  resumed from, the wall-clock start). A run is stopped and restarted many
  times; each startup drops a fresh dated file, so the sequence of these files
  is the run's session ledger.

The writers are split out of ``loop`` (and kept torch-free) so they can be unit
tested without a training run, mirroring the ``artifacts`` / ``runs`` split. The
caller supplies the timestamp strings so these functions stay pure.
"""

from __future__ import annotations

import pathlib

import pydantic

from wingspan.training import artifacts, config

# Up to this many same-second restarts get a unique ``process_<stamp>-N.json``
# before the writer gives up de-duplicating and overwrites the base name.
_MAX_SESSION_SUFFIX = 1000


class ModelConfig(pydantic.BaseModel):
    """The network-shape descriptor written to ``model_config.json``.

    These shape fields are exactly ``TrainConfig.architecture_key`` — the
    signature two checkpoints must share to be weight-compatible — plus the run
    name for context. A change to any of them invalidates previously-trained
    weights.
    """

    run_name: str
    state_dim: int
    choice_dim: int
    hidden: int
    family_order: tuple[str, ...]
    card_embed_dim: int


class SessionRecord(pydantic.BaseModel):
    """One training session's process record (``process_<stamp>.json``).

    Embeds the full :class:`config.TrainConfig` in force (so the batch sizes and
    every other knob are captured without field drift) plus the runtime context
    the session began in.
    """

    run_name: str
    started_at: str  # ISO-8601 local start time
    git_sha: str | None
    resumed: bool  # whether this session continued an existing run
    resumed_from_iteration: int  # 0 for a fresh start
    config: config.TrainConfig


def write_model_config(checkpoint_dir: str, cfg: config.TrainConfig) -> pathlib.Path:
    """Write (overwriting) ``model_config.json`` for ``cfg`` and return its path."""
    descriptor = ModelConfig(
        run_name=cfg.run_name,
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        hidden=cfg.hidden,
        family_order=cfg.family_order,
        card_embed_dim=cfg.card_embed_dim,
    )
    path = _ensure_dir(checkpoint_dir) / artifacts.MODEL_CONFIG_JSON
    path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_session_record(
    checkpoint_dir: str,
    cfg: config.TrainConfig,
    *,
    stamp: str,
    started_at: str,
    git_sha: str | None,
    resumed_from_iteration: int,
) -> pathlib.Path:
    """Write a fresh dated ``process_<stamp>.json`` for this session and return
    its path. ``stamp`` is a filesystem-safe timestamp supplied by the caller (so
    this stays pure / testable); a same-second collision gets a ``-N`` suffix so
    rapid restarts never clobber a prior session's record."""
    record = SessionRecord(
        run_name=cfg.run_name,
        started_at=started_at,
        git_sha=git_sha,
        resumed=resumed_from_iteration > 0,
        resumed_from_iteration=resumed_from_iteration,
        config=cfg,
    )
    path = _unique_session_path(_ensure_dir(checkpoint_dir), stamp)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


###### PRIVATE #######


def _ensure_dir(checkpoint_dir: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_session_path(directory: pathlib.Path, stamp: str) -> pathlib.Path:
    """``process_<stamp>.json`` if free, else ``…-1`` / ``…-2`` / … so two
    same-second startups never overwrite each other's session record."""
    base = directory / f"{artifacts.PROCESS_PREFIX}{stamp}.json"
    if not base.exists():
        return base
    for index in range(1, _MAX_SESSION_SUFFIX):
        candidate = directory / f"{artifacts.PROCESS_PREFIX}{stamp}-{index}.json"
        if not candidate.exists():
            return candidate
    return base
