"""Tournament competitors: their specs, on-disk discovery, and agent loading.

A competitor is either a trained model (a checkpoint directory holding
``last.pt`` + ``model_config.json``) or the built-in random agent. The same
:class:`ParticipantSpec` describes both; :func:`load_player` turns one into an
:class:`engine.Agent` the runner seats in a game.

Discovery (:func:`discover_runs`) reuses the configurator's
:func:`runs.inspect_run` to enumerate the active run plus every archived run
under a base checkpoint directory, so the picker and the CLI offer the exact
same "trained models on disk" the FLIGHT PLAN configurator manages.
"""

from __future__ import annotations

import enum
import pathlib
import random
import typing

import pydantic
import torch

from wingspan import agents, engine, model
from wingspan.training import artifacts
from wingspan.training import config as train_config
from wingspan.training import policy, runmeta
from wingspan.training.configure import runs


class ParticipantKind(enum.StrEnum):
    """Whether a competitor is a trained model or the random agent."""

    MODEL = "model"
    RANDOM = "random"


class ParticipantSpec(pydantic.BaseModel):
    """One tournament competitor, identified by a stable ``id``.

    ``checkpoint_dir`` is the run directory for a ``MODEL`` and ``None`` for the
    ``RANDOM`` agent. Frozen so a spec can be a dict key and shipped to worker
    processes as immutable pool ``initargs``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    display_name: str
    kind: ParticipantKind
    checkpoint_dir: str | None = None


class RunOption(pydantic.BaseModel):
    """One discoverable trained run, as shown to the user in the picker."""

    checkpoint_dir: str
    display_name: str
    iteration: int | None = None
    best_win_rate: float | None = None
    modified: float | None = None

    def to_spec(self) -> ParticipantSpec:
        """The competitor spec for selecting this run as a model player."""
        return ParticipantSpec(
            id=self.display_name,
            display_name=self.display_name,
            kind=ParticipantKind.MODEL,
            checkpoint_dir=self.checkpoint_dir,
        )


def random_spec() -> ParticipantSpec:
    """The spec for the built-in random agent competitor."""
    return ParticipantSpec(
        id="random", display_name="random", kind=ParticipantKind.RANDOM
    )


def discover_runs(base_dir: str) -> list[RunOption]:
    """Every selectable trained run under ``base_dir``: the active run plus each
    archived run (``<base_dir>/archive/<label>/``) that still holds a loadable
    checkpoint. Reuses :func:`runs.inspect_run`, which never raises — an
    unreadable run is simply skipped."""
    options: list[RunOption] = []
    summary = runs.inspect_run(base_dir)
    if _loadable(base_dir, summary):
        options.append(_option_from_summary(base_dir, summary))
    for entry in summary.archives:
        if not entry.has_checkpoint:
            continue
        archive_dir = str(
            pathlib.Path(base_dir) / artifacts.ARCHIVE_SUBDIR / entry.label
        )
        archive_summary = runs.inspect_run(archive_dir)
        if _loadable(archive_dir, archive_summary):
            options.append(
                _option_from_summary(archive_dir, archive_summary, label=entry.label)
            )
    return options


def spec_from_dir(checkpoint_dir: str) -> ParticipantSpec:
    """Build a model competitor spec for an explicit checkpoint dir (the
    ``--ai <dir>`` path). The display name is the run's name + iteration when the
    checkpoint is readable, else the directory's own name."""
    summary = runs.inspect_run(checkpoint_dir)
    if summary.exists and summary.readable:
        return _option_from_summary(checkpoint_dir, summary).to_spec()
    fallback = pathlib.Path(checkpoint_dir).name or checkpoint_dir
    return ParticipantSpec(
        id=fallback,
        display_name=fallback,
        kind=ParticipantKind.MODEL,
        checkpoint_dir=checkpoint_dir,
    )


def with_unique_ids(specs: typing.Sequence[ParticipantSpec]) -> list[ParticipantSpec]:
    """Return ``specs`` with any duplicate ``id`` disambiguated by a ``#N`` suffix
    (two runs can share a run name), so every competitor keys uniquely in the
    schedule, ELO table, and report."""
    seen: dict[str, int] = {}
    unique: list[ParticipantSpec] = []
    for spec in specs:
        count = seen.get(spec.id, 0) + 1
        seen[spec.id] = count
        if count == 1:
            unique.append(spec)
        else:
            new_id = f"{spec.id}#{count}"
            unique.append(
                spec.model_copy(update={"id": new_id, "display_name": new_id})
            )
    return unique


def load_player(
    spec: ParticipantSpec, device: torch.device, rng: random.Random
) -> engine.Agent:
    """Build the :class:`engine.Agent` for a competitor.

    A ``MODEL`` is reconstructed from its ``model_config.json`` descriptor, loaded
    with its ``last.pt`` weights, and wrapped as a greedy (argmax) strength agent.
    A ``RANDOM`` competitor returns a uniform-random agent seeded from ``rng`` so
    the caller controls its reproducibility per game.
    """
    if spec.kind is ParticipantKind.RANDOM:
        return agents.random_agent(rng)
    assert spec.checkpoint_dir is not None, "a MODEL competitor needs a checkpoint dir"
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(
            pathlib.Path(spec.checkpoint_dir) / artifacts.LAST_CKPT,
            map_location=device,
            weights_only=False,
        ),
    )
    net = _reconstruct_net(spec.checkpoint_dir, payload)
    state_dict = typing.cast("dict[str, torch.Tensor]", payload["model"])
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return policy.greedy_agent(net, device)


###### PRIVATE #######


def _reconstruct_net(
    checkpoint_dir: str, payload: dict[str, typing.Any]
) -> model.PolicyValueNet:
    """Rebuild the (fresh-weight) net in a run's saved shape, ready for
    ``load_state_dict``.

    Prefers the ``model_config.json`` descriptor; falls back to the full
    :class:`TrainConfig` embedded in the checkpoint when the descriptor is absent
    or predates a field (older runs), mirroring the training resume path so even
    pre-descriptor checkpoints play."""
    try:
        descriptor = runmeta.read_model_config(checkpoint_dir)
        return model.PolicyValueNet.from_model_config(descriptor)
    except (FileNotFoundError, pydantic.ValidationError):
        cfg = train_config.TrainConfig.model_validate(payload["config"])
        return model.PolicyValueNet(
            state_dim=cfg.state_dim,
            choice_dim=cfg.choice_dim,
            num_families=len(cfg.family_order),
            arch=cfg.arch,
            spec=cfg.encoding_spec,
        )


def _loadable(checkpoint_dir: str, summary: runs.RunSummary) -> bool:
    """Whether a run dir can be played: a readable ``last.pt`` plus either a
    ``model_config.json`` descriptor or the embedded :class:`TrainConfig` that
    :func:`_reconstruct_net` rebuilds the net from."""
    if not (summary.exists and summary.readable):
        return False
    has_descriptor = (
        pathlib.Path(checkpoint_dir) / artifacts.MODEL_CONFIG_JSON
    ).exists()
    return has_descriptor or summary.train_config is not None


def _option_from_summary(
    checkpoint_dir: str, summary: runs.RunSummary, *, label: str | None = None
) -> RunOption:
    """Build a :class:`RunOption` from an inspected run. Archived runs display by
    their archive label; the active run displays by its run name."""
    run_name = (
        summary.train_config.run_name if summary.train_config is not None else "run"
    )
    return RunOption(
        checkpoint_dir=checkpoint_dir,
        display_name=label if label is not None else run_name,
        iteration=summary.iteration,
        best_win_rate=summary.best_win_rate,
        modified=summary.modified,
    )
