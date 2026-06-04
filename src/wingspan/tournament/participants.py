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

from wingspan import agents, decisions, encode, engine, model
from wingspan.training import artifacts, policy, runmeta
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
    net = _reconstruct_net(spec.checkpoint_dir)
    state_dict = typing.cast("dict[str, torch.Tensor]", payload["model"])
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return policy.greedy_agent(net, device)


###### PRIVATE #######


def _reconstruct_net(checkpoint_dir: str) -> model.PolicyValueNet:
    """Rebuild the (fresh-weight) net in a run's saved shape, ready for
    ``load_state_dict``, from the run's ``model_config.json`` descriptor (every
    run dir carries one; a run without it is not seatable). Raises ``ValueError``
    when the saved encoding no longer matches the live encoder — freshly-encoded
    states could not feed the net — so an explicitly-named stale run fails at
    seating with a clear message instead of a mid-game tensor-shape error."""
    descriptor = runmeta.read_model_config(checkpoint_dir)
    saved = _descriptor_encoding_key(descriptor)
    live = _live_encoding_key(
        encode.EncodingSpec(include_setup=descriptor.include_setup)
    )
    if saved != live:
        raise ValueError(
            f"Run {checkpoint_dir!r} was trained against an encoding layout "
            "incompatible with the current code:\n"
            f"  saved: {saved}\n"
            f"  live:  {live}\n"
            "It cannot consume freshly-encoded states and is not seatable."
        )
    return model.PolicyValueNet.from_model_config(descriptor)


def _loadable(checkpoint_dir: str, summary: runs.RunSummary) -> bool:
    """Whether a run dir can be played: a readable ``last.pt``, the
    ``model_config.json`` descriptor :func:`_reconstruct_net` rebuilds the net
    from, and a saved encoding matching the live encoder (a stale-dim run
    cannot consume freshly-encoded states, so it is not offered)."""
    if not (summary.exists and summary.readable):
        return False
    return _encoding_compatible(checkpoint_dir)


def _encoding_compatible(checkpoint_dir: str) -> bool:
    """Whether the run's saved encoding descriptor matches the live encoder,
    mirroring ``selfplay._encoding_key``: the ``(state_dim, choice_dim,
    family_order)`` triple must agree for freshly-encoded inputs to feed the
    run's net. The descriptor's own ``include_setup`` selects which spec the
    live dims are computed for. A missing or unparseable descriptor returns
    ``False`` (the run is simply not seatable); never raises."""
    try:
        descriptor = runmeta.read_model_config(checkpoint_dir)
    except (OSError, pydantic.ValidationError):
        return False
    return _descriptor_encoding_key(descriptor) == _live_encoding_key(
        encode.EncodingSpec(include_setup=descriptor.include_setup)
    )


def _descriptor_encoding_key(
    descriptor: runmeta.ModelConfig,
) -> tuple[int, int, tuple[str, ...]]:
    """The encoding-compatibility signature a run was trained against."""
    return (descriptor.state_dim, descriptor.choice_dim, descriptor.family_order)


def _live_encoding_key(
    spec: encode.EncodingSpec,
) -> tuple[int, int, tuple[str, ...]]:
    """The live encoder's encoding-compatibility signature for ``spec``, built
    exactly the way ``config.TrainConfig._sync_encoding_dims`` derives it."""
    family_order = tuple(
        family.value
        for family in decisions.active_decision_families(spec.include_setup)
    )
    return (encode.state_size(spec), encode.choice_feature_dim(spec), family_order)


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
