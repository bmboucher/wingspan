"""Tournament competitors: on-disk discovery and agent loading.

A competitor is either a trained model (a checkpoint directory holding
``last.pt`` plus its config descriptor — ``run_config_<stamp>.json`` for ≥0.5
runs, legacy ``model_config.json`` for ≤0.4) or the built-in random agent. The data
shapes (:class:`~models.ParticipantSpec`, :class:`~models.RunOption`,
:class:`~models.ParticipantKind`) live in :mod:`models`; this module provides
the operational functions that turn those specs into live agents and discover
available runs on disk.

Discovery (:func:`discover_runs`) reuses the configurator's
:func:`runs.inspect_run` to enumerate the active run plus every archived run
under a base checkpoint directory, so the picker and the CLI offer the exact
same "trained models on disk" the FLIGHT PLAN configurator manages.

:func:`resolve_regime_flags` derives the setup/food engine regimes the games
run under from the competitors' training configs (via the same
``players.resolve_*`` functions ``wingspan play`` uses), so tournament games
mirror how the nets were trained and mixed-regime fields are refused up front.
"""

from __future__ import annotations

import pathlib
import random
import typing

import pydantic
import torch

from wingspan import agents, engine, players, version
from wingspan.tournament import models
from wingspan.training import artifacts, config, policy, runmeta
from wingspan.training.configure import runs


def random_spec() -> models.ParticipantSpec:
    """The spec for the built-in random agent competitor."""
    return models.ParticipantSpec(
        id="random", display_name="random", kind=models.ParticipantKind.RANDOM
    )


def discover_runs(base_dir: str) -> list[models.RunOption]:
    """Every selectable trained run under ``base_dir``: the active run plus each
    archived run (``<base_dir>/archive/<label>/``) that still holds a loadable
    checkpoint. Reuses :func:`runs.inspect_run`, which never raises — an
    unreadable run is simply skipped."""
    options: list[models.RunOption] = []
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


def spec_from_dir(checkpoint_dir: str) -> models.ParticipantSpec:
    """Build a model competitor spec for an explicit checkpoint dir (the
    ``--ai <dir>`` path). The display name is the run's name + iteration when the
    checkpoint is readable, else the directory's own name."""
    summary = runs.inspect_run(checkpoint_dir)
    if summary.exists and summary.readable:
        return _option_from_summary(checkpoint_dir, summary).to_spec()
    fallback = pathlib.Path(checkpoint_dir).name or checkpoint_dir
    return models.ParticipantSpec(
        id=fallback,
        display_name=fallback,
        kind=models.ParticipantKind.MODEL,
        checkpoint_dir=checkpoint_dir,
    )


def with_unique_ids(
    specs: typing.Sequence[models.ParticipantSpec],
) -> list[models.ParticipantSpec]:
    """Return ``specs`` with any duplicate ``id`` disambiguated by a ``#N`` suffix
    (two runs can share a run name), so every competitor keys uniquely in the
    schedule, ELO table, and report."""
    seen: dict[str, int] = {}
    unique: list[models.ParticipantSpec] = []
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
    spec: models.ParticipantSpec, device: torch.device, rng: random.Random
) -> engine.Agent:
    """Build the :class:`engine.Agent` for a competitor.

    A ``MODEL`` is loaded through the shared run-dir loader
    (:func:`players.load_policy_net_from_run_dir`: descriptor-reconstructed,
    version-checked) and wrapped as a greedy (argmax) strength agent. A
    ``RANDOM`` competitor returns a uniform-random agent seeded from ``rng`` so
    the caller controls its reproducibility per game.
    """
    if spec.kind is models.ParticipantKind.RANDOM:
        return agents.random_agent(rng)
    assert spec.checkpoint_dir is not None, "a MODEL competitor needs a checkpoint dir"
    net = players.load_policy_net_from_run_dir(spec.checkpoint_dir, device)
    return policy.greedy_agent(net, device)


def resolve_regime_flags(
    specs: typing.Sequence[models.ParticipantSpec],
) -> models.RegimeFlags:
    """The setup/food engine regimes the tournament's games must run under,
    derived from every model competitor's stored :class:`~config.RunConfig` so
    games mirror how the nets were trained — the tournament-wide analogue of the
    per-matchup resolution ``wingspan play`` performs (the shared
    ``players.resolve_*`` functions). ``RANDOM`` competitors carry no config and
    express no preference; an all-random field resolves to the engine's default
    (all flags off).

    Raises ``ValueError`` when two competitors were trained under different
    regimes: a game runs under exactly one regime, so mixed-regime seats cannot
    share a faithful game and their scores are not comparable — the tournament
    refuses rather than feeding one net decisions in a shape it never saw.
    """
    configs = [_config_for_spec(spec) for spec in specs]
    return models.RegimeFlags(
        split_setup_bonus=players.resolve_split_setup_bonus(configs),
        split_setup_food=players.resolve_split_setup_food(configs),
        combine_gain_food=players.resolve_combine_gain_food(configs),
    )


###### PRIVATE #######


def _config_for_spec(spec: models.ParticipantSpec) -> config.RunConfig | None:
    """The ``RunConfig`` a model competitor was trained under, read from its
    run-dir descriptor via the never-raising :func:`runs.inspect_run`. ``None``
    for the ``RANDOM`` agent (no config) or a model whose config is unreadable
    (it then expresses no regime preference, exactly as a config-free seat)."""
    if spec.kind is not models.ParticipantKind.MODEL:
        return None
    assert spec.checkpoint_dir is not None, "a MODEL competitor needs a checkpoint dir"
    return runs.inspect_run(spec.checkpoint_dir).train_config


def _loadable(checkpoint_dir: str, summary: runs.RunSummary) -> bool:
    """Whether a run dir can be played: a readable ``last.pt``, the config
    descriptor the shared run-dir loader rebuilds the net from (read via the
    dispatching ``runmeta.read_model_config``), and a saved encoding matching the
    live encoder (a stale-dim run cannot consume freshly-encoded states, so it is
    not offered)."""
    if not (summary.exists and summary.readable):
        return False
    return _encoding_compatible(checkpoint_dir)


def _encoding_compatible(checkpoint_dir: str) -> bool:
    """Whether the run's saved encoding descriptor matches the encoder its
    artifact era promises (the ``players.loaders`` encoding keys): the
    ``(state_dim, choice_dim, family_order)`` triple must agree for the net to
    consume the inputs it will be fed. The descriptor's own ``include_setup``
    selects the spec, and its artifact version selects the era (a pre-0.1 run
    is seatable through the ``compat.v0_0`` encoder). A missing, unparseable,
    or version-incompatible descriptor returns ``False`` (the run is simply
    not seatable); never raises."""
    try:
        descriptor = runmeta.read_model_config(checkpoint_dir)
    except (OSError, pydantic.ValidationError, version.IncompatibleArtifactError):
        return False
    return players.descriptor_encoding_key(descriptor) == players.expected_encoding_key(
        descriptor
    )


def _option_from_summary(
    checkpoint_dir: str, summary: runs.RunSummary, *, label: str | None = None
) -> models.RunOption:
    """Build a :class:`~models.RunOption` from an inspected run. Archived runs
    display by their archive label; the active run displays by its run name."""
    run_name = (
        summary.train_config.run.run_name if summary.train_config is not None else "run"
    )
    return models.RunOption(
        checkpoint_dir=checkpoint_dir,
        display_name=label if label is not None else run_name,
        iteration=summary.iteration,
        best_win_rate=summary.best_win_rate,
        modified=summary.modified,
    )
