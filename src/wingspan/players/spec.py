"""The unified player-spec grammar shared by the ``play`` and ``tournament`` CLIs.

A *player spec* is the string a CLI user types to seat one player: ``human``
(the interactive CLI agent), ``random`` (the uniform-random agent), a named
checkpoint (``last`` / ``best`` / ``opponent``, resolved against a checkpoint
directory), a run directory (its ``last.pt`` is seated), or a direct path to a
``.pt`` checkpoint file. :func:`parse_player_spec` turns the string into a
:class:`PlayerSpec`; ``players.factory.build_agent`` turns that into an Agent.
"""

from __future__ import annotations

import enum
import pathlib

import pydantic

from wingspan.training import artifacts

# The named checkpoint specs accepted by ``--p0`` / ``--p1``, mapped to the
# on-disk artifact filenames inside the checkpoint directory. Any other spec
# (besides ``human`` / ``random``) is treated as a filesystem path.
NAMED_SPECS: dict[str, str] = {
    "last": artifacts.LAST_CKPT,
    "best": artifacts.BEST_CKPT,
    "opponent": artifacts.OPPONENT_CKPT,
}


class PlayerKind(enum.StrEnum):
    """How a seat is driven: interactive human, uniform random, or a trained model."""

    HUMAN = "human"
    RANDOM = "random"
    MODEL = "model"


class PlayerSpec(pydantic.BaseModel):
    """One parsed seat spec.

    ``raw`` is the original spec string, kept for display in run headers.
    ``checkpoint_path`` (the ``.pt`` file to load) and ``run_dir`` (the
    directory the optional setup net resolves from) are set only for ``MODEL``
    seats.
    """

    kind: PlayerKind
    raw: str
    checkpoint_path: pathlib.Path | None = None
    run_dir: pathlib.Path | None = None


def parse_player_spec(raw: str, checkpoint_dir: pathlib.Path) -> PlayerSpec:
    """Parse one ``--p0`` / ``--p1`` value into a :class:`PlayerSpec`.

    ``human`` and ``random`` are the built-in agents; ``last`` / ``best`` /
    ``opponent`` name checkpoint artifacts under ``checkpoint_dir``; a path to
    an existing directory seats that run's ``last.pt``; anything else is a
    direct path to a ``.pt`` file, whose own directory provides the optional
    setup net.
    """
    if raw == PlayerKind.HUMAN:
        return PlayerSpec(kind=PlayerKind.HUMAN, raw=raw)
    if raw == PlayerKind.RANDOM:
        return PlayerSpec(kind=PlayerKind.RANDOM, raw=raw)
    if raw in NAMED_SPECS:
        return PlayerSpec(
            kind=PlayerKind.MODEL,
            raw=raw,
            checkpoint_path=checkpoint_dir / NAMED_SPECS[raw],
            run_dir=checkpoint_dir,
        )
    path = pathlib.Path(raw)
    if path.is_dir():
        return PlayerSpec(
            kind=PlayerKind.MODEL,
            raw=raw,
            checkpoint_path=path / artifacts.LAST_CKPT,
            run_dir=path,
        )
    return PlayerSpec(
        kind=PlayerKind.MODEL, raw=raw, checkpoint_path=path, run_dir=path.parent
    )
