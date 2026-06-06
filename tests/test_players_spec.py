"""Tests for the unified player-spec grammar (``wingspan.players.spec``).

``--p0`` / ``--p1`` accept ``human``, ``random``, a named checkpoint
(``last`` / ``best`` / ``opponent``), a path to a ``.pt`` file, or a run
directory; each must resolve to the right kind, checkpoint path, and
setup-net run directory.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import players  # noqa: E402
from wingspan.training import artifacts  # noqa: E402


def test_human_and_random_specs_have_no_paths():
    """The built-in agents carry no checkpoint or run directory."""
    for raw, kind in (
        ("human", players.PlayerKind.HUMAN),
        ("random", players.PlayerKind.RANDOM),
    ):
        player_spec = players.parse_player_spec(raw, pathlib.Path("checkpoints"))
        assert player_spec.kind is kind
        assert player_spec.raw == raw
        assert player_spec.checkpoint_path is None
        assert player_spec.run_dir is None


def test_named_specs_resolve_against_checkpoint_dir():
    """``last`` / ``best`` / ``opponent`` name artifacts under the checkpoint
    directory, which also provides the setup net."""
    checkpoint_dir = pathlib.Path("some") / "run"
    for name, filename in players.NAMED_SPECS.items():
        player_spec = players.parse_player_spec(name, checkpoint_dir)
        assert player_spec.kind is players.PlayerKind.MODEL
        assert player_spec.checkpoint_path == checkpoint_dir / filename
        assert player_spec.run_dir == checkpoint_dir


def test_run_directory_spec_seats_its_last_checkpoint(tmp_path: pathlib.Path):
    """A spec naming an existing directory seats that run's ``last.pt`` and
    resolves the setup net from the same directory."""
    run_dir = tmp_path / "archived-run"
    run_dir.mkdir()
    player_spec = players.parse_player_spec(str(run_dir), pathlib.Path("checkpoints"))
    assert player_spec.kind is players.PlayerKind.MODEL
    assert player_spec.checkpoint_path == run_dir / artifacts.LAST_CKPT
    assert player_spec.run_dir == run_dir


def test_pt_path_spec_uses_its_own_directory_for_the_setup_net(
    tmp_path: pathlib.Path,
):
    """A direct ``.pt`` path is loaded as-is; the optional setup net resolves
    from the checkpoint's own directory, not the default checkpoint dir."""
    ckpt_path = tmp_path / "exp" / "best.pt"
    player_spec = players.parse_player_spec(str(ckpt_path), pathlib.Path("checkpoints"))
    assert player_spec.kind is players.PlayerKind.MODEL
    assert player_spec.checkpoint_path == ckpt_path
    assert player_spec.run_dir == ckpt_path.parent
