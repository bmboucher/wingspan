"""Seat players from CLI specs: the spec grammar, checkpoint loaders, factory.

This package is the shared "seat a player" machinery behind the game-running
CLIs (``wingspan play`` and ``wingspan tournament``):

- ``spec``    — the unified player-spec grammar (``human`` / ``random`` /
  named checkpoint / ``.pt`` path / run directory)
- ``loaders`` — the two self-describing checkpoint load paths (embedded
  ``TrainConfig`` and run-dir ``model_config.json`` descriptor) plus the
  encoding-compatibility keys, with the artifact-version check enforced
- ``factory`` — spec → Agent construction and opening-bonus regime resolution

It is deliberately separate from the lean, torch-free ``wingspan.agents``
package: seating a trained model pulls in torch and the training stack.
"""

from wingspan.players.factory import (
    build_agent,
    resolve_split_setup_bonus,
    resolve_split_setup_food,
)
from wingspan.players.loaders import (
    descriptor_encoding_key,
    encoding_key,
    expected_encoding_key,
    load_policy_net,
    load_policy_net_from_run_dir,
    load_setup_net,
)
from wingspan.players.spec import (
    NAMED_SPECS,
    PlayerKind,
    PlayerSpec,
    parse_player_spec,
)

__all__ = [
    "NAMED_SPECS",
    "PlayerKind",
    "PlayerSpec",
    "build_agent",
    "descriptor_encoding_key",
    "encoding_key",
    "expected_encoding_key",
    "load_policy_net",
    "load_policy_net_from_run_dir",
    "load_setup_net",
    "parse_player_spec",
    "resolve_split_setup_bonus",
    "resolve_split_setup_food",
]
