"""Round-robin tournament between trained Wingspan AIs.

Pit N trained models (plus, optionally, the random agent) against each other in
a round-robin: every pair plays the same number of games with each competitor
going first equally often, live Elo ratings update on a ``rich`` dashboard as
games complete, and a JSON report records per-pair win rates and point margins
(as first / second player and overall) plus each competitor's final Elo. Games
fan out across worker processes so a full tournament finishes quickly.

The public surface is re-exported here:

- ``run_tournament`` / ``TournamentConfig`` — play a configured tournament
- ``ParticipantSpec`` / ``discover_runs`` — competitors and on-disk run discovery
- ``TournamentReport`` — the aggregated result (the JSON root)
- ``main`` — the ``wingspan-tournament`` CLI entry point
"""

from wingspan.tournament.app import main
from wingspan.tournament.models import (
    ParticipantSpec,
    TournamentConfig,
    TournamentReport,
)
from wingspan.tournament.participants import discover_runs
from wingspan.tournament.runner import run_tournament

__all__ = [
    "main",
    "run_tournament",
    "TournamentConfig",
    "ParticipantSpec",
    "discover_runs",
    "TournamentReport",
]
