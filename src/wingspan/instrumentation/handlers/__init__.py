"""Built-in instrumentation handlers.

Importing this package self-populates the handler registry: each submodule
decorates its handler class with ``@registry.register``, and the imports below
run those decorators (mirroring ``engine/powers/__init__``). Project-specific
handlers can register the same way from anywhere they are imported.

- ``decision_logger`` — ``DecisionLogger``: one JSONL row per genuine decision
- ``card_visits``     — ``CardVisitRecorder``: per-game per-bird play tally
- ``game_log_html``   — ``GameLogHtml``: one navigable HTML log viewer per game
"""

from wingspan.instrumentation.handlers import (
    card_visits,
    decision_logger,
    game_log_html,
)

# The submodules are imported for their ``@registry.register`` side effects; the
# reference below keeps the otherwise-unused imports live.
_ = (card_visits, decision_logger, game_log_html)

__all__ = [
    "card_visits",
    "decision_logger",
    "game_log_html",
]
