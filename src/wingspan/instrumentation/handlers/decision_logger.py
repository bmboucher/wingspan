"""A handler that logs every genuine decision as one JSONL row.

Records, per decision: the round, the decision class (which maps to a judgment
family), the deciding seat, the number of legal options, and the chosen option's
label. A minimal worked example of a single-event handler and the
``open`` / ``close`` file lifecycle.
"""

from __future__ import annotations

import json
import typing

import pydantic

from wingspan.instrumentation import events, registry

if typing.TYPE_CHECKING:
    from wingspan import decisions
    from wingspan.engine import core
    from wingspan.instrumentation import config


@registry.register("DecisionLogger")
class DecisionLogger(events.MadeDecisionHandler):
    """Append one JSONL row per genuine decision to ``output_path``."""

    output_path: str

    _file: typing.TextIO | None = pydantic.PrivateAttr(default=None)

    def open(self, context: config.RunContext) -> None:
        self._file = (context.output_dir / self.output_path).open("a", encoding="utf-8")

    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
        if self._file is None:
            return
        row = {
            "round": engine.state.round_idx,
            "decision": type(decision).__name__,
            "player_id": decision.player_id,
            "n_choices": len(decision.choices),
            "chosen": choice.display_label(),
        }
        self._file.write(json.dumps(row) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
