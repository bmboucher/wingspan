"""A handler that tallies how often each bird is played, per game.

Demonstrates a multi-event handler with state shared across events: it
accumulates a per-bird count on every ``bird_placed`` and flushes one JSONL row
(the whole tally) on ``game_end``, then resets for the next game. The same
instance is assigned to both events in the config, so the count survives between
them.
"""

from __future__ import annotations

import json
import typing

import pydantic

from wingspan.instrumentation import events, registry

if typing.TYPE_CHECKING:
    from wingspan import cards, state
    from wingspan.engine import core
    from wingspan.instrumentation import config


@registry.register("CardVisitRecorder")
class CardVisitRecorder(events.BirdPlacedHandler, events.GameEndHandler):
    """Count birds played per game; write one tally row per game to
    ``output_path``."""

    output_path: str

    _counts: dict[str, int] = pydantic.PrivateAttr(default_factory=dict[str, int])
    _file: typing.TextIO | None = pydantic.PrivateAttr(default=None)

    def open(self, context: config.RunContext) -> None:
        self._file = (context.output_dir / self.output_path).open("a", encoding="utf-8")

    def bird_placed(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        bird: cards.Bird,
        habitat: cards.Habitat,
        played_bird: state.PlayedBird,
    ) -> None:
        self._counts[bird.name] = self._counts.get(bird.name, 0) + 1

    def game_end(self, *, engine: core.Engine) -> None:
        if self._file is not None:
            self._file.write(json.dumps(self._counts) + "\n")
            self._file.flush()
        self._counts = {}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
