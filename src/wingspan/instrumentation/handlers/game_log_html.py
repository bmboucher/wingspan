"""A handler that records each game as a navigable HTML log viewer.

On every game it snapshots the full game state at each phase boundary and, at
``game_end``, writes a self-contained HTML file rendered by
:mod:`wingspan.reporting.game_log_html`.

Phase/segment alignment: each phase capture must fire at the same code point as
the ``=== ... ===`` log-section header that opens its segment so that
``zip(phases, segments)`` correctly assigns each segment's narration to the
phase whose state it describes.

  game_start    → "=== GAME START ==="             (always)
  setup_start   → "=== SETUP: P0 CHOOSING ... ===" (always, one per player)
  setup_applied → "=== SETUP: P0 CHOOSING BONUS CARD ===" (deferred-bonus path
                   only; non-deferred path fires setup_applied without creating
                   a new segment, so this handler skips phase creation for it)
  round_start   → "=== ROUND N ... ==="
  turn_start    → "=== P0, ROUND N, TURN M ... ==="
  game_end      → "=== GAME END ==="

The handler also subscribes to ``MADE_DECISION`` when the CLI injects
``ValueProbe`` objects (one per seat) via :meth:`configure_timeline`. Each
``made_decision`` call appends a
:class:`~wingspan.reporting.game_log_capture.RawTimelinePoint` to
``_raw_timeline``; at ``game_end``,
:func:`~wingspan.reporting.game_log_capture.build_timeline` finalizes the
timestamps and computes the chart coordinates. Without injected probes, the
timeline is score-only (no value/target lines).

The actual state→model conversion and narration slicing live in
:mod:`wingspan.reporting.game_log_capture`, imported lazily inside the event
methods to avoid the ``engine`` ↔ ``instrumentation`` import cycle.
"""

from __future__ import annotations

import pathlib
import typing

import pydantic

from wingspan.instrumentation import events, registry

if typing.TYPE_CHECKING:
    from wingspan import cards, decisions, state
    from wingspan.engine import core
    from wingspan.instrumentation import config
    from wingspan.players import value_sink
    from wingspan.reporting import game_log_capture, game_log_html
    from wingspan.training import config as train_config


@registry.register("GameLogHtml")
class GameLogHtmlHandler(
    events.GameStartHandler,
    events.SetupStartHandler,
    events.SetupAppliedHandler,
    events.RoundStartHandler,
    events.TurnStartHandler,
    events.MadeDecisionHandler,
    events.GameEndHandler,
):
    """Capture per-phase state snapshots and write one HTML log file per game.

    ``output_path`` is resolved against the run's output directory; when
    ``index_suffix`` is set the game index is inserted before the ``.html``
    extension (``log.html`` -> ``log.0.html``) so a multi-game series writes one
    file per game.

    Call :meth:`configure_timeline` after construction to inject per-seat
    ``ValueProbe`` objects and ``TrainConfig`` instances; without them the
    timeline chart shows scores only (value/target lines are omitted)."""

    output_path: str
    index_suffix: bool = False

    _phases: list[game_log_html.PhaseRecord] = pydantic.PrivateAttr(
        default_factory=list["game_log_html.PhaseRecord"]
    )
    _output_dir: pathlib.Path = pydantic.PrivateAttr(default_factory=pathlib.Path)
    _seed: int | None = pydantic.PrivateAttr(default=None)
    _matchup: tuple[str, str] | None = pydantic.PrivateAttr(default=None)
    _game_index: int = pydantic.PrivateAttr(default=0)
    _raw_timeline: list[game_log_capture.RawTimelinePoint] = pydantic.PrivateAttr(
        default_factory=list
    )
    _seat_configs: tuple[
        train_config.TrainConfig | None, train_config.TrainConfig | None
    ] = pydantic.PrivateAttr(default=(None, None))
    _probes: tuple[
        value_sink.ValueProbe | None, value_sink.ValueProbe | None
    ] = pydantic.PrivateAttr(default=(None, None))

    # ----- lifecycle ------------------------------------------------------

    def open(self, context: config.RunContext) -> None:
        self._output_dir = context.output_dir
        self._seed = context.seed
        self._matchup = context.matchup

    # ----- capture events -------------------------------------------------

    def game_start(self, *, engine: core.Engine) -> None:
        self._phases = []
        self._capture(engine, title="Game start", kind="game_start", active=None)

    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        from wingspan.reporting import game_log_capture

        self._phases.append(
            game_log_capture.capture_setup_start_phase(
                engine,
                index=len(self._phases),
                title=f"{player.name} — Choose hand",
                kind="setup_start",
                active=player.id,
                dealt_bonus=dealt_bonus,
            )
        )

    def setup_applied(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        choice: decisions.SetupChoice,
    ) -> None:
        # The non-deferred bonus path fires setup_applied with choice.bonus_card
        # set — setup_start already created the phase for that segment, so adding
        # another here would shift the zip(phases, segments) alignment off by one.
        # Only create a phase for the deferred path (choice.bonus_card is None),
        # which fires setup_applied at the start of the "CHOOSING BONUS CARD" segment.
        if choice.bonus_card is not None:
            return
        self._capture(
            engine, title=f"Setup — {player.name}", kind="setup", active=player.id
        )

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        self._capture(
            engine, title=f"Round {round_num + 1} begins", kind="round", active=None
        )

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        round_cubes = _round_cubes(engine.state.round_idx)
        turn_number = round_cubes - player.action_cubes_left + 1
        self._capture(
            engine,
            title=(
                f"{player.name} — Round {engine.state.round_idx + 1}, "
                f"Turn {turn_number}"
            ),
            kind="turn",
            active=player.id,
        )

    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
        from wingspan import decisions as decisions_module
        from wingspan.engine import scoring
        from wingspan.reporting import game_log_capture
        from wingspan.training import timestamps

        probe = self._probes[decision.player_id]
        value_pov = probe.take() if probe is not None else None
        gs = engine.state
        score_p0 = scoring.running_score(gs.players[0])
        score_p1 = scoring.running_score(gs.players[1])
        margin = (
            float(score_p0 - score_p1)
            if decision.player_id == 0
            else float(score_p1 - score_p0)
        )
        self._raw_timeline.append(
            game_log_capture.RawTimelinePoint(
                player_id=decision.player_id,
                margin_before=margin,
                provisional_timestamp=timestamps.provisional_timestamp(
                    decision, gs.turn_counter
                ),
                family_idx=decisions_module.family_index_for(type(decision)),
                score_p0=score_p0,
                score_p1=score_p1,
                phase_index=len(self._phases) - 1,
                value_pov=value_pov,
            )
        )

    def game_end(self, *, engine: core.Engine) -> None:
        from wingspan.reporting import game_log_capture, game_log_html

        self._capture(engine, title="Final scoring", kind="game_end", active=None)
        timeline = game_log_capture.build_timeline(
            engine=engine,
            raw_points=self._raw_timeline,
            seat_configs=self._seat_configs,
        )
        report = game_log_capture.build_report(
            engine=engine,
            phases=self._phases,
            seed=self._seed,
            matchup=self._matchup,
            timeline=timeline,
        )
        game_log_html.write_game_log_html(report, self._resolve_path())
        self._game_index += 1
        self._phases = []
        self._raw_timeline = []

    def configure_timeline(
        self,
        seat_configs: tuple[
            train_config.TrainConfig | None, train_config.TrainConfig | None
        ],
        probes: tuple[
            value_sink.ValueProbe | None, value_sink.ValueProbe | None
        ],
    ) -> None:
        """Inject per-seat configs and value probes for timeline chart support.

        Must be called before any game starts. Without this call the timeline
        shows score lines only (value/target lines require the probes)."""
        self._seat_configs = seat_configs
        self._probes = probes

    ###### PRIVATE #######

    def _capture(
        self, engine: core.Engine, *, title: str, kind: str, active: int | None
    ) -> None:
        """Snapshot the current game state as a narration-less phase record."""
        from wingspan.reporting import game_log_capture

        self._phases.append(
            game_log_capture.capture_phase(
                engine,
                index=len(self._phases),
                title=title,
                kind=kind,
                active=active,
            )
        )

    def _resolve_path(self) -> pathlib.Path:
        """The output path for the current game, suffixed by game index when a
        series writes more than one file."""
        path = self._output_dir / self.output_path
        if not self.index_suffix:
            return path
        suffix = path.suffix or ".html"
        return path.with_name(f"{path.stem}.{self._game_index}{suffix}")


def _round_cubes(round_idx: int) -> int:
    """Action cubes each player starts a round with — read lazily from ``state``
    to keep ``engine``/``state`` off this module's import-time path."""
    from wingspan import state as state_module

    return state_module.ROUND_CUBES[round_idx]
