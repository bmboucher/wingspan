"""A handler that records each game as a navigable HTML log viewer.

On every game it snapshots the full game state at each phase boundary and, at
``game_end``, writes a self-contained HTML file rendered by
:mod:`wingspan.reporting.game_log_html`.

Phase/segment alignment: each phase capture must fire at the same code point as
the ``=== ... ===`` log-section header that opens its segment so that
``zip(phases, segments)`` correctly assigns each segment's log items to the
phase whose state it describes.

  game_start  → "=== GAME START ==="              (always)
  setup_start → "=== SETUP: P0 CHOOSING ... ===" (always, one per player)
  round_start → "=== ROUND N ... ==="
  turn_start  → "=== P0, ROUND N, TURN M ... ==="
  game_end    → "=== GAME END ==="

In the deferred-bonus regime the engine also emits a secondary
``=== SETUP: P0 CHOOSING BONUS CARD ===`` header per player.  We create only
one combined setup phase per player (at ``setup_start``), so
:func:`~wingspan.reporting.game_log_capture._merge_secondary_setup_segments`
folds that extra segment into the primary before the zip runs.

The handler subscribes to ``MADE_DECISION`` when the CLI injects
``DecisionProbe`` objects (one per seat) via :meth:`configure_timeline`. Each
``made_decision`` call appends a
:class:`~wingspan.reporting.game_log_capture.RawTimelinePoint` to
``_raw_timeline`` (for the Timeline chart). During setup, the decision is also
routed into a per-player ``SetupCaptureState`` bucket rather than the flat
``_decision_items`` list.  At ``game_end``,
:func:`~wingspan.reporting.game_log_capture.finalize_setup_phase` assembles the
grouped setup log, then :func:`~wingspan.reporting.game_log_capture.build_report`
merges remaining decision items with the text log.

The actual state→model conversion lives in
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
    from wingspan.players import decision_probe
    from wingspan.reporting import game_log_capture, game_log_html
    from wingspan.training import config as train_config


@registry.register("GameLogHtml")
class GameLogHtmlHandler(
    events.GameStartHandler,
    events.SetupStartHandler,
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
    ``DecisionProbe`` objects and ``TrainConfig`` instances; without them the
    timeline chart shows scores only and decision boxes show no option bars."""

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
        default_factory=list["game_log_capture.RawTimelinePoint"]
    )
    # (phase_index, LogItem) pairs, one per genuine AI decision outside setup.
    _decision_items: list[tuple[int, game_log_html.LogItem]] = pydantic.PrivateAttr(
        default_factory=list[tuple[int, "game_log_html.LogItem"]]
    )
    # Per-player setup capture bucket; keyed by player_id.
    _setup_captures: dict[int, game_log_capture.SetupCaptureState] = (
        pydantic.PrivateAttr(
            default_factory=dict[int, "game_log_capture.SetupCaptureState"]
        )
    )
    _seat_configs: tuple[
        train_config.TrainConfig | None, train_config.TrainConfig | None
    ] = pydantic.PrivateAttr(default=(None, None))
    _probes: tuple[
        decision_probe.DecisionProbe | None, decision_probe.DecisionProbe | None
    ] = pydantic.PrivateAttr(default=(None, None))

    # ----- lifecycle ------------------------------------------------------

    def open(self, context: config.RunContext) -> None:
        self._output_dir = context.output_dir
        self._seed = context.seed
        self._matchup = context.matchup

    # ----- capture events -------------------------------------------------

    def game_start(self, *, engine: core.Engine) -> None:
        self._phases = []
        self._decision_items = []
        self._setup_captures = {}
        self._capture(engine, title="Game start", kind="game_start", active=None)

    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        from wingspan.reporting import game_log_capture

        phase = game_log_capture.capture_setup_phase(
            engine,
            index=len(self._phases),
            title=f"{player.name} — Setup",
            active=player.id,
            dealt_bonus=dealt_bonus,
        )
        self._phases.append(phase)
        self._setup_captures[player.id] = game_log_capture.SetupCaptureState(
            phase_index=phase.index,
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
        value_pov, annotation = probe.take() if probe is not None else (None, None)

        phase_index = len(self._phases) - 1
        current_kind = self._phases[phase_index].kind if self._phases else ""

        if current_kind == "setup":
            # Route into the per-player capture bucket instead of _decision_items.
            capture = self._setup_captures.get(decision.player_id)
            if capture is not None:
                game_log_capture.record_setup_decision(
                    capture, engine, decision, choice, annotation
                )
        elif annotation is not None:
            decision_item = game_log_capture.build_decision_item(
                engine, decision, choice, annotation
            )
            self._decision_items.append((phase_index, decision_item))

        # Record the timeline point (value + score margin) regardless of phase type.
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
                phase_index=phase_index,
                value_pov=value_pov,
            )
        )

    def game_end(self, *, engine: core.Engine) -> None:
        from wingspan.reporting import game_log_capture, game_log_html

        self._capture(engine, title="Final scoring", kind="game_end", active=None)

        # Finalize each player's setup phase before building the report.
        for phase in self._phases:
            if phase.kind == "setup" and phase.active_player_id is not None:
                capture = self._setup_captures.get(phase.active_player_id)
                if capture is not None:
                    game_log_capture.finalize_setup_phase(phase, capture)

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
            decision_items=self._decision_items,
        )
        game_log_html.write_game_log_html(report, self._resolve_path())
        self._game_index += 1
        self._phases = []
        self._raw_timeline = []
        self._decision_items = []
        self._setup_captures = {}

    def configure_timeline(
        self,
        seat_configs: tuple[
            train_config.TrainConfig | None, train_config.TrainConfig | None
        ],
        probes: tuple[
            decision_probe.DecisionProbe | None, decision_probe.DecisionProbe | None
        ],
    ) -> None:
        """Inject per-seat configs and decision probes.

        Must be called before any game starts. Without this call the timeline
        shows score lines only and decision boxes omit option bars."""
        self._seat_configs = seat_configs
        self._probes = probes

    ###### PRIVATE #######

    def _capture(
        self, engine: core.Engine, *, title: str, kind: str, active: int | None
    ) -> None:
        """Snapshot the current game state as an empty-log-items phase record."""
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
