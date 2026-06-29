"""A handler that records each game as a navigable HTML log viewer.

On every game it snapshots the full game state at each phase boundary and, at
``game_end``, writes a self-contained HTML file rendered by
:mod:`wingspan.reporting.game_log_html`.

Phase alignment: each phase capture fires at the same code point as the
corresponding ``events.begin_phase`` call in the recorder so that
``zip(handler._phases, engine.events.root.phases)`` correctly pairs each
phase snapshot with its event-tree node.

  game_start  → PhaseNode "game_start"   (begin_game)
  setup_start → PhaseNode "setup"        (begin_phase in _resolve_setup_choice)
  round_start → PhaseNode "round"        (begin_phase in _play_round)
  turn_start  → PhaseNode "turn"         (begin_phase in _take_turn)
  game_end    → PhaseNode "game_end"     (end_game)

At ``game_end`` the handler reads the finished tree from ``engine.events.root``
and calls :func:`~wingspan.reporting.game_log_capture.build_report` to merge
the phase snapshots with the tree's log items. The
:class:`~wingspan.gamelog.recorder.EventRecorder` is the sole
``DecisionProbe`` consumer — this handler does not subscribe to
``MADE_DECISION``.
"""

from __future__ import annotations

import pathlib
import typing

import pydantic

from wingspan.instrumentation import events, registry

if typing.TYPE_CHECKING:
    from wingspan import cards, state
    from wingspan.engine import core
    from wingspan.instrumentation import config
    from wingspan.reporting import game_log_html
    from wingspan.training import config as train_config


@registry.register("GameLogHtml")
class GameLogHtmlHandler(
    events.GameStartHandler,
    events.SetupStartHandler,
    events.RoundStartHandler,
    events.TurnStartHandler,
    events.GameEndHandler,
):
    """Capture per-phase state snapshots and write one HTML log file per game.

    ``output_path`` is resolved against the run's output directory; when
    ``index_suffix`` is set the game index is inserted before the ``.html``
    extension (``log.html`` -> ``log.0.html``) so a multi-game series writes one
    file per game.

    Call :meth:`configure_timeline` after construction to inject per-seat
    ``TrainConfig`` instances; without them the timeline chart shows scores only
    and decision boxes show no option bars."""

    output_path: str
    index_suffix: bool = False

    _phases: list[game_log_html.PhaseRecord] = pydantic.PrivateAttr(
        default_factory=list["game_log_html.PhaseRecord"]
    )
    _output_dir: pathlib.Path = pydantic.PrivateAttr(default_factory=pathlib.Path)
    _seed: int | None = pydantic.PrivateAttr(default=None)
    _matchup: tuple[str, str] | None = pydantic.PrivateAttr(default=None)
    _game_index: int = pydantic.PrivateAttr(default=0)
    _seat_configs: tuple[
        train_config.TrainConfig | None, train_config.TrainConfig | None
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
            game_log_capture.capture_setup_phase(
                engine,
                index=len(self._phases),
                title=f"{player.name} — Setup",
                active=player.id,
                dealt_bonus=dealt_bonus,
            )
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

    def game_end(self, *, engine: core.Engine) -> None:
        from wingspan.gamelog import recorder as gamelog_recorder
        from wingspan.reporting import game_log_capture, game_log_html

        self._capture(engine, title="Final scoring", kind="game_end", active=None)
        rec = engine.events
        assert isinstance(rec, gamelog_recorder.EventRecorder), (
            "GameLogHtmlHandler requires an EventRecorder — "
            "pass event_recorder=gamelog_recorder.EventRecorder(...) to play_one_game"
        )
        tree = rec.root
        timeline_points = game_log_capture.extract_timeline_points(tree)
        timeline = game_log_capture.build_timeline(
            engine=engine,
            raw_points=timeline_points,
            seat_configs=self._seat_configs,
        )
        report = game_log_capture.build_report(
            engine=engine,
            phases=self._phases,
            tree=tree,
            seed=self._seed,
            matchup=self._matchup,
            timeline=timeline,
        )
        game_log_html.write_game_log_html(report, self._resolve_path())
        self._game_index += 1
        self._phases = []

    def configure_timeline(
        self,
        seat_configs: tuple[
            train_config.TrainConfig | None, train_config.TrainConfig | None
        ],
    ) -> None:
        """Inject per-seat training configs for the timeline chart.

        Must be called before any game starts. Without this call the timeline
        shows score lines only and decision boxes omit option bars."""
        self._seat_configs = seat_configs

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
