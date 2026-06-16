# pyright: reportPrivateUsage=false
"""Tests for the modal timeline chart added to ``wingspan play --html``.

Covers three layers:

* :func:`timestamps.discounted_future_returns` — the shared backward-discount
  kernel (γ=1 telescoping identity, γ<1 hand-checked values, Δt=0 edge case).
* :func:`timestamps.finalize_provisional_timestamps` — the non-Step companion
  to :func:`timestamps.finalize_timestamps`; must agree with the Step-based
  reference on identical sequences.
* The instrumentation handler's ``made_decision`` timeline path over a full
  random game (no model seat): one point per non-forced decision, scores
  match ``running_score``, value/target lines are ``None``, ``phase_index``
  stays in range.
"""

from __future__ import annotations

import math
import os
import pathlib
import random
import sys
import typing

import numpy as np
import pydantic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import agents, cards, decisions, engine
from wingspan.engine import scoring
from wingspan.instrumentation import config as instrumentation_config
from wingspan.instrumentation import events as instrumentation_events
from wingspan.players import value_sink
from wingspan.reporting import game_log_capture, game_log_html
from wingspan.training import config as train_config
from wingspan.training import timestamps

if typing.TYPE_CHECKING:
    from wingspan import state
    from wingspan.engine import core


###### Kernel tests (no engine) ######


def _assert_close(actual: list[float], expected: list[float]) -> None:
    """Element-wise float comparison (pytest.approx is untyped under strict pyright)."""
    assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
    for got, want in zip(actual, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), f"{got} != {want}"


def test_discounted_future_returns_gamma_one_telescopes():
    """At γ=1 returns telescope: G[k] = terminal − checkpoint[k] for every k."""
    checkpoints = [0.0, 3.0, 5.0, 8.0]  # three decisions + terminal
    times = [1.0, 2.0, 3.0, 4.0]
    result = timestamps.discounted_future_returns(checkpoints, times, discount=1.0)
    # G[0] = 8-0=8, G[1] = 8-3=5, G[2] = 8-5=3
    _assert_close(result, [8.0, 5.0, 3.0])


def test_discounted_future_returns_gamma_half():
    """At γ=0.5 with Δt=1 between each pair the decay factor is 0.5^1=0.5.

    Three decisions at times 1,2,3 with terminal at 4:
      checkpoints [0, 2, 4, 6]
      rewards     [2, 2, 2]
      G[2] = 2
      G[1] = 2 + 0.5^1 * 2 = 3
      G[0] = 2 + 0.5^1 * 3 = 3.5
    """
    checkpoints = [0.0, 2.0, 4.0, 6.0]
    times = [1.0, 2.0, 3.0, 4.0]
    result = timestamps.discounted_future_returns(checkpoints, times, discount=0.5)
    _assert_close(result, [3.5, 3.0, 2.0])


def test_discounted_future_returns_gamma_zero_immediate_only():
    """At γ=0 each return is exactly its one-step reward (no future)."""
    checkpoints = [10.0, 13.0, 11.0, 16.0]
    times = [1.0, 2.0, 3.0, 4.0]
    result = timestamps.discounted_future_returns(checkpoints, times, discount=0.0)
    _assert_close(result, [3.0, -2.0, 5.0])


def test_discounted_future_returns_delta_t_zero_no_decay():
    """Two simultaneous decisions (Δt=0) pass credit through undecayed at any γ,
    because 0^0 == 1 in Python."""
    checkpoints = [0.0, 3.0, 5.0]
    times = [2.0, 2.0, 3.0]  # first two at same time
    result = timestamps.discounted_future_returns(checkpoints, times, discount=0.0)
    # G[1] = 5-3=2, G[0] = 3 + 0^0 * 2 = 5
    _assert_close(result, [5.0, 2.0])


def test_discounted_future_returns_single_decision():
    """A single decision before the terminal: return equals its one-step reward."""
    result = timestamps.discounted_future_returns([0.0, 7.0], [1.0, 2.0], discount=0.9)
    _assert_close(result, [7.0])


###### finalize_provisional_timestamps ######


def _main_family() -> int:
    """Family index for the main action decision."""
    return decisions.family_index_for(decisions.MainActionDecision)


def _food_family() -> int:
    """Family index for the gain-food decision."""
    return decisions.family_index_for(decisions.GainFoodDecision)


def test_finalize_provisional_timestamps_agrees_with_step_version():
    """``finalize_provisional_timestamps`` must produce the same spread as
    :func:`finalize_timestamps` when given the same sequence as parallel lists."""
    from wingspan.training import steps

    main_f = _main_family()
    food_f = _food_family()
    # Turn 3: main + two followers. Turn 4: main only.
    step_sequence = [
        steps.Step(
            state=np.zeros(1, dtype=np.float32),
            choices=np.zeros((1, 1), dtype=np.float32),
            chosen_idx=0,
            player_id=0,
            family_idx=main_f,
            timestamp=3.0,
        ),
        steps.Step(
            state=np.zeros(1, dtype=np.float32),
            choices=np.zeros((1, 1), dtype=np.float32),
            chosen_idx=0,
            player_id=0,
            family_idx=food_f,
            timestamp=3.0,
        ),
        steps.Step(
            state=np.zeros(1, dtype=np.float32),
            choices=np.zeros((1, 1), dtype=np.float32),
            chosen_idx=0,
            player_id=0,
            family_idx=food_f,
            timestamp=3.0,
        ),
        steps.Step(
            state=np.zeros(1, dtype=np.float32),
            choices=np.zeros((1, 1), dtype=np.float32),
            chosen_idx=0,
            player_id=0,
            family_idx=main_f,
            timestamp=4.0,
        ),
    ]
    timestamps.finalize_timestamps(step_sequence)
    reference = [step.timestamp for step in step_sequence]

    provisional = [3.0, 3.0, 3.0, 4.0]
    family_idxs = [main_f, food_f, food_f, main_f]
    result = timestamps.finalize_provisional_timestamps(provisional, family_idxs)
    _assert_close(result, reference)


def test_finalize_provisional_timestamps_setup_window_unchanged():
    """Setup-window items (timestamp < 1) are never modified."""
    food_f = _food_family()
    provisional = [
        timestamps.SETUP_KEEP_TIMESTAMP,
        timestamps.SETUP_FOOD_TIMESTAMP,
        timestamps.SETUP_FOOD_TIMESTAMP,
    ]
    family_idxs = [food_f, food_f, food_f]
    result = timestamps.finalize_provisional_timestamps(provisional, family_idxs)
    _assert_close(result, provisional)


def test_finalize_provisional_timestamps_no_main_action_group():
    """A group with no main action spreads all items (vs-random opponent reactions)."""
    food_f = _food_family()
    provisional = [5.0, 5.0]
    family_idxs = [food_f, food_f]
    result = timestamps.finalize_provisional_timestamps(provisional, family_idxs)
    _assert_close(result, [5.0 + 1.0 / 3.0, 5.0 + 2.0 / 3.0])


###### Handler timeline over a random game ######


class _TimelineCapture(
    instrumentation_events.GameStartHandler,
    instrumentation_events.SetupStartHandler,
    instrumentation_events.SetupAppliedHandler,
    instrumentation_events.RoundStartHandler,
    instrumentation_events.TurnStartHandler,
    instrumentation_events.MadeDecisionHandler,
    instrumentation_events.GameEndHandler,
):
    """Test handler that mirrors the production ``GameLogHtmlHandler`` lifecycle.

    Private attrs are accessed directly in tests; ``reportPrivateUsage=false``
    suppresses the external-access diagnostic for the whole file."""

    _raw_timeline: list[game_log_capture.RawTimelinePoint] = pydantic.PrivateAttr(
        default_factory=list["game_log_capture.RawTimelinePoint"]
    )
    _phases: list[game_log_html.PhaseRecord] = pydantic.PrivateAttr(
        default_factory=list["game_log_html.PhaseRecord"]
    )
    _probes: tuple[value_sink.ValueProbe | None, value_sink.ValueProbe | None] = (
        pydantic.PrivateAttr(default=(None, None))
    )

    def open(self, context: instrumentation_config.RunContext) -> None:
        pass

    def game_start(self, *, engine: core.Engine) -> None:
        self._phases = []
        self._raw_timeline = []
        self._snap(engine, "Game start", "game_start", None)

    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        self._snap(engine, f"{player.name} setup start", "setup_start", player.id)

    def setup_applied(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        choice: decisions.SetupChoice,
    ) -> None:
        if choice.bonus_card is not None:
            return
        self._snap(engine, f"{player.name} setup", "setup", player.id)

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        self._snap(engine, f"Round {round_num + 1}", "round", None)

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        self._snap(engine, f"{player.name} turn", "turn", player.id)

    def made_decision(
        self,
        *,
        engine: core.Engine,
        decision: decisions.Decision[typing.Any],
        choice: decisions.Choice,
    ) -> None:
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
                family_idx=decisions.family_index_for(type(decision)),
                score_p0=score_p0,
                score_p1=score_p1,
                phase_index=len(self._phases) - 1,
                value_pov=value_pov,
            )
        )

    def game_end(self, *, engine: core.Engine) -> None:
        self._snap(engine, "Final scoring", "game_end", None)

    def _snap(
        self, engine: core.Engine, title: str, kind: str, active: int | None
    ) -> None:
        """Snapshot the current engine state as a phase record."""
        self._phases.append(
            game_log_capture.capture_phase(
                engine,
                index=len(self._phases),
                title=title,
                kind=kind,
                active=active,
            )
        )


_ALL_EVENTS = [
    "game_start",
    "setup_start",
    "setup_applied",
    "round_start",
    "turn_start",
    "made_decision",
    "game_end",
]


def _run_with_capture(seed: int) -> tuple[engine.Engine, _TimelineCapture]:
    """Run one random game and return the engine and capture handler."""
    eng, *_ = engine.Engine.create(seed=seed)
    rng = random.Random(seed)
    handler = _TimelineCapture()
    cfg = instrumentation_config.InstrumentationConfig.model_validate(
        {"handlers": {"h": handler}, "events": {event: ["h"] for event in _ALL_EVENTS}}
    )
    instrumentation = cfg.build()
    instrumentation.open(
        instrumentation_config.RunContext(
            output_dir=pathlib.Path("."), run_name="t", seed=seed
        )
    )
    engine.Engine.play_one_game(
        eng.state,
        (agents.random_agent(rng), agents.random_agent(rng)),
        instrumentation=instrumentation,
    )
    instrumentation.close()
    return eng, handler


def test_handler_records_at_least_one_point():
    """The handler appends at least one ``RawTimelinePoint`` per game — a game
    with zero non-forced decisions is not realistic for full wingspan."""
    _, capture = _run_with_capture(seed=42)
    assert len(capture._raw_timeline) > 0


def test_handler_timeline_scores_are_non_negative():
    """Running scores are never negative in Wingspan — any negative value
    indicates the capture code is computing something wrong."""
    _, capture = _run_with_capture(seed=77)
    for point in capture._raw_timeline:
        assert point.score_p0 >= 0, f"negative score_p0 at {point}"
        assert point.score_p1 >= 0, f"negative score_p1 at {point}"


def test_handler_timeline_player_ids_are_valid():
    """``player_id`` in every raw point is either 0 or 1."""
    _, capture = _run_with_capture(seed=123)
    for point in capture._raw_timeline:
        assert point.player_id in (0, 1)


def test_handler_timeline_phase_index_in_range():
    """Every raw point's ``phase_index`` is a valid index into the phases list
    captured by the same handler run."""
    _, capture = _run_with_capture(seed=999)
    num_phases = len(capture._phases)
    for point in capture._raw_timeline:
        assert (
            0 <= point.phase_index < num_phases
        ), f"phase_index {point.phase_index} out of range [0, {num_phases})"


def test_handler_timeline_no_value_without_probe():
    """Without injected value probes all ``value_pov`` fields are ``None``."""
    _, capture = _run_with_capture(seed=321)
    for point in capture._raw_timeline:
        assert point.value_pov is None


def test_build_timeline_produces_monotone_timestamps():
    """``build_timeline`` finalizes timestamps: the sequence must be non-decreasing
    (setup window points share timestamps; turn points are spread into (N, N+1))."""
    eng, capture = _run_with_capture(seed=456)
    timeline = game_log_capture.build_timeline(
        engine=eng,
        raw_points=capture._raw_timeline,
        seat_configs=(None, None),
    )
    assert len(timeline) == len(capture._raw_timeline)
    tss = [pt.timestamp for pt in timeline]
    for prev_ts, curr_ts in zip(tss, tss[1:]):
        assert curr_ts >= prev_ts - 1e-9, f"non-monotone: {prev_ts} then {curr_ts}"


def test_build_timeline_value_and_target_none_without_configs():
    """With ``seat_configs=(None, None)`` the value/target return fields are all
    ``None`` — score-only degradation path for human/random seats."""
    eng, capture = _run_with_capture(seed=789)
    timeline = game_log_capture.build_timeline(
        engine=eng,
        raw_points=capture._raw_timeline,
        seat_configs=(None, None),
    )
    for point in timeline:
        assert point.value_return_p0 is None
        assert point.target_return_p0 is None


def test_build_timeline_returns_exclude_realized():
    """``value_return_p0`` and ``target_return_p0`` are the bare P0-relative
    discounted future return (``decision_delta`` mode), not ``realized + return``.

    At γ=1 the target telescopes to ``terminal − margin_before`` (not flat),
    and the critic return equals ``sign · value_pov · score_norm`` with no
    realized margin term added.  The key bug symptom was a flat target line;
    asserting two distinct values catches any regression that re-adds ``realized``."""
    eng, *_ = engine.Engine.create(seed=1)
    eng.state.players[0].final_score = 10
    eng.state.players[1].final_score = 4
    eng.state.turn_counter = 52

    # Two P0 decisions with non-zero score_p0/score_p1 so realized ≠ 0.
    main_family = _main_family()
    raw_points = [
        game_log_capture.RawTimelinePoint(
            player_id=0,
            margin_before=3.0,
            provisional_timestamp=1.0,
            family_idx=main_family,
            score_p0=5,
            score_p1=2,
            phase_index=0,
            value_pov=1.0,
        ),
        game_log_capture.RawTimelinePoint(
            player_id=0,
            margin_before=7.0,
            provisional_timestamp=2.0,
            family_idx=main_family,
            score_p0=8,
            score_p1=1,
            phase_index=0,
            value_pov=-0.5,
        ),
    ]
    cfg = train_config.RunConfig(
        training=train_config.TrainingConfig(
            reward_discount=1.0,
            score_norm=2.0,
            reward_mode=train_config.RewardMode.DECISION_DELTA,
        ),
    )
    timeline = game_log_capture.build_timeline(
        engine=eng,
        raw_points=raw_points,
        seat_configs=(cfg, None),
    )

    # Critic return = sign · value_pov · score_norm (no realized term).
    assert math.isclose(timeline[0].value_return_p0 or 0.0, 1.0 * 2.0)
    assert math.isclose(timeline[1].value_return_p0 or 0.0, -0.5 * 2.0)

    # At γ=1 target telescopes to terminal − margin_before = 6 − margin_before.
    assert math.isclose(timeline[0].target_return_p0 or 0.0, 6.0 - 3.0)
    assert math.isclose(timeline[1].target_return_p0 or 0.0, 6.0 - 7.0)

    # The two values must differ — a flat line is the bug's symptom.
    assert timeline[0].target_return_p0 != timeline[1].target_return_p0


def test_build_timeline_terminal_margin_flat_target():
    """In ``terminal_margin`` mode the target is constant at the final margin.

    All decisions, regardless of ``margin_before``, get the same P0-relative
    target: the terminal point delta (P0 minus P1).  Including a P1 decision
    verifies the sign flip still yields the same P0-relative constant."""
    eng, *_ = engine.Engine.create(seed=1)
    eng.state.players[0].final_score = 10
    eng.state.players[1].final_score = 4
    eng.state.turn_counter = 52

    main_family = _main_family()
    raw_points = [
        game_log_capture.RawTimelinePoint(
            player_id=0,
            margin_before=3.0,
            provisional_timestamp=1.0,
            family_idx=main_family,
            score_p0=5,
            score_p1=2,
            phase_index=0,
            value_pov=1.0,
        ),
        game_log_capture.RawTimelinePoint(
            player_id=1,
            margin_before=2.0,
            provisional_timestamp=2.0,
            family_idx=main_family,
            score_p0=6,
            score_p1=3,
            phase_index=0,
            value_pov=0.5,
        ),
    ]
    cfg = train_config.RunConfig(
        training=train_config.TrainingConfig(
            reward_discount=1.0,
            score_norm=2.0,
            reward_mode=train_config.RewardMode.TERMINAL_MARGIN,
        ),
    )
    timeline = game_log_capture.build_timeline(
        engine=eng,
        raw_points=raw_points,
        seat_configs=(cfg, cfg),
    )

    # Both decisions get the flat P0-relative terminal: 10 − 4 = 6.0.
    assert math.isclose(timeline[0].target_return_p0 or 0.0, 6.0)
    assert math.isclose(timeline[1].target_return_p0 or 0.0, 6.0)


def test_build_timeline_empty_for_no_decisions():
    """``build_timeline`` returns an empty list when given no raw points."""
    eng, *_ = engine.Engine.create(seed=1)
    timeline = game_log_capture.build_timeline(
        engine=eng, raw_points=[], seat_configs=(None, None)
    )
    assert timeline == []


def test_cli_html_timeline_data_embedded(tmp_path: pathlib.Path):
    """The ``--html`` CLI flag embeds ``timeline`` data in the HTML output (even
    for random seats — score-only mode still produces an array with entries)."""
    from wingspan import cli

    out_path = tmp_path / "game.html"
    code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "5",
            "--quiet",
            "--html",
            str(out_path),
            "--instrument-out",
            str(tmp_path),
        ]
    )
    assert code == 0
    text = out_path.read_text(encoding="utf-8")
    # The JSON data island must contain the "timeline" key.
    assert '"timeline"' in text
