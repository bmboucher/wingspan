"""Convert live game state and the engine's text log into a game-log report.

This is the engine-aware half of the HTML game-log feature: it reads a
``GameState`` (and the bonus-scoring helpers) to flatten each phase into the
primitive display models in :mod:`wingspan.reporting.game_log_html`, and it
splits the engine's interleaved text log into one decision-narration block per
phase. It is imported lazily by
:class:`wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler` —
never at import time — so its dependence on ``engine`` does not close the
``engine`` ↔ ``instrumentation`` import cycle.

Public API: :func:`capture_phase` (one narration-less snapshot),
:func:`build_timeline` (finalize timestamps and compute per-decision chart
points), and :func:`build_report` (attach narration from the log and assemble
the report).
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import cards, state
from wingspan.agents import display
from wingspan.engine import scoring
from wingspan.reporting import game_log_html

if typing.TYPE_CHECKING:
    from wingspan.engine import core
    from wingspan.training import config as train_config

# Marks a phase boundary in the interleaved log: the engine writes one such
# header per captured phase, in the same order the capture events fire.
_HEADER_PREFIX = "==="

# Display labels for the three habitat rows, in board order.
_HABITAT_LABELS: dict[cards.Habitat, str] = {
    cards.Habitat.FOREST: "Forest",
    cards.Habitat.GRASSLAND: "Grassland",
    cards.Habitat.WETLAND: "Wetland",
}


def capture_setup_start_phase(
    engine: core.Engine,
    *,
    index: int,
    title: str,
    kind: str,
    active: int,
    dealt_bonus: list[cards.BonusCard],
) -> game_log_html.PhaseRecord:
    """Snapshot state just before a player makes their setup choices.

    Like :func:`capture_phase` but populates ``setup_bonus_options`` with the
    two offered bonus cards (marked ``pending=True``) so the viewer can show
    them as dimmed/unselected in the bonus panel."""
    phase = capture_phase(engine, index=index, title=title, kind=kind, active=active)
    phase.setup_bonus_options = [
        game_log_html.BonusCardInfo(
            name=bc.name,
            text=display.strip_ansi(bc.vp_text),
            vp_now=0,
            count=0,
            pending=True,
        )
        for bc in dealt_bonus
    ]
    return phase


def capture_phase(
    engine: core.Engine, *, index: int, title: str, kind: str, active: int | None
) -> game_log_html.PhaseRecord:
    """Snapshot the current game state as a narration-less phase record.

    The narration is filled in later by :func:`build_report` once the whole log
    is available; everything else (both seats' boards, hands, food, scores,
    bonus cards, the shared tray / birdfeeder / round goals) is read from the
    live state now."""
    gs = engine.state
    return game_log_html.PhaseRecord(
        index=index,
        title=title,
        kind=kind,
        round_idx=gs.round_idx,
        active_player_id=active,
        panels=[_player_panel(player, gs) for player in gs.players],
        tray=[_bird_cell_info(bird) for bird in gs.tray],
        feeder_text=gs.birdfeeder.format(),
        feeder_slots=_feeder_slots(gs.birdfeeder),
        round_goals=_round_goal_infos(gs),
        narration=[],
    )


class RawTimelinePoint(pydantic.BaseModel):
    """One recorded decision's raw data for the timeline chart.

    Populated by the instrumentation handler at ``made_decision`` time, before
    timestamp finalization (provisional timestamps are spread into turn windows
    only after the game ends). ``value_pov`` is the critic's output for the
    deciding player's POV (divided by ``score_norm``); ``None`` when no net
    backed that seat."""

    player_id: int
    margin_before: float
    provisional_timestamp: float
    family_idx: int
    score_p0: int
    score_p1: int
    phase_index: int
    value_pov: float | None = None


def build_timeline(
    *,
    engine: core.Engine,
    raw_points: list[RawTimelinePoint],
    seat_configs: tuple[
        train_config.TrainConfig | None, train_config.TrainConfig | None
    ],
) -> list[game_log_html.TimelinePoint]:
    """Finalize timestamps and compute per-decision chart coordinates.

    Reads the game's final scores from ``engine`` to compute the discounted-
    return targets that match what ``learner._decision_delta_returns`` computes.
    Returns an empty list when ``raw_points`` is empty (game with no decisions)."""
    from wingspan.training import timestamps

    if not raw_points:
        return []

    # Finalize provisional timestamps for all points in recording order.
    provisional_ts = [pt.provisional_timestamp for pt in raw_points]
    family_idxs = [pt.family_idx for pt in raw_points]
    final_ts = timestamps.finalize_provisional_timestamps(provisional_ts, family_idxs)
    game_end_ts = timestamps.final_timestamp(engine.state.turn_counter)

    # Compute discounted returns per player (matching learner._decision_delta_returns).
    final_score_p0 = engine.state.players[0].final_score or 0
    final_score_p1 = engine.state.players[1].final_score or 0
    terminal_per_player = (
        float(final_score_p0 - final_score_p1),
        float(final_score_p1 - final_score_p0),
    )

    # Build a {index: target_raw} map (raw = before / score_norm division).
    target_raw: dict[int, float] = {}
    for player_id in (0, 1):
        cfg = seat_configs[player_id]
        if cfg is None:
            continue
        indices = [i for i, pt in enumerate(raw_points) if pt.player_id == player_id]
        if not indices:
            continue
        checkpoints = [raw_points[i].margin_before for i in indices]
        checkpoints.append(terminal_per_player[player_id])
        times = [final_ts[i] for i in indices]
        times.append(game_end_ts)
        raw_returns = timestamps.discounted_future_returns(
            checkpoints, times, cfg.reward_discount
        )
        for position, idx in enumerate(indices):
            target_raw[idx] = raw_returns[position]

    # Assemble TimelinePoint objects — value/target are P0-relative future returns, in VP.
    result: list[game_log_html.TimelinePoint] = []
    for idx, (pt, ts) in enumerate(zip(raw_points, final_ts)):
        cfg = seat_configs[pt.player_id]
        score_norm = cfg.score_norm if cfg is not None else 1.0
        sign = 1 if pt.player_id == 0 else -1

        # Critic: denormalize the predicted return to P0-relative VP.
        value_return: float | None = None
        if pt.value_pov is not None and cfg is not None:
            value_return = sign * pt.value_pov * score_norm

        # Target: raw return is already in VP (not normalized); convert to P0.
        target_return: float | None = None
        if idx in target_raw:
            target_return = sign * target_raw[idx]

        result.append(
            game_log_html.TimelinePoint(
                timestamp=ts,
                player_id=pt.player_id,
                score_p0=pt.score_p0,
                score_p1=pt.score_p1,
                phase_index=pt.phase_index,
                value_return_p0=value_return,
                target_return_p0=target_return,
            )
        )
    return result


def build_report(
    *,
    engine: core.Engine,
    phases: list[game_log_html.PhaseRecord],
    seed: int | None,
    matchup: tuple[str, str] | None,
    timeline: list[game_log_html.TimelinePoint] | None = None,
) -> game_log_html.GameLogReport:
    """Attach each log segment's narration to its phase and assemble the report.

    Segments and phases are paired in order; any count mismatch (no engine path
    produces one) degrades gracefully by pairing the common prefix."""
    segments = _split_log_into_segments(engine.state.log_entries)
    for phase, segment in zip(phases, segments):
        phase.narration = _segment_narration(segment, is_turn=phase.kind == "turn")
    final_scores = [player.final_score for player in engine.state.players]
    return game_log_html.GameLogReport(
        seed=seed,
        matchup=matchup,
        player_names=[player.name for player in engine.state.players],
        final_scores=(
            [score for score in final_scores if score is not None]
            if all(score is not None for score in final_scores)
            else None
        ),
        phases=phases,
        timeline=timeline if timeline is not None else [],
    )


###### PRIVATE #######


#### State -> reporting-model conversion (primitives only) ####


def _player_panel(
    player: state.Player, gs: state.GameState
) -> game_log_html.PlayerPanel:
    """Flatten one player's full visible state into a display panel."""
    return game_log_html.PlayerPanel(
        player_id=player.id,
        name=player.name,
        action_cubes_left=player.action_cubes_left,
        rows=[_habitat_row(player, habitat) for habitat in cards.ALL_HABITATS],
        hand=[
            cell for bird in player.hand if (cell := _bird_cell_info(bird)) is not None
        ],
        food=[
            game_log_html.FoodCount(label=food.value, count=player.food[food])
            for food in cards.ALL_FOODS
        ],
        score=_score_breakdown(player, gs),
        bonus_cards=[_bonus_card_info(player, bc) for bc in player.bonus_cards],
    )


def _habitat_row(
    player: state.Player, habitat: cards.Habitat
) -> game_log_html.HabitatRow:
    """One habitat row padded to a fixed ``BOARD_COLUMNS`` width."""
    cells = [
        game_log_html.BoardCell(bird=_bird_cell_info(pb.bird, played=pb))
        for pb in player.board[habitat]
    ]
    while len(cells) < game_log_html.BOARD_COLUMNS:
        cells.append(game_log_html.BoardCell(bird=None))
    return game_log_html.HabitatRow(label=_HABITAT_LABELS[habitat], cells=cells)


def _bird_cell_info(
    bird: cards.Bird | None, played: state.PlayedBird | None = None
) -> game_log_html.BirdCellInfo | None:
    """Flatten a bird (and, when on the board, its per-game state) to a cell."""
    if bird is None:
        return None

    # Build food cost as a slot list for emoji rendering: specific foods
    # repeated by count, then wild slots.
    slots: list[str] = []
    for food, specific_count in zip(cards.ALL_FOODS, bird.food_cost.specific):
        slots.extend([food.value] * specific_count)
    slots.extend(["wild"] * bird.food_cost.wild)

    return game_log_html.BirdCellInfo(
        name=bird.name,
        vp=bird.points,
        nest=bird.nest.value,
        wingspan_cm=bird.wingspan_cm,
        habitats="/".join(habitat.value for habitat in bird.habitats),
        food_cost=display.format_cost(bird.food_cost),
        food_cost_slots=slots,
        egg_limit=bird.egg_limit,
        eggs=played.eggs if played is not None else 0,
        tucked=played.tucked_cards if played is not None else 0,
        cached=played.cached_food.total() if played is not None else 0,
        power_color=bird.power.color.value,
        power_text=bird.plain_power_text,
    )


def _score_breakdown(
    player: state.Player, gs: state.GameState
) -> game_log_html.ScoreBreakdown:
    """The seven score columns, matching the game log's score table.

    Round-goal points are projected: for unscored rounds we compute the live
    standing as if the round ended right now, so the score reflects progress
    toward goals mid-game. Scored rounds stay frozen at their actual award."""
    bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
    bonus_pts = sum(scoring.bonus_score(player, bc) for bc in player.bonus_cards)

    # Use live projections for all rounds, not just the accumulated total.
    goals_pts = sum(
        scoring.round_goal_standing_for_round(gs, player, ri).vp
        for ri in range(min(len(gs.round_goals), 4))
    )
    total = (
        bird_pts
        + bonus_pts
        + player.total_eggs
        + player.total_tucked
        + player.total_cached
        + goals_pts
    )
    return game_log_html.ScoreBreakdown(
        birds=bird_pts,
        eggs=player.total_eggs,
        tucked=player.total_tucked,
        cached=player.total_cached,
        bonus=bonus_pts,
        goals=goals_pts,
        total=total,
    )


def _bonus_card_info(
    player: state.Player, bc: cards.BonusCard
) -> game_log_html.BonusCardInfo:
    """A held bonus card with its current VP and qualifying count for this player."""
    return game_log_html.BonusCardInfo(
        name=bc.name,
        text=display.strip_ansi(bc.vp_text),
        vp_now=scoring.bonus_score(player, bc),
        count=scoring.bonus_qualifying_count(player, bc),
    )


def _round_goal_infos(gs: state.GameState) -> list[game_log_html.RoundGoalInfo]:
    """All four round goals with payouts, scored flags, VP projections, and counts."""
    infos: list[game_log_html.RoundGoalInfo] = []
    for round_idx, (goal, payout) in enumerate(
        zip(gs.round_goals[:4], state.ROUND_GOAL_PAYOUTS_2P)
    ):
        first_vp, second_vp = payout

        p0_standing = scoring.round_goal_standing_for_round(
            gs, gs.players[0], round_idx
        )
        p1_standing = (
            scoring.round_goal_standing_for_round(gs, gs.players[1], round_idx)
            if len(gs.players) > 1
            else None
        )
        infos.append(
            game_log_html.RoundGoalInfo(
                round_num=round_idx + 1,
                description=goal.description,
                first_vp=first_vp,
                second_vp=second_vp,
                scored=len(gs.scored_goals) > round_idx,
                p0_vp=p0_standing.vp,
                p1_vp=p1_standing.vp if p1_standing is not None else 0,
                p0_count=p0_standing.count,
                p1_count=p1_standing.count if p1_standing is not None else 0,
            )
        )
    return infos


def _feeder_slots(feeder: state.Birdfeeder) -> list[str | None]:
    """5 feeder slots: food-type value string, 'choice', or None for empty."""
    slots: list[str | None] = []
    for food in cards.ALL_FOODS:
        slots.extend([food.value] * feeder.counts[food])
    slots.extend(["choice"] * feeder.choice_dice)
    while len(slots) < 5:
        slots.append(None)
    return slots[:5]


#### Log segmentation ####


def _split_log_into_segments(
    entries: list[state.LogEntry],
) -> list[list[state.LogEntry]]:
    """Split the interleaved log into one segment per ``=== ... ===`` header.

    Each segment starts at a header line and runs up to (excluding) the next
    one. Any lines before the first header are dropped (there are none in
    practice — the log opens with the GAME START banner)."""
    segments: list[list[state.LogEntry]] = []
    for entry in entries:
        if entry.text.startswith(_HEADER_PREFIX):
            segments.append([entry])
        elif segments:
            segments[-1].append(entry)
    return segments


def _segment_narration(
    segment: list[state.LogEntry], *, is_turn: bool
) -> list[game_log_html.NarrationLine]:
    """The decision lines of one phase: the segment minus its header (the title
    already shows it) and, for a turn, minus the verbose state-summary block.

    A turn segment opens with ``log_turn_summary``'s board / hand / tray / score
    / bonus lines, terminated by the single blank line the engine logs before
    the action; that whole prefix is dropped because the pinned panel already
    shows the same state structurally."""
    body = segment[1:]
    if is_turn:
        body = _drop_summary_block(body)
    return [
        game_log_html.NarrationLine(
            player_id=entry.player_id,
            text=display.strip_ansi(entry.text),
            is_decision_start=_is_decision_start(entry.text),
        )
        for entry in body
    ]


def _drop_summary_block(body: list[state.LogEntry]) -> list[state.LogEntry]:
    """Drop everything up to and including the first blank line — the turn-start
    state summary. Returns ``body`` unchanged if no blank line is present."""
    for index, entry in enumerate(body):
        if entry.text == "":
            return body[index + 1 :]
    return body


def _is_decision_start(text: str) -> bool:
    """True when this log line is a decision header (the start of a new event).

    Decision headers are emitted by the display agent in the form
    ``[P0] SomeDecision | N choices | head:...``; they start with ``[`` and
    contain the word ``Decision``."""
    return text.startswith("[") and "Decision" in text
