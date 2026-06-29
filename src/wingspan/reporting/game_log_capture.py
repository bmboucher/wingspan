"""Convert live game state and the structured event tree into a game-log report.

This module is the engine-aware half of the HTML game-log feature. It reads a
``GameState`` and ``GameEventTree`` to assemble the primitive display models in
:mod:`wingspan.reporting.game_log_html`. It is imported lazily by
:class:`wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler`
to keep the ``engine`` <-> ``instrumentation`` import cycle at bay.

Public API: :func:`capture_phase`, :func:`capture_setup_phase` (state
snapshots), :func:`tree_to_log_items` (phase → ``LogItem`` list from tree),
:func:`extract_timeline_points` (DFS → ``RawTimelinePoint`` list),
:func:`build_timeline` (finalize timestamps / compute chart points), and
:func:`build_report` (merge tree with phase snapshots and assemble the report).
"""

from __future__ import annotations

import functools
import typing

import pydantic

from wingspan import cards, state
from wingspan.agents import display
from wingspan.engine import scoring
from wingspan.gamelog import models as gamelog_models
from wingspan.reporting import card_view, game_log_html

if typing.TYPE_CHECKING:
    from wingspan.engine import core
    from wingspan.training import config as train_config


# Display labels for the three habitat rows, in board order.
_HABITAT_LABELS: dict[cards.Habitat, str] = {
    cards.Habitat.FOREST: "Forest",
    cards.Habitat.GRASSLAND: "Grassland",
    cards.Habitat.WETLAND: "Wetland",
}


@functools.cache
def _birds_by_name() -> dict[str, cards.Bird]:
    """Name→Bird map for the full card table, cached per process."""
    birds, _, _ = cards.load_all()
    return {bird.name: bird for bird in birds}


# ---------------------------------------------------------------------------
# Phase snapshot helpers


def capture_setup_phase(
    engine: core.Engine,
    *,
    index: int,
    title: str,
    active: int,
    dealt_bonus: list[cards.BonusCard],
) -> game_log_html.PhaseRecord:
    """Snapshot state at the start of one player's combined setup phase.

    Like :func:`capture_phase` but populates ``setup_bonus_options`` with the
    two offered bonus cards (marked ``pending=True``).  :func:`build_report`
    sets ``selected`` on the kept card and clears the ``pending`` flag."""
    phase = capture_phase(engine, index=index, title=title, kind="setup", active=active)
    phase.setup_bonus_options = [
        game_log_html.BonusCardInfo(
            name=bc.name,
            condition=bc.condition,
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

    The narration is filled in later by :func:`build_report` once the tree is
    available; everything else (boards, hands, food, scores, bonus cards, the
    shared tray / birdfeeder / round goals) is read from the live state now."""
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
        log_items=[],
    )


# ---------------------------------------------------------------------------
# Timeline data model and builder


class RawTimelinePoint(pydantic.BaseModel):
    """One recorded decision's raw data for the timeline chart.

    Populated by :func:`extract_timeline_points` from the event tree, before
    timestamp finalization. ``value_pov`` is the critic's output for the
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

    Reads the game's final scores from ``engine`` and each seat's
    ``reward_mode`` to compute the target line so it matches the training
    signal the critic was actually trained on.  Returns an empty list when
    ``raw_points`` is empty (game with no decisions)."""
    from wingspan.training import config as train_config
    from wingspan.training import timestamps

    if not raw_points:
        return []

    # Finalize provisional timestamps for all points in recording order.
    provisional_ts = [pt.provisional_timestamp for pt in raw_points]
    family_idxs = [pt.family_idx for pt in raw_points]
    final_ts = timestamps.finalize_provisional_timestamps(provisional_ts, family_idxs)
    game_end_ts = timestamps.final_timestamp(engine.state.turn_counter)

    # Terminal value for each seat under each reward_basis.
    final_score_p0 = engine.state.players[0].final_score or 0
    final_score_p1 = engine.state.players[1].final_score or 0
    terminal_margin = (
        float(final_score_p0 - final_score_p1),
        float(final_score_p1 - final_score_p0),
    )
    terminal_own_score = (float(final_score_p0), float(final_score_p1))

    # Build a {index: target_raw} map (raw = before / score_norm division),
    # matching the actual training signal for each seat's reward_mode.
    target_raw: dict[int, float] = {}
    for player_id in (0, 1):
        cfg = seat_configs[player_id]
        if cfg is None:
            continue
        indices = [i for i, pt in enumerate(raw_points) if pt.player_id == player_id]
        if not indices:
            continue

        basis = cfg.training.reward_basis
        own_score_basis = basis is train_config.RewardBasis.OWN_SCORE
        terminal = (
            terminal_own_score[player_id]
            if own_score_basis
            else terminal_margin[player_id]
        )

        # terminal_margin: critic target is the flat end-of-game value broadcast
        # to every decision.
        if cfg.training.reward_mode is train_config.RewardMode.TERMINAL_MARGIN:
            for idx in indices:
                target_raw[idx] = terminal
            continue

        # decision_delta: critic target telescopes toward 0 at the end.
        if own_score_basis:
            checkpoints = [
                float(
                    raw_points[i].score_p0 if player_id == 0 else raw_points[i].score_p1
                )
                for i in indices
            ]
        else:
            checkpoints = [raw_points[i].margin_before for i in indices]
        checkpoints.append(terminal)
        times = [final_ts[i] for i in indices]
        times.append(game_end_ts)
        raw_returns = timestamps.discounted_future_returns(
            checkpoints, times, cfg.training.reward_discount
        )
        for position, idx in enumerate(indices):
            target_raw[idx] = raw_returns[position]

    # Assemble TimelinePoint objects (value/target are P0-relative future returns, in VP).
    result: list[game_log_html.TimelinePoint] = []
    for idx, (pt, ts) in enumerate(zip(raw_points, final_ts)):
        cfg = seat_configs[pt.player_id]
        score_norm = cfg.training.score_norm if cfg is not None else 1.0
        sign = 1 if pt.player_id == 0 else -1

        value_return: float | None = None
        if pt.value_pov is not None and cfg is not None:
            value_return = sign * pt.value_pov * score_norm

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


# ---------------------------------------------------------------------------
# Report assembly


def build_report(
    *,
    engine: core.Engine,
    phases: list[game_log_html.PhaseRecord],
    tree: gamelog_models.GameEventTree,
    seed: int | None,
    matchup: tuple[str, str] | None,
    timeline: list[game_log_html.TimelinePoint] | None = None,
) -> game_log_html.GameLogReport:
    """Assemble the HTML game-log report from phase snapshots and the event tree.

    For each non-setup phase, ``tree_to_log_items`` maps the tree's events to
    ``LogItem`` objects.  Setup phases additionally get hand and bonus-card
    highlights applied from the ``SetupEvent`` stored in the tree."""
    for phase, tree_phase in zip(phases, tree.phases):
        if phase.kind == "setup":
            setup_events = [
                ev
                for ev in tree_phase.events
                if isinstance(ev, gamelog_models.SetupEvent)
            ]
            if setup_events:
                _apply_setup_highlights(phase, setup_events[0])
        phase.log_items = tree_to_log_items(tree_phase)

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


# ---------------------------------------------------------------------------
# Tree → LogItem conversion


def tree_to_log_items(phase: gamelog_models.PhaseNode) -> list[game_log_html.LogItem]:
    """Convert a tree phase's events into ``LogItem`` list for the HTML decision log.

    Each :class:`~gamelog_models.GameEvent` type maps to one or more ``LogItem``
    objects: decisions become collapsible boxes, notes become muted lines,
    play-bird events become grouped parents, and scoring events are skipped
    (they render as phase snapshots, not log items)."""
    items: list[game_log_html.LogItem] = []
    for event in phase.events:
        items.extend(_game_event_to_items(event))
    return items


def extract_timeline_points(
    tree: gamelog_models.GameEventTree,
) -> list[RawTimelinePoint]:
    """Extract timeline data by DFS-walking all ``DecisionSubEvent``s in the tree.

    Each :class:`~gamelog_models.DecisionSubEvent` stores the timeline scalars
    recorded at decision time; this function reconstructs
    :class:`RawTimelinePoint` objects from them, assigning ``phase_index`` from
    the tree's phase structure (positionally aligned with the handler's
    ``PhaseRecord`` list)."""
    from wingspan.training import timestamps

    slot_ts = (
        timestamps.SETUP_KEEP_TIMESTAMP,
        timestamps.SETUP_BONUS_TIMESTAMP,
        timestamps.SETUP_FOOD_TIMESTAMP,
    )
    points: list[RawTimelinePoint] = []
    for phase_idx, phase_node in enumerate(tree.phases):
        for event in phase_node.events:
            _collect_decision_points(event, phase_idx, points, slot_ts)
    return points


###### PRIVATE #######


#### State → reporting-model conversion ####


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
    cell = card_view.bird_cell_info(bird)
    if played is None:
        return cell
    return cell.model_copy(
        update={
            "eggs": played.eggs,
            "tucked": played.tucked_cards,
            "cached": played.cached_food.total(),
        }
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
        condition=bc.condition,
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


#### Tree-to-item conversion helpers ####


def _apply_setup_highlights(
    phase: game_log_html.PhaseRecord,
    setup_event: gamelog_models.SetupEvent,
) -> None:
    """Apply hand and bonus-card highlights from a ``SetupEvent`` to its phase record.

    Sets ``selected=True`` on kept hand cards and the kept bonus card option,
    clearing ``pending`` on the chosen bonus. Replaces the old ``finalize_setup_phase``.
    """
    if phase.active_player_id is not None:
        active_panels = [
            panel for panel in phase.panels if panel.player_id == phase.active_player_id
        ]
        if active_panels:
            kept = set(setup_event.kept_card_names)
            for cell in active_panels[0].hand:
                cell.selected = cell.name in kept

    kept_bonus = setup_event.kept_bonus_name
    for bonus_opt in phase.setup_bonus_options:
        if bonus_opt.name == kept_bonus:
            bonus_opt.selected = True
            bonus_opt.pending = False


def _game_event_to_items(
    event: gamelog_models.GameEvent,
) -> list[game_log_html.LogItem]:
    """Map one :class:`~gamelog_models.GameEvent` to zero or more ``LogItem``s."""
    if isinstance(event, gamelog_models.PlayBirdEvent):
        return _play_bird_event_to_items(event)
    elif isinstance(event, gamelog_models.SetupEvent):
        return _sub_events_to_items(event.sub_events, event.player_id)
    elif isinstance(
        event, (gamelog_models.RoundGoalEvent, gamelog_models.FinalScoringEvent)
    ):
        # Rendered as phase snapshots; no log-item representation.
        return []
    else:
        # MainActionEvent, ActivateBaseEvent, ActivateBrownEvent, ReactionEvent, LooseEvent
        result: list[game_log_html.LogItem] = []
        result.extend(_sub_events_to_items(event.sub_events, event.player_id))
        for child in event.children:
            result.extend(_game_event_to_items(child))
        return result


def _play_bird_event_to_items(
    event: gamelog_models.PlayBirdEvent,
) -> list[game_log_html.LogItem]:
    """A ``PlayBirdEvent`` becomes a 'group' headed by the bird-selection decision.

    Sub-events (selection, egg payments, food payment) are children. Any
    ``WhitePowerEvent`` child contributes a trailing power-activation note plus
    its own decisions."""
    child_items = _sub_events_to_items(event.sub_events, event.player_id)
    if not child_items:
        # No decisions recorded (shouldn't happen in normal play); fall back flat.
        items: list[game_log_html.LogItem] = []
        for child in event.children:
            items.extend(_game_event_to_items(child))
        return items

    group = game_log_html.LogItem(
        kind="group",
        player_id=event.player_id,
        text=child_items[0].text,
        children=child_items,
    )
    result: list[game_log_html.LogItem] = [group]

    # Each WhitePowerEvent child contributes a power-activation note and its decisions.
    for child_event in event.children:
        if isinstance(child_event, gamelog_models.WhitePowerEvent):
            bird = _birds_by_name().get(child_event.bird_name)
            if bird is not None and bird.plain_power_text:
                result.append(
                    game_log_html.LogItem(
                        kind="note",
                        player_id=event.player_id,
                        text=f"{child_event.bird_name}: {bird.plain_power_text}",
                        power_color="white",
                    )
                )
            result.extend(_sub_events_to_items(child_event.sub_events, event.player_id))
        else:
            result.extend(_game_event_to_items(child_event))

    return result


def _sub_events_to_items(
    sub_events: list[gamelog_models.SubEvent],
    player_id: int | None,
) -> list[game_log_html.LogItem]:
    """Convert a list of ``SubEvent``s to ``LogItem``s, dropping empty notes."""
    items: list[game_log_html.LogItem] = []
    for sub in sub_events:
        item = _sub_event_to_item(sub, player_id)
        if item is not None:
            items.append(item)
    return items


def _sub_event_to_item(
    sub: gamelog_models.SubEvent,
    player_id: int | None = None,
) -> game_log_html.LogItem | None:
    """Convert one ``SubEvent`` to a ``LogItem``, or ``None`` to suppress it."""
    pid = sub.player_id if sub.player_id is not None else player_id
    if isinstance(sub, gamelog_models.DecisionSubEvent):
        return game_log_html.LogItem(
            kind="decision",
            player_id=pid,
            text=sub.outcome_text,
            options=list(sub.options),
            state_stripes=sub.state_stripes,
        )
    elif isinstance(sub, gamelog_models.ForcedSubEvent):
        return game_log_html.LogItem(
            kind="forced",
            player_id=pid,
            text=sub.text,
            forced=True,
        )
    elif isinstance(sub, gamelog_models.NoteSubEvent):
        if sub.text:
            return game_log_html.LogItem(
                kind="note",
                player_id=pid,
                text=sub.text,
            )
    return None


def _collect_decision_points(
    event: gamelog_models.GameEvent,
    phase_idx: int,
    points: list[RawTimelinePoint],
    slot_ts: tuple[float, float, float],
) -> None:
    """DFS helper: collect ``DecisionSubEvent`` data into ``RawTimelinePoint``s.

    Visits ``event.sub_events`` first (in recording order), then recurses into
    ``event.children`` so the point sequence matches the game's decision order."""
    for sub in event.sub_events:
        if isinstance(sub, gamelog_models.DecisionSubEvent):
            slot = sub.setup_slot
            if slot is None:
                prov_ts = float(sub.turn_counter)
            else:
                prov_ts = slot_ts[slot] if 0 <= slot < 3 else slot_ts[2]
            points.append(
                RawTimelinePoint(
                    player_id=sub.player_id or 0,
                    margin_before=sub.margin_before,
                    provisional_timestamp=prov_ts,
                    family_idx=sub.family_idx,
                    score_p0=sub.score_p0,
                    score_p1=sub.score_p1,
                    phase_index=phase_idx,
                    value_pov=sub.value,
                )
            )
    for child in event.children:
        _collect_decision_points(child, phase_idx, points, slot_ts)
