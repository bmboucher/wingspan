"""Convert live game state and the engine's text log into a game-log report.

This is the engine-aware half of the HTML game-log feature: it reads a
``GameState`` (and the bonus-scoring helpers) to flatten each phase into the
primitive display models in :mod:`wingspan.reporting.game_log_html`, and it
splits the engine's interleaved text log into one ``LogItem`` block per phase.
It is imported lazily by
:class:`wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler` —
never at import time — so its dependence on ``engine`` does not close the
``engine`` ↔ ``instrumentation`` import cycle.

Public API: :func:`capture_phase` (one snapshot without log items),
:func:`capture_setup_phase` (snapshot for the combined per-player setup phase),
:func:`build_decision_item` (build a structured ``LogItem`` from a
``PolicyAnnotation``), :func:`record_setup_decision` (route one setup decision
into a per-player capture bucket), :func:`finalize_setup_phase` (assemble
highlighted cards + grouped food log after all setup decisions are recorded),
:func:`build_timeline` (finalize timestamps and compute per-decision chart
points), and :func:`build_report` (merge decision items with the text log and
assemble the report).
"""

from __future__ import annotations

import collections
import typing

import pydantic

from wingspan import cards, decisions, state
from wingspan.agents import display
from wingspan.engine import scoring
from wingspan.players import decision_probe
from wingspan.reporting import game_log_html, humanize

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
    two offered bonus cards (marked ``pending=True``).  The viewer dims
    un-selected options until :func:`finalize_setup_phase` sets ``selected``
    on the kept card and clears the ``pending`` flag."""
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
        log_items=[],
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


def build_decision_item(
    engine: core.Engine,
    decision: decisions.Decision[typing.Any],
    choice: decisions.Choice,
    annotation: decision_probe.PolicyAnnotation,
) -> game_log_html.LogItem:
    """Build a structured ``LogItem`` for one genuine AI decision.

    Selects up to ``_MAX_DECISION_OPTIONS`` options by probability (always
    including the chosen option even if it falls outside the top N) and builds
    a ``DecisionOption`` for each using the humanizer.  The ``text`` field
    holds the humanized outcome summary shown in the collapsed header."""
    gs = engine.state
    n_choices = len(decision.choices)
    ranked = sorted(
        range(n_choices), key=lambda idx: annotation.probs[idx], reverse=True
    )

    # Top-N by probability, then force-include the selected option if absent.
    shown_indices = ranked[:_MAX_DECISION_OPTIONS]
    if annotation.chosen_idx not in shown_indices:
        shown_indices = shown_indices[:-1] + [annotation.chosen_idx]

    options: list[game_log_html.DecisionOption] = []
    for idx in shown_indices:
        idx_choice = decision.choices[idx]
        options.append(
            game_log_html.DecisionOption(
                label=humanize.humanize_choice(
                    idx_choice, gs, player_id=decision.player_id
                ),
                prob=annotation.probs[idx],
                score=annotation.scores[idx] if annotation.scores is not None else None,
                selected=(idx == annotation.chosen_idx),
            )
        )

    return game_log_html.LogItem(
        kind="decision",
        player_id=decision.player_id,
        text=humanize.humanize_outcome(decision, choice, gs),
        options=options,
    )


# Maximum options shown in a decision box (selected always included).
_MAX_DECISION_OPTIONS = 5

# Substring that identifies the secondary setup-bonus log segment, emitted only
# in the deferred-bonus regime.  Must not match the combined-regime header
# "… BIRDS, FOOD, AND BONUS CARD".
_BONUS_SEGMENT_MARKER = "CHOOSING BONUS CARD"


class SetupCaptureState(pydantic.BaseModel):
    """Transient accumulator for one player's setup decisions.

    Created at ``setup_start`` and consumed at ``game_end`` (``finalize_setup_phase``).
    All fields are filled in incrementally as ``record_setup_decision`` processes
    each decision; nothing outside this module needs to inspect the fields directly."""

    phase_index: int
    kept_card_names: set[str] = pydantic.Field(default_factory=set)
    kept_bonus_name: str | None = None
    keep_item: game_log_html.LogItem | None = None
    bonus_item: game_log_html.LogItem | None = None
    food_items: list[game_log_html.LogItem] = pydantic.Field(
        default_factory=list[game_log_html.LogItem]
    )
    food_spent: list[str] = pydantic.Field(default_factory=list)
    food_gained: list[str] = pydantic.Field(default_factory=list)
    combined_food_labels: list[str] = pydantic.Field(default_factory=list)


def record_setup_decision(
    capture: SetupCaptureState,
    engine: core.Engine,
    decision: decisions.Decision[typing.Any],
    choice: decisions.Choice,
    annotation: decision_probe.PolicyAnnotation | None,
) -> None:
    """Route one setup-context decision into the capture bucket.

    Called by the instrumentation handler instead of the normal decision-item
    path whenever the current phase is a setup phase.  Branches on the choice
    type to fill the correct slot(s) in ``capture``."""
    # SetupChoice: records kept cards and, in non-split regimes, food + bonus.
    if isinstance(choice, decisions.SetupChoice):
        capture.kept_card_names = {bird.name for bird in choice.kept_cards}
        if choice.bonus_card is not None:
            capture.kept_bonus_name = choice.bonus_card.name
        if choice.kept_foods:
            capture.combined_food_labels = [food.value for food in choice.kept_foods]
        if annotation is not None:
            item = build_decision_item(engine, decision, choice, annotation)
            names = ", ".join(bird.name for bird in choice.kept_cards) or "no birds"
            item.text = f"Keeps {names}"
            capture.keep_item = item
        return

    # FoodChoice under SpendFoodDecision: player discards one food token.
    if isinstance(choice, decisions.FoodChoice) and isinstance(
        decision, decisions.SpendFoodDecision
    ):
        food_value = choice.food.value
        capture.food_spent.append(food_value)
        if annotation is not None:
            item = build_decision_item(engine, decision, choice, annotation)
            item.text = f"Discards {food_value}"
            capture.food_items.append(item)
        return

    # FoodChoice under GainFoodDecision: player gains one food token.
    if isinstance(choice, decisions.FoodChoice) and isinstance(
        decision, decisions.GainFoodDecision
    ):
        food_value = choice.food.value
        capture.food_gained.append(food_value)
        if annotation is not None:
            item = build_decision_item(engine, decision, choice, annotation)
            item.text = f"Gains {food_value}"
            capture.food_items.append(item)
        return

    # BonusCardChoice: deferred-bonus path; player picks which bonus to keep.
    if isinstance(choice, decisions.BonusCardChoice):
        capture.kept_bonus_name = choice.bonus_card.name
        if annotation is not None:
            item = build_decision_item(engine, decision, choice, annotation)
            capture.bonus_item = item
        return


def finalize_setup_phase(
    phase: game_log_html.PhaseRecord,
    capture: SetupCaptureState,
) -> None:
    """Assemble the completed setup phase log from the capture bucket.

    Sets the ``selected`` flag on the kept hand cards and the kept bonus card,
    clears the ``pending`` flag on the chosen bonus, computes the kept-food
    label list, and builds the three-node decision log (keep, food group, bonus).
    This runs once per player at ``game_end``, before ``build_report``."""
    # Highlight kept hand cards.
    if phase.active_player_id is not None:
        active_panels = [
            panel for panel in phase.panels if panel.player_id == phase.active_player_id
        ]
        if active_panels:
            for cell in active_panels[0].hand:
                cell.selected = cell.name in capture.kept_card_names

    # Highlight kept bonus and mark the chosen one as no-longer-pending.
    for bonus_opt in phase.setup_bonus_options:
        if bonus_opt.name == capture.kept_bonus_name:
            bonus_opt.selected = True
            bonus_opt.pending = False

    # Derive the kept-food label list.
    if capture.combined_food_labels:
        kept_food_labels = capture.combined_food_labels
    elif capture.food_spent:
        all_food_values = [food.value for food in cards.ALL_FOODS]
        kept_food_labels = [f for f in all_food_values if f not in capture.food_spent]
    else:
        kept_food_labels = capture.food_gained

    # Build the food group node (only when there were food decisions to record).
    food_group: game_log_html.LogItem | None = None
    if capture.food_items:
        kept_label = ", ".join(kept_food_labels) if kept_food_labels else "no food"
        food_group = game_log_html.LogItem(
            kind="group",
            player_id=phase.active_player_id,
            text=f"Keeps {kept_label}",
            children=list(capture.food_items),
        )

    # Assemble the phase log: [keep-choice, food-group, bonus-choice] (skipping None).
    phase.log_items = [
        item
        for item in (capture.keep_item, food_group, capture.bonus_item)
        if item is not None
    ]


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

    # Terminal margin from each seat's POV (P0's is positive when P0 leads).
    final_score_p0 = engine.state.players[0].final_score or 0
    final_score_p1 = engine.state.players[1].final_score or 0
    terminal_per_player = (
        float(final_score_p0 - final_score_p1),
        float(final_score_p1 - final_score_p0),
    )

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

        # terminal_margin: critic is trained on the flat end-of-game margin
        # broadcast to every decision, so the target is constant at the terminal.
        if cfg.training.reward_mode is train_config.RewardMode.TERMINAL_MARGIN:
            for idx in indices:
                target_raw[idx] = terminal_per_player[player_id]
            continue

        # decision_delta: critic is trained on the discounted future margin
        # change, so the target telescopes toward 0 at the end.
        checkpoints = [raw_points[i].margin_before for i in indices]
        checkpoints.append(terminal_per_player[player_id])
        times = [final_ts[i] for i in indices]
        times.append(game_end_ts)
        raw_returns = timestamps.discounted_future_returns(
            checkpoints, times, cfg.training.reward_discount
        )
        for position, idx in enumerate(indices):
            target_raw[idx] = raw_returns[position]

    # Assemble TimelinePoint objects — value/target are P0-relative future returns, in VP.
    result: list[game_log_html.TimelinePoint] = []
    for idx, (pt, ts) in enumerate(zip(raw_points, final_ts)):
        cfg = seat_configs[pt.player_id]
        score_norm = cfg.training.score_norm if cfg is not None else 1.0
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
    decision_items: list[tuple[int, game_log_html.LogItem]] | None = None,
) -> game_log_html.GameLogReport:
    """Merge structured decision items with the text log and assemble the report.

    For each non-setup phase, the pre-built ``decision_items`` (keyed by
    ``phase_index``) are consumed in order as decision headers are encountered in
    the text log; remaining lines become ``note`` or ``forced`` ``LogItem``s.
    Setup phases already have their ``log_items`` set by
    :func:`finalize_setup_phase` and are skipped in the text-log loop.

    ``_merge_secondary_setup_segments`` folds any secondary CHOOSING BONUS CARD
    segment into the preceding segment first so the ``zip(phases, segments)``
    count stays 1:1 after we create only one phase per player."""
    # Group non-setup decision items by phase index for per-phase lookup.
    items_by_phase: dict[int, collections.deque[game_log_html.LogItem]] = (
        collections.defaultdict(collections.deque)
    )
    for phase_index, log_item in decision_items or []:
        items_by_phase[phase_index].append(log_item)

    segments = _split_log_into_segments(engine.state.log_entries)
    segments = _merge_secondary_setup_segments(segments)

    for phase, segment in zip(phases, segments):
        if phase.kind == "setup":
            # log_items already set by finalize_setup_phase; skip text-log pass.
            continue
        phase_queue = items_by_phase.get(phase.index, collections.deque())
        phase.log_items = _segment_log_items(
            segment, is_turn=phase.kind == "turn", phase_decision_items=phase_queue
        )

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


#### Log segmentation ####


def _merge_secondary_setup_segments(
    segments: list[list[state.LogEntry]],
) -> list[list[state.LogEntry]]:
    """Fold deferred-bonus setup segments into the preceding primary segment.

    In the split-bonus regime the engine emits two ``=== ===`` headers per
    player: the primary CHOOSING BIRDS header and a secondary CHOOSING BONUS
    CARD header.  After this change we create only one phase per player, so the
    secondary segment must be folded into the primary to keep the
    ``zip(phases, segments)`` count aligned.

    Any segment whose header line contains ``_BONUS_SEGMENT_MARKER`` is appended
    (without its header line) to the previous segment and removed.  In the
    combined or food-split regimes no such header exists, so the list is
    returned unchanged."""
    merged: list[list[state.LogEntry]] = []
    for segment in segments:
        header_text = segment[0].text if segment else ""
        if merged and _BONUS_SEGMENT_MARKER in header_text:
            merged[-1].extend(segment[1:])
        else:
            merged.append(segment)
    return merged


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


def _segment_log_items(
    segment: list[state.LogEntry],
    *,
    is_turn: bool,
    phase_decision_items: collections.deque[game_log_html.LogItem],
) -> list[game_log_html.LogItem]:
    """Convert one log segment into a list of ``LogItem``s.

    Decision header lines (``is_decision_start``) pop a pre-built item from
    ``phase_decision_items`` (always 1:1 for AI seats) and skip the distribution
    text block through and including the following ``chose:`` line.  Forced
    single-choice lines become ``"forced"`` items.  Everything else becomes a
    ``"note"`` item processed through the humanizer.

    A turn segment opens with the verbose board/hand/score summary; the whole
    prefix up to the first blank line is dropped (the panel already shows it)."""
    body = segment[1:]  # skip the === header line
    if is_turn:
        body = _drop_summary_block(body)

    result: list[game_log_html.LogItem] = []
    body_iter = iter(body)

    for entry in body_iter:
        text = display.strip_ansi(entry.text)

        if _is_decision_start(text):
            # Consume the structured decision item for this AI decision.
            if phase_decision_items:
                result.append(phase_decision_items.popleft())
            # Skip the distribution block (option lines + chose: line).
            for skip_entry in body_iter:
                if "chose:" in display.strip_ansi(skip_entry.text):
                    break

        elif "skipping decision, only 1 choice: " in text:
            # Engine logged a forced single-option decision.
            label = text.split("only 1 choice: ", 1)[-1]
            result.append(
                game_log_html.LogItem(
                    kind="forced",
                    player_id=entry.player_id,
                    text=humanize.humanize_forced(label),
                    forced=True,
                )
            )

        else:
            # Regular notification: humanize and show as a note.
            cleaned = humanize.humanize_note(text)
            if cleaned:
                result.append(
                    game_log_html.LogItem(
                        kind="note",
                        player_id=entry.player_id,
                        text=cleaned,
                    )
                )

    return result


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
