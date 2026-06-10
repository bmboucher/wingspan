"""Verbose turn-state summary lines for the game log.

Provides :func:`log_turn_summary`, called once at the top of every turn from
``engine.core._take_turn``, to print a detailed snapshot of the game state so
a reader of the log does not need to reconstruct context from the event stream.

Also provides :func:`log_game_setup` (called once after ``=== GAME START ===``)
and :func:`log_dealt_hand` (called once per player inside each SETUP block).
"""

from __future__ import annotations

import itertools
import typing

from wingspan import cards, state
from wingspan.agents import display
from wingspan.engine import scoring

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# Column headers for the score table.
_SCORE_HEADERS = ["Birds", "Eggs", "Tuck", "Cache", "Bonus", "Goals", "TOTAL"]

# Column headers for the food table (aligned to cards.ALL_FOODS order), plus
# the birdfeeder's inv/seed choice-face column shown on the Feeder row.
_FOOD_HEADERS = ["Inv", "Seed", "Fish", "Berry", "Rod", "Inv/Seed"]


def log_game_setup(engine: "core.Engine") -> None:
    """Emit the initial shared board state once, immediately after GAME START.

    Logs the face-up tray (3 birds with full stat + power lines), the
    birdfeeder die state on a single line, and all four round goals with their
    2-player VP payouts."""
    gs = engine.state

    _log_tray(engine, gs)
    engine.log("")
    engine.log(f"Birdfeeder: {gs.birdfeeder.format()}")
    engine.log("")

    # All four round goals with their per-round 2P payouts (1st/2nd VP).
    engine.log("Round Goals:")
    for round_idx, (goal, payout) in enumerate(
        zip(gs.round_goals[:4], state.ROUND_GOAL_PAYOUTS_2P)
    ):
        first, second = payout
        engine.log(f"  Round {round_idx + 1} ({first}/{second} VP): {goal.description}")


def log_dealt_hand(
    engine: "core.Engine", player: state.Player, dealt_cards: list[cards.Bird]
) -> None:
    """Log a player's full dealt hand at the start of their SETUP block.

    Shows every dealt bird (including cards the player will later discard) with
    the same stat + power text format used elsewhere in the log."""
    engine.log(f"Dealt hand ({len(dealt_cards)}):")
    for bird in dealt_cards:
        bird_text = display.format_bird_full(bird, indent="        ")
        _log_multiline(engine, bird_text, indent="  ")


def log_dealt_bonus(
    engine: "core.Engine",
    dealt_cards: list[cards.Bird],
    dealt_bonus: list[cards.BonusCard],
    player: state.Player,
) -> None:
    """Log the two dealt bonus cards during the CHOOSING BIRDS setup block.

    For each bonus card, logs its full text followed by an indented line
    showing how many matching birds appear in the dealt hand and the current
    face-up tray — the pool the player can actually see at setup time."""
    if not dealt_bonus:
        return
    tray_birds: list[cards.Bird] = [
        bird for bird in engine.state.tray if bird is not None
    ]
    engine.log(f"Dealt bonus ({len(dealt_bonus)}):")
    for bc in dealt_bonus:
        engine.log(f"  {display.format_bonus(bc)}")
        help_text = display.format_bonus_with_setup_help(
            bc,
            hand_birds=dealt_cards,
            tray_birds=tray_birds,
            selected_hand_birds=[],
        )
        # The help text may include a second line (for type-counting cards).
        for bonus_line in help_text.split("\n")[1:]:
            engine.log(f"    {bonus_line.strip()}")


def log_turn_summary(engine: "core.Engine") -> None:
    """Emit a detailed snapshot of the game state at the start of a turn.

    Writes the active player's board (by habitat), hand, tray, a combined
    score+food table (including a Feeder row with die counts), and their
    bonus card status into the game log."""
    gs = engine.state
    player = gs.me()

    _log_board(engine, player)
    _log_hand(engine, player)
    _log_tray(engine, gs)
    _log_score_food_tables(engine, gs)
    _log_bonus_cards(engine, player)


###### PRIVATE #######


#### Bird sections ####


def _log_multiline(engine: "core.Engine", text: str, indent: str = "") -> None:
    """Split a possibly multi-line formatter result and log each sub-line."""
    for line in text.split("\n"):
        engine.log(indent + line)


def _log_board(engine: "core.Engine", player: state.Player) -> None:
    """Log the board as three habitat sections, each bird on its own line(s)."""
    habitat_labels: dict[cards.Habitat, str] = {
        cards.Habitat.FOREST: "Forest",
        cards.Habitat.GRASSLAND: "Grass",
        cards.Habitat.WETLAND: "Wetland",
    }
    engine.log("Board:")
    for habitat in cards.ALL_HABITATS:
        engine.log(f"  {habitat_labels[habitat]}:")
        row = player.board[habitat]
        if row:
            for pb in row:
                bird_text = display.format_played_bird_full(pb, indent="          ")
                _log_multiline(engine, bird_text, indent="    ")
        else:
            engine.log("    (empty)")


def _log_hand(engine: "core.Engine", player: state.Player) -> None:
    """Log each bird in hand on its own line(s) with full stats and power text."""
    engine.log(f"Hand ({len(player.hand)}):")
    if player.hand:
        for bird in player.hand:
            bird_text = display.format_bird_full(bird, indent="        ")
            _log_multiline(engine, bird_text, indent="  ")
    else:
        engine.log("  (empty)")


def _log_tray(engine: "core.Engine", gs: state.GameState) -> None:
    """Log each tray card on its own line(s)."""
    engine.log("Tray:")
    for bird in gs.tray:
        if bird is not None:
            bird_text = display.format_bird_full(bird, indent="        ")
            _log_multiline(engine, bird_text, indent="  ")
        else:
            engine.log("  (empty slot)")


#### Score + food tables ####


def _score_row(player: state.Player) -> list[int | str]:
    """Compute the 7 score columns for one player."""
    bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
    bonus_pts = sum(scoring.bonus_score(player, bc) for bc in player.bonus_cards)
    total = (
        bird_pts
        + bonus_pts
        + player.total_eggs
        + player.total_tucked
        + player.total_cached
        + player.round_goal_points
    )
    return [
        bird_pts,
        player.total_eggs,
        player.total_tucked,
        player.total_cached,
        bonus_pts,
        player.round_goal_points,
        total,
    ]


def _food_row(player: state.Player) -> list[int | str]:
    """Compute the 6 food-table columns for one player.

    The first 5 are food token counts; the 6th (Inv/Seed) is blank because
    that column belongs to the birdfeeder's choice-face dice, not players."""
    return [*[player.food[food] for food in cards.ALL_FOODS], ""]


def _feeder_food_row(feeder: state.Birdfeeder) -> list[int | str]:
    """Compute the 6 food-table columns for the Feeder row.

    Five die-face counts plus the inv/seed choice-face count."""
    return [*[feeder.counts[food] for food in cards.ALL_FOODS], feeder.choice_dice]


def _table_lines(
    row_labels: list[str],
    headers: list[str],
    rows: list[list[int | str]],
) -> list[str]:
    """Format a table as strings: header row, then one data row per label.

    Columns are right-aligned; each column is as wide as the widest of its
    header and all row values in that column."""
    col_widths = [
        max(len(headers[col_idx]), *(len(str(row[col_idx])) for row in rows))
        for col_idx in range(len(headers))
    ]
    name_width = max(len(label) for label in row_labels)

    def _format_row(label: str, values: list[int | str]) -> str:
        cells = " │ ".join(
            str(values[col_idx]).rjust(col_widths[col_idx])
            for col_idx in range(len(values))
        )
        return f"{label.ljust(name_width)} │ {cells}"

    header_cells = " │ ".join(
        headers[col_idx].rjust(col_widths[col_idx]) for col_idx in range(len(headers))
    )
    header_line = f"{''.ljust(name_width)} │ {header_cells}"
    data_lines = [
        _format_row(row_labels[row_idx], rows[row_idx]) for row_idx in range(len(rows))
    ]
    return [header_line] + data_lines


def _log_score_food_tables(engine: "core.Engine", gs: state.GameState) -> None:
    """Log score and food counts as a single combined side-by-side table.

    The food side carries an extra Feeder row below the player rows showing
    the current birdfeeder die counts (five fixed faces + inv/seed choice
    faces). The score side has no entry for Feeder, so zip_longest pads it
    with a blank line to keep the combined output rectangular."""
    player_names = [player.name for player in gs.players]

    score_rows: list[list[int | str]] = [_score_row(player) for player in gs.players]
    food_rows: list[list[int | str]] = [_food_row(player) for player in gs.players]
    food_rows.append(_feeder_food_row(gs.birdfeeder))

    score_lines = _table_lines(player_names, _SCORE_HEADERS, score_rows)
    food_lines = _table_lines(player_names + ["Feeder"], _FOOD_HEADERS, food_rows)

    # Pad score lines to uniform width so the food table aligns vertically.
    score_width = max(len(line) for line in score_lines)
    separator = "  ║  "
    for score_line, food_line in itertools.zip_longest(
        score_lines, food_lines, fillvalue=""
    ):
        engine.log(score_line.ljust(score_width) + separator + food_line)


#### Bonus cards ####


def _log_bonus_cards(engine: "core.Engine", player: state.Player) -> None:
    """Log each bonus card with its scoring text and current VP, if any."""
    if not player.bonus_cards:
        return
    engine.log("Bonus:")
    for bc in player.bonus_cards:
        engine.log(f"  {display.format_bonus(bc)}")
        engine.log(f"    {display.format_bonus_score_now(bc, player)}")
