"""Verbose turn-state summary lines for the game log.

Provides :func:`log_turn_summary`, called once at the top of every turn from
``engine.core._take_turn``, to print a detailed snapshot of the game state so
a reader of the log does not need to reconstruct context from the event stream.
"""

from __future__ import annotations

import typing

from wingspan import cards, state
from wingspan.agents import display
from wingspan.engine import scoring

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# Food abbreviations for the feeder line.
_FOOD_ABBR: dict[cards.Food, str] = {
    cards.Food.INVERTEBRATE: "inv",
    cards.Food.SEED: "seed",
    cards.Food.FISH: "fish",
    cards.Food.FRUIT: "berry",
    cards.Food.RODENT: "rod",
}

# Column headers for the score table.
_SCORE_HEADERS = ["Birds", "Eggs", "Tuck", "Cache", "Bonus", "Goals", "TOTAL"]

# Column headers for the food table (aligned to cards.ALL_FOODS order).
_FOOD_HEADERS = ["Inv", "Seed", "Fish", "Berry", "Rod"]


def log_turn_summary(engine: "core.Engine") -> None:
    """Emit a detailed snapshot of the game state at the start of a turn.

    Writes the active player's board (by habitat), hand, tray, feeder, a
    combined score+food table, and their bonus card status into the game log."""
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
    """Log each tray card on its own line(s), then the feeder food counts."""
    engine.log("Tray:")
    for bird in gs.tray:
        if bird is not None:
            bird_text = display.format_bird_full(bird, indent="        ")
            _log_multiline(engine, bird_text, indent="  ")
        else:
            engine.log("  (empty slot)")

    feeder_parts = [
        f"{abbr}={gs.birdfeeder.counts[food]}" for food, abbr in _FOOD_ABBR.items()
    ]
    if gs.birdfeeder.choice_dice:
        feeder_parts.append(f"+{gs.birdfeeder.choice_dice} choice")
    engine.log("Feeder: " + "  ".join(feeder_parts))


#### Score + food tables ####


def _score_row(player: state.Player) -> list[int]:
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


def _food_row(player: state.Player) -> list[int]:
    """Compute the 5 food columns for one player."""
    return [player.food[food] for food in cards.ALL_FOODS]


def _table_lines(
    player_names: list[str],
    headers: list[str],
    rows: list[list[int]],
) -> list[str]:
    """Format a table as 3 strings: header row, then one row per player.

    Columns are right-aligned; each column is as wide as the widest of its
    header and all player values in that column."""
    col_widths = [
        max(len(headers[col_idx]), *(len(str(row[col_idx])) for row in rows))
        for col_idx in range(len(headers))
    ]
    name_width = max(len(name) for name in player_names)

    def _format_row(label: str, values: list[int]) -> str:
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
        _format_row(player_names[row_idx], rows[row_idx])
        for row_idx in range(len(rows))
    ]
    return [header_line] + data_lines


def _log_score_food_tables(engine: "core.Engine", gs: state.GameState) -> None:
    """Log score and food counts as a single combined side-by-side table."""
    player_names = [player.name for player in gs.players]
    score_rows = [_score_row(player) for player in gs.players]
    food_rows = [_food_row(player) for player in gs.players]

    score_lines = _table_lines(player_names, _SCORE_HEADERS, score_rows)
    food_lines = _table_lines(player_names, _FOOD_HEADERS, food_rows)

    # Pad score lines to uniform width so the food table aligns vertically.
    score_width = max(len(line) for line in score_lines)
    separator = "  ║  "
    for score_line, food_line in zip(score_lines, food_lines):
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
