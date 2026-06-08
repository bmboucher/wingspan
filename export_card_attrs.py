"""Export every core-set bird card as a CSV row.

Columns: name, power_text, then the 44 values of the card_attributes stripe
in unnormalized (raw) form — integers and booleans as they appear on the card.

Usage (from the repo root with the project venv active):

    python export_card_attrs.py > cards.csv
    python export_card_attrs.py cards.csv       # write directly to a file
"""

import csv
import sys

import numpy as np

import wingspan.cards as cards
import wingspan.encode.layout as layout
import wingspan.encode.state_encode as state_encode

# ── Column names ──────────────────────────────────────────────────────────────

_FOOD_COLS = [f"food_cost_{food.value}" for food in cards.ALL_FOODS] + ["food_cost_wild"]
_NEST_COLS = ["nest_bowl", "nest_cavity", "nest_ground", "nest_platform"]
_HABITAT_COLS = [f"habitat_{hab.value}" for hab in cards.ALL_HABITATS]
_COLOR_COLS = ["color_brown", "color_white", "color_pink", "color_yellow"]
_BONUS_COLS = [
    "bonus_anatomist",
    "bonus_backyard_birder",
    "bonus_cartographer",
    "bonus_historian",
    "bonus_large_bird_specialist",
    "bonus_passerine_specialist",
    "bonus_photographer",
]
_EXCHANGE_COLS = [
    "px_cards_to_discard",
    "px_food_to_pay",
    "px_eggs_to_pay",
    "px_food_to_gain",
    "px_eggs_to_gain",
    "px_cards_to_draw",
    "px_cards_to_tuck",
    "px_opp_food_to_gain",
    "px_opp_eggs_to_gain",
    "px_opp_cards_to_draw",
    "px_opp_cards_to_tuck",
    "px_plays_to_gain",
    "px_cache_to_gain",
]

_ATTR_COLS = (
    ["points"]
    + _FOOD_COLS
    + _NEST_COLS
    + _HABITAT_COLS
    + ["flocking", "predator", "wingspan", "egg_limit"]
    + _COLOR_COLS
    + ["plays_another_bird", "caches_food"]
    + _BONUS_COLS
    + _EXCHANGE_COLS
)

_FIELDNAMES = ["name", "power_text"] + _ATTR_COLS


def _fmt(value: float) -> str:
    """Format a value as an integer string when it is whole, else as a float."""
    return str(int(value)) if value == int(value) else str(value)


def _nest_bits(bird: cards.Bird) -> list[int]:
    """4-bit nest encoding matching the stripe layout (STAR = all ones)."""
    if bird.nest == cards.NestType.STAR:
        return [1, 1, 1, 1]
    return [1 if bird.nest == nest else 0 for nest in layout._NEST_BASE_TYPES]


def _color_bits(bird: cards.Bird) -> list[int]:
    """4-bit power-color one-hot (NONE = all zeros)."""
    return [1 if bird.color == color else 0 for color in layout._COLORS]


def _bird_row(bird: cards.Bird) -> dict[str, object]:
    """Build one CSV row for ``bird``: metadata then all 44 attr values (unnormalized)."""

    # Food cost: 5 specific foods then wild, all raw integers.
    food_counts = bird.food_cost.counts
    food_vals = list(food_counts[:5]) + [food_counts[5]]

    # Power exchange: accumulate unnormalized by scaling the normalized vector back.
    exchange_raw: np.ndarray = (
        state_encode._bird_power_exchange_vector(bird) * layout._EXCHANGE_SCALE
    )

    values: list[object] = (
        [bird.points]
        + food_vals
        + _nest_bits(bird)
        + [1 if hab in bird.habitats else 0 for hab in cards.ALL_HABITATS]
        + [int(bird.flocking), int(bird.predator)]
        + [bird.wingspan_cm, bird.egg_limit]
        + _color_bits(bird)
        + [int(bird.plays_another_bird), int(state_encode._is_caching_bird(bird))]
        + [1 if name in bird.bonus_categories else 0 for name in layout._KEPT_BONUS_NAMES]
        + [_fmt(v) for v in exchange_raw.tolist()]
    )

    row: dict[str, object] = {
        "name": bird.name,
        "power_text": bird.plain_power_text,
    }
    for col, value in zip(_ATTR_COLS, values, strict=True):
        row[col] = _fmt(float(value)) if not isinstance(value, str) else value
    return row


def main() -> None:
    """Load all core-set birds and write one CSV row per bird to stdout or a file."""
    birds, _bonuses, _goals = cards.load_all()

    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    if out_path is not None:
        file_obj = open(out_path, "w", newline="", encoding="utf-8")
    else:
        file_obj = sys.stdout

    try:
        writer = csv.DictWriter(file_obj, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for bird in birds:
            writer.writerow(_bird_row(bird))
    finally:
        if out_path is not None:
            file_obj.close()


if __name__ == "__main__":
    main()
