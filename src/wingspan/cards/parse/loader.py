"""Bundled wingsearch JSON loader: the core-set birds, bonus cards, and goals.

``load_all`` reads the three bundled JSON files, validates each core-set row
into an input record, and converts each via ``record.load()`` into its parsed
card model. The parse runs once per process (the card models are immutable, so
the cached instances are shared); each ``load_all`` call returns fresh outer
lists so callers may reorder or consume them freely.
"""

from __future__ import annotations

import functools
import json
from importlib import resources
from importlib.resources import abc

import pydantic

from wingspan.cards import schema

# A handful of fan-made bonus cards in the wingsearch data are mistakenly
# tagged ``"Set": "core"`` and prefixed with this marker in their name. They
# are not part of the published base game, so they are excluded at load time.
_FAN_MADE_PREFIX = "[Fan Made]"


def load_all() -> (
    tuple[list[schema.Bird], list[schema.BonusCard], list[schema.EndRoundGoal]]
):
    """Read every core-set bird, bonus card, and end-of-round goal from the
    bundled JSON data. Returns three parallel lists in source order.

    The underlying parse is cached per process; the returned lists are fresh
    copies sharing the immutable (frozen) card models, so mutating a list —
    e.g. ``state.new_game`` shuffling a deck copy — never leaks between
    callers."""
    birds, bonuses, goals = _load_all_cached()
    return list(birds), list(bonuses), list(goals)


def power_coverage(birds: list[schema.Bird]) -> tuple[int, int]:
    """Return ``(implemented, total)``. Birds with no power text are counted
    as implemented (there is nothing to model)."""
    impl = sum(
        1
        for bird in birds
        if not any(
            effect.kind == schema.EffectKind.UNIMPLEMENTED
            for effect in bird.power.effects
        )
    )
    return impl, len(birds)


@functools.cache
def _load_all_cached() -> tuple[
    tuple[schema.Bird, ...],
    tuple[schema.BonusCard, ...],
    tuple[schema.EndRoundGoal, ...],
]:
    """One-time parse of the bundled JSON into immutable card tuples (the
    shared instances behind every ``load_all`` call's list copies)."""
    base = resources.files("wingspan.data")
    bird_records = _load_core_records(base / "master.json", schema.BirdRecord)
    bonus_records = [
        bonus_record
        for bonus_record in _load_core_records(base / "bonus.json", schema.BonusRecord)
        if not bonus_record.bonus_card.startswith(_FAN_MADE_PREFIX)
    ]
    goal_records = _load_core_records(base / "goals.json", schema.GoalRecord)
    birds = tuple(
        bird
        for bird in (record.load(bonus_records) for record in bird_records)
        if bird is not None
    )
    bonuses = tuple(bonus_record.load() for bonus_record in bonus_records)
    goals = tuple(goal_record.load() for goal_record in goal_records)
    return birds, bonuses, goals


def _load_core_records[R: pydantic.BaseModel](
    path: abc.Traversable, model: type[R]
) -> list[R]:
    """Load and validate only the core-set rows from a wingsearch JSON file.

    Non-core rows are skipped before validation: other expansions use values
    (e.g. ``"Wingspan": "*"`` for variable-wingspan birds) that fall outside
    the core-set schema and would otherwise fail validation here."""
    return [
        model.model_validate(row)
        for row in json.loads(path.read_text(encoding="utf-8"))
        if row.get("Set") == "core"
    ]
