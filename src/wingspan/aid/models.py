"""Pydantic data shapes for the aid session.

Pure data layer for the ``wingspan aid`` package: the physical-table facts
entered at setup (:class:`SetupEntry`, :class:`FeederEntry`), the running
scratch the relay/advisor hooks consult mid-turn (:class:`TurnNotes`,
:class:`OpponentPlayNote`), the advisor's combined setup-keep recommendation
(:class:`SetupPreview`), and the end-of-session summary
(:class:`SessionReport`). No behavior beyond field-level conversions and
cross-field validation lives here -- see ``oracle.py`` / ``oracle_state.py``
for the interactive machinery that produces these shapes and ``preview.py``
for the simulation that produces a :class:`SetupPreview`.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import state
from wingspan.cards import schema

# Aid sessions are 2-player only (see the plan's "user-approved scope"), so
# the only valid seats are 0 (the user) and 1 (the opponent).
_LAST_SEAT_INDEX = 1

# Number of end-of-round goals a full setup entry carries -- one per round.
_N_ROUNDS = len(state.ROUND_CUBES)


def _new_opponent_play_note_list() -> list["OpponentPlayNote"]:
    return []


class FeederEntry(pydantic.BaseModel):
    """One birdfeeder-roll observation entered by the user: the physical
    dice face counts, aligned to :data:`wingspan.cards.ALL_FOODS`, plus how
    many dice landed on the invertebrate/seed choice face. The total die
    count must equal :data:`wingspan.state.BIRDFEEDER_DICE`."""

    counts: typing.Annotated[
        list[int],
        pydantic.Field(min_length=schema.N_FOODS, max_length=schema.N_FOODS),
    ]
    choice_dice: typing.Annotated[int, pydantic.Field(ge=0)]

    @pydantic.model_validator(mode="after")
    def _check_total_dice(self) -> "FeederEntry":
        total = sum(self.counts) + self.choice_dice
        if total != state.BIRDFEEDER_DICE:
            raise ValueError(
                f"feeder entry totals {total} dice, expected "
                f"{state.BIRDFEEDER_DICE}"
            )
        return self

    def to_food_pool(self) -> state.FoodPool:
        """The single-face counts as a :class:`wingspan.state.FoodPool`
        (``choice_dice`` stays a separate count, mirroring
        ``Birdfeeder.counts``/``Birdfeeder.choice_dice``)."""
        return state.FoodPool(counts=list(self.counts))


class SetupEntry(pydantic.BaseModel):
    """Everything dealt at the start of a physical game, as entered by the
    user: the starting hand, the two dealt bonus cards (one is kept), the
    four round goals, the initial face-up tray, the initial feeder roll, and
    which seat takes the first turn."""

    hand: typing.Annotated[
        tuple[schema.Bird, ...],
        pydantic.Field(
            min_length=state.STARTING_HAND_SIZE,
            max_length=state.STARTING_HAND_SIZE,
        ),
    ]
    bonus_pair: typing.Annotated[
        tuple[schema.BonusCard, ...],
        pydantic.Field(
            min_length=state.STARTING_BONUS_CARDS_DEAL,
            max_length=state.STARTING_BONUS_CARDS_DEAL,
        ),
    ]
    goals: typing.Annotated[
        tuple[schema.EndRoundGoal, ...],
        pydantic.Field(min_length=_N_ROUNDS, max_length=_N_ROUNDS),
    ]
    tray: typing.Annotated[
        tuple[schema.Bird, ...],
        pydantic.Field(min_length=state.TRAY_SIZE, max_length=state.TRAY_SIZE),
    ]
    feeder: FeederEntry
    start_player: typing.Annotated[int, pydantic.Field(ge=0, le=_LAST_SEAT_INDEX)]


class OpponentPlayNote(pydantic.BaseModel):
    """One bird the opponent played this turn, entered by the relay agent
    so the engine's ``PlayBirdDecision`` can substitute the real card for
    the placeholder sitting in the opponent's tracked hand (stage 3)."""

    bird: schema.Bird
    habitat: schema.Habitat


class TurnNotes(pydantic.BaseModel):
    """Mutable per-turn scratch consumed by the relay/advisor hooks (stage
    3): which opponent plays were reported this turn, and how many of them
    the subsequent decisions have already consumed."""

    plays: list[OpponentPlayNote] = pydantic.Field(
        default_factory=_new_opponent_play_note_list
    )
    play_consumed_count: int = 0

    def clear(self) -> None:
        """Reset both fields; called at the start of every turn."""
        self.plays = []
        self.play_consumed_count = 0


class SetupPreview(pydantic.BaseModel):
    """The model's own preferred setup keep, replayed through the real
    deferred-resolution code path (:mod:`wingspan.engine.setup_flow`) on a
    throwaway cloned state -- see ``preview.py``'s ``preview_setup``.

    Under a split-setup regime the advisor's setup ``SetupDecision`` only
    offers a card-keep pick, with the bonus and/or food picks resolved by
    later separate in-game decisions; this model captures what those later
    picks would resolve to under the model's own preferred keep, so the
    advisor can show one combined upfront recommendation line."""

    kept_cards: tuple[schema.Bird, ...]
    bonus_card: schema.BonusCard | None
    kept_foods: state.FoodPool

    def format_line(self) -> str:
        """Render the one-line combined recommendation, e.g. ``model
        recommends: keep [Bird A, Bird B] + bonus [Card X] + foods [fish,
        seed x2]``.

        An empty keep renders ``[none]``; the ``bonus [...]`` segment is
        omitted entirely when :attr:`bonus_card` is ``None``; foods with a
        count of exactly one are listed by name alone, higher counts get an
        ``xN`` suffix, and zero-count foods are dropped."""
        keep_names = [bird.name for bird in self.kept_cards] or ["none"]
        segments = [f"keep [{', '.join(keep_names)}]"]
        if self.bonus_card is not None:
            segments.append(f"bonus [{self.bonus_card.name}]")
        food_names = [
            food.value if count == 1 else f"{food.value} x{count}"
            for food, count in self.kept_foods.items()
            if count > 0
        ] or ["none"]
        segments.append(f"foods [{', '.join(food_names)}]")
        return "model recommends: " + " + ".join(segments)


class SessionReport(pydantic.BaseModel):
    """End-of-session results (populated in stage 4): final scores in seat
    order, the winning seat id (``None`` for a tie), and whether the
    opponent's bonus cards were entered for exact final scoring."""

    scores: list[int]
    winner: int | None
    opponent_bonus_entered: bool
