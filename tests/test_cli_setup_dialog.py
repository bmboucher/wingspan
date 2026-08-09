"""Tests for the split-aware setup sub-dialog in ``wingspan.agents.cli``.

Under the split-setup regimes (``split_setup_bonus`` / ``split_setup_food``),
the engine's ``SetupDecision`` offers ``SetupChoice``s that pin the deferred
axis (or axes) to their empty value on every option: ``bonus_card=None`` for
a deferred bonus pick, ``kept_foods=()`` for a deferred food pick. These
tests exercise ``setup_dialog_axes`` directly against hand-built choice
lists, and ``resolve_setup_choice_dialog`` end-to-end by monkeypatching the
``interactive.select_form`` seam with a recording fake that both records
which sections were presented and returns scripted answers.
"""

from __future__ import annotations

import typing

import pytest

from wingspan import cards, decisions
from wingspan.agents import cli as agents_cli
from wingspan.agents import interactive

_BIRDS, _BONUSES, _GOALS = cards.load_all()


class _RecordingSelectForm:
    """Fake ``interactive.select_form`` that records the sections it was
    shown (proving which sub-dialogs actually ran) and replays one scripted
    answer per call, in call order."""

    def __init__(self, responses: list[list[list[int]]]) -> None:
        self._responses = responses
        self.calls: list[list[interactive.Section]] = []

    def __call__(
        self,
        sections: typing.Sequence[interactive.Section],
        *,
        header: str,
        instructions: str = "",
        live_options: typing.Callable[[list[set[int]]], list[list[str]]] | None = None,
        live_footer: typing.Callable[[list[set[int]]], list[str]] | None = None,
    ) -> list[list[int]]:
        response = self._responses[len(self.calls)]
        self.calls.append(list(sections))
        return response


def _setup_choice(
    kept_cards: tuple[cards.Bird, ...] = (),
    kept_foods: tuple[cards.Food, ...] = (),
    bonus_card: cards.BonusCard | None = None,
) -> decisions.SetupChoice:
    return decisions.SetupChoice(
        kept_cards=kept_cards, kept_foods=kept_foods, bonus_card=bonus_card
    )


# ---------------------------------------------------------------------------
# setup_dialog_axes — direct unit tests


def test_axes_both_split_when_every_choice_empty_on_both() -> None:
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=_BIRDS[:2],
        dealt_bonus=_BONUSES[:2],
        choices=[
            _setup_choice(kept_cards=(_BIRDS[0],)),
            _setup_choice(kept_cards=()),
        ],
    )
    assert agents_cli.setup_dialog_axes(decision) == (False, False)


def test_axes_bonus_only_when_every_choice_carries_a_bonus_but_no_foods() -> None:
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=_BIRDS[:2],
        dealt_bonus=_BONUSES[:2],
        choices=[
            _setup_choice(kept_cards=(_BIRDS[0],), bonus_card=_BONUSES[0]),
            _setup_choice(kept_cards=(), bonus_card=_BONUSES[1]),
        ],
    )
    assert agents_cli.setup_dialog_axes(decision) == (True, False)


def test_axes_food_only_when_every_choice_carries_foods_but_no_bonus() -> None:
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=_BIRDS[:1],
        dealt_bonus=_BONUSES[:2],
        choices=[
            _setup_choice(
                kept_cards=(_BIRDS[0],), kept_foods=tuple(cards.ALL_FOODS[:4])
            ),
            _setup_choice(kept_cards=(), kept_foods=tuple(cards.ALL_FOODS)),
        ],
    )
    assert agents_cli.setup_dialog_axes(decision) == (False, True)


def test_axes_neither_split_when_both_present() -> None:
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=_BIRDS[:1],
        dealt_bonus=_BONUSES[:2],
        choices=[
            _setup_choice(
                kept_cards=(_BIRDS[0],),
                kept_foods=tuple(cards.ALL_FOODS[:4]),
                bonus_card=_BONUSES[0],
            ),
        ],
    )
    assert agents_cli.setup_dialog_axes(decision) == (True, True)


def test_axes_mixed_choices_any_over_all_not_first() -> None:
    """A single choice carrying a bonus is enough to flip ``ask_bonus`` True,
    even when ``choices[0]`` does not — ``any(...)`` must scan every choice,
    not just the first."""
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=_BIRDS[:2],
        dealt_bonus=_BONUSES[:2],
        choices=[
            _setup_choice(kept_cards=(_BIRDS[0],), bonus_card=None),
            _setup_choice(kept_cards=(_BIRDS[1],), bonus_card=_BONUSES[0]),
        ],
    )
    assert agents_cli.setup_dialog_axes(decision) == (True, False)


# ---------------------------------------------------------------------------
# resolve_setup_choice_dialog — end-to-end sub-dialog behavior


def test_both_axes_split_asks_only_for_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    dealt_cards = _BIRDS[:2]
    dealt_bonus = _BONUSES[:2]
    choice_keep_first = _setup_choice(kept_cards=(dealt_cards[0],))
    choice_keep_none = _setup_choice(kept_cards=())
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
        choices=[choice_keep_first, choice_keep_none],
    )

    # One select_form call for the cards/bonus step (bird section only) picks
    # index 0 (keep the first dealt bird); no second call for foods.
    fake = _RecordingSelectForm(responses=[[[0]]])
    monkeypatch.setattr(interactive, "select_form", fake)

    result = agents_cli.resolve_setup_choice_dialog(decision, tray=[])

    assert result == choice_keep_first
    assert len(fake.calls) == 1, "food dialog must not run when food is split"
    assert len(fake.calls[0]) == 1, "no bonus section when bonus is split"
    (bird_section,) = fake.calls[0]
    assert bird_section.mode is interactive.Mode.MULTI


def test_bonus_only_split_asks_for_food_not_bonus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dealt_cards = _BIRDS[:1]
    dealt_bonus = _BONUSES[:2]
    non_kept_foods = tuple(cards.ALL_FOODS[1:])  # keep 4 foods after 1 kept card
    choice = _setup_choice(kept_cards=(dealt_cards[0],), kept_foods=non_kept_foods)
    other_choice = _setup_choice(kept_cards=(), kept_foods=tuple(cards.ALL_FOODS))
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
        choices=[choice, other_choice],
    )
    food_indices = [cards.ALL_FOODS.index(food) for food in non_kept_foods]

    fake = _RecordingSelectForm(
        responses=[
            [[0]],  # step 1: keep the one dealt bird (no bonus section)
            [food_indices],  # step 2: keep the matching foods
        ]
    )
    monkeypatch.setattr(interactive, "select_form", fake)

    result = agents_cli.resolve_setup_choice_dialog(decision, tray=[])

    assert result == choice
    assert len(fake.calls) == 2, "food dialog must run when food is not split"
    assert len(fake.calls[0]) == 1, "no bonus section when bonus is split"


def test_food_only_split_asks_for_bonus_not_food(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dealt_cards = _BIRDS[:1]
    dealt_bonus = _BONUSES[:2]
    choice = _setup_choice(kept_cards=(dealt_cards[0],), bonus_card=dealt_bonus[1])
    other_choice = _setup_choice(kept_cards=(), bonus_card=dealt_bonus[0])
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
        choices=[choice, other_choice],
    )

    # Step 1 has both a bird section and a bonus section: pick bird index 0
    # and bonus index 1 (matching dealt_bonus[1]). No step 2 call.
    fake = _RecordingSelectForm(responses=[[[0], [1]]])
    monkeypatch.setattr(interactive, "select_form", fake)

    result = agents_cli.resolve_setup_choice_dialog(decision, tray=[])

    assert result == choice
    assert len(fake.calls) == 1, "food dialog must not run when food is split"
    assert len(fake.calls[0]) == 2, "bonus section must appear when bonus is not split"
    bird_section, bonus_section = fake.calls[0]
    assert bird_section.mode is interactive.Mode.MULTI
    assert bonus_section.mode is interactive.Mode.SINGLE


def test_non_split_asks_for_everything_like_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dealt_cards = _BIRDS[:1]
    dealt_bonus = _BONUSES[:2]
    kept_foods = tuple(cards.ALL_FOODS[1:])
    choice = _setup_choice(
        kept_cards=(dealt_cards[0],),
        kept_foods=kept_foods,
        bonus_card=dealt_bonus[0],
    )
    other_choice = _setup_choice(
        kept_cards=(), kept_foods=tuple(cards.ALL_FOODS), bonus_card=dealt_bonus[1]
    )
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
        choices=[choice, other_choice],
    )
    food_indices = [cards.ALL_FOODS.index(food) for food in kept_foods]

    fake = _RecordingSelectForm(
        responses=[
            [[0], [0]],  # step 1: keep the bird, pick bonus index 0
            [food_indices],  # step 2: keep the matching foods
        ]
    )
    monkeypatch.setattr(interactive, "select_form", fake)

    result = agents_cli.resolve_setup_choice_dialog(decision, tray=[])

    assert result == choice
    assert len(fake.calls) == 2
    assert len(fake.calls[0]) == 2, "both bird and bonus sections presented"
    assert len(fake.calls[1]) == 1, "food section presented"


def test_empty_dealt_bonus_omits_bonus_section_without_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dealt_cards = _BIRDS[:1]
    kept_foods = tuple(cards.ALL_FOODS[1:])
    choice = _setup_choice(kept_cards=(dealt_cards[0],), kept_foods=kept_foods)
    other_choice = _setup_choice(kept_cards=(), kept_foods=tuple(cards.ALL_FOODS))
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose",
        dealt_cards=dealt_cards,
        dealt_bonus=[],
        choices=[choice, other_choice],
    )
    food_indices = [cards.ALL_FOODS.index(food) for food in kept_foods]

    fake = _RecordingSelectForm(
        responses=[
            [[0]],  # step 1: keep the bird (no bonus section: none dealt)
            [food_indices],  # step 2: keep the matching foods
        ]
    )
    monkeypatch.setattr(interactive, "select_form", fake)

    result = agents_cli.resolve_setup_choice_dialog(decision, tray=[])

    assert result == choice
    assert len(fake.calls[0]) == 1, "no bonus section when no bonus cards were dealt"
