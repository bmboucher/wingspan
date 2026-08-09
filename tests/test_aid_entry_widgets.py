# pyright: reportPrivateUsage=false
# (calls entry._collect_hand / _collect_bonus_pair / _collect_tray /
# _collect_goals directly to exercise each private per-block interactive
# branch -- deliberate)
"""Tests for the interactive typeahead wiring in ``wingspan.aid.entry`` and
``wingspan.aid.oracle`` (stage A2): every converted call site branches once
on ``Console.supports_interactive()``, routing to ``widgets.typeahead_pick``
on the True side while the False side keeps the pre-existing scripted text
path byte-for-byte (covered already by ``test_aid_oracle.py``).

``_FakeTypeahead`` stands in for ``widgets.typeahead_pick``, monkeypatched on
the ``widgets`` module itself -- both ``entry.py`` and ``oracle.py``
reference it through the module attribute, so one patch covers both call
sites. It records every call's ``(prompt, find, render, allow_blank,
initial)`` (as a ``_TypeaheadCall``) so a test can invoke the captured
``find``/``render`` directly, then replays a fixed sequence of scripted
results. ``aid_helpers.InteractiveConsole``/``interactive_console`` stand in
for a real tty without touching stdin/stdout; their ``read``/``write`` are
still scripted like ``aid_helpers.scripted_console`` so the confirm-echo
("Correct?") prompts that still flow through ``Console.ask``/``confirm`` on
the interactive path resolve headlessly too.
"""

from __future__ import annotations

import collections.abc

import pydantic
import pytest

import aid_helpers
from wingspan import cards, state
from wingspan.aid import console as console_module
from wingspan.aid import entry, oracle, placeholders, widgets
from wingspan.cards.parse import catalog

# How many of the initial hand/tray birds test fixtures below carve off the
# front of the catalog -- kept as one constant so the hand and tray slices
# below never accidentally overlap.
_HAND_SLICE_END = state.STARTING_HAND_SIZE
_TRAY_SLICE_START = _HAND_SLICE_END
_TRAY_SLICE_END = _TRAY_SLICE_START + state.TRAY_SIZE


class _TypeaheadCall[T](pydantic.BaseModel):
    """One recorded call to the faked ``widgets.typeahead_pick``: the prompt
    text, the ``find``/``render`` callables actually passed (so a test can
    invoke ``find`` directly to check what an ``exclude`` filter narrowed
    away, or ``render`` to check its output), and the
    ``allow_blank``/``initial`` keyword arguments."""

    prompt: str
    find: collections.abc.Callable[[str], list[T]]
    render: collections.abc.Callable[[T], str]
    allow_blank: bool
    initial: list[T] | None


class _FakeTypeahead[T]:
    """Records every call and replays a fixed sequence of scripted results,
    standing in for the real ``widgets.typeahead_pick`` tty shell."""

    def __init__(self, results: collections.abc.Sequence[T | None]) -> None:
        self.calls: list[_TypeaheadCall[T]] = []
        self._results: collections.deque[T | None] = collections.deque(results)

    def __call__(
        self,
        con: console_module.Console,
        prompt: str,
        find: collections.abc.Callable[[str], list[T]],
        render: collections.abc.Callable[[T], str],
        *,
        allow_blank: bool = False,
        initial: collections.abc.Sequence[T] | None = None,
    ) -> T | None:
        """Record this call's arguments, then pop and return the next
        scripted result."""
        self.calls.append(
            _TypeaheadCall(
                prompt=prompt,
                find=find,
                render=render,
                allow_blank=allow_blank,
                initial=list(initial) if initial is not None else None,
            )
        )
        return self._results.popleft()


def _fresh_oracle(
    con: console_module.Console,
) -> tuple[oracle.SessionOracle, placeholders.PlaceholderRegistry]:
    """A ``SessionOracle`` wired to ``con`` and a fresh placeholder
    registry."""
    echo = console_module.LogEcho(con)
    registry = placeholders.PlaceholderRegistry()
    return oracle.SessionOracle(con, echo, registry), registry


# ---------------------------------------------------------------------------
# identify_bird / identify_bonus


def test_identify_bird_interactive_routes_through_typeahead_and_excludes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bird = catalog.birds_ordered()[5]
    fake = _FakeTypeahead[cards.Bird]([bird, bird])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con = aid_helpers.InteractiveConsole()

    picked = entry.identify_bird(con, "Which bird?")

    assert picked is bird
    assert len(fake.calls) == 1
    first_call = fake.calls[0]
    assert first_call.prompt == "Which bird?"
    assert bird in first_call.find(bird.name)

    entry.identify_bird(con, "Which bird?", exclude=[bird])

    second_call = fake.calls[1]
    assert bird not in second_call.find(bird.name)


def test_identify_bird_fallback_scripted_console_resolves_by_name() -> None:
    bird = catalog.birds_ordered()[5]
    con, _transcript = aid_helpers.scripted_console([bird.name])

    picked = entry.identify_bird(con, "Which bird?")

    assert picked.name == bird.name


def test_identify_bonus_interactive_routes_through_typeahead_and_excludes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bonus_card = catalog.bonus_cards_ordered()[3]
    fake = _FakeTypeahead[cards.BonusCard]([bonus_card, bonus_card])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con = aid_helpers.InteractiveConsole()

    picked = entry.identify_bonus(con, "Which bonus card?")

    assert picked is bonus_card
    first_call = fake.calls[0]
    assert bonus_card in first_call.find(bonus_card.name)

    entry.identify_bonus(con, "Which bonus card?", exclude=[bonus_card])

    second_call = fake.calls[1]
    assert bonus_card not in second_call.find(bonus_card.name)


def test_identify_bonus_fallback_scripted_console_resolves_by_name() -> None:
    bonus_card = catalog.bonus_cards_ordered()[3]
    con, _transcript = aid_helpers.scripted_console([bonus_card.name])

    picked = entry.identify_bonus(con, "Which bonus card?")

    assert picked.name == bonus_card.name


# ---------------------------------------------------------------------------
# _collect_hand


def test_collect_hand_interactive_dedup_and_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    birds = catalog.birds_ordered()[:_HAND_SLICE_END]
    fake = _FakeTypeahead[cards.Bird](birds)
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, transcript = aid_helpers.interactive_console([""])  # "Correct?" -> blank = yes

    hand = entry._collect_hand(con)

    assert hand == tuple(birds)
    assert len(fake.calls) == state.STARTING_HAND_SIZE
    assert f"1 of {state.STARTING_HAND_SIZE}" in fake.calls[0].prompt
    assert (
        f"{state.STARTING_HAND_SIZE} of {state.STARTING_HAND_SIZE}"
        in fake.calls[-1].prompt
    )

    last_find = fake.calls[-1].find
    assert birds[0] not in last_find(birds[0].name)

    assert any("Correct?" in line for line in transcript)


# ---------------------------------------------------------------------------
# _collect_bonus_pair


def test_collect_bonus_pair_interactive_excludes_first_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bonus_cards = catalog.bonus_cards_ordered()[: state.STARTING_BONUS_CARDS_DEAL]
    fake = _FakeTypeahead[cards.BonusCard](bonus_cards)
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([""])

    bonus_pair = entry._collect_bonus_pair(con)

    assert bonus_pair == tuple(bonus_cards)
    second_find = fake.calls[1].find
    assert bonus_cards[0] not in second_find(bonus_cards[0].name)


# ---------------------------------------------------------------------------
# _collect_tray


def test_collect_tray_interactive_excludes_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand = tuple(catalog.birds_ordered()[:_HAND_SLICE_END])
    tray_birds = catalog.birds_ordered()[_TRAY_SLICE_START:_TRAY_SLICE_END]
    fake = _FakeTypeahead[cards.Bird](tray_birds)
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([""])

    tray = entry._collect_tray(con, hand)

    assert tray == tuple(tray_birds)
    first_find = fake.calls[0].find
    assert hand[0] not in first_find(hand[0].name)


# ---------------------------------------------------------------------------
# _collect_goals


def test_collect_goals_interactive_initial_shrinks_and_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, all_goals = cards.load_all()
    picks = all_goals[: len(state.ROUND_CUBES)]
    fake = _FakeTypeahead[cards.EndRoundGoal](picks)
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([""])

    goals = entry._collect_goals(con)

    assert goals == tuple(picks)
    assert len(fake.calls) == len(state.ROUND_CUBES)

    first_initial = fake.calls[0].initial
    assert first_initial is not None
    assert len(first_initial) == len(all_goals)

    last_initial = fake.calls[-1].initial
    assert last_initial is not None
    assert len(last_initial) == len(all_goals) - (len(state.ROUND_CUBES) - 1)

    render = fake.calls[0].render
    assert render(picks[0]) == picks[0].description


# ---------------------------------------------------------------------------
# SessionOracle.reveal_bird / reveal_bonus


def test_reveal_bird_interactive_blank_mints_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTypeahead[cards.Bird]([None])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle, registry = _fresh_oracle(con)

    revealed = session_oracle.reveal_bird()

    assert registry.is_placeholder(revealed)
    assert fake.calls[0].allow_blank is True


def test_reveal_bird_interactive_returns_picked_card_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bird = catalog.birds_ordered()[9]
    fake = _FakeTypeahead[cards.Bird]([bird])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle, registry = _fresh_oracle(con)

    revealed = session_oracle.reveal_bird()

    assert revealed is bird
    assert not registry.is_placeholder(revealed)


def test_reveal_bonus_interactive_blank_mints_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTypeahead[cards.BonusCard]([None])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle, registry = _fresh_oracle(con)

    revealed = session_oracle.reveal_bonus()

    assert registry.is_placeholder(revealed)
    assert fake.calls[0].allow_blank is True


def test_reveal_bonus_interactive_returns_picked_card_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bonus_card = catalog.bonus_cards_ordered()[6]
    fake = _FakeTypeahead[cards.BonusCard]([bonus_card])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle, registry = _fresh_oracle(con)

    revealed = session_oracle.reveal_bonus()

    assert revealed is bonus_card
    assert not registry.is_placeholder(revealed)


def test_reveal_bird_queued_reveal_skips_widget_even_when_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTypeahead[cards.Bird]([])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle, _registry = _fresh_oracle(con)
    queued_bird = catalog.birds_ordered()[3]
    session_oracle.queue_bird_reveals([queued_bird])

    revealed = session_oracle.reveal_bird()

    assert revealed is queued_bird
    assert fake.calls == []


def test_reveal_bonus_queued_reveal_skips_widget_even_when_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTypeahead[cards.BonusCard]([])
    monkeypatch.setattr(widgets, "typeahead_pick", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle, _registry = _fresh_oracle(con)
    queued_bonus = catalog.bonus_cards_ordered()[1]
    session_oracle.queue_bonus_reveals([queued_bonus])

    revealed = session_oracle.reveal_bonus()

    assert revealed is queued_bonus
    assert fake.calls == []
