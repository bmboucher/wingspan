# pyright: reportPrivateUsage=false
# (calls entry._collect_feeder / relay._ask_distinct_foods directly to
# exercise each private interactive branch -- deliberate)
"""Tests for the counts-widget wiring in ``wingspan.aid.entry``,
``wingspan.aid.oracle``, and ``wingspan.aid.relay`` (stage A3): every
converted call site branches once on ``Console.supports_interactive()``,
routing to ``widgets.counts_entry`` on the True side while the False side
keeps the pre-existing die-face/comma-separated text path, with one
sanctioned improvement -- the die-face fallback's retry message now names
any unrecognized token instead of only restating the grammar.

``_FakeCounts`` stands in for ``widgets.counts_entry``, monkeypatched on the
``widgets`` module itself -- ``entry.py``, ``oracle.py``, and ``relay.py``
all reference it through the module attribute, so one patch covers every
call site. It records every call's ``(prompt, labels, target_total,
field_caps)`` (as a ``_CountsCall``) then replays a fixed sequence of
scripted ``list[int]`` results. ``aid_helpers.InteractiveConsole``/
``interactive_console`` stand in for a real tty without touching
stdin/stdout, mirroring ``test_aid_entry_widgets.py``'s stage-A2
conventions.
"""

from __future__ import annotations

import collections
import collections.abc

import pydantic
import pytest

import aid_helpers
from wingspan import cards, state
from wingspan.aid import console as console_module
from wingspan.aid import entry, models, oracle, placeholders, relay, widgets

# The counts-widget field labels a die-face entry (feeder roll / out-of-feeder
# dice) offers: each food's display value, in ``cards.ALL_FOODS`` order, plus
# the choice-face token.
_DIE_FACE_LABELS = [food.value for food in cards.ALL_FOODS] + [oracle.CHOICE_FACE_TOKEN]


class _CountsCall(pydantic.BaseModel):
    """One recorded call to the faked ``widgets.counts_entry``: the prompt
    text, the field labels, the target total, and any per-field caps."""

    prompt: str
    labels: list[str]
    target_total: int
    field_caps: list[int] | None


class _FakeCounts:
    """Records every call and replays a fixed sequence of scripted
    ``list[int]`` results, standing in for the real ``widgets.counts_entry``
    tty shell."""

    def __init__(self, results: collections.abc.Sequence[list[int]]) -> None:
        self.calls: list[_CountsCall] = []
        self._results: collections.deque[list[int]] = collections.deque(results)

    def __call__(
        self,
        con: console_module.Console,
        prompt: str,
        labels: collections.abc.Sequence[str],
        target_total: int,
        *,
        field_caps: collections.abc.Sequence[int] | None = None,
    ) -> list[int]:
        """Record this call's arguments, then pop and return the next
        scripted result."""
        self.calls.append(
            _CountsCall(
                prompt=prompt,
                labels=list(labels),
                target_total=target_total,
                field_caps=list(field_caps) if field_caps is not None else None,
            )
        )
        return self._results.popleft()


def _fresh_oracle(
    con: console_module.Console,
) -> oracle.SessionOracle:
    """A ``SessionOracle`` wired to ``con`` and a fresh placeholder
    registry."""
    echo = console_module.LogEcho(con)
    registry = placeholders.PlaceholderRegistry()
    return oracle.SessionOracle(con, echo, registry)


# ---------------------------------------------------------------------------
# entry._collect_feeder


def test_collect_feeder_interactive_splits_counts_and_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCounts([[1, 1, 1, 1, 0, 1]])  # 4 single faces + 1 choice = 5
    monkeypatch.setattr(widgets, "counts_entry", fake)
    con, transcript = aid_helpers.interactive_console([""])  # "Correct?" -> yes

    feeder = entry._collect_feeder(con)

    assert feeder.counts == [1, 1, 1, 1, 0]
    assert feeder.choice_dice == 1
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call.labels == _DIE_FACE_LABELS
    assert call.target_total == state.BIRDFEEDER_DICE
    assert any("Correct?" in line for line in transcript)


# ---------------------------------------------------------------------------
# SessionOracle.feeder_roll


def test_feeder_roll_interactive_splits_counts_and_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCounts([[0, 2, 1, 0, 0, 2]])  # 3 single faces + 2 choice = 5
    monkeypatch.setattr(widgets, "counts_entry", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle = _fresh_oracle(con)

    feeder = session_oracle.feeder_roll()

    assert feeder.counts == [0, 2, 1, 0, 0]
    assert feeder.choice_dice == 2
    assert len(fake.calls) == 1
    assert fake.calls[0].labels == _DIE_FACE_LABELS
    assert fake.calls[0].target_total == state.BIRDFEEDER_DICE


def test_feeder_roll_queued_entry_skips_widget_even_when_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCounts([])
    monkeypatch.setattr(widgets, "counts_entry", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle = _fresh_oracle(con)
    queued = models.FeederEntry(counts=[1, 1, 1, 1, 0], choice_dice=1)
    session_oracle.queue_feeder_roll(queued)

    feeder = session_oracle.feeder_roll()

    assert feeder == queued
    assert fake.calls == []


# ---------------------------------------------------------------------------
# SessionOracle.dice_roll


def test_dice_roll_interactive_uses_n_as_target_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCounts([[1, 0, 1, 0, 0, 0]])  # 2 single faces, 0 choice = 2
    monkeypatch.setattr(widgets, "counts_entry", fake)
    con, _transcript = aid_helpers.interactive_console([])
    session_oracle = _fresh_oracle(con)

    pool, choice_dice = session_oracle.dice_roll(2)

    assert pool.counts == [1, 0, 1, 0, 0]
    assert choice_dice == 0
    assert len(fake.calls) == 1
    assert fake.calls[0].labels == _DIE_FACE_LABELS
    assert fake.calls[0].target_total == 2


# ---------------------------------------------------------------------------
# relay._ask_distinct_foods


def test_ask_distinct_foods_interactive_caps_each_field_at_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCounts([[1, 0, 1, 0, 0]])  # invertebrate + fish
    monkeypatch.setattr(widgets, "counts_entry", fake)
    con = aid_helpers.InteractiveConsole()

    kept = relay._ask_distinct_foods(con, 2)

    assert kept == (cards.Food.INVERTEBRATE, cards.Food.FISH)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call.labels == [food.value for food in cards.ALL_FOODS]
    assert call.target_total == 2
    assert call.field_caps == [1] * len(cards.ALL_FOODS)


# ---------------------------------------------------------------------------
# invalid_die_face_tokens


def test_invalid_die_face_tokens_returns_bad_tokens_in_order() -> None:
    assert oracle.invalid_die_face_tokens("seed xyz fish abc") == ["xyz", "abc"]


def test_invalid_die_face_tokens_empty_when_all_valid() -> None:
    assert oracle.invalid_die_face_tokens("seed fish choice rodent") == []


def test_invalid_die_face_tokens_choice_token_is_case_insensitive() -> None:
    assert oracle.invalid_die_face_tokens("CHOICE ChOiCe choice") == []


# ---------------------------------------------------------------------------
# Fallback retry message: bad token vs. wrong-count-all-valid


def test_dice_roll_fallback_names_unrecognized_token() -> None:
    con, transcript = aid_helpers.scripted_console(["xyz fish", "fish rodent"])
    session_oracle = _fresh_oracle(con)

    counts, _choice_dice = session_oracle.dice_roll(2)

    assert counts.counts == [0, 0, 1, 0, 1]
    assert any("xyz" in line for line in transcript)


def test_dice_roll_fallback_wrong_count_all_valid_keeps_generic_message() -> None:
    con, transcript = aid_helpers.scripted_console(["fish", "fish rodent"])
    session_oracle = _fresh_oracle(con)

    session_oracle.dice_roll(2)

    assert any("Enter exactly" in line for line in transcript)
    assert not any("Unrecognized" in line for line in transcript)


def test_collect_feeder_fallback_names_unrecognized_token() -> None:
    con, transcript = aid_helpers.scripted_console(
        ["seed seed fish choice bogus", "seed seed fish choice choice", ""]
    )

    feeder = entry._collect_feeder(con)

    assert feeder.choice_dice == 2
    assert any("bogus" in line for line in transcript)


def test_collect_feeder_fallback_wrong_count_all_valid_keeps_generic_message() -> None:
    con, transcript = aid_helpers.scripted_console(
        ["seed seed fish choice", "seed seed fish choice choice", ""]
    )

    entry._collect_feeder(con)

    assert any("Enter exactly" in line for line in transcript)
    assert not any("Unrecognized" in line for line in transcript)
