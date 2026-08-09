"""Tests for ``wingspan.aid.widgets`` -- the typeahead/counts input widgets.

Covers all three layers: the pure reducers/renderers (``step_typeahead`` /
``step_counts`` / ``render_typeahead_frame`` / ``render_counts_frame``), the
headless drivers (``run_typeahead`` / ``run_counts``) exercised end-to-end
with scripted ``keys.KeyEvent`` streams and a recording draw function, and
``Console.supports_interactive`` (the capability flag layer 3 branches on --
the tty shells themselves, ``typeahead_pick`` / ``counts_entry``, are thin
wiring with no logic of their own and are not exercised here).
"""

from __future__ import annotations

import sys

import pytest

import aid_helpers
from wingspan.aid import console, models, widgets
from wingspan.training.configure import keys

# ---- step_typeahead ----


def _char(char: str) -> keys.KeyEvent:
    return keys.KeyEvent(kind=keys.KeyKind.CHAR, char=char)


def _key(kind: keys.KeyKind) -> keys.KeyEvent:
    return keys.KeyEvent(kind=kind)


def test_step_typeahead_char_appends_and_resets_highlight():
    state = models.TypeaheadState(query="w", highlight=3)
    new_state, outcome = widgets.step_typeahead(
        state, n_matches=5, event=_char("o"), allow_blank=False
    )
    assert new_state.query == "wo"
    assert new_state.highlight == 0
    assert outcome is models.TypeaheadOutcome.CONTINUE


def test_step_typeahead_backspace_drops_last_char():
    state = models.TypeaheadState(query="wo", highlight=2)
    new_state, outcome = widgets.step_typeahead(
        state, n_matches=3, event=_key(keys.KeyKind.BACKSPACE), allow_blank=False
    )
    assert new_state.query == "w"
    assert new_state.highlight == 0
    assert outcome is models.TypeaheadOutcome.CONTINUE


def test_step_typeahead_backspace_on_empty_query_is_noop():
    state = models.TypeaheadState(query="", highlight=0)
    new_state, _outcome = widgets.step_typeahead(
        state, n_matches=0, event=_key(keys.KeyKind.BACKSPACE), allow_blank=False
    )
    assert new_state.query == ""


def test_step_typeahead_up_down_wrap_within_visible_range():
    state = models.TypeaheadState(query="a", highlight=0)
    up_state, _ = widgets.step_typeahead(
        state, n_matches=3, event=_key(keys.KeyKind.UP), allow_blank=False
    )
    assert up_state.highlight == 2  # wraps to the last of 3 visible matches

    down_state, _ = widgets.step_typeahead(
        up_state, n_matches=3, event=_key(keys.KeyKind.DOWN), allow_blank=False
    )
    assert down_state.highlight == 0


def test_step_typeahead_up_down_wrap_at_max_visible_boundary():
    # 20 matches truncate to MAX_VISIBLE_MATCHES (8); highlight must never
    # reach a truncated row.
    state = models.TypeaheadState(query="a", highlight=0)
    up_state, _ = widgets.step_typeahead(
        state, n_matches=20, event=_key(keys.KeyKind.UP), allow_blank=False
    )
    assert up_state.highlight == widgets.MAX_VISIBLE_MATCHES - 1

    down_state, _ = widgets.step_typeahead(
        up_state, n_matches=20, event=_key(keys.KeyKind.DOWN), allow_blank=False
    )
    assert down_state.highlight == 0


def test_step_typeahead_up_down_noop_with_no_matches():
    state = models.TypeaheadState(query="zzz", highlight=0)
    new_state, _ = widgets.step_typeahead(
        state, n_matches=0, event=_key(keys.KeyKind.UP), allow_blank=False
    )
    assert new_state.highlight == 0


def test_step_typeahead_enter_with_matches_accepts():
    state = models.TypeaheadState(query="wo", highlight=1)
    new_state, outcome = widgets.step_typeahead(
        state, n_matches=3, event=_key(keys.KeyKind.ENTER), allow_blank=False
    )
    assert outcome is models.TypeaheadOutcome.ACCEPT
    assert new_state.highlight == 1


def test_step_typeahead_enter_empty_query_allow_blank_accepts_blank():
    state = models.TypeaheadState(query="", highlight=0)
    _new_state, outcome = widgets.step_typeahead(
        state, n_matches=0, event=_key(keys.KeyKind.ENTER), allow_blank=True
    )
    assert outcome is models.TypeaheadOutcome.ACCEPT_BLANK


def test_step_typeahead_enter_no_matches_no_allow_blank_continues():
    state = models.TypeaheadState(query="zzz", highlight=0)
    _new_state, outcome = widgets.step_typeahead(
        state, n_matches=0, event=_key(keys.KeyKind.ENTER), allow_blank=False
    )
    assert outcome is models.TypeaheadOutcome.CONTINUE


def test_step_typeahead_enter_no_matches_allow_blank_nonempty_query_continues():
    state = models.TypeaheadState(query="zzz", highlight=0)
    _new_state, outcome = widgets.step_typeahead(
        state, n_matches=0, event=_key(keys.KeyKind.ENTER), allow_blank=True
    )
    assert outcome is models.TypeaheadOutcome.CONTINUE


def test_step_typeahead_escape_clears_query():
    state = models.TypeaheadState(query="wood", highlight=2)
    new_state, outcome = widgets.step_typeahead(
        state, n_matches=3, event=_key(keys.KeyKind.ESCAPE), allow_blank=False
    )
    assert new_state.query == ""
    assert new_state.highlight == 0
    assert outcome is models.TypeaheadOutcome.CONTINUE


def test_step_typeahead_interrupt_raises():
    state = models.TypeaheadState()
    with pytest.raises(KeyboardInterrupt):
        widgets.step_typeahead(
            state, n_matches=0, event=_key(keys.KeyKind.INTERRUPT), allow_blank=False
        )


def test_step_typeahead_other_keys_are_noop():
    state = models.TypeaheadState(query="a", highlight=0)
    new_state, outcome = widgets.step_typeahead(
        state, n_matches=3, event=_key(keys.KeyKind.TAB), allow_blank=False
    )
    assert new_state.query == "a"
    assert new_state.highlight == 0
    assert outcome is models.TypeaheadOutcome.CONTINUE


def test_step_typeahead_does_not_mutate_input_state():
    state = models.TypeaheadState(query="a", highlight=2)
    widgets.step_typeahead(state, n_matches=3, event=_char("b"), allow_blank=False)
    assert state.query == "a"
    assert state.highlight == 2


# ---- step_counts ----


def test_step_counts_left_right_wrap_focus():
    state = models.CountsState(values=[0, 0, 0], focus=0)
    left_state, _ = widgets.step_counts(
        state, target_total=5, event=_key(keys.KeyKind.LEFT)
    )
    assert left_state.focus == 2  # wraps below zero to the last field

    right_state, _ = widgets.step_counts(
        left_state, target_total=5, event=_key(keys.KeyKind.RIGHT)
    )
    assert right_state.focus == 0


def test_step_counts_down_clamped_at_zero():
    state = models.CountsState(values=[0, 0], focus=0)
    new_state, _ = widgets.step_counts(
        state, target_total=5, event=_key(keys.KeyKind.DOWN)
    )
    assert new_state.values[0] == 0


def test_step_counts_up_clamped_at_target_total_by_default():
    state = models.CountsState(values=[5, 0], focus=0)
    new_state, _ = widgets.step_counts(
        state, target_total=5, event=_key(keys.KeyKind.UP)
    )
    assert new_state.values[0] == 5


def test_step_counts_field_caps_respected():
    state = models.CountsState(values=[2, 0], focus=0)
    new_state, _ = widgets.step_counts(
        state, target_total=5, event=_key(keys.KeyKind.UP), field_caps=[2, 10]
    )
    assert new_state.values[0] == 2  # capped below target_total


def test_step_counts_digit_char_sets_value():
    state = models.CountsState(values=[0, 0], focus=1)
    new_state, _ = widgets.step_counts(state, target_total=5, event=_char("3"))
    assert new_state.values == [0, 3]


def test_step_counts_digit_char_respects_cap():
    state = models.CountsState(values=[0], focus=0)
    new_state, _ = widgets.step_counts(
        state, target_total=5, event=_char("9"), field_caps=[3]
    )
    assert new_state.values == [3]


def test_step_counts_non_digit_char_ignored():
    state = models.CountsState(values=[2], focus=0)
    new_state, _ = widgets.step_counts(state, target_total=5, event=_char("x"))
    assert new_state.values == [2]


def test_step_counts_backspace_zeroes_focused_only():
    state = models.CountsState(values=[3, 4], focus=1)
    new_state, _ = widgets.step_counts(
        state, target_total=10, event=_key(keys.KeyKind.BACKSPACE)
    )
    assert new_state.values == [3, 0]


def test_step_counts_escape_zeroes_all_and_resets_focus():
    state = models.CountsState(values=[3, 4], focus=1)
    new_state, _ = widgets.step_counts(
        state, target_total=10, event=_key(keys.KeyKind.ESCAPE)
    )
    assert new_state.values == [0, 0]
    assert new_state.focus == 0


def test_step_counts_enter_accepted_only_when_sum_matches():
    matching = models.CountsState(values=[2, 3], focus=0)
    _new_state, accepted = widgets.step_counts(
        matching, target_total=5, event=_key(keys.KeyKind.ENTER)
    )
    assert accepted is True

    mismatched = models.CountsState(values=[2, 2], focus=0)
    _new_state, accepted = widgets.step_counts(
        mismatched, target_total=5, event=_key(keys.KeyKind.ENTER)
    )
    assert accepted is False


def test_step_counts_non_enter_events_never_accept():
    state = models.CountsState(values=[5], focus=0)
    _new_state, accepted = widgets.step_counts(
        state, target_total=5, event=_key(keys.KeyKind.UP)
    )
    assert accepted is False


def test_step_counts_interrupt_raises():
    state = models.CountsState(values=[0], focus=0)
    with pytest.raises(KeyboardInterrupt):
        widgets.step_counts(state, target_total=5, event=_key(keys.KeyKind.INTERRUPT))


def test_step_counts_does_not_mutate_input_state():
    state = models.CountsState(values=[1, 2], focus=0)
    widgets.step_counts(state, target_total=5, event=_char("9"))
    assert state.values == [1, 2]
    assert state.focus == 0


# ---- render_typeahead_frame ----


def test_render_typeahead_frame_empty_query_hint_no_allow_blank():
    lines = widgets.render_typeahead_frame(
        "Pick a bird:", models.TypeaheadState(), [], allow_blank=False
    )
    assert lines == [
        "Pick a bird:",
        "> ",
        "  (type to search)",
    ]


def test_render_typeahead_frame_empty_query_hint_allow_blank():
    lines = widgets.render_typeahead_frame(
        "Pick a bird:", models.TypeaheadState(), [], allow_blank=True
    )
    assert lines == [
        "Pick a bird:",
        "> ",
        "  (type to search; Enter = face-down/unknown)",
    ]


def test_render_typeahead_frame_no_matches_nonempty_query():
    state = models.TypeaheadState(query="xyz")
    lines = widgets.render_typeahead_frame("Pick a bird:", state, [], allow_blank=False)
    assert lines == [
        "Pick a bird:",
        "> xyz",
        "  (no matches — Backspace or Esc to edit)",
    ]


def test_render_typeahead_frame_highlighted_list():
    state = models.TypeaheadState(query="wo", highlight=1)
    lines = widgets.render_typeahead_frame(
        "Pick a bird:",
        state,
        ["Wood Duck", "Wood Thrush", "Woodpecker"],
        allow_blank=False,
    )
    assert lines == [
        "Pick a bird:",
        "> wo",
        "  Wood Duck",
        "* Wood Thrush",
        "  Woodpecker",
    ]


def test_render_typeahead_frame_truncated_list():
    state = models.TypeaheadState(query="b", highlight=0)
    labels = [f"Bird {index}" for index in range(10)]
    lines = widgets.render_typeahead_frame("Pick:", state, labels, allow_blank=False)
    assert lines[:2] == ["Pick:", "> b"]
    assert lines[2] == "* Bird 0"
    assert len(lines) == 2 + widgets.MAX_VISIBLE_MATCHES + 1
    assert lines[-1] == "  ... 2 more (keep typing)"


# ---- render_counts_frame ----


def test_render_counts_frame_focused_mid_field():
    state = models.CountsState(values=[2, 1, 0], focus=1)
    lines = widgets.render_counts_frame(
        "Enter counts:", ["seed", "fish", "rodent"], state, target_total=3
    )
    assert lines[0] == "Enter counts:"
    assert lines[1] == " seed:2  [fish:1]  rodent:0 "


def test_render_counts_frame_total_line_not_matching():
    state = models.CountsState(values=[2, 2], focus=0)
    lines = widgets.render_counts_frame(
        "Enter counts:", ["seed", "fish"], state, target_total=5
    )
    assert lines[2] == "total 4/5 (Enter accepts when the total matches)"


def test_render_counts_frame_total_line_matching():
    state = models.CountsState(values=[2, 3], focus=0)
    lines = widgets.render_counts_frame(
        "Enter counts:", ["seed", "fish"], state, target_total=5
    )
    assert lines[2] == "total 5/5 — Enter to accept"


# ---- drivers: run_typeahead / run_counts ----


def _recording_draw() -> tuple[widgets.DrawFn, list[list[str]]]:
    """A ``DrawFn`` that records every frame it is asked to draw."""
    frames: list[list[str]] = []

    def draw(lines: list[str], _prev_line_count: int) -> int:
        frames.append(list(lines))
        return len(lines)

    return draw, frames


def test_run_typeahead_type_narrow_then_enter_selects():
    candidates = ["Wood Duck", "Wood Thrush", "Woodpecker", "Robin"]

    def find(query: str) -> list[str]:
        return [
            candidate for candidate in candidates if query.lower() in candidate.lower()
        ]

    draw, frames = _recording_draw()
    events = iter([_char("w"), _char("o"), _key(keys.KeyKind.ENTER)])

    picked = widgets.run_typeahead(
        events, "Pick a bird:", find, lambda item: item, draw
    )

    assert picked == "Wood Duck"
    assert len(frames) == 3  # initial + after 'w' + after 'wo'


def test_run_typeahead_initial_list_arrow_only_selection():
    def find(_query: str) -> list[str]:
        raise AssertionError("find must not be called while the query is empty")

    draw, _frames = _recording_draw()
    events = iter(
        [_key(keys.KeyKind.DOWN), _key(keys.KeyKind.DOWN), _key(keys.KeyKind.ENTER)]
    )

    picked = widgets.run_typeahead(
        events,
        "Pick:",
        find,
        lambda item: item,
        draw,
        initial=["A", "B", "C", "D"],
    )

    assert picked == "C"


def test_run_typeahead_blank_accept_returns_none():
    def find(_query: str) -> list[str]:
        return []

    draw, _frames = _recording_draw()
    events = iter([_key(keys.KeyKind.ENTER)])

    picked = widgets.run_typeahead(
        events, "Pick:", find, lambda item: item, draw, allow_blank=True
    )

    assert picked is None


def test_run_typeahead_exhausted_events_raises():
    def find(_query: str) -> list[str]:
        return []

    draw, _frames = _recording_draw()

    with pytest.raises(RuntimeError, match="typeahead event stream ended"):
        widgets.run_typeahead(iter([]), "Pick:", find, lambda item: item, draw)


def test_run_counts_rejects_wrong_total_then_accepts():
    draw, frames = _recording_draw()
    events = iter(
        [
            _char("1"),
            _key(keys.KeyKind.RIGHT),
            _char("1"),
            _key(keys.KeyKind.ENTER),  # sum is 2, target is 3 -- rejected
            _key(keys.KeyKind.UP),
            _key(keys.KeyKind.ENTER),  # sum is now 3 -- accepted
        ]
    )

    values = widgets.run_counts(events, "Enter counts:", ["seed", "fish"], 3, draw)

    assert values == [1, 2]
    assert len(frames) == 6  # initial + one redraw per non-accepting event


def test_run_counts_exhausted_events_raises():
    draw, _frames = _recording_draw()

    with pytest.raises(RuntimeError, match="counts event stream ended"):
        widgets.run_counts(iter([]), "Enter counts:", ["seed", "fish"], 3, draw)


# ---- Console.supports_interactive ----


def test_supports_interactive_false_for_scripted_console():
    con, _transcript = aid_helpers.scripted_console([])
    assert con.supports_interactive() is False


def test_supports_interactive_true_for_real_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert console.Console().supports_interactive() is True


def test_supports_interactive_false_for_non_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert console.Console().supports_interactive() is False
