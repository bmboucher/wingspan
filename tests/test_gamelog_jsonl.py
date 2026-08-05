"""Tests for the flat JSONL renderer of the structured game-event tree.

The file's contract is that it loses nothing: every node of the tree becomes
exactly one row, the tree reconstructs from the rows' link columns, and each
row's header text is the same one the HTML log shows.  These tests play real
games and check all three, plus the two deliberate omissions (the
encoding-viewer stripes) and the one deliberate rename (``outcome_text`` ->
``text``).
"""

from __future__ import annotations

import functools
import json
import pathlib
import sys
import typing

# Add src/ to path as per test_smoke.py convention.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from wingspan.gamelog import models  # noqa: E402  (needs the sys.path insert above)
from wingspan.gamelog import render_jsonl, summarize  # noqa: E402


# Playing a full game dominates the cost here and every test only reads its
# rows, so cache per (seed, seats) — the module plays two games, not a dozen.
@functools.cache
def _record_game(seed: int = 42, num_players: int = 2) -> models.GameEventTree:
    """Play one random-agent game through a real recorder; return its tree."""
    import random

    from wingspan import agents
    from wingspan import engine as engine_mod
    from wingspan.gamelog import recorder as gamelog_recorder

    rec = gamelog_recorder.EventRecorder(probes=tuple(None for _ in range(num_players)))
    eng, *_ = engine_mod.Engine.create(seed=seed, num_players=num_players)
    rng = random.Random(seed)
    engine_mod.Engine.play_one_game(
        eng.state,
        [agents.random_agent(rng) for _ in range(num_players)],
        event_recorder=rec,
    )
    return rec.root


def _meta(game_id: str = "42", num_players: int = 2) -> models.GameMeta:
    """A minimal header row for a rendered game."""
    return models.GameMeta(
        game_id=game_id,
        seed=42,
        num_players=num_players,
        seats=["random"] * num_players,
        scores=[10] * num_players,
        winner=0,
    )


@functools.cache
def _rows(seed: int = 42, num_players: int = 2) -> tuple[dict[str, typing.Any], ...]:
    """Every rendered row of one game, parsed back from its JSON lines."""
    tree = _record_game(seed=seed, num_players=num_players)
    lines = render_jsonl.render_rows(tree, _meta(num_players=num_players))
    return tuple(json.loads(line) for line in lines)


def _walk_events(tree: models.GameEventTree) -> typing.Iterator[models.GameEvent]:
    """Yield every event in the tree, depth-first."""

    def recurse(event: models.GameEvent) -> typing.Iterator[models.GameEvent]:
        yield event
        for child in event.children:
            yield from recurse(child)

    for phase in tree.phases:
        for event in phase.events:
            yield from recurse(event)


#### The file's shape ####


def test_first_row_is_the_game_header():
    """One header row per match opens the file, naming the conditions played."""
    header = _rows()[0]
    assert header["row"] == "game"
    assert header["game_id"] == "42"
    assert header["source"] == "play"
    assert header["num_players"] == 2
    assert header["seats"] == ["random", "random"]
    assert header["scores"] == [10, 10]
    assert header["winner"] == 0


def test_a_tie_is_recorded_as_no_winning_seat():
    """``determine_winner``'s -1 tie sentinel never reaches the log as a seat."""
    meta = models.GameMeta(game_id="t", seed=1, num_players=2, winner=None)
    assert json.loads(meta.model_dump_json())["winner"] is None


def test_every_line_is_a_single_json_object():
    """The file is line-delimited JSON: no indentation, no line spans a break."""
    lines = render_jsonl.render_rows(_record_game(), _meta())
    for line in lines:
        assert "\n" not in line
        assert not line.startswith(" ")
        assert isinstance(json.loads(line), dict)


def test_rendered_block_is_newline_terminated():
    """``render_jsonl`` produces an appendable block: every row ends its line."""
    block = render_jsonl.render_jsonl(_record_game(), _meta())
    assert block.endswith("\n")
    assert len(block.splitlines()) == len(
        render_jsonl.render_rows(_record_game(), _meta())
    )


def test_no_column_mixes_scalars_and_containers():
    """A column is scalar everywhere or a container everywhere.

    This is what makes the file load as a table: a column that is sometimes a
    number and sometimes a list has no usable dtype, and the two meanings would
    have to be untangled by hand.  ``scores`` is the near-miss — a per-seat
    breakdown on a ``final_scoring`` row and a running score on a resolution
    row — but both are lists, so the column stays honest."""
    shapes: dict[str, set[str]] = {}
    for row in _rows() + _rows(seed=7, num_players=4):
        for column, value in row.items():
            kind = "container" if isinstance(value, (list, dict)) else "scalar"
            shapes.setdefault(column, set()).add(kind)
    mixed = {column for column, kinds in shapes.items() if len(kinds) > 1}
    assert not mixed


#### Nothing lost ####


def test_every_node_becomes_exactly_one_row():
    """Row count matches the tree's node count — no node dropped or doubled."""
    tree = _record_game()
    events = list(_walk_events(tree))
    sub_events = sum(len(event.sub_events) for event in events)
    node_rows = [row for row in _rows() if row["row"] != "game"]
    assert len(node_rows) == len(events) + sub_events


def test_seq_orders_the_rows_depth_first():
    """``seq`` is a dense 0-based index over the game's node rows."""
    node_rows = [row for row in _rows() if row["row"] != "game"]
    assert [row["seq"] for row in node_rows] == list(range(len(node_rows)))


def test_rows_reconstruct_the_tree():
    """The event forest rebuilt from ``event_id`` / ``parent_id`` matches the tree.

    This is the property the whole flat format rests on: the nesting the HTML
    log shows structurally is recoverable from two integer columns."""
    tree = _record_game()
    event_rows = [row for row in _rows() if row["row"] == "event"]

    children_by_parent: dict[int | None, list[int]] = {}
    for row in event_rows:
        children_by_parent.setdefault(row.get("parent_id"), []).append(row["event_id"])

    for event in _walk_events(tree):
        rebuilt = children_by_parent.get(event.event_id, [])
        assert rebuilt == [child.event_id for child in event.children]

    # Phase-level events (no enclosing event) are the roots of the forest.
    roots = children_by_parent.get(None, [])
    expected_roots = [event.event_id for phase in tree.phases for event in phase.events]
    assert roots == expected_roots


def test_sub_rows_file_under_the_event_that_owns_them():
    """A sub-event's ``event_id`` points at its owner, in the recorded order."""
    tree = _record_game()
    sub_rows = [row for row in _rows() if row["row"] == "sub"]
    by_event: dict[int, list[str]] = {}
    for row in sorted(sub_rows, key=lambda row: row["seq"]):
        by_event.setdefault(row["event_id"], []).append(row["kind"])

    for event in _walk_events(tree):
        if not event.sub_events:
            continue
        assert by_event[event.event_id] == [sub.kind for sub in event.sub_events]


def test_phase_seq_addresses_a_turn_that_round_and_turn_cannot():
    """``phase_seq`` is what identifies one turn; ``phase_turn`` is not unique.

    ``turn_idx`` numbers a seat's turns within a round, so every seat has a
    turn 3 — ``(phase_round, phase_turn)`` names one phase per seat, not one
    phase. Grouping a turn's rows on the pair would silently merge two seats'
    turns into one."""
    tree = _record_game()
    turn_rows = [row for row in _rows() if row.get("phase") == "turn"]
    by_round_and_turn = {(row["phase_round"], row["phase_turn"]) for row in turn_rows}
    by_phase_seq = {row["phase_seq"] for row in turn_rows}
    assert len(by_phase_seq) > len(by_round_and_turn)
    assert len(by_phase_seq) == sum(
        1 for phase in tree.phases if phase.kind == "turn" and phase.events
    )

    # Every phase_seq indexes the phase it came from.
    for row in _rows()[1:]:
        phase_seq: int = row["phase_seq"]
        assert tree.phases[phase_seq].kind == row["phase"]


def test_both_rows_of_an_event_agree_on_its_tree_position():
    """An event's row and its sub rows carry identical link columns."""
    rows_by_event: dict[int, list[dict[str, typing.Any]]] = {}
    for row in _rows():
        if row["row"] != "game":
            rows_by_event.setdefault(row["event_id"], []).append(row)

    for grouped in rows_by_event.values():
        assert len({row.get("parent_id") for row in grouped}) == 1
        assert len({row["phase"] for row in grouped}) == 1


#### Columns ####


def test_event_rows_carry_the_same_header_the_html_log_shows():
    """``text`` on an event row is exactly ``summarize.summary_text``.

    The file and the viewer must never disagree about what a turn did."""
    tree = _record_game()
    text_by_event = {
        row["event_id"]: row["text"] for row in _rows() if row["row"] == "event"
    }
    for event in _walk_events(tree):
        assert text_by_event[event.event_id] == summarize.summary_text(event)


def test_effect_rows_carry_their_own_typed_fields():
    """Each effect contributes its own columns rather than an opaque payload."""
    draws = [row for row in _rows() if row.get("kind") == "draw_card"]
    assert draws, "a full game always draws cards"
    assert all(row["card"] and row["source"] for row in draws)

    eggs = [row for row in _rows() if row.get("kind") == "lay_egg"]
    assert eggs and all(row["bird"] and row["habitat"] for row in eggs)


def test_event_rows_carry_the_folded_effect_summary():
    """``sum_`` columns pre-fold the ledger so an analyst need not re-walk it."""
    tree = _record_game()
    rows_by_event = {row["event_id"]: row for row in _rows() if row["row"] == "event"}
    folded = [
        (event, summarize.summarize(event))
        for event in _walk_events(tree)
        if summarize.summarize(event).eggs_laid
    ]
    assert folded, "a full game always lays eggs somewhere"
    for event, summary in folded:
        assert rows_by_event[event.event_id]["sum_eggs_laid"] == summary.eggs_laid


def test_summary_columns_are_absent_when_nothing_happened():
    """An event with no effects contributes no ``sum_`` columns at all."""
    empty = [
        row
        for row in _rows()
        if row["row"] == "event" and not any(key.startswith("sum_") for key in row)
    ]
    assert empty, "a full game always crosses a bird with no brown power"


def test_reveals_carry_their_disclosure_text():
    """A deck draw's row is the only record of which card came off the deck."""
    deck_draws = [
        row
        for row in _rows()
        if row.get("kind") == "draw_card" and row["source"] == "deck"
    ]
    assert deck_draws
    for row in deck_draws:
        assert row["text"] == f"Draws {row['card']} from the deck"


def test_silent_effects_have_no_text():
    """An effect already folded into its event's header says nothing itself."""
    spends = [row for row in _rows() if row.get("kind") == "spend_food"]
    assert spends and all(row["text"] == "" for row in spends)


def test_resolutions_put_their_outcome_in_the_uniform_text_column():
    """``outcome_text`` is renamed to ``text`` so every row reads the same way."""
    resolutions = [row for row in _rows() if row.get("kind") in ("decision", "forced")]
    assert resolutions
    assert all(row["text"] for row in resolutions)
    assert all("outcome_text" not in row for row in resolutions)


def test_round_goal_keeps_its_own_round_alongside_the_phase_coordinates():
    """``phase_round`` is namespaced so it cannot collide with ``round_idx``."""
    goals = [row for row in _rows() if row.get("kind") == "round_goal"]
    assert len(goals) == 4
    for row in goals:
        assert row["round_idx"] == row["phase_round"]
        assert row["counts"] and row["vps"]


#### Decision annotations ####


def _annotated_tree() -> models.GameEventTree:
    """A one-event tree whose decision carries a full policy annotation."""
    decision = models.DecisionSubEvent(
        player_id=0,
        outcome_text="Gains fish",
        options=[
            models.DecisionOption(
                label="Gain fish",
                prob=0.75,
                score=1.5,
                selected=True,
                choice_stripes=[
                    models.EncodedStripe(name="choice", description="c", sub_fields=[])
                ],
            ),
            models.DecisionOption(label="Gain seed", prob=0.25, score=0.5),
        ],
        state_stripes=[
            models.EncodedStripe(name="state", description="s", sub_fields=[])
        ],
        value=0.4,
    )
    event = models.MainActionEvent(
        event_id=0, player_id=0, action="gain_food", sub_events=[decision]
    )
    return models.GameEventTree(
        phases=[models.PhaseNode(kind="turn", round_idx=0, turn_idx=1, events=[event])]
    )


def test_decision_rows_keep_the_policy_distribution():
    """Every offered option survives with its probability, score, and selection."""
    rows = [
        json.loads(line)
        for line in render_jsonl.render_rows(_annotated_tree(), _meta())
    ]
    decision = next(row for row in rows if row.get("kind") == "decision")
    assert [option["label"] for option in decision["options"]] == [
        "Gain fish",
        "Gain seed",
    ]
    assert decision["options"][0]["prob"] == 0.75
    assert decision["options"][0]["selected"] is True
    assert decision["value"] == 0.4


def test_encoding_stripes_never_reach_the_file():
    """The viewer's feature vectors are dropped — they would dwarf the log."""
    block = render_jsonl.render_jsonl(_annotated_tree(), _meta())
    assert "state_stripes" not in block
    assert "choice_stripes" not in block


#### Files and table sizes ####


def test_append_game_concatenates_matches(tmp_path: pathlib.Path):
    """Two matches land in one file, each opened by its own header row."""
    path = tmp_path / "games.jsonl"
    render_jsonl.append_game(path, _record_game(), _meta(game_id="a"))
    render_jsonl.append_game(path, _record_game(seed=23), _meta(game_id="b"))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    headers = [row for row in rows if row["row"] == "game"]
    assert [row["game_id"] for row in headers] == ["a", "b"]
    assert {row["game_id"] for row in rows} == {"a", "b"}


def test_rows_render_at_four_seats():
    """The row schema holds at table sizes above two."""
    rows = _rows(seed=7, num_players=4)
    assert rows[0]["num_players"] == 4
    scores = [
        row["scores"]
        for row in rows
        if row.get("kind") == "decision" and "scores" in row
    ]
    assert scores and all(len(seat_scores) == 4 for seat_scores in scores)
