"""Flat JSONL renderer for the structured game-event tree.

The tree is the shape a renderer wants; this is the shape an analyst wants.
One match becomes a :class:`~wingspan.gamelog.models.GameMeta` header row
followed by one row per node — every event and every sub-event — in depth-first
order, each a single-line JSON object.  Files concatenate: a ``--games`` series
and a whole tournament both land in one file, keyed by ``game_id``.

The nesting the tree carries structurally survives as columns (``event_id`` /
``parent_id`` / ``depth`` / ``seq``), so ``pandas.read_json(path, lines=True)``
loads the whole thing and the tree reconstructs from the ``event`` rows alone.
Each node's own typed fields become its own columns — a ``draw_card`` row has
``card`` and ``source``; a ``lay_egg`` row has ``bird`` and ``count`` — rather
than being folded into an opaque payload.  Event rows additionally carry the
folded effect summary as ``sum_``-prefixed columns and the same ``text`` header
the HTML log shows, so the file and the viewer never disagree about what a turn
did.

Two things are deliberately dropped: the encoding-viewer stripes
(:attr:`~wingspan.gamelog.models.DecisionSubEvent.state_stripes` and each
option's ``choice_stripes``), which are a full feature vector per decision and
would dwarf everything else.  The policy distribution itself — every offered
option's label, probability, score, and whether it was taken — is kept.

**Import discipline:** standard library, ``pydantic``, and the sibling
:mod:`~wingspan.gamelog.models` / :mod:`~wingspan.gamelog.summarize` modules
only, matching the rest of the package.
"""

from __future__ import annotations

import pathlib

from wingspan.gamelog import models, summarize

# Event fields the common columns already carry, plus the two tree links this
# flat format replaces with ``event_id`` / ``parent_id``.
_EVENT_EXCLUDE = {"kind", "event_id", "player_id", "sub_events", "children"}

# Sub-event fields handled elsewhere: ``outcome_text`` becomes the uniform
# ``text`` column, the stripes are viewer-only bulk, and ``options`` is rebuilt
# below without its per-option stripes.
_SUB_EXCLUDE = {"kind", "player_id", "outcome_text", "state_stripes", "options"}

# Namespace for the folded EventSummary columns, keeping them from colliding
# with a node's own field names (an event has ``count``; so could a summary).
_SUMMARY_PREFIX = "sum_"


def render_rows(tree: models.GameEventTree, meta: models.GameMeta) -> list[str]:
    """Every JSONL line for one match: the game header, then one row per node.

    Lines carry no trailing newline — :func:`render_jsonl` adds them."""
    lines = [meta.model_dump_json()]
    lines.extend(
        row.model_dump_json(exclude_none=True) for row in _node_rows(tree, meta.game_id)
    )
    return lines


def render_jsonl(tree: models.GameEventTree, meta: models.GameMeta) -> str:
    """One match as a newline-terminated JSONL block, ready to append."""
    return "".join(f"{line}\n" for line in render_rows(tree, meta))


def append_game(
    path: pathlib.Path, tree: models.GameEventTree, meta: models.GameMeta
) -> None:
    """Append one match's rows to ``path``, creating the file if absent.

    Appending rather than writing is the point of the format: a series, a
    tournament worker's shard, and a single game all use this one call."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(render_jsonl(tree, meta))


###### PRIVATE #######


def _node_rows(tree: models.GameEventTree, game_id: str) -> list[models.NodeRow]:
    """Every node in the tree as a row, in depth-first order."""
    rows: list[models.NodeRow] = []
    for phase_seq, phase in enumerate(tree.phases):
        for event in phase.events:
            _walk_event(
                event,
                phase,
                game_id,
                phase_seq=phase_seq,
                parent_id=None,
                depth=0,
                rows=rows,
            )
    return rows


def _walk_event(
    event: models.AnyGameEvent,
    phase: models.PhaseNode,
    game_id: str,
    *,
    phase_seq: int,
    parent_id: int | None,
    depth: int,
    rows: list[models.NodeRow],
) -> None:
    """Append ``event``'s row, then its sub-events, then recurse into children.

    Sub-events precede child events, matching the order both renderers show
    them in: an event's own decisions and effects, then whatever nested inside
    it.  ``seq`` falls out of the append order."""
    rows.append(
        _event_row(
            event,
            phase,
            game_id,
            phase_seq=phase_seq,
            parent_id=parent_id,
            depth=depth,
            seq=len(rows),
        )
    )
    for sub in event.sub_events:
        rows.append(
            _sub_row(
                sub,
                event,
                phase,
                game_id,
                phase_seq=phase_seq,
                parent_id=parent_id,
                depth=depth + 1,
                seq=len(rows),
            )
        )
    for child in event.children:
        _walk_event(
            child,
            phase,
            game_id,
            phase_seq=phase_seq,
            parent_id=event.event_id,
            depth=depth + 1,
            rows=rows,
        )


def _event_row(
    event: models.AnyGameEvent,
    phase: models.PhaseNode,
    game_id: str,
    *,
    phase_seq: int,
    parent_id: int | None,
    depth: int,
    seq: int,
) -> models.NodeRow:
    """One event's row: common columns, its own fields, its folded summary."""
    columns = _common_columns(game_id, phase, phase_seq, seq)
    columns.update(
        {
            "row": models.RowKind.EVENT,
            "event_id": event.event_id,
            "parent_id": parent_id,
            "depth": depth,
            "kind": event.kind,
            "player_id": event.player_id,
            "text": summarize.summary_text(event),
        }
    )
    columns.update(
        event.model_dump(mode="json", exclude=_EVENT_EXCLUDE, exclude_none=True)
    )
    columns.update(_summary_columns(summarize.summarize(event)))
    return models.NodeRow.model_validate(columns)


def _sub_row(
    sub: models.AnySubEvent,
    event: models.AnyGameEvent,
    phase: models.PhaseNode,
    game_id: str,
    *,
    phase_seq: int,
    parent_id: int | None,
    depth: int,
    seq: int,
) -> models.NodeRow:
    """One sub-event's row, filed under the event that owns it."""
    columns = _common_columns(game_id, phase, phase_seq, seq)
    columns.update(
        {
            "row": models.RowKind.SUB,
            "event_id": event.event_id,
            "parent_id": parent_id,
            "depth": depth,
            "kind": sub.kind,
            "player_id": sub.player_id,
            "text": _sub_text(sub),
        }
    )
    columns.update(sub.model_dump(mode="json", exclude=_SUB_EXCLUDE, exclude_none=True))
    if isinstance(sub, models.DecisionSubEvent) and sub.options:
        columns["options"] = [
            option.model_dump(mode="json", exclude={"choice_stripes"})
            for option in sub.options
        ]
    return models.NodeRow.model_validate(columns)


def _common_columns(
    game_id: str, phase: models.PhaseNode, phase_seq: int, seq: int
) -> dict[str, object]:
    """The game and phase coordinates every node row carries.

    The phase's round and turn are namespaced (``phase_round`` / ``phase_turn``)
    because a :class:`~wingspan.gamelog.models.RoundGoalEvent` has a
    ``round_idx`` field of its own, and two different meanings must not share a
    column."""
    return {
        "game_id": game_id,
        "seq": seq,
        "phase": phase.kind,
        "phase_seq": phase_seq,
        "phase_round": phase.round_idx,
        "phase_turn": phase.turn_idx,
    }


def _summary_columns(summary: summarize.EventSummary) -> dict[str, object]:
    """The event's folded effect summary as ``sum_``-prefixed columns.

    Only non-default entries appear, so an event that laid no eggs contributes
    no ``sum_eggs_laid`` column at all and the row stays narrow."""
    folded: dict[str, object] = summary.model_dump(mode="json", exclude_defaults=True)
    return {f"{_SUMMARY_PREFIX}{name}": value for name, value in folded.items()}


def _sub_text(sub: models.AnySubEvent) -> str:
    """A sub-event's human-readable line.

    A resolution says what was chosen; a reveal says what it disclosed; every
    other effect is silent here, because it has already been folded into its
    event's ``text``."""
    if isinstance(sub, models.ResolvedSubEvent):
        return sub.outcome_text
    return summarize.reveal_text(sub)
