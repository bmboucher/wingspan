"""Golden N=2 encoding fixture — replay test.

Replays the exact seeds/agents ``golden_capture.py`` used to build
``tests/data/golden_n2.json`` and asserts the recorded decision-hash
sequence is unchanged. This is the byte-identity backstop for the whole
N-player Stage 1 refactor: it freezes both the 2-player decision
*sequence* and the ``encode_state`` / ``encode_choices`` *bytes*, so any
change that alters 2-player behavior — an extra/missing decision, a
reordered one, or a shifted encoding — fails here with the first
divergent (seed, decision index, decision class).

Per the Stage 1 plan, nothing in this stage should trip this test: the
round-goal payout rename, the winner/tiebreak kernel, the Oystercatcher and
Hummingbird rewrites, and the ``opponents_clockwise`` helper are all
designed to reduce to today's exact 2-player behavior.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

import golden_capture


def _load_fixture() -> golden_capture.GoldenFixture:
    if not golden_capture.FIXTURE_PATH.exists():
        pytest.fail(
            f"golden fixture missing: {golden_capture.FIXTURE_PATH} — run "
            "`python tests/golden_capture.py` to (re)capture it"
        )
    return golden_capture.GoldenFixture.model_validate_json(
        golden_capture.FIXTURE_PATH.read_text(encoding="utf-8")
    )


def test_golden_fixture_file_exists():
    """The fixture must be committed — this test has nothing to replay
    against otherwise."""
    assert (
        golden_capture.FIXTURE_PATH.exists()
    ), f"missing {golden_capture.FIXTURE_PATH}; run golden_capture.py"


@pytest.mark.parametrize("seed", golden_capture.SEEDS)
def test_replay_matches_golden_hashes(seed: int):
    """Replaying ``seed`` today must reproduce the exact recorded hash
    sequence — same decision classes, same player ids, same encoded bytes,
    in the same order."""
    fixture = _load_fixture()
    expected = fixture.seeds[seed]
    actual = golden_capture.play_seed_hashes(seed)

    assert len(actual) == len(expected), (
        f"seed {seed}: decision count changed "
        f"({len(actual)} now vs {len(expected)} recorded) — the 2-player "
        "decision sequence drifted"
    )
    for idx, (want, got) in enumerate(zip(expected, actual)):
        assert (want.decision_class, want.player_id) == (
            got.decision_class,
            got.player_id,
        ), (
            f"seed {seed}, decision {idx}: decision identity changed — "
            f"expected {want.decision_class!r}/player {want.player_id}, "
            f"got {got.decision_class!r}/player {got.player_id}"
        )
        assert want.hashes == got.hashes, (
            f"seed {seed}, decision {idx} ({want.decision_class}): encoded "
            f"tensor hashes changed — expected {want.hashes!r}, got {got.hashes!r}"
        )
