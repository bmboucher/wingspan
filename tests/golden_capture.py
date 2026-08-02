"""Golden N=2 encoding fixture — capture script.

Plays seeded 2-player random-vs-random games through today's (pre-N-player)
engine and records a SHA-256 hash of the encoded state/choice tensors at
every genuine decision point. The resulting ``tests/data/golden_n2.json``
fixture is the byte-identity proof that Stage 1's N-player engine changes
leave 2-player play untouched — both the decision *sequence* (which
decisions get asked, in what order, to which player) and the encoder
*bytes* (``encode_state`` / ``encode_choices`` under both
``EncodingSpec.include_setup`` settings).

``test_encoding_golden_n2.py`` replays the same seeds and asserts the hash
sequence matches this fixture exactly.

Run directly to (re)capture the fixture:

    python tests/golden_capture.py

Before writing anything, every seed is played *twice* and the two hash
sequences are asserted identical — a fixture built from a non-reproducible
game would be worthless as a byte-identity proof, so a divergence here
aborts the capture instead of silently writing a flaky fixture.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pydantic

from wingspan import agents, decisions, encode, engine

# Seeds captured into the fixture. Five games spread across a handful of
# board-shuffle/agent-choice streams exercise a wide variety of decision
# types without a slow capture run. Public: ``test_encoding_golden_n2.py``
# parametrizes its replay test over the same tuple.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Public: read by ``test_encoding_golden_n2.py`` to locate the committed
# fixture.
FIXTURE_PATH = pathlib.Path(__file__).parent / "data" / "golden_n2.json"

# The two encoding specs every decision is hashed under — same pair
# ``test_encoding_golden_n2.py`` (and the live model configs) use.
_SPEC_SETUP = encode.EncodingSpec(include_setup=True)
_SPEC_NOSETUP = encode.EncodingSpec(include_setup=False)


class DecisionHashes(pydantic.BaseModel):
    """SHA-256 hex digests of one decision's encoded tensors, at both
    ``EncodingSpec.include_setup`` settings.

    The ``_nosetup`` fields are ``None`` for a ``SetupDecision``: the
    setup-excluded spec drops the ``SetupDecision`` decision-type column
    entirely, so there is nothing to encode."""

    state_setup: str
    state_nosetup: str | None
    choices_setup: str
    choices_nosetup: str | None


class DecisionRecord(pydantic.BaseModel):
    """One genuine (multi-choice) decision point: its class, deciding
    player, and encoded-tensor hashes.

    ``Engine.ask`` auto-resolves single-choice decisions without consulting
    the agent, so only real forks ever become a ``DecisionRecord``."""

    decision_class: str
    player_id: int
    hashes: DecisionHashes


class GoldenFixture(pydantic.BaseModel):
    """The full golden fixture: an ordered decision-hash sequence per
    captured seed, keyed by seed number."""

    seeds: dict[int, list[DecisionRecord]]


def play_seed_hashes(seed: int) -> list[DecisionRecord]:
    """Play one full 2-player game at ``seed``, recording a
    :class:`DecisionRecord` at every genuine decision point.

    Both seats share one ``wingspan.agents.random_agent`` closed over a
    single ``random.Random(seed)`` — the same deterministic random-vs-random
    shape ``test_smoke.py`` uses. The game itself is built exactly as
    ``Engine.create`` builds it today, so this mirrors the production
    2-player path rather than a bespoke test-only construction."""
    eng, *_ = engine.Engine.create(seed=seed)
    rng = random.Random(seed)
    base_agent = agents.random_agent(rng)
    records: list[DecisionRecord] = []

    def recording_agent[C: decisions.Choice](
        live_engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        records.append(_hash_decision(live_engine, decision))
        return base_agent(live_engine, decision)

    engine.Engine.play_one_game(eng.state, (recording_agent, recording_agent))
    return records


def capture_fixture() -> GoldenFixture:
    """Play every seed twice, assert bit-for-bit determinism between the two
    passes, and return the fixture built from the first pass."""
    seed_records: dict[int, list[DecisionRecord]] = {}
    for seed in SEEDS:
        first_pass = play_seed_hashes(seed)
        second_pass = play_seed_hashes(seed)
        _assert_identical_sequences(seed, first_pass, second_pass)
        seed_records[seed] = first_pass
    return GoldenFixture(seeds=seed_records)


def main() -> None:
    """Capture the fixture and write it to :data:`_FIXTURE_PATH`."""
    fixture = capture_fixture()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(fixture.model_dump_json(indent=2), encoding="utf-8")
    total_decisions = sum(len(records) for records in fixture.seeds.values())
    print(
        f"Wrote {FIXTURE_PATH} — {len(fixture.seeds)} seeds, "
        f"{total_decisions} decisions"
    )


###### PRIVATE #######


def _hash_decision(
    live_engine: engine.Engine, decision: decisions.Decision[typing.Any]
) -> DecisionRecord:
    """Hash ``decision``'s encoded state/choices under both encoding specs,
    skipping the ``_nosetup`` pair for a ``SetupDecision`` (see
    :class:`DecisionHashes`)."""
    state_setup = encode.encode_state(live_engine.state, decision, _SPEC_SETUP)
    choices_setup = encode.encode_choices(decision, live_engine.state, _SPEC_SETUP)
    is_setup = decisions.is_setup_decision(decision)
    state_nosetup = (
        None
        if is_setup
        else encode.encode_state(live_engine.state, decision, _SPEC_NOSETUP)
    )
    choices_nosetup = (
        None
        if is_setup
        else encode.encode_choices(decision, live_engine.state, _SPEC_NOSETUP)
    )
    return DecisionRecord(
        decision_class=type(decision).__name__,
        player_id=decision.player_id,
        hashes=DecisionHashes(
            state_setup=_hash_array(state_setup),
            state_nosetup=(
                None if state_nosetup is None else _hash_array(state_nosetup)
            ),
            choices_setup=_hash_array(choices_setup),
            choices_nosetup=(
                None if choices_nosetup is None else _hash_array(choices_nosetup)
            ),
        ),
    )


def _hash_array(arr: np.ndarray) -> str:
    """SHA-256 hex digest of ``arr``'s contiguous float32 bytes."""
    payload = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _assert_identical_sequences(
    seed: int, first: list[DecisionRecord], second: list[DecisionRecord]
) -> None:
    """Assert two capture passes over the same seed produced an identical
    hash sequence — the reproducibility precondition for a golden fixture.

    Raises with the first divergent index so a non-deterministic agent /
    engine path is easy to locate rather than silently writing a flaky
    fixture."""
    if len(first) != len(second):
        raise AssertionError(
            f"seed {seed}: decision count differs between passes "
            f"({len(first)} vs {len(second)}) — the game is not reproducible"
        )
    for idx, (left, right) in enumerate(zip(first, second)):
        if left != right:
            raise AssertionError(
                f"seed {seed}: capture is not reproducible — passes diverge at "
                f"decision {idx} ({left.decision_class}): {left!r} != {right!r}"
            )


# Placed after every definition it transitively calls (``main`` ->
# ``capture_fixture`` -> ``play_seed_hashes`` -> ``_hash_decision`` /
# ``_assert_identical_sequences``) since running a top-level statement before
# the rest of the module has finished executing would leave the later names
# unbound.
if __name__ == "__main__":
    main()
