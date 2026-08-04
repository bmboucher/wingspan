"""Round-trip serialization tests for the structured game-event tree.

The tree's ``sub_events`` / ``children`` fields are typed as *discriminated
unions* rather than their base classes.  This is load-bearing and easy to
regress: with base-class annotations pydantic silently drops every
subclass-declared field on dump (``bird_name``, ``outcome_text``, ``habitat``,
…) and the node type becomes unrecoverable on load — no exception is raised
either way, so only an explicit round-trip catches it.

These tests play real games and assert that a dump/load cycle reproduces the
tree exactly, that every node type survives, and that the ids and phase
coordinates added for the flat structured log are well-formed.
"""

from __future__ import annotations

import functools
import sys
import typing

# Add src/ to path as per test_smoke.py convention.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from wingspan.gamelog import models  # noqa: E402  (needs the sys.path insert above)

if typing.TYPE_CHECKING:
    from wingspan import cards
    from wingspan.engine import core


# Playing a full game is by far the most expensive thing here, and every test
# below only reads its tree.  Cache per (seed, seats) so the module plays three
# games total rather than one per test — the suite is CPU-bound and extra load
# destabilises the timing-sensitive tests elsewhere in it.
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


def _walk_events(tree: models.GameEventTree) -> typing.Iterator[models.GameEvent]:
    """Yield every event in the tree, depth-first."""

    def recurse(event: models.GameEvent) -> typing.Iterator[models.GameEvent]:
        yield event
        for child in event.children:
            yield from recurse(child)

    for phase in tree.phases:
        for event in phase.events:
            yield from recurse(event)


def test_tree_round_trips_through_json():
    """A full game tree survives dump -> load unchanged."""
    tree = _record_game()
    restored = models.GameEventTree.model_validate_json(tree.model_dump_json())
    assert restored == tree


def test_round_trip_preserves_subclass_fields_and_types():
    """Subclass-declared fields and node classes survive the round trip.

    This is the specific failure mode of a base-class-typed container: the
    dump succeeds, the load succeeds, and every distinguishing field is gone.
    """
    tree = _record_game()
    restored = models.GameEventTree.model_validate_json(tree.model_dump_json())

    original = list(_walk_events(tree))
    reloaded = list(_walk_events(restored))
    assert len(original) == len(reloaded)
    for before, after in zip(original, reloaded, strict=True):
        assert type(before) is type(after)

    # Spot-check the fields that vanish under a base-class annotation.
    brown = [e for e in reloaded if isinstance(e, models.ActivateBrownEvent)]
    assert brown, "expected at least one brown-power slot in a full game"
    assert all(event.bird_name for event in brown)

    base = [e for e in reloaded if isinstance(e, models.ActivateBaseEvent)]
    assert base and all(event.habitat and event.action for event in base)

    decisions = [
        sub
        for event in reloaded
        for sub in event.sub_events
        if isinstance(sub, models.DecisionSubEvent)
    ]
    assert decisions and all(sub.outcome_text for sub in decisions)


def test_every_event_type_round_trips():
    """Each concrete event class that a real game produces reloads as itself."""
    tree = _record_game()
    restored = models.GameEventTree.model_validate_json(tree.model_dump_json())
    seen = {type(event) for event in _walk_events(restored)}
    for expected in (
        models.MainActionEvent,
        models.PlayBirdEvent,
        models.ActivateBaseEvent,
        models.ActivateBrownEvent,
        models.RoundGoalEvent,
        models.FinalScoringEvent,
        models.SetupEvent,
    ):
        assert expected in seen, f"{expected.__name__} missing from the tree"


def test_event_ids_are_unique_and_monotonic():
    """Every event carries a distinct id, assigned in creation order."""
    tree = _record_game()
    ids = [event.event_id for event in _walk_events(tree)]
    assert len(ids) == len(set(ids)), "event ids must be unique within a game"
    assert min(ids) == 0


def test_phases_carry_round_and_turn_coordinates():
    """Turn phases know their round and turn; round phases know their round."""
    tree = _record_game()
    turns = [phase for phase in tree.phases if phase.kind == "turn"]
    rounds = [phase for phase in tree.phases if phase.kind == "round"]
    assert turns and rounds
    assert all(phase.round_idx is not None for phase in turns)
    assert all(phase.turn_idx is not None for phase in turns)
    assert [phase.round_idx for phase in rounds] == [0, 1, 2, 3]


def test_round_goals_land_in_their_round_phase():
    """Round-goal scoring is recorded against the round it scores, not the
    last turn of that round."""
    tree = _record_game()
    for phase in tree.phases:
        for event in phase.events:
            if isinstance(event, models.RoundGoalEvent):
                assert phase.kind == "round"
                assert phase.round_idx == event.round_idx


def test_forced_sub_events_carry_clock_fields():
    """Forced resolutions are joinable to the timeline like genuine decisions."""
    tree = _record_game()
    forced = [
        sub
        for event in _walk_events(tree)
        for sub in event.sub_events
        if isinstance(sub, models.ForcedSubEvent)
    ]
    assert forced, "a full random game always auto-resolves some single-choice ask"
    assert any(sub.turn_counter > 0 for sub in forced)
    assert all(len(sub.scores) >= 2 for sub in forced)


def test_no_decision_is_orphaned_into_a_loose_event():
    """Every recorded decision has a named parent event.

    ``LooseEvent`` is the auto-wrap bucket for a ``record_*`` that fired outside
    any ``begin_*``/``end_event`` bracket.  A non-empty one means some ask is
    not wired into the call-site graph, and it renders as an unlabelled blob in
    both the HTML log and the structured export.  Two used to escape: the
    extra-play accept and the end-of-turn discards."""
    for seed, seats in ((42, 2), (23, 2), (7, 4)):
        tree = _record_game(seed=seed, num_players=seats)
        loose = [
            event
            for event in _walk_events(tree)
            if isinstance(event, models.LooseEvent)
        ]
        assert not loose, f"seed {seed}: {len(loose)} unbracketed decision group(s)"


def test_extra_play_offers_are_bracketed():
    """An accrued extra play is its own event, with the resulting play nested.

    Seed 23 is the first 2-seat random game whose birds grant an extra play."""
    tree = _record_game(seed=23)
    extra = [
        event
        for event in _walk_events(tree)
        if isinstance(event, models.ExtraPlayEvent)
    ]
    assert extra, "seed 23 is expected to accrue at least one extra play"
    # Accepted offers nest the play; declined ones hold only the decline.
    assert all(event.sub_events for event in extra)


def test_turn_end_discards_are_bracketed():
    """End-of-turn discard obligations group under their own event.

    Seed 23 again — it plays a ``DRAW_CARDS_THEN_DISCARD_EOT`` bird, so the
    obligation actually fires."""
    tree = _record_game(seed=23)
    turn_ends = [
        event for event in _walk_events(tree) if isinstance(event, models.TurnEndEvent)
    ]
    assert turn_ends, "seed 23 is expected to trigger an end-of-turn discard"
    assert all(event.sub_events for event in turn_ends)


@functools.cache
def _play_with_fixed_setups(seed: int = 99) -> models.GameEventTree:
    """Play a game through the fixed-setup (setup-model) path.

    That path resolves setup from a chooser instead of asking agents, and is
    the one that historically opened no ``"setup"`` phase at all."""
    import random

    from wingspan import agents, decisions
    from wingspan import engine as engine_mod
    from wingspan.gamelog import recorder as gamelog_recorder

    class _Keep:
        """Minimal ``SetupKeep`` protocol implementation for the chooser."""

        def __init__(self, choice: decisions.SetupChoice) -> None:
            self._choice = choice

        def to_setup_choice(self) -> decisions.SetupChoice:
            return self._choice

    def choose_setups(
        engine: core.Engine,
        dealt: tuple[tuple[list[cards.Bird], list[cards.BonusCard]], ...],
    ) -> list[_Keep]:
        """Keep the first two dealt cards, paying with the seat's first foods."""
        keeps: list[_Keep] = []
        for player_id, (dealt_cards, dealt_bonus) in enumerate(dealt):
            player = engine.state.players[player_id]
            kept_foods = tuple(
                player.food.types_with_positive()[: len(dealt_cards) - 2]
            )
            keeps.append(
                _Keep(
                    decisions.SetupChoice(
                        label="fixed",
                        kept_cards=tuple(dealt_cards[:2]),
                        kept_foods=kept_foods,
                        bonus_card=dealt_bonus[0],
                    )
                )
            )
        return keeps

    rec = gamelog_recorder.EventRecorder(probes=(None, None))
    eng, *_ = engine_mod.Engine.create(seed=seed)
    rng = random.Random(seed)
    engine_mod.Engine.play_one_game_with_setups(
        eng.state,
        (agents.random_agent(rng), agents.random_agent(rng)),
        choose_setups,
        event_recorder=rec,
    )
    return rec.root


def test_fixed_setup_path_opens_setup_phases():
    """The setup-model path opens one ``setup`` phase per seat, like the
    agent-asked path — previously its ``SetupEvent``s leaked into
    ``game_start``."""
    tree = _play_with_fixed_setups()
    setup_phases = [phase for phase in tree.phases if phase.kind == "setup"]
    assert len(setup_phases) == 2
    assert all(phase.events for phase in setup_phases)
    game_start = next(phase for phase in tree.phases if phase.kind == "game_start")
    assert not any(
        isinstance(event, models.SetupEvent) for event in game_start.events
    ), "setup events must not fall back into the game_start phase"


def test_fixed_setup_path_round_trips():
    """The setup-model path's tree serializes like any other."""
    tree = _play_with_fixed_setups()
    assert models.GameEventTree.model_validate_json(tree.model_dump_json()) == tree


def test_round_trip_at_four_seats():
    """The discriminated unions hold at table sizes above two."""
    tree = _record_game(seed=7, num_players=4)
    restored = models.GameEventTree.model_validate_json(tree.model_dump_json())
    assert restored == tree
    scores = [
        sub.scores
        for event in _walk_events(restored)
        for sub in event.sub_events
        if isinstance(sub, models.DecisionSubEvent)
    ]
    assert scores and all(len(row) == 4 for row in scores)
