"""Reconciliation tests: the effect ledger must account for the whole game.

The game log records what changed as an explicit ledger — every mutation is
performed by a :mod:`wingspan.engine.ledger` function that also records the
matching :class:`~wingspan.gamelog.models.Effect`.  That is only trustworthy if
*every* mutation goes through it: a bird power that pokes ``pb.eggs`` directly
would leave the log quietly under-reporting, with nothing to catch it.

These tests are that catch.  They replay the recorded ledger from an empty game
and assert it reproduces the final :class:`~wingspan.state.GameState` exactly —
per-seat food, per-bird eggs / tucked cards / caches, hand sizes, and board
contents.  A mutation that bypasses the ledger shows up here as a mismatch, so
the seam cannot rot silently.

They run over several seeds at 2 and 4 seats, because a lot of the engine's
mutation sites (pink reactors, the clockwise card draft, all-players powers) are
only reachable at particular table sizes.
"""

from __future__ import annotations

import collections
import functools
import random
import sys
import typing

# Add src/ to path as per test_smoke.py convention.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from wingspan import agents  # noqa: E402  (needs the sys.path insert above)
from wingspan import cards  # noqa: E402
from wingspan import engine as engine_module  # noqa: E402
from wingspan import state as state_module  # noqa: E402
from wingspan.gamelog import models  # noqa: E402
from wingspan.gamelog import recorder as gamelog_recorder  # noqa: E402

# Seeds and table sizes the reconciliation runs over. Kept small: each entry is
# a full game, and the suite is CPU-bound.
_CASES = ((42, 2), (23, 2), (7, 4), (101, 4))


class _Recorded(typing.NamedTuple):
    """One played game: its final state and the ledger recorded alongside."""

    state: state_module.GameState
    effects: list[models.AnyEffect]


@functools.cache
def _play(seed: int, num_players: int) -> _Recorded:
    """Play one random-agent game with a live recorder attached."""
    rec = gamelog_recorder.EventRecorder(probes=tuple(None for _ in range(num_players)))
    eng, *_ = engine_module.Engine.create(seed=seed, num_players=num_players)
    rng = random.Random(seed)
    engine_module.Engine.play_one_game(
        eng.state,
        [agents.random_agent(rng) for _ in range(num_players)],
        event_recorder=rec,
    )
    return _Recorded(state=eng.state, effects=_all_effects(rec.root))


def _all_effects(tree: models.GameEventTree) -> list[models.AnyEffect]:
    """Every recorded effect in the tree, in chronological (DFS) order."""
    found: list[models.AnyEffect] = []

    def walk(events: typing.Sequence[models.GameEvent]) -> None:
        for event in events:
            for sub in event.sub_events:
                if isinstance(sub, models.Effect):
                    found.append(sub)
            walk(event.children)

    for phase in tree.phases:
        walk(phase.events)
    return found


def _of_type[E: models.Effect](
    effects: typing.Sequence[models.AnyEffect], wanted: type[E]
) -> list[E]:
    """The recorded effects of one concrete class."""
    return [effect for effect in effects if isinstance(effect, wanted)]


###### Food ######


def _ledger_food(
    effects: typing.Sequence[models.AnyEffect],
) -> dict[tuple[int, str], int]:
    """Net food per ``(seat, food)`` implied by the ledger.

    A cache taken from the player's own supply is recorded as a spend *and* a
    cache, so only the spend side counts here; a power that caches straight off
    the supply never touches the pool at all."""
    net: dict[tuple[int, str], int] = collections.defaultdict(int)
    for effect in effects:
        if isinstance(effect, models.GainFoodEffect):
            assert effect.player_id is not None
            net[(effect.player_id, effect.food)] += effect.amount
        elif isinstance(effect, models.SpendFoodEffect):
            assert effect.player_id is not None
            net[(effect.player_id, effect.food)] -= effect.amount
    return net


def test_food_pools_reconcile():
    """Every seat's food pool equals gains minus spends, food type by food type."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        net = _ledger_food(played.effects)
        for player in played.state.players:
            for food in cards.ALL_FOODS:
                assert player.food[food] == net[(player.id, food.value)], (
                    f"seed {seed}/{seats}p: seat {player.id} {food.value} is "
                    f"{player.food[food]} but the ledger accounts for "
                    f"{net[(player.id, food.value)]}"
                )


def test_caches_reconcile():
    """Every bird's cached food equals what the ledger cached minus uncached."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        net: dict[tuple[int, str, str], int] = collections.defaultdict(int)
        for effect in played.effects:
            if isinstance(effect, models.CacheFoodEffect):
                assert effect.player_id is not None
                net[(effect.player_id, effect.bird, effect.food)] += effect.amount
            elif isinstance(effect, models.UncacheFoodEffect):
                assert effect.player_id is not None
                net[(effect.player_id, effect.bird, effect.food)] -= effect.amount

        for player in played.state.players:
            for row in player.board.values():
                for played_bird in row:
                    for food in cards.ALL_FOODS:
                        key = (player.id, played_bird.bird.name, food.value)
                        assert played_bird.cached_food[food] == net[key], (
                            f"seed {seed}/{seats}p: {played_bird.bird.name} holds "
                            f"{played_bird.cached_food[food]} {food.value} but the "
                            f"ledger accounts for {net[key]}"
                        )


###### Eggs and tucked cards ######


def test_eggs_reconcile():
    """Every bird's egg count equals eggs laid minus eggs removed on it.

    Keyed on ``(seat, bird name)`` rather than the recorded ``(habitat, slot)``:
    a bird can be moved between habitats mid-game and keeps its eggs, so the
    coordinates on an effect are a snapshot, not an identity. Each bird card
    exists once in the core set, so its name identifies it within a seat."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        net: dict[tuple[int, str], int] = collections.defaultdict(int)
        for effect in played.effects:
            if isinstance(effect, models.LayEggEffect):
                assert effect.player_id is not None
                net[(effect.player_id, effect.bird)] += effect.count
            elif isinstance(effect, models.RemoveEggEffect):
                assert effect.player_id is not None
                net[(effect.player_id, effect.bird)] -= effect.count

        for player in played.state.players:
            for habitat, row in player.board.items():
                for slot, played_bird in enumerate(row):
                    key = (player.id, played_bird.bird.name)
                    assert played_bird.eggs == net[key], (
                        f"seed {seed}/{seats}p: {played_bird.bird.name}"
                        f"@{habitat.value}[{slot}] holds {played_bird.eggs} eggs but "
                        f"the ledger accounts for {net[key]}"
                    )


def test_tucked_cards_reconcile():
    """Every bird's tucked count equals the tucks recorded behind it."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        net: dict[tuple[int, str], int] = collections.defaultdict(int)
        for effect in _of_type(played.effects, models.TuckCardEffect):
            assert effect.player_id is not None
            net[(effect.player_id, effect.bird)] += 1

        for player in played.state.players:
            for row in player.board.values():
                for played_bird in row:
                    key = (player.id, played_bird.bird.name)
                    assert played_bird.tucked_cards == net[key], (
                        f"seed {seed}/{seats}p: {played_bird.bird.name} has "
                        f"{played_bird.tucked_cards} tucked but the ledger accounts "
                        f"for {net[key]}"
                    )


###### Hands and boards ######


def test_hands_reconcile():
    """Every seat's hand is exactly the multiset of cards the ledger left there.

    Cards arrive by draw or by an incoming pass, and leave by discard, tuck,
    play, or an outgoing pass. Comparing the whole multiset (not just its size)
    catches a mis-attributed transfer that a count alone would hide."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        hands: dict[int, collections.Counter[str]] = {
            player.id: collections.Counter() for player in played.state.players
        }
        for effect in played.effects:
            if isinstance(effect, models.DrawCardEffect):
                assert effect.player_id is not None
                hands[effect.player_id][effect.card] += 1
            elif isinstance(
                effect,
                (
                    models.DiscardCardEffect,
                    models.PlayBirdEffect,
                ),
            ):
                assert effect.player_id is not None
                hands[effect.player_id][effect.card] -= 1
            elif isinstance(effect, models.TuckCardEffect):
                # Only a tuck from hand leaves a hand; a deck tuck never entered one.
                if effect.source == models.CardSource.HAND:
                    assert effect.player_id is not None
                    hands[effect.player_id][effect.card] -= 1
            elif isinstance(effect, models.PassCardEffect):
                assert effect.player_id is not None
                hands[effect.player_id][effect.card] -= 1
                hands[effect.to_player_id][effect.card] += 1

        for player in played.state.players:
            actual = collections.Counter(bird.name for bird in player.hand)
            expected = +hands[player.id]  # drop zero/negative entries for comparison
            assert actual == expected, (
                f"seed {seed}/{seats}p: seat {player.id} hand mismatch; "
                f"only in hand={actual - expected}, only in ledger={expected - actual}"
            )


def test_boards_reconcile():
    """Replaying the ledger's placements and moves rebuilds every board exactly.

    This is the strongest of the reconciliations: it reproduces row *order*, not
    just membership, so a placement recorded at the wrong slot or a habitat move
    recorded as a fresh play both fail here."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        rows: dict[tuple[int, str], list[str]] = collections.defaultdict(list)
        for effect in played.effects:
            if isinstance(effect, models.PlayBirdEffect):
                assert effect.player_id is not None
                rows[(effect.player_id, effect.habitat)].append(effect.card)
            elif isinstance(effect, models.MoveBirdEffect):
                assert effect.player_id is not None
                source = rows[(effect.player_id, effect.from_habitat)]
                assert source.pop() == effect.card, (
                    f"seed {seed}/{seats}p: move of {effect.card} does not come "
                    f"from the rightmost slot of {effect.from_habitat}"
                )
                rows[(effect.player_id, effect.to_habitat)].append(effect.card)

        for player in played.state.players:
            for habitat, row in player.board.items():
                replayed = rows[(player.id, habitat.value)]
                assert [pb.bird.name for pb in row] == replayed, (
                    f"seed {seed}/{seats}p: seat {player.id} {habitat.value} row is "
                    f"{[pb.bird.name for pb in row]} but the ledger replays to "
                    f"{replayed}"
                )


###### Ledger completeness and shape ######


def test_ledger_is_non_trivial():
    """The reconciliation above is only meaningful if effects actually fire.

    Guards against a regression that silently stops recording: every assertion
    in this module passes vacuously against an empty ledger."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        assert (
            len(played.effects) > 100
        ), f"seed {seed}/{seats}p: only {len(played.effects)} effects recorded"
        kinds = {type(effect) for effect in played.effects}
        for expected in (
            models.GainFoodEffect,
            models.SpendFoodEffect,
            models.LayEggEffect,
            models.DrawCardEffect,
            models.DiscardCardEffect,
            models.PlayBirdEffect,
            models.FeederRerollEffect,
            models.TrayRefillEffect,
        ):
            assert expected in kinds, f"seed {seed}/{seats}p: no {expected.__name__}"


def test_every_seat_owned_effect_names_a_seat():
    """Effects on a seat's own resources are attributed to that seat.

    Per-seat reconciliation depends on it: an effect with no ``player_id`` would
    silently drop out of every sum above. The face-up tray is the one genuinely
    seatless resource — the end-of-round reset belongs to no player — so its
    refills are exempt."""
    for seed, seats in _CASES:
        played = _play(seed, seats)
        orphans = [
            effect
            for effect in played.effects
            if effect.player_id is None
            and not isinstance(effect, models.TrayRefillEffect)
        ]
        assert not orphans, (
            f"seed {seed}/{seats}p: {len(orphans)} unattributed effects, e.g. "
            f"{orphans[0]!r}"
        )


def test_reveals_carry_their_hidden_information():
    """The four reveal effects record what was previously hidden."""
    played = _play(42, 2)
    deck_draws = [
        effect
        for effect in _of_type(played.effects, models.DrawCardEffect)
        if effect.source is models.CardSource.DECK
    ]
    assert deck_draws and all(effect.card for effect in deck_draws)

    rerolls = _of_type(played.effects, models.FeederRerollEffect)
    assert rerolls and all(
        len(effect.faces) == state_module.BIRDFEEDER_DICE for effect in rerolls
    )

    refills = _of_type(played.effects, models.TrayRefillEffect)
    assert refills and all(
        effect.card and 0 <= effect.slot < state_module.TRAY_SIZE for effect in refills
    )


def test_ledger_survives_serialization():
    """Effects round-trip through JSON with their fields intact.

    They are ordinary ``sub_events``, so they ride the same discriminated union
    as decisions — and would silently lose every field if one were added without
    joining it (see ``tests/test_gamelog_serialize.py``)."""
    rec = gamelog_recorder.EventRecorder(probes=(None, None))
    eng, *_ = engine_module.Engine.create(seed=5)
    rng = random.Random(5)
    engine_module.Engine.play_one_game(
        eng.state, [agents.random_agent(rng) for _ in range(2)], event_recorder=rec
    )
    restored = models.GameEventTree.model_validate_json(rec.root.model_dump_json())
    before = _all_effects(rec.root)
    after = _all_effects(restored)
    assert before and before == after
