# pyright: reportPrivateUsage=false
# (tests read the package-private layout offsets to pin the frozen stripes,
# matching the test_compat_v1_4.py convention)
"""Tests for the v1.6 compat shim (``wingspan.compat.v1_5``): era-1.5
SKIP_OPTIONAL accept-row pricing and the pre-v1.6 forecast effect mapper.

v1.6 made every accept row carry the full exchange ledger. An era-1.5 artifact
must keep computing the old rows byte for byte:

* ``TuckActivateChoice`` rows: special-kind bit + the ``cards_to_tuck`` exchange
  slot only (no paid card, no payoff, no consequence stripes);
* ``PayCostChoice`` rows: no ``gained_food`` type in the GAIN_FOOD stripe, an
  any-feeder-die ``becomes_playable`` for every food gain, and a ``bonus_delta``
  from the hand-card flow only;
* the card-table ``power_ex`` block and every habitat-row ``MainActionChoice``
  ``exchange`` stripe, priced with the pre-v1.6 effect mapper (the expected
  values below are written out by hand, never by calling the mapper).

Geometry is unchanged, so era 1.5 keeps the live dims. As for every era a
committed LFS checkpoint is deferred: the round-trip test builds an era-1.5 net
and reloads it through the production ``load_policy_net`` path.
"""

from __future__ import annotations

import pathlib
import random
import typing

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import (
    architecture,
    cards,
    compat,
    decisions,
    encode,
    model,
    setup_model,
    state,
)
from wingspan.compat import v1_4 as compat_v1_4
from wingspan.compat import v1_5 as compat_v1_5
from wingspan.encode import layout
from wingspan.players import loaders
from wingspan.training import config
from wingspan.training import setup_net as setup_net_module

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}
_BONUS_DELTA = slice(
    layout._OFF_BONUS_DELTA, layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_DIM
)


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_net(net_cls: type[model.PolicyValueNet], era: str) -> model.PolicyValueNet:
    """A net of ``net_cls`` built at ``era``'s dims — how the load path builds it."""
    spec = encode.DEFAULT_SPEC
    state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
    return net_cls(
        state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
    )


def _game() -> state.GameState:
    return state.new_game(random.Random(3), _BIRDS, _BONUSES, _GOALS)


def _tuck_decision(ledger: decisions.ExchangeLedger) -> decisions.ActivateTuckDecision:
    return decisions.ActivateTuckDecision(
        player_id=0,
        prompt="tuck?",
        choices=[
            decisions.TuckActivateChoice(label="tuck", **ledger.model_dump()),
            decisions.SkipChoice(label="skip"),
        ],
    )


def _pay_decision(ledger: decisions.ExchangeLedger) -> decisions.AcceptExchangeDecision:
    return decisions.AcceptExchangeDecision(
        player_id=0,
        prompt="accept?",
        choices=[
            decisions.PayCostChoice(label="pay", **ledger.model_dump()),
            decisions.SkipChoice(label="skip"),
        ],
    )


def _old_tuck_row(tuck_count: int, width: int) -> np.ndarray:
    """The era-1.5 tuck yes row, written out by hand: special kind and the
    ``cards_to_tuck`` exchange slot, nothing else."""
    row = np.zeros(width, dtype=np.float32)
    row[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    row[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_TUCK] = (
        tuck_count / layout._EXCHANGE_SCALE
    )
    return row


_FULL_TUCK = decisions.ExchangeLedger(
    paid_card_count=1,
    gained_tuck_count=1,
    gained_egg_count=1,
    gained_card_count=1,
    gained_food_count=1,
    gained_food=cards.Food.FRUIT,
)


class TestRouting:
    def test_era_1_5_routes_to_the_shim_with_live_dims(self) -> None:
        assert (
            model.PolicyValueNet.class_for_version("1.5")
            is compat_v1_5.PolicyValueNetV1_5
        )
        spec = encode.DEFAULT_SPEC
        assert compat.encoding_dims_for_era("1.5", spec) == (
            encode.state_size(spec),
            encode.choice_feature_dim(spec),
        )

    def test_live_era_is_the_live_net(self) -> None:
        assert model.PolicyValueNet.class_for_version("1.6") is model.PolicyValueNet

    def test_older_shims_inherit_the_1_5_shim(self) -> None:
        assert issubclass(
            compat_v1_4.PolicyValueNetV1_4, compat_v1_5.PolicyValueNetV1_5
        )


class TestEra15AcceptRowsAreFrozen:
    def test_tuck_row_is_only_the_old_tuck_slot(self) -> None:
        game_state = _game()
        net = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        rows = net.encode_choices(_tuck_decision(_FULL_TUCK), game_state)

        np.testing.assert_array_equal(rows[0], _old_tuck_row(1, rows.shape[1]))
        # The live encoding of the same row is richer, so the shim really changed it.
        live = encode.encode_choices(_tuck_decision(_FULL_TUCK), game_state)
        assert not np.array_equal(live[0], rows[0])

    def test_skip_row_is_untouched(self) -> None:
        game_state = _game()
        net = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        decision = _tuck_decision(_FULL_TUCK)
        live = encode.encode_choices(decision, game_state)
        np.testing.assert_array_equal(
            net.encode_choices(decision, game_state)[1], live[1]
        )

    def test_unaffected_pay_row_matches_live(self) -> None:
        """A ledger the v1.6 changes do not touch encodes identically live."""
        game_state = _game()
        net = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        decision = _pay_decision(
            decisions.ExchangeLedger(
                paid_card_count=1, paid_egg_count=1, gained_card_count=1
            )
        )
        np.testing.assert_array_equal(
            net.encode_choices(decision, game_state),
            encode.encode_choices(decision, game_state),
        )

    def test_typed_food_is_priced_as_an_untyped_gain(self) -> None:
        """Era 1.5 ignored ``gained_food``: the row equals the live encoding of
        the same ledger with the type left unspecified."""
        game_state = _game()
        game_state.players[0].hand = list(_BIRDS[:5])
        net = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        typed = decisions.ExchangeLedger(
            gained_food_count=1, gained_food=cards.Food.FRUIT
        )
        untyped = decisions.ExchangeLedger(gained_food_count=1)
        np.testing.assert_array_equal(
            net.encode_choices(_pay_decision(typed), game_state),
            encode.encode_choices(_pay_decision(untyped), game_state),
        )

    def test_egg_gain_bonus_delta_is_hand_flow_only(self) -> None:
        """Era 1.5 left the egg-counting cards out of ``bonus_delta``: priced
        with Oologist held, the row equals the live row with Oologist absent."""
        egg_ledger = decisions.ExchangeLedger(
            paid_card_count=1, gained_tuck_count=1, gained_egg_count=1
        )
        bird = next(candidate for candidate in _BIRDS if candidate.egg_limit >= 1)

        def staged(bonus_names: list[str]) -> state.GameState:
            game_state = _game()
            player = game_state.players[0]
            player.bonus_cards = [_BONUS_BY_NAME[name] for name in bonus_names]
            player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird)]
            return game_state

        net = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        decision = _pay_decision(egg_ledger)
        era_row = net.encode_choices(
            decision, staged(["Oologist", "Visionary Leader"])
        )[0]
        hand_only_row = encode.encode_choices(decision, staged(["Visionary Leader"]))[0]
        live_row = encode.encode_choices(
            decision, staged(["Oologist", "Visionary Leader"])
        )[0]

        np.testing.assert_array_equal(era_row, hand_only_row)
        assert not np.array_equal(era_row[_BONUS_DELTA], live_row[_BONUS_DELTA])

    def test_era_1_4_net_chains_the_same_freeze(self) -> None:
        """``PolicyValueNetV1_4`` inherits the shim: its tuck row is the old
        row narrowed by its own tail strip."""
        game_state = _game()
        net = _era_net(compat_v1_4.PolicyValueNetV1_4, "1.4")
        rows = net.encode_choices(_tuck_decision(_FULL_TUCK), game_state)
        live_width = encode.choice_feature_dim(encode.DEFAULT_SPEC)
        start = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
        end = start + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        expected = np.delete(_old_tuck_row(1, live_width), slice(start, end))
        np.testing.assert_array_equal(rows[0], expected)


# ---------------------------------------------------------------------------
# The forecast effect-mapper freeze (card-table power_ex + MAIN_ACTION exchange)

_POWER_EX = slice(
    layout._OFF_ATTR_POWER_EX, layout._OFF_ATTR_POWER_EX + layout._EXCHANGE_DIM
)
_EXCHANGE = slice(layout._OFF_EXCHANGE, layout._OFF_EXCHANGE + layout._EXCHANGE_DIM)

# bird -> (old ledger, new ledger) of its single power effect, written out by
# hand from the pre-v1.6 / v1.6 mapper rules (never by calling the mapper).
_PowerLedgers = tuple[dict[str, int], dict[str, int]]
_FIXED_BIRDS: dict[str, _PowerLedgers] = {
    "Cedar Waxwing": (
        {"paid_card_count": 1, "gained_food_count": 1},
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_food_count": 1},
    ),
    "Pygmy Nuthatch": (
        {"paid_card_count": 1},
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_food_count": 1},
    ),
    "American Coot": (
        {"paid_card_count": 1, "gained_card_count": 1},
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_card_count": 1},
    ),
    "Brewer's Blackbird": (
        {"paid_card_count": 1, "gained_egg_count": 1},
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_egg_count": 1},
    ),
    "White-Throated Swift": (
        {"paid_card_count": 1, "gained_egg_count": 1},
        {"paid_card_count": 1, "gained_tuck_count": 1, "gained_egg_count": 1},
    ),
    "Sandhill Crane": (
        {"paid_egg_count": 1, "gained_tuck_count": 2},
        {"paid_food_count": 1, "gained_tuck_count": 2},
    ),
    "Say's Phoebe": (
        {"gained_egg_count": 1, "opp_gained_egg_count": 1},
        {"gained_egg_count": 1},
    ),
    "American Bittern": (
        {},
        {"gained_card_count": 1, "opp_gained_card_count": 1},
    ),
    "American Oystercatcher": (
        {},
        {"gained_card_count": 2, "opp_gained_card_count": 1},
    ),
    "American Avocet": ({}, {"gained_egg_count": 1}),
    "Hermit Thrush": ({}, {"gained_food_count": 1, "opp_gained_food_count": 1}),
    "Barred Owl": ({}, {"gained_tuck_count": 1}),
    "Horned Lark": (
        {"gained_tuck_count": 1},
        {"paid_card_count": 1, "gained_tuck_count": 1},
    ),
    "Anna's Hummingbird": (
        {"gained_food_count": 1, "opp_gained_food_count": 1},
        {"gained_food_count": 1, "opp_gained_food_count": 1},
    ),
}
_BIRD_BY_NAME = {bird.name: bird for bird in _BIRDS}


def _power_ex_vector(counts: dict[str, int]) -> np.ndarray:
    """The card-table power_ex block for ``counts``, as the mapper projects it
    (float32 slot accumulation, then ``/ _EXCHANGE_SCALE``)."""
    vec = np.zeros(layout._EXCHANGE_DIM, dtype=np.float32)
    for field, count in counts.items():
        vec[layout._EXCHANGE_SLOT_FOR_LEDGER_FIELD[field]] += count
    return vec / layout._EXCHANGE_SCALE


def _setup_net(
    net_cls: type[setup_net_module.SetupNet],
) -> setup_net_module.SetupNet:
    return net_cls(
        encoding=setup_model.SetupEncoding(),
        arch=setup_model.SetupArchitecture(
            trunk_layers=(8,), choice_layers=(8,), head_layers=(8,), value_layers=()
        ),
        main_arch=_small_arch(),
    )


def _card_table_row(net: model.PolicyValueNet, bird_name: str) -> np.ndarray:
    row = cards.bird_index(_BIRD_BY_NAME[bird_name]) + 1
    return net.card_features.numpy()[row, _POWER_EX]


class TestEra15ForecastMapperCardTableFrozen:
    def test_every_fixed_kind_keeps_its_old_power_ex(self) -> None:
        shim = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        live = _era_net(model.PolicyValueNet, "1.6")
        for bird_name, (old, new) in _FIXED_BIRDS.items():
            assert len(_BIRD_BY_NAME[bird_name].power.effects) == 1, bird_name
            np.testing.assert_array_equal(
                _card_table_row(shim, bird_name), _power_ex_vector(old), bird_name
            )
            np.testing.assert_array_equal(
                _card_table_row(live, bird_name), _power_ex_vector(new), bird_name
            )
            if old != new:
                assert not np.array_equal(
                    _card_table_row(shim, bird_name), _card_table_row(live, bird_name)
                ), bird_name

    def test_unrelated_rows_match_live(self) -> None:
        """Only the birds with a fixed kind differ between the era table and the
        live table; every other row (EOT-discard birds included — era 1.5 has
        the discard side) is identical."""
        shim = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        live = _era_net(model.PolicyValueNet, "1.6")
        fixed_rows = {
            cards.bird_index(_BIRD_BY_NAME[name]) + 1
            for name in _FIXED_BIRDS
            if _FIXED_BIRDS[name][0] != _FIXED_BIRDS[name][1]
        }
        shim_block = shim.card_features.numpy()
        live_block = live.card_features.numpy()
        differing = {
            row
            for row in range(shim_block.shape[0])
            if not np.array_equal(shim_block[row], live_block[row])
        }
        # Every differing row is a catalog bird carrying one of the fixed kinds
        # (a superset of the sampled birds: all birds sharing those kinds).
        assert fixed_rows <= differing
        bird_by_row = {cards.bird_index(bird) + 1: bird for bird in _BIRDS}
        for row in differing:
            bird = bird_by_row[row]
            assert any(
                effect.kind in _FIXED_KINDS for effect in bird.power.effects
            ), bird.name

    def test_era_1_4_chain_freezes_the_same_values(self) -> None:
        """``PolicyValueNetV1_4`` inherits the shim; its own pre-EOT refill
        keeps the old mapper too (none of the sampled birds has an EOT power)."""
        chained = _era_net(compat_v1_4.PolicyValueNetV1_4, "1.4")
        for bird_name, (old, _new) in _FIXED_BIRDS.items():
            np.testing.assert_array_equal(
                _card_table_row(chained, bird_name), _power_ex_vector(old), bird_name
            )

    def test_setup_net_routes_and_freezes_the_card_table(self) -> None:
        assert (
            setup_net_module.SetupNet.class_for_version("1.5")
            is compat_v1_5.SetupNetV1_5
        )
        assert issubclass(compat_v1_4.SetupNetV1_4, compat_v1_5.SetupNetV1_5)
        assert (
            setup_net_module.SetupNet.class_for_version("1.6")
            is setup_net_module.SetupNet
        )
        for net_cls in (compat_v1_5.SetupNetV1_5, compat_v1_4.SetupNetV1_4):
            shim = _setup_net(net_cls)
            for bird_name, (old, _new) in _FIXED_BIRDS.items():
                row = cards.bird_index(_BIRD_BY_NAME[bird_name]) + 1
                np.testing.assert_array_equal(
                    shim.card_features.numpy()[row, _POWER_EX],
                    _power_ex_vector(old),
                    f"{net_cls.__name__} {bird_name}",
                )
        live = _setup_net(setup_net_module.SetupNet)
        sandhill = cards.bird_index(_BIRD_BY_NAME["Sandhill Crane"]) + 1
        np.testing.assert_array_equal(
            live.card_features.numpy()[sandhill, _POWER_EX],
            _power_ex_vector(_FIXED_BIRDS["Sandhill Crane"][1]),
        )


_FIXED_KINDS = frozenset(
    {
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY,
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE,
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS,
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY,
        cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW,
        cards.EffectKind.TUCK_FROM_HAND,
        cards.EffectKind.TUCK_FROM_DECK_PAID,
        cards.EffectKind.LAY_EGG_ALL_NEST,
        cards.EffectKind.FEWEST_WETLAND_DRAWS_CARD,
        cards.EffectKind.FEWEST_FOREST_GAINS_DIE,
        cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
        cards.EffectKind.PREDATOR_HUNT,
        cards.EffectKind.PINK_PLAY_BIRD_TUCK,
        cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER,
    }
)


class _Scenario(typing.NamedTuple):
    """One staged MAIN_ACTION row: ``bird`` alone in ``habitat``'s row."""

    bird: str
    habitat: cards.Habitat
    action: decisions.MainAction
    hand: int
    seed_food: int


_SCENARIOS = [
    _Scenario(
        "Cedar Waxwing", cards.Habitat.FOREST, decisions.MainAction.GAIN_FOOD, 2, 0
    ),
    _Scenario(
        "Pygmy Nuthatch", cards.Habitat.FOREST, decisions.MainAction.GAIN_FOOD, 2, 0
    ),
    _Scenario("Barred Owl", cards.Habitat.FOREST, decisions.MainAction.GAIN_FOOD, 0, 0),
    _Scenario(
        "American Coot", cards.Habitat.WETLAND, decisions.MainAction.DRAW_CARDS, 2, 0
    ),
    _Scenario(
        "American Bittern", cards.Habitat.WETLAND, decisions.MainAction.DRAW_CARDS, 0, 0
    ),
    # The old mapper charged an egg the player lacks, so the power was skipped.
    _Scenario(
        "Sandhill Crane", cards.Habitat.WETLAND, decisions.MainAction.DRAW_CARDS, 0, 1
    ),
]
# (The white-power kinds — LAY_EGG_ALL_NEST — and the pink Horned Lark never
# fire from a main-action row, so only the card table carries them.)


def _stage(scenario: _Scenario) -> state.GameState:
    game_state = _game()
    player = game_state.players[0]
    player.hand = list(_BIRDS[: scenario.hand])
    player.food = state.FoodPool.from_dict({cards.Food.SEED: scenario.seed_food})
    player.board[scenario.habitat] = [
        state.PlayedBird(bird=_BIRD_BY_NAME[scenario.bird])
    ]
    return game_state


def _expected_exchange_row(
    scenario: _Scenario, game_state: state.GameState, *, old: bool
) -> np.ndarray:
    """The 13-slot MAIN_ACTION exchange stripe, written out by hand: the row's
    base track gain, the trade-arrow conversion where the staged hand / food /
    eggs allow it, then the bird's single power at the old or new mapping."""
    player = game_state.players[0]
    board = player.board
    counts: dict[str, int] = {}

    def add(field: str, amount: int) -> None:
        counts[field] = counts.get(field, 0) + amount

    hand_left = scenario.hand
    if scenario.action == decisions.MainAction.GAIN_FOOD:
        add("gained_food_count", board.gain_food_count())
        if board.action_offers_convert(scenario.habitat) and hand_left >= 1:
            add("paid_card_count", 1)
            add("gained_food_count", 1)
            hand_left -= 1
    else:  # DRAW_CARDS: no eggs staged, so the egg->card conversion never fires
        drawn = board.draw_cards_count()
        add("gained_card_count", drawn)
        hand_left += drawn

    # The power fires only when its costs fit what is left (no eggs are staged).
    power_old, power_new = _FIXED_BIRDS[scenario.bird]
    power = power_old if old else power_new
    affordable = (
        power.get("paid_card_count", 0) <= hand_left
        and power.get("paid_food_count", 0) <= scenario.seed_food
        and power.get("paid_egg_count", 0) <= 0
    )
    for field, amount in (power if affordable else {}).items():
        add(field, amount)

    row = np.zeros(layout._EXCHANGE_DIM, dtype=np.float32)
    for field, slot in layout._EXCHANGE_SLOT_FOR_LEDGER_FIELD.items():
        row[slot] = counts.get(field, 0) / layout._EXCHANGE_SCALE
    return row


class TestEra15ForecastMapperMainActionRowsFrozen:
    @pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda sc: sc.bird)
    def test_main_action_exchange_row_is_the_old_forecast(
        self, scenario: _Scenario
    ) -> None:
        game_state = _stage(scenario)
        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(
                    label=scenario.action.value, action=scenario.action
                )
            ],
        )
        shim = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        shim_row = shim.encode_choices(decision, game_state)[0]
        live_row = encode.encode_choices(decision, game_state)[0]

        np.testing.assert_array_equal(
            shim_row[_EXCHANGE],
            _expected_exchange_row(scenario, game_state, old=True),
        )
        np.testing.assert_array_equal(
            live_row[_EXCHANGE],
            _expected_exchange_row(scenario, game_state, old=False),
        )
        # Only the exchange stripe is frozen: every other column is the live one.
        outside = np.ones(shim_row.shape[0], dtype=bool)
        outside[_EXCHANGE] = False
        np.testing.assert_array_equal(shim_row[outside], live_row[outside])

    def test_play_bird_row_is_untouched(self) -> None:
        game_state = _game()
        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(
                    label="play", action=decisions.MainAction.PLAY_BIRD
                )
            ],
        )
        shim = _era_net(compat_v1_5.PolicyValueNetV1_5, "1.5")
        np.testing.assert_array_equal(
            shim.encode_choices(decision, game_state),
            encode.encode_choices(decision, game_state),
        )

    def test_era_1_4_chain_zeroes_the_exchange_stripe(self) -> None:
        """Era <= 1.4 never had the forecast: its rows stay zero through the
        inherited era-1.5 refill."""
        scenario = _SCENARIOS[0]
        game_state = _stage(scenario)
        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(
                    label=scenario.action.value, action=scenario.action
                )
            ],
        )
        chained = _era_net(compat_v1_4.PolicyValueNetV1_4, "1.4")
        assert np.all(chained.encode_choices(decision, game_state)[:, _EXCHANGE] == 0.0)


def test_v1_5_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.5-stamped checkpoint loads under live code via ``load_policy_net`` as
    the shim class (at the live dims — v1.6 changed no shape) and forward-passes a
    tuck-gate decision through the frozen accept-row encoder."""
    base = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(
            run_name="v15-roundtrip",
            checkpoint_dir=str(tmp_path),
            games_per_iter=2,
            eval_games=2,
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(8, 8),
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
        ),
    )
    cfg = config.with_encoding_version(base, "1.5")
    assert cfg.encoding_version == "1.5"
    assert cfg.state_dim == encode.state_size(cfg.encoding_spec)
    assert cfg.choice_dim == encode.choice_feature_dim(cfg.encoding_spec)

    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_5.PolicyValueNetV1_5
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )
    ckpt = tmp_path / "v15.pt"
    torch.save(
        {"config": cfg.model_dump(), "model": net.state_dict(), "version": "1.5"},
        ckpt,
    )

    loaded, saved_cfg = loaders.load_policy_net(ckpt, torch.device("cpu"))
    assert type(loaded) is compat_v1_5.PolicyValueNetV1_5
    assert saved_cfg.encoding_version == "1.5"

    game_state = _game()
    decision = _tuck_decision(_FULL_TUCK)
    state_vec = loaded.encode_state(game_state, decision)
    choice_feats = loaded.encode_choices(decision, game_state)
    logits, _value = loaded(
        torch.from_numpy(state_vec).unsqueeze(0),
        torch.from_numpy(choice_feats).unsqueeze(0),
        torch.ones(1, choice_feats.shape[0]),
        torch.tensor([decisions.family_index_for(type(decision))], dtype=torch.long),
    )
    assert logits.shape == (1, choice_feats.shape[0])
