# pyright: reportPrivateUsage=false
# (tests verify shim output row-by-row against the live encoder — a deliberate
# compat coupling matching test_compat_shim_v0_4.py)
"""Unit tests for the ``wingspan.compat.v0_7`` ``becomes_playable`` food shim.

Pins the shim's semantics directly:

* The version predicate covers exactly 0.7 (not 0.6, not 0.8).
* ``encode_choices_v07`` reproduces the eggs-included ``becomes_playable`` bits
  that a v0.7 checkpoint was trained against.
* On an egg-blocked board, the live encoder (v0.8) and the shim diverge on a
  food-gain choice row: the live encoder lights up the food-affordable bird; the
  shim leaves it dark (because the egg cost is not met).
* ``PolicyValueNetV06.encode_choices`` delegates to ``encode_choices_v07`` and
  produces the same result.
* Version routing: 0.7 → ``PolicyValueNetV07``, 0.6 → ``PolicyValueNetV06``,
  0.8 / live → ``PolicyValueNet``.
* ``PolicyValueNetV07`` forward pass produces finite logits and value.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import (
    architecture,
    cards,
    decisions,
    encode,
    engine,
    model,
    state,
    version,
)
from wingspan.compat import v0_6, v0_7
from wingspan.encode import layout
from wingspan.training import runmeta

_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)

# ---------------------------------------------------------------------------
# Version predicate


def test_version_predicate_covers_0_7():
    """``uses_v0_7_becomes_playable_encoding`` is True for exactly 0.7."""
    assert v0_7.uses_v0_7_becomes_playable_encoding("0.7")


def test_version_predicate_excludes_0_6():
    """0.6 artifacts are handled by the v0_6 card-encoder shim, not this one."""
    assert not v0_7.uses_v0_7_becomes_playable_encoding("0.6")


def test_version_predicate_excludes_0_8_and_beyond():
    """0.8+ artifacts use the live encoding."""
    assert not v0_7.uses_v0_7_becomes_playable_encoding("0.8")
    assert not v0_7.uses_v0_7_becomes_playable_encoding("1.0")


def test_version_predicate_excludes_pre_0_7():
    """Pre-0.7 artifacts fall under earlier shims."""
    assert not v0_7.uses_v0_7_becomes_playable_encoding("0.5")
    assert not v0_7.uses_v0_7_becomes_playable_encoding("0.0")


# ---------------------------------------------------------------------------
# Egg-blocked board: live vs. shim diverge on the food-gain row


def _build_egg_blocked_eng() -> tuple[
    engine.Engine,
    list[cards.Bird],
    cards.Bird | None,
    decisions.GainFoodDecision | None,
]:
    """Create an engine with one FOREST bird in play (0 eggs), player food zeroed,
    and a FOREST-only 1-SEED bird in hand. Returns ``(eng, birds_list, target,
    decision)`` where ``target`` / ``decision`` are None if the catalog has no
    suitable bird."""
    eng, birds_list, *_ = engine.Engine.create(seed=50)
    player = eng.state.players[0]
    seed_idx = cards.food_index(cards.Food.SEED)
    fish_idx = cards.food_index(cards.Food.FISH)

    # Place one bird in the forest (eggs=0) so slot 2 costs 1 egg.
    forest_occupant = next(
        (bird for bird in birds_list if cards.Habitat.FOREST in bird.habitats), None
    )
    if forest_occupant is None:
        return eng, birds_list, None, None
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=forest_occupant))

    # Target: FOREST-only bird needing exactly 1 SEED.
    target = next(
        (
            bird
            for bird in birds_list
            if (
                bird.habitats == (cards.Habitat.FOREST,)
                and bird.food_cost.counts[seed_idx] == 1
                and bird.food_cost.counts[fish_idx] == 0
                and bird.food_cost.total == 1
            )
        ),
        None,
    )
    if target is None:
        return eng, birds_list, None, None

    player.food = state.FoodPool(counts=[0] * cards.N_FOODS)
    player.hand = [target]

    decision: decisions.GainFoodDecision = decisions.GainFoodDecision(
        player_id=0,
        prompt="gain food",
        choices=[
            decisions.FoodChoice(
                label="seed", food=cards.Food.SEED, from_choice_die=False
            ),
            decisions.FoodChoice(
                label="fish", food=cards.Food.FISH, from_choice_die=False
            ),
        ],
    )
    return eng, birds_list, target, decision


def test_live_encoder_flags_seed_for_egg_blocked_bird():
    """v0.8 live encoder lights up the egg-blocked SEED bird on the SEED row."""
    eng, _, target, decision = _build_egg_blocked_eng()
    if target is None or decision is None:
        import pytest

        pytest.skip("no suitable forest-only SEED bird in catalog seed 50")

    rows = encode.encode_choices(decision, eng.state)
    seed_row = rows[0]  # SEED choice is index 0
    offset = layout.CHOICE_BECOMES_PLAYABLE_OFFSET
    bird_bit = seed_row[offset + cards.bird_index(target)]
    assert (
        bird_bit == 1.0
    ), "live encoder should flag SEED bird on the SEED row (ignore_eggs=True)"


def test_shim_v07_does_not_flag_egg_blocked_bird():
    """v0.7 shim (eggs-included) does NOT light up the egg-blocked SEED bird."""
    eng, _, target, decision = _build_egg_blocked_eng()
    if target is None or decision is None:
        import pytest

        pytest.skip("no suitable forest-only SEED bird in catalog seed 50")

    rows = v0_7.encode_choices_v07(decision, eng.state)
    seed_row = rows[0]  # SEED choice is index 0
    offset = layout.CHOICE_BECOMES_PLAYABLE_OFFSET
    bird_bit = seed_row[offset + cards.bird_index(target)]
    assert (
        bird_bit == 0.0
    ), "shim should NOT flag egg-blocked bird (eggs-included semantics)"


def test_v06_encode_choices_matches_v07_shim():
    """``PolicyValueNetV06.encode_choices`` delegates to v0_7 and produces the
    same rows as ``encode_choices_v07`` on the egg-blocked board."""
    eng, _, target, decision = _build_egg_blocked_eng()
    if target is None or decision is None:
        import pytest

        pytest.skip("no suitable forest-only SEED bird in catalog seed 50")

    net = v0_6.PolicyValueNetV06(
        arch=_SMALL,
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
    )
    net_rows = net.encode_choices(decision, eng.state)  # type: ignore[arg-type]
    shim_rows = v0_7.encode_choices_v07(decision, eng.state)
    assert np.array_equal(
        net_rows, shim_rows
    ), "PolicyValueNetV06.encode_choices must match encode_choices_v07 row-for-row"


def test_live_and_shim_rows_differ_on_seed_row():
    """The live encoder and the shim produce different becomes_playable bits on
    the SEED row when the target bird is egg-blocked — this is the key regression
    check for the shim."""
    eng, _, target, decision = _build_egg_blocked_eng()
    if target is None or decision is None:
        import pytest

        pytest.skip("no suitable forest-only SEED bird in catalog seed 50")

    live_rows = encode.encode_choices(decision, eng.state)
    shim_rows = v0_7.encode_choices_v07(decision, eng.state)
    assert not np.array_equal(
        live_rows, shim_rows
    ), "live and shim rows must differ on the egg-blocked seed scenario"


# ---------------------------------------------------------------------------
# PolicyValueNetV07


def test_policy_value_net_v07_encode_choices_uses_shim():
    """``PolicyValueNetV07.encode_choices`` produces the same rows as ``encode_choices_v07``."""
    eng, _, target, decision = _build_egg_blocked_eng()
    if target is None or decision is None:
        import pytest

        pytest.skip("no suitable forest-only SEED bird in catalog seed 50")

    net = v0_7.PolicyValueNetV07(
        arch=_SMALL,
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
    )
    net_rows = net.encode_choices(decision, eng.state)  # type: ignore[arg-type]
    shim_rows = v0_7.encode_choices_v07(decision, eng.state)
    assert np.array_equal(net_rows, shim_rows)


def test_policy_value_net_v07_forward_pass_finite():
    """A batch of synthetic inputs through ``PolicyValueNetV07`` produces finite
    logits and value."""
    net = v0_7.PolicyValueNetV07(
        arch=_SMALL,
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
    )
    net.eval()
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, net.state_dim)
    choice_vec = torch.randn(batch_size, n_choices, net.choice_dim)
    mask = torch.ones(batch_size, n_choices)
    family = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_vec, choice_vec, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# Version routing


def test_from_model_config_routes_0_7_to_v07():
    """A v0.7 descriptor reconstructs as ``PolicyValueNetV07``."""
    v07_config = runmeta.ModelConfig(
        run_name="routing-v07",
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.7",
    )
    net = model.PolicyValueNet.from_model_config(v07_config)
    assert isinstance(net, v0_7.PolicyValueNetV07)


def test_from_model_config_routes_0_6_to_v06():
    """A v0.6 descriptor still reconstructs as ``PolicyValueNetV06`` (card shim)."""
    v06_config = runmeta.ModelConfig(
        run_name="routing-v06",
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.6",
    )
    net = model.PolicyValueNet.from_model_config(v06_config)
    assert isinstance(net, v0_6.PolicyValueNetV06)
    assert not isinstance(net, v0_7.PolicyValueNetV07)


def test_from_model_config_routes_live_to_base():
    """A current-version descriptor reconstructs as the live ``PolicyValueNet``."""
    live_config = runmeta.ModelConfig(
        run_name="routing-live",
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version=version.MODEL_VERSION,
    )
    net = model.PolicyValueNet.from_model_config(live_config)
    assert type(net) is model.PolicyValueNet
    assert not isinstance(net, v0_7.PolicyValueNetV07)
    assert not isinstance(net, v0_6.PolicyValueNetV06)
