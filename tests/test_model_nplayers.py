"""N-player model construction and forward pass.

``ModelArchitecture.num_players`` (default 2) joins ``ShapeKey`` alongside
``use_board_attention``: N=2 must stay the checkpoint-compat anchor (a tiny
default net's ``state_dict`` shape table pinned against a committed fixture),
while N=3/4 nets must construct, forward, and — with board attention on —
share their *state_dict key set* with an N=2 net at the same topology (only
the input-facing Linear shapes differ, since every opponent board attends
through the one shared ``board_attn_opp`` module).
"""

from __future__ import annotations

import json
import pathlib
import random
import typing

import pytest

torch = pytest.importorskip("torch")

from wingspan import (
    agents,
    architecture,
    cards,
    decisions,
    encode,
    engine,
    model,
    state,
)
from wingspan.training import runmeta

_FIXTURE_PATH = pathlib.Path(__file__).parent / "data" / "state_dict_shape_n2.json"
_N_CHOICES = 4


def _tiny_arch(
    num_players: int, *, use_board_attention: bool = False
) -> architecture.ModelArchitecture:
    """A deliberately tiny architecture — fast to build/forward, but wide
    enough to exercise every block (card/trunk/choice/scorer/value, plus
    board attention when requested)."""
    return architecture.ModelArchitecture(
        trunk_layers=(8,),
        choice_layers=(8,),
        head_layers=(4,),
        value_layers=(),
        card_embed_dim=4,
        card_encoder_layers=(),
        num_players=num_players,
        use_board_attention=use_board_attention,
    )


def _net(
    num_players: int, *, use_board_attention: bool = False
) -> model.PolicyValueNet:
    spec = encode.EncodingSpec(num_players=num_players)
    arch = _tiny_arch(num_players, use_board_attention=use_board_attention)
    return model.PolicyValueNet(arch=arch, spec=spec)


# ---------------------------------------------------------------------------
# N=2 checkpoint-compat anchor: state_dict {key: shape} vs a committed fixture


def test_n2_default_net_state_dict_shape_matches_committed_fixture():
    """Protects Stages 3-4: the N=2 tiny-arch net's state_dict key order and
    per-key shape must never drift from the committed fixture without a
    deliberate, reviewed change (and, if the shape actually changes, a
    MODEL_VERSION bump). The fixture JSON preserves key order as written
    (registration order — part of the checkpoint format), so both the key
    SET/shapes (dict equality) and the ORDER (list equality) are checked."""
    assert _FIXTURE_PATH.exists(), f"missing fixture: {_FIXTURE_PATH}"
    raw_text = _FIXTURE_PATH.read_text(encoding="utf-8")
    expected = json.loads(raw_text)

    net = _net(2)
    actual = {key: list(tensor.shape) for key, tensor in net.state_dict().items()}
    assert actual == expected
    assert list(actual.keys()) == list(expected.keys())


# ---------------------------------------------------------------------------
# N=3/4 construction + forward on synthetic batches


@pytest.mark.parametrize("num_players", [2, 3, 4, 5])
def test_net_constructs_and_forwards_at_every_n(num_players: int):
    net = _net(num_players).eval()
    spec = encode.EncodingSpec(num_players=num_players)
    batch = 2
    state_t = torch.zeros(batch, encode.state_size(spec))
    choices_t = torch.zeros(batch, _N_CHOICES, encode.choice_feature_dim(spec))
    mask = torch.ones(batch, _N_CHOICES)
    family_idx = torch.zeros(batch, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_t, choices_t, mask, family_idx)
    assert logits.shape == (batch, _N_CHOICES)
    assert value.shape == (batch,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_net_forwards_on_a_real_encoded_n3_decision():
    """Construct a real N=3 game, encode a genuine decision through the live
    encoders, and forward it through a tiny N=3 net — not a synthetic zero
    batch."""
    rng = random.Random(3)
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(rng, birds, bonuses, goals, num_players=3)
    spec = encode.EncodingSpec(num_players=3)
    net = _net(3).eval()

    agent = agents.random_agent(random.Random(4))
    captured_decisions: list[decisions.Decision[typing.Any]] = []
    captured_states: list[state.GameState] = []

    def probe[C: decisions.Choice](
        live_engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        # Skip SetupDecision: it is scored under include_setup=True only, and
        # this test's spec (matching the tiny net's include_setup=False arch)
        # deliberately exercises an ordinary in-game decision instead.
        if (
            not captured_decisions
            and len(decision.choices) > 1
            and not decisions.is_setup_decision(decision)
        ):
            captured_decisions.append(decision)
            captured_states.append(live_engine.state.model_copy(deep=True))
        return agent(live_engine, decision)

    engine.Engine.play_one_game(game_state, (probe, probe, probe))

    decision = captured_decisions[0]
    played_state = captured_states[0]
    state_vec = encode.encode_state(played_state, decision, spec)
    choices_mat = encode.encode_choices(decision, played_state, spec)

    state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    choices_t = torch.tensor(choices_mat, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, choices_mat.shape[0])
    family_idx = torch.tensor(
        [decisions.family_index_for(type(decision))], dtype=torch.long
    )
    with torch.no_grad():
        logits, value = net(state_t, choices_t, mask, family_idx)
    assert logits.shape == (1, choices_mat.shape[0])
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# Board attention: shared board_attn_opp keeps the state_dict key SET fixed
# across N


def test_board_attention_state_dict_key_set_identical_across_n():
    net2 = _net(2, use_board_attention=True)
    net3 = _net(3, use_board_attention=True)
    net4 = _net(4, use_board_attention=True)
    keys2 = set(net2.state_dict().keys())
    keys3 = set(net3.state_dict().keys())
    keys4 = set(net4.state_dict().keys())
    assert keys2 == keys3 == keys4
    # No new modules: exactly board_attn_me / board_attn_opp, at every N.
    attn_keys = {key for key in keys2 if key.startswith("board_attn_")}
    assert {key.split(".")[0] for key in attn_keys} == {
        "board_attn_me",
        "board_attn_opp",
    }


def test_board_attention_only_input_facing_linears_change_shape_with_n():
    net2 = _net(2, use_board_attention=True)
    net3 = _net(3, use_board_attention=True)
    sd2, sd3 = net2.state_dict(), net3.state_dict()
    differing = {key for key in sd2 if sd2[key].shape != sd3[key].shape}
    assert differing == {"state_trunk.0.weight", "choice_encoder.0.weight"}
    # The shared board_attn_opp module's own parameters are shape-identical —
    # it is genuinely one module reused per opponent, not resized.
    for key in sd2:
        if key.startswith("board_attn_"):
            assert sd2[key].shape == sd3[key].shape, key


def test_board_attention_n3_forward_runs():
    net = _net(3, use_board_attention=True).eval()
    spec = encode.EncodingSpec(num_players=3)
    batch = 2
    state_t = torch.zeros(batch, encode.state_size(spec))
    choices_t = torch.zeros(batch, _N_CHOICES, encode.choice_feature_dim(spec))
    mask = torch.ones(batch, _N_CHOICES)
    family_idx = torch.zeros(batch, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_t, choices_t, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# arch/spec seat-count mismatch


def test_mismatched_num_players_raises():
    with pytest.raises(ValueError, match="num_players"):
        model.PolicyValueNet(
            arch=_tiny_arch(3), spec=encode.EncodingSpec(num_players=2)
        )


# ---------------------------------------------------------------------------
# from_model_config: N=3 roundtrip + default-to-2 when the field is absent


def test_from_model_config_roundtrips_at_n3():
    spec = encode.EncodingSpec(num_players=3)
    arch = _tiny_arch(3)
    descriptor = runmeta.ModelConfig(
        run_name="t",
        state_dim=encode.state_size(spec),
        choice_dim=encode.choice_feature_dim(spec),
        family_order=tuple(
            f.value for f in decisions.active_decision_families(spec.include_setup)
        ),
        architecture=arch,
        include_setup=False,
        num_players=3,
    )
    net = model.PolicyValueNet.from_model_config(descriptor)
    assert net.spec.num_players == 3
    assert net.arch.num_players == 3
    assert net.state_dim == encode.state_size(spec)
    assert net.choice_dim == encode.choice_feature_dim(spec)


def test_model_config_defaults_num_players_to_2_when_field_absent():
    """A descriptor JSON that predates the num_players field (the key entirely
    absent, simulating a pre-Stage-2 model_config.json) rehydrates to
    num_players=2 — the rehydration guarantee: pre-N artifacts compute
    identically."""
    spec = encode.DEFAULT_SPEC
    arch = _tiny_arch(2)
    raw = {
        "run_name": "t",
        "state_dim": encode.state_size(spec),
        "choice_dim": encode.choice_feature_dim(spec),
        "family_order": [
            f.value for f in decisions.active_decision_families(spec.include_setup)
        ],
        "architecture": arch.model_dump(),
        "include_setup": False,
        # num_players intentionally omitted.
    }
    assert "num_players" not in raw
    descriptor = runmeta.ModelConfig.model_validate(raw)
    assert descriptor.num_players == 2
    net = model.PolicyValueNet.from_model_config(descriptor)
    assert net.spec.num_players == 2
