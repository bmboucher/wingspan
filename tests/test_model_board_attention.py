# pyright: reportPrivateUsage=false
# (white-box tests access _board_position_matrix / _apply_board_attention /
# _board_attention_modules / _embed_state to verify invariants directly)
"""Board self-attention layer (``use_board_attention``) unit tests.

Covers the forward pass (including fully-empty boards at game start), parameter
accounting accuracy, ShapeKey/architecture_key incompatibility with the base net,
trunk-input-width invariance, and the save/reload round-trip.
"""

from __future__ import annotations

import typing

import pydantic
import pytest

torch = pytest.importorskip("torch")

from wingspan import architecture, encode
from wingspan.model import core

# ---------------------------------------------------------------------------
# Helpers

_N_CHOICES = 4


def _arch(
    use_board_attention: bool,
    positions: bool = False,
    heads: int = 1,
    *,
    shared: bool = False,
) -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(64, 64),
        choice_layers=(64, 64),
        head_layers=(64,),
        card_embed_dim=64,
        card_encoder_layers=(64,),
        use_distinct_hand_model=True,
        hand_encoder_layers=(64,),
        use_board_attention=use_board_attention,
        board_attention_positions=positions,
        board_attention_heads=heads,
        board_attention_shared=shared,
    )


def _net(
    use_board_attention: bool,
    positions: bool = False,
    heads: int = 1,
    *,
    shared: bool = False,
) -> core.PolicyValueNet:
    return core.PolicyValueNet(
        arch=_arch(use_board_attention, positions, heads, shared=shared)
    ).eval()


def _zero_state(batch: int = 2):
    """A batch of all-zero state vectors (empty boards, game start).

    Unannotated: ``torch`` arrives via ``pytest.importorskip`` (an ``Any``)."""
    state_dim = encode.state_size()
    return torch.zeros(batch, state_dim)


def _inputs(batch: int = 2):
    """A minimal (state, choices, mask, family_idx) batch.

    Unannotated: ``torch`` arrives via ``pytest.importorskip`` (an ``Any``)."""
    state = _zero_state(batch)
    choice_dim = encode.CHOICE_FEATURE_DIM
    choices = torch.zeros(batch, _N_CHOICES, choice_dim)
    mask = torch.ones(batch, _N_CHOICES)
    family_idx = torch.zeros(batch, dtype=torch.long)
    return state, choices, mask, family_idx


# ---------------------------------------------------------------------------
# Tests


def test_forward_no_nan_empty_boards() -> None:
    """Forward pass on fully-empty boards (game start) must produce finite outputs."""
    net = _net(use_board_attention=True)
    state, choices, mask, family_idx = _inputs(batch=3)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert logits.shape == (3, _N_CHOICES)
    assert value.shape == (3,)
    assert torch.isfinite(logits).all(), "logits contain NaN/inf"
    assert torch.isfinite(value).all(), "value contains NaN/inf"


def test_forward_no_nan_partial_boards() -> None:
    """Forward with a non-trivial state (non-zero card indices) is also NaN-free."""
    net = _net(use_board_attention=True)
    state, choices, mask, family_idx = _inputs(batch=2)
    # Set slot 0 and slot 1 of own board to card indices 1 and 2.
    state[:, encode.OFF_CARD_INDEX + 0] = 1.0
    state[:, encode.OFF_CARD_INDEX + 1] = 2.0
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_base_net_has_no_attention_modules() -> None:
    """With use_board_attention=False the two attention modules must not exist."""
    net = _net(use_board_attention=False)
    assert not hasattr(net, "board_attn_me")
    assert not hasattr(net, "board_attn_opp")


def test_attention_net_has_attention_modules() -> None:
    """With use_board_attention=True both attention modules are registered."""
    net = _net(use_board_attention=True)
    assert hasattr(net, "board_attn_me")
    assert hasattr(net, "board_attn_opp")
    assert isinstance(net.board_attn_me, torch.nn.MultiheadAttention)
    assert isinstance(net.board_attn_opp, torch.nn.MultiheadAttention)


def test_param_count_matches_accounting() -> None:
    """The actual parameter count of the attention net equals count_parameters."""
    arch_attn = _arch(use_board_attention=True)
    net = core.PolicyValueNet(arch=arch_attn)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_attn,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_attn.card_embed_dim,
            use_distinct_hand_model=arch_attn.use_distinct_hand_model,
            hand_embed_dim=arch_attn.hand_embed_dim,
            tray_set_embedding=arch_attn.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM,
            arch_attn.card_embed_dim,
            pooled_hand_width=arch_attn.pooled_hand_width,
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
        slot_scalar_dim=encode.SLOT_SCALAR_DIM,
    )
    assert actual == report.total, (
        f"actual {actual} != accounting {report.total} "
        f"(board_attention block: {report.board_attention})"
    )


def test_base_param_count_unchanged() -> None:
    """With use_board_attention=False the net's parameter count equals the accounting."""
    arch_base = _arch(use_board_attention=False)
    net = core.PolicyValueNet(arch=arch_base)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_base,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_base.card_embed_dim,
            use_distinct_hand_model=arch_base.use_distinct_hand_model,
            hand_embed_dim=arch_base.hand_embed_dim,
            tray_set_embedding=arch_base.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM,
            arch_base.card_embed_dim,
            pooled_hand_width=arch_base.pooled_hand_width,
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )
    assert actual == report.total
    assert report.board_attention is None


def test_shape_key_differs() -> None:
    """shape_key must differ between attention-on and attention-off architectures."""
    assert _arch(True).shape_key != _arch(False).shape_key


def test_architecture_key_incompatible() -> None:
    """architecture_key (used by the resume gate) must be incompatible."""
    from wingspan.training import config as train_config

    cfg_base = train_config.RunConfig()
    cfg_attn = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(use_board_attention=True)
        )
    )
    assert cfg_base.architecture_key != cfg_attn.architecture_key


def test_trunk_input_width_invariant() -> None:
    """The trunk's first-Linear in_features must be identical with/without attention."""
    net_base = core.PolicyValueNet(arch=_arch(False))
    net_attn = core.PolicyValueNet(arch=_arch(True))
    # state_trunk is an nn.Sequential; first child is a Linear.
    first_base = next(net_base.state_trunk.children())
    first_attn = next(net_attn.state_trunk.children())
    assert isinstance(first_base, torch.nn.Linear)
    assert isinstance(first_attn, torch.nn.Linear)
    assert first_base.in_features == first_attn.in_features, (
        f"trunk in_features differ: base={first_base.in_features}, "
        f"attn={first_attn.in_features}"
    )


def test_save_reload_roundtrip() -> None:
    """Save and reload the attention net; flag must survive and state_dict loads."""
    net = core.PolicyValueNet(arch=_arch(True))
    state_dict = net.state_dict()

    # Simulate model_config.json round-trip via architecture Pydantic model.
    arch_json = _arch(True).model_dump()
    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.use_board_attention is True
    assert arch_loaded.shape_key == _arch(True).shape_key

    # Rebuild and load weights.
    net2 = core.PolicyValueNet(arch=arch_loaded)
    net2.load_state_dict(state_dict)
    # Forward pass should still work.
    state, choices, mask, family_idx = _inputs()
    with torch.no_grad():
        logits, value = net2.eval()(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# board_attention_positions tests


def test_forward_no_nan_empty_boards_with_positions() -> None:
    """Forward pass with board_attention_positions on, fully-empty boards."""
    net = _net(use_board_attention=True, positions=True)
    state, choices, mask, family_idx = _inputs(batch=3)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert logits.shape == (3, _N_CHOICES)
    assert value.shape == (3,)
    assert torch.isfinite(logits).all(), "logits contain NaN/inf"
    assert torch.isfinite(value).all(), "value contains NaN/inf"


def test_forward_no_nan_partial_boards_with_positions() -> None:
    """Forward with board_attention_positions on and non-zero card indices."""
    net = _net(use_board_attention=True, positions=True)
    state, choices, mask, family_idx = _inputs(batch=2)
    state[:, encode.OFF_CARD_INDEX + 0] = 1.0
    state[:, encode.OFF_CARD_INDEX + 1] = 2.0
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_trunk_input_width_grows_with_positions() -> None:
    """The trunk's first-Linear in_features grows by N_BOARD_INDEX_SLOTS *
    BOARD_POSITION_DIM (240) when board_attention_positions turns on, matching
    encode.trunk_input_dim(..., board_position_dim=...)."""
    arch_attn = _arch(True, positions=False)
    arch_pos = _arch(True, positions=True)
    net_attn = core.PolicyValueNet(arch=arch_attn)
    net_pos = core.PolicyValueNet(arch=arch_pos)
    first_attn = next(net_attn.state_trunk.children())
    first_pos = next(net_pos.state_trunk.children())
    assert isinstance(first_attn, torch.nn.Linear)
    assert isinstance(first_pos, torch.nn.Linear)

    state_dim = encode.state_size(encode.DEFAULT_SPEC)
    expected_attn = encode.trunk_input_dim(
        state_dim,
        arch_attn.card_embed_dim,
        use_distinct_hand_model=arch_attn.use_distinct_hand_model,
        hand_embed_dim=arch_attn.hand_embed_dim,
        tray_set_embedding=arch_attn.tray_set_embedding,
        n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
    )
    expected_pos = encode.trunk_input_dim(
        state_dim,
        arch_pos.card_embed_dim,
        use_distinct_hand_model=arch_pos.use_distinct_hand_model,
        hand_embed_dim=arch_pos.hand_embed_dim,
        tray_set_embedding=arch_pos.tray_set_embedding,
        n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        board_position_dim=encode.BOARD_POSITION_DIM,
    )
    assert first_attn.in_features == expected_attn
    assert first_pos.in_features == expected_pos
    assert (
        expected_pos
        == expected_attn + encode.N_BOARD_INDEX_SLOTS * encode.BOARD_POSITION_DIM
    )


def test_param_count_matches_accounting_with_positions() -> None:
    """The actual parameter count of the attention+positions net equals
    count_parameters(..., board_position_dim=BOARD_POSITION_DIM)."""
    arch_pos = _arch(use_board_attention=True, positions=True)
    net = core.PolicyValueNet(arch=arch_pos)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_pos,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_pos.card_embed_dim,
            use_distinct_hand_model=arch_pos.use_distinct_hand_model,
            hand_embed_dim=arch_pos.hand_embed_dim,
            tray_set_embedding=arch_pos.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
            board_position_dim=encode.BOARD_POSITION_DIM,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM,
            arch_pos.card_embed_dim,
            pooled_hand_width=arch_pos.pooled_hand_width,
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
        slot_scalar_dim=encode.SLOT_SCALAR_DIM,
        board_position_dim=encode.BOARD_POSITION_DIM,
    )
    assert actual == report.total, (
        f"actual {actual} != accounting {report.total} "
        f"(board_attention block: {report.board_attention})"
    )
    assert report.board_attention is not None
    # W = 64 (card_embed_dim) + 9 (slot_scalar_dim) + 8 (board_position_dim) = 81.
    expected_attn_params = 2 * (4 * 81 * 81 + 4 * 81)
    assert report.board_attention.total == expected_attn_params == 53_136


def test_shape_key_differs_with_positions() -> None:
    """shape_key must differ between positions-on and positions-off (both
    attention-on) architectures."""
    attn_only = _arch(True, positions=False)
    attn_and_positions = _arch(True, positions=True)
    assert attn_only.shape_key != attn_and_positions.shape_key


def test_architecture_key_incompatible_with_positions() -> None:
    """architecture_key (the resume gate) must be incompatible between
    positions on/off, mirroring test_architecture_key_incompatible."""
    from wingspan.training import config as train_config

    cfg_attn = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(use_board_attention=True)
        )
    )
    cfg_pos = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(
                use_board_attention=True, board_attention_positions=True
            )
        )
    )
    assert cfg_attn.architecture_key != cfg_pos.architecture_key


def test_board_position_buffer_layout() -> None:
    """board_position is [15, 8]; slot 7 -> habitat 1, column 2 (slot = hab*5 +
    col); every row is a habitat one-hot concatenated with a column one-hot."""
    net = _net(use_board_attention=True, positions=True)
    position = net.board_position
    assert position.shape == (encode.SLOTS_PER_BOARD, encode.BOARD_POSITION_DIM)

    hab_block = position[:, : encode.BOARD_POSITION_HAB_DIM]
    col_block = position[:, encode.BOARD_POSITION_HAB_DIM :]
    assert torch.all(hab_block.sum(dim=1) == 1.0)
    assert torch.all(col_block.sum(dim=1) == 1.0)

    assert position[7, 1] == 1.0
    assert position[7, encode.BOARD_POSITION_HAB_DIM + 2] == 1.0


def test_board_position_not_in_state_dict() -> None:
    """board_position is a non-persistent buffer — absent from state_dict, like
    card_summary_matrix and card_pad_mask."""
    net = _net(use_board_attention=True, positions=True)
    assert "board_position" not in net.state_dict()


def test_empty_slot_token_exactly_zero_with_positions() -> None:
    """Empty-slot tokens stay exactly zero through attention+residual even with
    the position block concatenated: card_emb and scalars are already zero for
    an empty slot, and the position block is explicitly masked to zero there —
    this exercises the real _board_position_matrix / _apply_board_attention
    that back that invariant in _embed_state_board_attention."""
    batch = 2
    card_embed_dim = 64
    token_width = card_embed_dim + encode.SLOT_SCALAR_DIM + encode.BOARD_POSITION_DIM
    attn = torch.nn.MultiheadAttention(
        embed_dim=token_width, num_heads=1, batch_first=True
    )

    position = core._board_position_matrix()
    empty = torch.ones(batch, encode.SLOTS_PER_BOARD, dtype=torch.bool)
    masked_position = (
        position.unsqueeze(0)
        .expand(batch, -1, -1)
        .masked_fill(empty.unsqueeze(-1), 0.0)
    )
    tokens = torch.cat(
        [
            torch.zeros(batch, encode.SLOTS_PER_BOARD, card_embed_dim),
            torch.zeros(batch, encode.SLOTS_PER_BOARD, encode.SLOT_SCALAR_DIM),
            masked_position,
        ],
        dim=-1,
    )
    with torch.no_grad():
        flat = core._apply_board_attention(attn, tokens, empty)
    assert torch.all(flat == 0.0)


def test_save_reload_roundtrip_with_positions() -> None:
    """Save and reload the attention+positions net; flag must survive and
    state_dict loads, mirroring test_save_reload_roundtrip."""
    net = core.PolicyValueNet(arch=_arch(True, positions=True))
    state_dict = net.state_dict()

    arch_json = _arch(True, positions=True).model_dump()
    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.board_attention_positions is True
    assert arch_loaded.shape_key == _arch(True, positions=True).shape_key

    net2 = core.PolicyValueNet(arch=arch_loaded)
    net2.load_state_dict(state_dict)
    state, choices, mask, family_idx = _inputs()
    with torch.no_grad():
        logits, value = net2.eval()(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_positions_without_attention_is_inert_not_rejected() -> None:
    """board_attention_positions=True with use_board_attention=False constructs
    without error and is simply inert (no attention modules or position buffer
    registered, forward behaves identically to plain False/False) — rejecting
    this combination at the ModelArchitecture layer would surface as a "commit
    rejected" error the moment use_board_attention is toggled off while this
    was still on, before reset_hidden_fields runs (see
    training.config.validate_launchable for the actual launch-time check)."""
    arch = architecture.ModelArchitecture(
        use_board_attention=False,
        board_attention_positions=True,
    )
    assert arch.board_attention_positions is True

    net = core.PolicyValueNet(arch=arch).eval()
    assert not hasattr(net, "board_attn_me")
    assert not hasattr(net, "board_attn_opp")
    assert not hasattr(net, "board_position")

    state, choices, mask, family_idx = _inputs(batch=2)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_inert_positions_share_shape_key_with_plain_off() -> None:
    """An inert (positions=True, attention=False) architecture must share its
    shape_key with the plain (positions=False, attention=False) architecture —
    they build byte-identical modules, so shape_key uses board_attention_positions_active,
    not the raw field, to avoid a spurious fresh-run trigger."""
    inert = architecture.ModelArchitecture(
        use_board_attention=False, board_attention_positions=True
    )
    plain = architecture.ModelArchitecture(
        use_board_attention=False, board_attention_positions=False
    )
    assert inert.board_attention_positions_active is False
    assert inert.shape_key == plain.shape_key


# ---------------------------------------------------------------------------
# board_attention_heads tests


def test_board_attention_embed_dim_helper() -> None:
    """board_attention_embed_dim pads to the next multiple of num_heads (0 pad
    when it already divides evenly)."""
    assert architecture.board_attention_embed_dim(73, 1) == 73
    assert architecture.board_attention_embed_dim(73, 2) == 74
    assert architecture.board_attention_embed_dim(73, 8) == 80
    assert architecture.board_attention_embed_dim(81, 3) == 81
    assert architecture.board_attention_embed_dim(81, 2) == 82


def test_forward_no_nan_empty_boards_with_heads() -> None:
    """Forward pass on fully-empty boards with heads=2 (pad=1) must be finite."""
    net = _net(use_board_attention=True, heads=2)
    state, choices, mask, family_idx = _inputs(batch=3)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert logits.shape == (3, _N_CHOICES)
    assert value.shape == (3,)
    assert torch.isfinite(logits).all(), "logits contain NaN/inf"
    assert torch.isfinite(value).all(), "value contains NaN/inf"


def test_forward_no_nan_partial_boards_with_heads() -> None:
    """Forward with heads=2 and non-zero card indices is also NaN-free."""
    net = _net(use_board_attention=True, heads=2)
    state, choices, mask, family_idx = _inputs(batch=2)
    state[:, encode.OFF_CARD_INDEX + 0] = 1.0
    state[:, encode.OFF_CARD_INDEX + 1] = 2.0
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_attention_modules_padded_embed_dim_and_heads() -> None:
    """board_attn_me/opp are built at the padded embed_dim with num_heads set,
    proving _build_board_attention wires both through correctly."""
    net_heads = _net(use_board_attention=True, heads=2)
    assert net_heads.board_attn_me.embed_dim == 74
    assert net_heads.board_attn_me.num_heads == 2
    assert net_heads.board_attn_opp.embed_dim == 74
    assert net_heads.board_attn_opp.num_heads == 2

    net_single = _net(use_board_attention=True, heads=1)
    assert net_single.board_attn_me.embed_dim == 73
    assert net_single.board_attn_me.num_heads == 1


def test_trunk_input_width_invariant_with_heads() -> None:
    """The trunk's first-Linear in_features must be identical across head
    counts — the padding never leaves _apply_board_attention (proves the
    slice-back), both without and with board_attention_positions."""
    net_h1 = core.PolicyValueNet(arch=_arch(True, heads=1))
    net_h2 = core.PolicyValueNet(arch=_arch(True, heads=2))
    net_h8 = core.PolicyValueNet(arch=_arch(True, heads=8))
    first_h1 = next(net_h1.state_trunk.children())
    first_h2 = next(net_h2.state_trunk.children())
    first_h8 = next(net_h8.state_trunk.children())
    assert isinstance(first_h1, torch.nn.Linear)
    assert first_h1.in_features == first_h2.in_features == first_h8.in_features

    # Same invariant with positions on (81-wide token; heads=3 -> pad 0).
    net_pos_h1 = core.PolicyValueNet(arch=_arch(True, positions=True, heads=1))
    net_pos_h2 = core.PolicyValueNet(arch=_arch(True, positions=True, heads=2))
    net_pos_h3 = core.PolicyValueNet(arch=_arch(True, positions=True, heads=3))
    first_pos_h1 = next(net_pos_h1.state_trunk.children())
    first_pos_h2 = next(net_pos_h2.state_trunk.children())
    first_pos_h3 = next(net_pos_h3.state_trunk.children())
    assert isinstance(first_pos_h1, torch.nn.Linear)
    assert (
        first_pos_h1.in_features == first_pos_h2.in_features == first_pos_h3.in_features
    )


def test_param_count_matches_accounting_with_heads() -> None:
    """The actual parameter count of a heads=2 attention net equals
    count_parameters; the BOARD ATTN block total equals 2*(4*74^2 + 4*74)
    (padded embed_dim=74, not the true 73-wide token)."""
    arch_heads = _arch(use_board_attention=True, heads=2)
    net = core.PolicyValueNet(arch=arch_heads)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_heads,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_heads.card_embed_dim,
            use_distinct_hand_model=arch_heads.use_distinct_hand_model,
            hand_embed_dim=arch_heads.hand_embed_dim,
            tray_set_embedding=arch_heads.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM,
            arch_heads.card_embed_dim,
            pooled_hand_width=arch_heads.pooled_hand_width,
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
        slot_scalar_dim=encode.SLOT_SCALAR_DIM,
    )
    assert actual == report.total, (
        f"actual {actual} != accounting {report.total} "
        f"(board_attention block: {report.board_attention})"
    )
    assert report.board_attention is not None
    expected_attn_params = 2 * (4 * 74 * 74 + 4 * 74)
    assert report.board_attention.total == expected_attn_params


def test_heads_pad_zero_shapes_match_but_shape_key_differs() -> None:
    """heads=3 at the 81-wide positions token pads to 81 (pad 0) — identical
    state_dict keys/shapes to heads=1, but shape_key must still differ:
    weights trained under one head count compute a different function under
    another even when every tensor shape coincides."""
    arch_h1 = _arch(True, positions=True, heads=1)
    arch_h3 = _arch(True, positions=True, heads=3)
    net_h1 = core.PolicyValueNet(arch=arch_h1)
    net_h3 = core.PolicyValueNet(arch=arch_h3)

    sd_h1 = net_h1.state_dict()
    sd_h3 = net_h3.state_dict()
    assert sd_h1.keys() == sd_h3.keys()
    for key in sd_h1:
        assert sd_h1[key].shape == sd_h3[key].shape, key

    assert arch_h1.shape_key != arch_h3.shape_key


def test_shape_key_differs_with_heads() -> None:
    """shape_key must differ between heads=1 and heads=2 (both attention-on,
    no positions)."""
    assert _arch(True, heads=1).shape_key != _arch(True, heads=2).shape_key


def test_architecture_key_incompatible_with_heads() -> None:
    """architecture_key (the resume gate) must be incompatible between heads=1
    and heads=2, mirroring test_architecture_key_incompatible."""
    from wingspan.training import config as train_config

    cfg_h1 = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(use_board_attention=True)
        )
    )
    cfg_h2 = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(
                use_board_attention=True, board_attention_heads=2
            )
        )
    )
    assert cfg_h1.architecture_key != cfg_h2.architecture_key


def test_empty_slot_token_exactly_zero_with_heads_padding() -> None:
    """Empty-slot tokens stay exactly zero through the pad->attend->slice path:
    an all-zero 73-wide token fed to a 74-embed_dim (heads=2) module must
    produce an exactly-zero flattened output at the true 73-wide width,
    proving the pad/slice doesn't leak noise into empty slots."""
    batch = 2
    token_width = 64 + encode.SLOT_SCALAR_DIM  # 73
    embed_dim = architecture.board_attention_embed_dim(token_width, 2)  # 74
    attn = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=2, batch_first=True
    )
    tokens = torch.zeros(batch, encode.SLOTS_PER_BOARD, token_width)
    empty = torch.ones(batch, encode.SLOTS_PER_BOARD, dtype=torch.bool)
    with torch.no_grad():
        flat = core._apply_board_attention(attn, tokens, empty)
    assert flat.shape == (batch, encode.SLOTS_PER_BOARD * token_width)
    assert torch.all(flat == 0.0)


def test_save_reload_roundtrip_with_heads() -> None:
    """Save and reload a heads=2 attention net; flag must survive and
    state_dict loads, mirroring test_save_reload_roundtrip."""
    net = core.PolicyValueNet(arch=_arch(True, heads=2))
    state_dict = net.state_dict()

    arch_json = _arch(True, heads=2).model_dump()
    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.board_attention_heads == 2
    assert arch_loaded.shape_key == _arch(True, heads=2).shape_key

    net2 = core.PolicyValueNet(arch=arch_loaded)
    net2.load_state_dict(state_dict)
    state, choices, mask, family_idx = _inputs()
    with torch.no_grad():
        logits, value = net2.eval()(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_old_config_without_heads_field_rebuilds_single_head() -> None:
    """A config dict predating board_attention_heads (the key entirely absent,
    simulating a pre-this-feature model_config.json) must rehydrate to heads=1
    and build the original single-head (73-wide) module — the REGIME/no-shim
    claim that old artifacts load unchanged."""
    arch_json = _arch(True).model_dump()
    del arch_json["board_attention_heads"]
    assert "board_attention_heads" not in arch_json

    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.board_attention_heads == 1
    assert arch_loaded.shape_key == _arch(True).shape_key

    net = core.PolicyValueNet(arch=arch_loaded)
    assert net.board_attn_me.embed_dim == 73
    assert net.board_attn_me.num_heads == 1


def test_heads_without_attention_is_inert_not_rejected() -> None:
    """board_attention_heads != 1 with use_board_attention=False constructs
    without error and is simply inert — no attention modules registered,
    forward behaves identically to plain False/1 (see
    training.config.validate_launchable for the actual launch-time check),
    mirroring test_positions_without_attention_is_inert_not_rejected."""
    arch = architecture.ModelArchitecture(
        use_board_attention=False,
        board_attention_heads=4,
    )
    assert arch.board_attention_heads == 4

    net = core.PolicyValueNet(arch=arch).eval()
    assert not hasattr(net, "board_attn_me")
    assert not hasattr(net, "board_attn_opp")

    state, choices, mask, family_idx = _inputs(batch=2)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_inert_heads_share_shape_key_with_plain_off() -> None:
    """An inert (heads=4, attention=False) architecture must share its
    shape_key with the plain (heads=1, attention=False) architecture — they
    build byte-identical modules, so shape_key uses
    board_attention_heads_active, not the raw field, to avoid a spurious
    fresh-run trigger."""
    inert = architecture.ModelArchitecture(
        use_board_attention=False, board_attention_heads=4
    )
    plain = architecture.ModelArchitecture(
        use_board_attention=False, board_attention_heads=1
    )
    assert inert.board_attention_heads_active == 1
    assert inert.shape_key == plain.shape_key


def test_heads_zero_rejected() -> None:
    """board_attention_heads=0 is rejected by the ge=1 Field constraint."""
    with pytest.raises(pydantic.ValidationError):
        architecture.ModelArchitecture(board_attention_heads=0)


# ---------------------------------------------------------------------------
# board_attention_shared tests


def test_forward_no_nan_empty_boards_shared() -> None:
    """Forward pass on fully-empty boards (game start) with a shared board-
    attention module must produce finite outputs, mirroring
    test_forward_no_nan_empty_boards."""
    net = _net(use_board_attention=True, shared=True)
    state, choices, mask, family_idx = _inputs(batch=3)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert logits.shape == (3, _N_CHOICES)
    assert value.shape == (3,)
    assert torch.isfinite(logits).all(), "logits contain NaN/inf"
    assert torch.isfinite(value).all(), "value contains NaN/inf"


def test_forward_no_nan_partial_boards_shared() -> None:
    """Forward with a non-trivial state (non-zero card indices) and a shared
    board-attention module is also NaN-free, mirroring
    test_forward_no_nan_partial_boards."""
    net = _net(use_board_attention=True, shared=True)
    state, choices, mask, family_idx = _inputs(batch=2)
    state[:, encode.OFF_CARD_INDEX + 0] = 1.0
    state[:, encode.OFF_CARD_INDEX + 1] = 2.0
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_shared_net_has_only_the_shared_module() -> None:
    """With board_attention_shared=True the net registers a single board_attn
    module and neither of the unshared board_attn_me / board_attn_opp pair."""
    net = _net(use_board_attention=True, shared=True)
    assert hasattr(net, "board_attn")
    assert isinstance(net.board_attn, torch.nn.MultiheadAttention)
    assert not hasattr(net, "board_attn_me")
    assert not hasattr(net, "board_attn_opp")


def test_shared_state_dict_key_set_is_disjoint_from_unshared() -> None:
    """The payoff of the distinct-name decision: the shared net's board_attn*
    state_dict keys and the unshared net's board_attn_me*/board_attn_opp* keys
    never overlap, so a strict-mode cross-mode load fails loudly with BOTH
    missing and unexpected keys instead of silently reusing a stale copy.

    state_dict() does not dedupe shared submodules, so aliasing
    board_attn_opp = board_attn_me under sharing would still emit both key
    sets and let a shared checkpoint strict-load into an unshared net (or vice
    versa) silently — registering a genuinely distinct name is what makes that
    cross-mode load fail loudly instead."""
    shared_net = _net(use_board_attention=True, shared=True)
    unshared_net = _net(use_board_attention=True)

    shared_keys = {
        key for key in shared_net.state_dict() if key.startswith("board_attn")
    }
    unshared_keys = {
        key for key in unshared_net.state_dict() if key.startswith("board_attn")
    }
    # Disjoint by construction: "board_attn.*" vs "board_attn_me.*"/"board_attn_opp.*".
    assert shared_keys.isdisjoint(unshared_keys)

    with pytest.raises(RuntimeError):
        shared_net.load_state_dict(unshared_net.state_dict())
    with pytest.raises(RuntimeError):
        unshared_net.load_state_dict(shared_net.state_dict())


def test_shared_param_count_matches_accounting() -> None:
    """The actual parameter count of the shared attention net equals
    count_parameters, mirroring test_param_count_matches_accounting."""
    arch_shared = _arch(use_board_attention=True, shared=True)
    net = core.PolicyValueNet(arch=arch_shared)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_shared,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_shared.card_embed_dim,
            use_distinct_hand_model=arch_shared.use_distinct_hand_model,
            hand_embed_dim=arch_shared.hand_embed_dim,
            tray_set_embedding=arch_shared.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM,
            arch_shared.card_embed_dim,
            pooled_hand_width=arch_shared.pooled_hand_width,
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
        slot_scalar_dim=encode.SLOT_SCALAR_DIM,
    )
    assert actual == report.total, (
        f"actual {actual} != accounting {report.total} "
        f"(board_attention block: {report.board_attention})"
    )


def test_shared_halves_board_attention_params() -> None:
    """Sharing halves the BOARD ATTN parameter count at the same topology: one
    module (4E²+4E) instead of the own/opp pair (2·(4E²+4E)), mirroring the
    closing assertions of test_param_count_matches_accounting_with_positions."""
    spec = encode.DEFAULT_SPEC

    def _board_attn_report(
        arch: architecture.ModelArchitecture,
    ) -> architecture.ParamReport:
        return architecture.count_parameters(
            arch,
            card_feat_in=encode.CARD_FEATURE_DIM,
            trunk_in=encode.trunk_input_dim(
                encode.state_size(spec),
                arch.card_embed_dim,
                use_distinct_hand_model=arch.use_distinct_hand_model,
                hand_embed_dim=arch.hand_embed_dim,
                tray_set_embedding=arch.tray_set_embedding,
                n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
            ),
            choice_in=encode.choice_input_dim(
                encode.CHOICE_FEATURE_DIM,
                arch.card_embed_dim,
                pooled_hand_width=arch.pooled_hand_width,
            ),
            num_families=encode.num_families(spec),
            hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
            slot_scalar_dim=encode.SLOT_SCALAR_DIM,
        )

    unshared = _board_attn_report(_arch(use_board_attention=True))
    shared = _board_attn_report(_arch(use_board_attention=True, shared=True))
    assert unshared.board_attention is not None
    assert shared.board_attention is not None
    assert unshared.board_attention.total == 2 * shared.board_attention.total
    # W = 64 (card_embed_dim) + 9 (slot_scalar_dim) = 73.
    assert shared.board_attention.total == 4 * 73 * 73 + 4 * 73 == 21_608


def test_shared_param_count_with_positions_and_heads() -> None:
    """Sharing composes with positions and heads: the three _active resolvers
    (board_attention_positions_active, board_attention_heads_active,
    board_attention_shared_active) all apply simultaneously, mirroring
    test_param_count_matches_accounting_with_positions."""
    arch = _arch(True, positions=True, heads=2, shared=True)
    net = core.PolicyValueNet(arch=arch)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            tray_set_embedding=arch.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
            board_position_dim=encode.BOARD_POSITION_DIM,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM,
            arch.card_embed_dim,
            pooled_hand_width=arch.pooled_hand_width,
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
        slot_scalar_dim=encode.SLOT_SCALAR_DIM,
        board_position_dim=encode.BOARD_POSITION_DIM,
    )
    assert actual == report.total, (
        f"actual {actual} != accounting {report.total} "
        f"(board_attention block: {report.board_attention})"
    )
    assert report.board_attention is not None
    # W = 64 (card_embed_dim) + 9 (slot_scalar_dim) + 8 (board_position_dim) = 81;
    # heads=2 pads 81 -> 82.
    assert report.board_attention.total == 4 * 82 * 82 + 4 * 82 == 27_224


def test_shared_module_embed_dim_and_heads() -> None:
    """The shared board_attn module is built at the same padded embed_dim as
    the unshared pair — 73-wide/1 head at heads=1, 74-wide/2 heads at heads=2
    — mirroring test_attention_modules_padded_embed_dim_and_heads."""
    net_single = _net(use_board_attention=True, shared=True)
    assert typing.cast(int, net_single.board_attn.embed_dim) == 73
    assert net_single.board_attn.num_heads == 1

    net_heads = _net(use_board_attention=True, heads=2, shared=True)
    assert typing.cast(int, net_heads.board_attn.embed_dim) == 74
    assert net_heads.board_attn.num_heads == 2


def test_trunk_input_width_invariant_with_shared() -> None:
    """The trunk's first-Linear in_features is identical across attention-off,
    attention-on, and attention-on+shared — sharing changes only the BOARD
    ATTN block, never the trunk-visible width, mirroring
    test_trunk_input_width_invariant."""
    net_base = core.PolicyValueNet(arch=_arch(False))
    net_attn = core.PolicyValueNet(arch=_arch(True))
    net_shared = core.PolicyValueNet(arch=_arch(True, shared=True))
    first_base = next(net_base.state_trunk.children())
    first_attn = next(net_attn.state_trunk.children())
    first_shared = next(net_shared.state_trunk.children())
    assert isinstance(first_base, torch.nn.Linear)
    assert isinstance(first_attn, torch.nn.Linear)
    assert isinstance(first_shared, torch.nn.Linear)
    assert first_base.in_features == first_attn.in_features == first_shared.in_features


def test_shape_key_differs_with_shared() -> None:
    """shape_key must differ between shared and unshared (both attention-on)."""
    assert _arch(True, shared=True).shape_key != _arch(True).shape_key


def test_architecture_key_incompatible_with_shared() -> None:
    """architecture_key (the resume gate) must be incompatible between shared
    and unshared, mirroring test_architecture_key_incompatible_with_positions."""
    from wingspan.training import config as train_config

    cfg_attn = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(use_board_attention=True)
        )
    )
    cfg_shared = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(
                use_board_attention=True, board_attention_shared=True
            )
        )
    )
    assert cfg_attn.architecture_key != cfg_shared.architecture_key


def test_save_reload_roundtrip_shared() -> None:
    """Save and reload the shared attention net; flag must survive and
    state_dict loads, mirroring test_save_reload_roundtrip."""
    net = core.PolicyValueNet(arch=_arch(True, shared=True))
    state_dict = net.state_dict()

    arch_json = _arch(True, shared=True).model_dump()
    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.board_attention_shared is True
    assert arch_loaded.shape_key == _arch(True, shared=True).shape_key

    net2 = core.PolicyValueNet(arch=arch_loaded)
    net2.load_state_dict(state_dict)
    state, choices, mask, family_idx = _inputs()
    with torch.no_grad():
        logits, value = net2.eval()(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_old_config_without_shared_field_rebuilds_the_pair() -> None:
    """A config dict predating board_attention_shared (the key entirely
    absent, simulating a pre-this-feature model_config.json) must rehydrate to
    shared=False and build the original board_attn_me/board_attn_opp pair —
    the REGIME/no-shim claim that old artifacts load unchanged, mirroring
    test_old_config_without_heads_field_rebuilds_single_head. No MODEL_VERSION
    bump or compat shim is needed for this field."""
    arch_json = _arch(True).model_dump()
    del arch_json["board_attention_shared"]
    assert "board_attention_shared" not in arch_json

    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.board_attention_shared is False

    net = core.PolicyValueNet(arch=arch_loaded)
    assert hasattr(net, "board_attn_me")
    assert hasattr(net, "board_attn_opp")
    assert not hasattr(net, "board_attn")


def test_shared_inert_without_attention() -> None:
    """board_attention_shared=True with use_board_attention=False is inert:
    board_attention_shared_active is False, the architecture shares its
    shape_key with the plain (attention-off) config, and the built net has
    none of the attention modules — mirroring
    test_inert_heads_share_shape_key_with_plain_off /
    test_positions_without_attention_is_inert_not_rejected."""
    arch = _arch(False, shared=True)
    assert arch.board_attention_shared_active is False
    assert arch.shape_key == _arch(False).shape_key

    net = core.PolicyValueNet(arch=arch).eval()
    assert not hasattr(net, "board_attn")
    assert not hasattr(net, "board_attn_me")
    assert not hasattr(net, "board_attn_opp")


# The semantic proof: a shared module scores a board the same regardless of
# seat, and that symmetry propagates end-to-end through _embed_state; the
# contrast tests show neither claim holds for the unshared own/opp pair.


def _identical_board_tokens(
    batch: int = 2, token_width: int = 64 + encode.SLOT_SCALAR_DIM
):
    """One [B, 15, W] board-token batch with every slot filled (no empties).

    Unannotated: ``torch`` arrives via ``pytest.importorskip`` (an ``Any``)."""
    torch.manual_seed(0)
    return torch.randn(batch, encode.SLOTS_PER_BOARD, token_width)


def test_shared_module_scores_identical_boards_identically() -> None:
    """Under sharing, attn_pov and attn_opp are literally the same module
    object, so the same board tokens score identically whichever "seat" they
    are passed through as — the same board scores the same no matter whose
    seat it is, which is the whole point of the flag."""
    net = _net(True, shared=True)
    attn_pov, attn_opp = net._board_attention_modules()
    assert attn_pov is attn_opp

    batch = 2
    tokens = _identical_board_tokens(batch=batch)
    empty = torch.zeros(batch, encode.SLOTS_PER_BOARD, dtype=torch.bool)
    with torch.no_grad():
        pov_out = core._apply_board_attention(attn_pov, tokens, empty)
        opp_out = core._apply_board_attention(attn_opp, tokens, empty)
    assert torch.allclose(pov_out, opp_out)


def test_separate_modules_score_identical_boards_differently() -> None:
    """The contrast case proving
    test_shared_module_scores_identical_boards_identically has teeth: with
    independent (unshared) modules, attn_pov is not attn_opp, and the SAME
    board tokens score differently depending on which module processes them."""
    net = _net(True)
    attn_pov, attn_opp = net._board_attention_modules()
    assert attn_pov is not attn_opp

    batch = 2
    tokens = _identical_board_tokens(batch=batch)
    empty = torch.zeros(batch, encode.SLOTS_PER_BOARD, dtype=torch.bool)
    with torch.no_grad():
        pov_out = core._apply_board_attention(attn_pov, tokens, empty)
        opp_out = core._apply_board_attention(attn_opp, tokens, empty)
    assert not torch.allclose(pov_out, opp_out)


def _state_with_boards(first_ids: range, second_ids: range):
    """A zero state with seat-0 and seat-1 board card indices written.

    Board card indices live in the card-index block: seat 0 occupies slots
    0..14 and seat 1 slots 15..29 (see ``model.core``). Every other field —
    including the per-slot continuous scalars — stays zero, so the two boards
    differ ONLY in which cards they hold.

    Unannotated: ``torch`` arrives via ``pytest.importorskip`` (an ``Any``)."""
    state = _zero_state(1)
    for i, card_id in enumerate(first_ids):
        state[:, encode.OFF_CARD_INDEX + i] = float(card_id)
    for i, card_id in enumerate(second_ids):
        state[:, encode.OFF_CARD_INDEX + encode.SLOTS_PER_BOARD + i] = float(card_id)
    return state


def test_shared_embed_state_is_symmetric_under_board_swap() -> None:
    """End-to-end proof: with a shared board_attn module, swapping which seat
    holds which board content permutes but does not change the embedded
    state's values.

    Board content reaches _embed_state's output ONLY through the two
    flattened attention blocks (own_flat then opp_flats) — the board
    continuous stripes and the board card-index block are both excised from
    the continuous prefix. With one shared module scoring every board
    identically, "score this board" is the same function of board content
    whichever seat calls it, so swapping the two boards' seats merely
    exchanges two adjacent equal-width blocks in the flattened output — a
    permutation of bit-identical values, not a change in the values
    themselves. This also demonstrates the encoder is seat-agnostic: the same
    board content in seat 0 and seat 1 yields the same tokens."""
    net = _net(True, shared=True)
    board_a = range(1, 16)
    board_b = range(16, 31)

    with torch.no_grad():
        emb = net._embed_state(_state_with_boards(board_a, board_b), net.card_table())
        emb_swapped = net._embed_state(
            _state_with_boards(board_b, board_a), net.card_table()
        )
    assert torch.allclose(
        emb.flatten().sort().values, emb_swapped.flatten().sort().values
    )


def test_separate_modules_embed_state_is_not_symmetric_under_board_swap() -> None:
    """Contrast case for test_shared_embed_state_is_symmetric_under_board_swap:
    with independent board_attn_me / board_attn_opp modules, own_flat(board)
    and opp_flat(board) are different functions of the same board content, so
    swapping which seat holds which board is not a mere permutation of the
    embedded state's values."""
    net = _net(True)
    board_a = range(1, 16)
    board_b = range(16, 31)

    with torch.no_grad():
        emb = net._embed_state(_state_with_boards(board_a, board_b), net.card_table())
        emb_swapped = net._embed_state(
            _state_with_boards(board_b, board_a), net.card_table()
        )
    assert not torch.allclose(
        emb.flatten().sort().values, emb_swapped.flatten().sort().values
    )
