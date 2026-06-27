"""Post-embedding rewrite rules for card-index / identity stripes.

``_embed_layout`` rewrites a raw :class:`~descriptors.VectorLayout` into the
network's post-embedding input view; ``state_embed_rules``,
``choice_embed_rules``, and ``setup_embed_rules`` supply the per-run expansion
dicts for the state, choice, and setup layouts respectively.
"""

from __future__ import annotations

import pydantic

from wingspan import state
from wingspan.encode import layout
from wingspan.encode.stripes import descriptors


class _EmbedRule(pydantic.BaseModel):
    """How a raw card-index / identity stripe is shown at its post-embedding width."""

    new_size: int
    encoding: str
    value_range: str
    notes: str


def embed_layout(
    raw: descriptors.VectorLayout,
    rules: dict[str, _EmbedRule],
    expected_total: int,
) -> descriptors.VectorLayout:
    """Rewrite a raw vector layout into the network's post-embedding input view.

    Every card-index / identity stripe named in ``rules`` is replaced by its
    embedded-width stripe and all offsets are recomputed cumulatively (sizes change,
    so downstream offsets shift). A rule with ``new_size == 0`` *removes* its
    stripe — the raw dims were folded into another block (the hand summary
    redirected into the hand encoder). The result's total must equal
    ``expected_total`` — the trunk / choice-encoder first-``Linear`` input width.
    """
    stripes: list[descriptors.StripeDescriptor] = []
    off = 0
    for stripe in raw.stripes:
        rule = rules.get(stripe.name)
        if rule is None:
            stripes.append(stripe.model_copy(update={"offset": off}))
            off += stripe.size
            continue
        if rule.new_size == 0:
            continue
        stripes.append(
            stripe.model_copy(
                update={
                    "offset": off,
                    "size": rule.new_size,
                    "encoding": rule.encoding,
                    "value_range": rule.value_range,
                    "notes": rule.notes,
                }
            )
        )
        off += rule.new_size
    assert off == expected_total, (
        f"embedded stripe offsets sum to {off} but expected {expected_total} — "
        "stripes.py expansion is out of sync with layout.trunk/choice_input_dim"
    )
    return descriptors.VectorLayout(total_size=expected_total, stripes=tuple(stripes))


def state_embed_rules(
    card_embed_dim: int,
    *,
    use_distinct_hand_model: bool = False,
    use_board_attention: bool = False,
    hand_embed_dim: int | None = None,
    pooled_hand_width: int | None = None,
    tray_set_embedding: bool = False,
    n_playable_multihots: int = 0,
) -> dict[str, _EmbedRule]:
    """The card-index / hand stripes of the state vector, at embedded width.

    When ``use_board_attention`` is ``True`` the raw per-slot board and board-index
    stripes are folded into attention-output blocks:

    * ``board_me`` / ``board_opp`` each expand to ``BOARD_SLOTS × (card_embed_dim +
      SLOT_SCALAR_DIM)`` — one concat-of-card-embed-and-scalars vector per slot.
    * ``card_idx_board`` is removed (``new_size=0``): the per-slot card lookup
      is already included in the attention-output blocks above.

    The total is unchanged, so the ``embed_layout`` consistency check still holds.

    ``n_playable_multihots`` is the count of extra playability multi-hot stripes
    that follow ``hand_multihot`` in the v0.6+ state vector.  Each is embedded
    through the same shared card embedder at the same output width as the hand
    multi-hot (``hand_width``).  Pass ``N_HAND_PLAYABLE_MULTIHOTS`` for live-era
    artifacts; 0 for pre-0.6 compat layouts that lack these stripes.
    """
    n_board = layout.N_BOARD_INDEX_SLOTS
    tray = state.TRAY_SIZE
    hand = layout.HAND_MULTIHOT_DIM
    if use_distinct_hand_model:
        hand_width = hand_embed_dim if hand_embed_dim is not None else card_embed_dim
    else:
        hand_width = (
            pooled_hand_width if pooled_hand_width is not None else card_embed_dim
        )
    rules = {
        "card_idx_board": _EmbedRule(
            new_size=n_board * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{n_board} board slots (15 me + 15 opp) -> one {card_embed_dim}-dim "
                f"shared card embedding each ({n_board}x{card_embed_dim}). Raw encoding "
                "stores 30 integer indices (bird_index + 1; 0 = empty)."
            ),
        ),
        "card_idx_tray": _EmbedRule(
            new_size=tray * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{tray} tray slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({tray}x{card_embed_dim}). Raw encoding stores {tray} indices."
            ),
        ),
        "hand_multihot": _EmbedRule(
            new_size=hand_width,
            encoding="card-embedding (pooled)",
            value_range="learned",
            notes=(
                f"My hand -> one {hand_width}-dim embedding, pooled over the "
                f"held cards' shared card vectors. Raw encoding is a {hand}-wide "
                "multi-hot over all core birds."
            ),
        ),
    }
    if use_distinct_hand_model:
        # The dedicated hand encoder consumes [multi-hot ⊕ derived hand summary]:
        # the hand stripe becomes the encoder's N-wide output. In v0.9+ the
        # hand-summary stripe is absent from the live state (derived in-model);
        # in pre-0.9 frozen vectors it was physically present and redirected into
        # the encoder (handled in _embed_state via the hand_summary_end offset).
        rules["hand_multihot"] = _EmbedRule(
            new_size=hand_width,
            encoding="card-set-embedding (hand encoder)",
            value_range="learned",
            notes=(
                f"My hand -> one {hand_width}-dim set embedding from the dedicated "
                f"hand encoder over [multi-hot ({hand}) ⊕ 10-dim hand summary "
                "derived in-model from the multi-hot via set_summary_from_multihot]. "
                "Raw encoding is the multi-hot only (hand summary removed in v0.9)."
            ),
        )
    if use_board_attention:
        slots_per_seat = layout.N_BOARD_INDEX_SLOTS // 2
        slot_scalar_dim = layout.SLOT_SCALAR_DIM  # 9 scalars per slot
        attn_width = slots_per_seat * (card_embed_dim + slot_scalar_dim)
        rules["board_me"] = _EmbedRule(
            new_size=attn_width,
            encoding="board-attention output",
            value_range="learned",
            notes=(
                f"{slots_per_seat} board slots → one ({card_embed_dim}+{slot_scalar_dim})-dim "
                "concat (card embedding + per-slot scalars) each, shaped for the "
                "board-attention transformer."
            ),
        )
        rules["board_opp"] = _EmbedRule(
            new_size=attn_width,
            encoding="board-attention output",
            value_range="learned",
            notes=(
                f"{slots_per_seat} board slots → one ({card_embed_dim}+{slot_scalar_dim})-dim "
                "concat (card embedding + per-slot scalars) each, shaped for the "
                "board-attention transformer."
            ),
        )
        # card_idx_board is folded into the attention blocks above.
        rules["card_idx_board"] = _EmbedRule(
            new_size=0,
            encoding="folded",
            value_range="-",
            notes="Folded into board_me / board_opp attention-output blocks.",
        )
    if tray_set_embedding:
        rules["card_idx_tray"] = _EmbedRule(
            new_size=tray * card_embed_dim + hand_width,
            encoding="card-embedding + card-set-embedding",
            value_range="learned",
            notes=(
                f"{tray} tray slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({tray}x{card_embed_dim}) plus one {hand_width}-dim tray-*set* "
                "embedding from the hand encoder (multi-hot + summary derived "
                f"in-model from the index columns). Raw encoding stores {tray} "
                "indices."
            ),
        )
    if n_playable_multihots >= 1:
        rules["hand_playable_me"] = _EmbedRule(
            new_size=hand_width,
            encoding="card-embedding (playable set, pooled)",
            value_range="learned",
            notes=(
                f"Hand birds currently playable (food+habitat+eggs) -> one "
                f"{hand_width}-dim embedding, pooled over the birds' shared card "
                f"vectors. Raw encoding is a {hand}-wide multi-hot over all core birds."
            ),
        )
    if n_playable_multihots >= 2:
        rules["hand_playable_eggs_me"] = _EmbedRule(
            new_size=hand_width,
            encoding="card-embedding (egg-blocked set, pooled)",
            value_range="learned",
            notes=(
                f"Hand birds where food is affordable and a habitat slot is open, "
                f"but the egg cost is not yet met -> one {hand_width}-dim embedding, "
                f"pooled over the birds' shared card vectors. Raw encoding is a "
                f"{hand}-wide multi-hot over all core birds."
            ),
        )
    return rules


def setup_embed_rules(
    card_embed_dim: int,
    set_width: int,
    *,
    use_distinct: bool = False,
) -> dict[str, _EmbedRule]:
    """The card-set multi-hot and tray integer-index stripes of the setup vector,
    shown at their post-embedding widths.

    ``set_width`` is the width of one embedded card-set: ``pooled_hand_width``
    when ``use_distinct=False`` (the default pooling path), or
    ``hand_embed_width`` when ``use_distinct=True`` (dedicated hand encoder).
    The tray rule covers per-slot card-table rows only — no tray-set embedding.
    Everything else passes through unchanged.
    """
    tray = state.TRAY_SIZE
    kept_dim = layout.HAND_MULTIHOT_DIM
    if use_distinct:
        set_encoding = "card-set-embedding (hand encoder)"
        set_notes_suffix = "via the frozen copy of the main net's hand encoder."
    else:
        set_encoding = "card-embedding (pooled)"
        set_notes_suffix = (
            "pooled over the shared card table's bird rows "
            "(same pooling mode as the main net's hand stripe)."
        )
    _card_set_rule = _EmbedRule(
        new_size=set_width,
        encoding=set_encoding,
        value_range="learned",
        notes=f"180-dim multi-hot -> one {set_width}-dim set embedding {set_notes_suffix}",
    )
    return {
        "kept_cards": _EmbedRule(
            new_size=set_width,
            encoding=set_encoding,
            value_range="learned",
            notes=(
                f"Kept-card multi-hot ({kept_dim} dims) -> one {set_width}-dim "
                f"set embedding {set_notes_suffix}"
            ),
        ),
        "tray": _EmbedRule(
            new_size=tray * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{tray} tray slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({tray}×{card_embed_dim} = {tray * card_embed_dim} total). "
                f"Raw encoding stores {tray} integer indices."
            ),
        ),
        # Appended card-set multi-hots: each 180-dim stripe is embedded as one
        # set vector.  Rules are looked up by name so they only fire when the
        # stripe is present (embed_layout skips names not in the layout).
        "turn1_playable": _card_set_rule,
        "playable_kept_cards": _card_set_rule,
    }


def choice_embed_rules(
    card_embed_dim: int, pooled_hand_width: int
) -> dict[str, _EmbedRule]:
    """The bird-index / kept-set stripes of the choice vector, embedded.
    The ``board_hab``/``board_col`` one-hots and ``board_target`` scalars pass
    through unchanged (no rule). The ``kept_multihot`` rule only fires when the
    stripe is present (``include_setup`` layouts) — ``embed_layout`` looks rules
    up by name.

    ``pooled_hand_width`` is the width produced by ``pool_card_set`` under the
    run's ``hand_pooling`` mode (equal to ``card_embed_dim`` for MEAN/SUM;
    ``2*card_embed_dim+1`` for CONCAT_MAX_SUM).  Both ``becomes_playable`` and
    ``becomes_unplayable`` are card sets embedded through the same pooling, so
    they each expand to ``pooled_hand_width``."""
    kept = layout.CHOICE_KEPT_MULTIHOT_DIM
    return {
        "bird_id": _EmbedRule(
            new_size=card_embed_dim,
            encoding="card-embedding (candidate)",
            value_range="learned",
            notes=(
                f"Candidate bird -> one {card_embed_dim}-dim shared card embedding, "
                "zeroed when no bird. Raw encoding is a single integer index column "
                "(bird_index + 1; 0 = no bird)."
            ),
        ),
        "kept_multihot": _EmbedRule(
            new_size=card_embed_dim,
            encoding="card-embedding (kept set, summed)",
            value_range="learned",
            notes=(
                f"A setup pick's kept set -> one {card_embed_dim}-dim embedding, "
                f"summed over the kept cards' shared card vectors. Raw encoding is "
                f"a {kept}-wide multi-hot over all core birds."
            ),
        ),
        "becomes_playable": _EmbedRule(
            new_size=pooled_hand_width,
            encoding="card-embedding (becomes-playable set, pooled)",
            value_range="learned",
            notes=(
                f"The set of hand birds that would become playable by accepting this "
                f"choice -> one {pooled_hand_width}-dim embedding, pooled over the "
                f"birds' shared card vectors (same pooling mode as the hand stripe). "
                f"Raw encoding is a "
                f"{layout.CHOICE_BECOMES_PLAYABLE_DIM}-wide multi-hot over all core birds."
            ),
        ),
        "becomes_unplayable": _EmbedRule(
            new_size=pooled_hand_width,
            encoding="card-embedding (becomes-unplayable set, pooled)",
            value_range="learned",
            notes=(
                f"The set of currently-playable hand birds that would become "
                f"unplayable by accepting this choice -> one {pooled_hand_width}-dim "
                f"embedding, pooled over the birds' shared card vectors (same pooling "
                f"mode as the hand stripe). Raw encoding is a "
                f"{layout.CHOICE_BECOMES_UNPLAYABLE_DIM}-wide multi-hot over all "
                f"core birds."
            ),
        ),
    }
