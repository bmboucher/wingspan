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
    hand_embed_dim: int | None = None,
    pooled_hand_width: int | None = None,
    tray_set_embedding: bool = False,
) -> dict[str, _EmbedRule]:
    """The card-index / hand stripes of the state vector, at embedded width."""
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
        # The dedicated hand encoder consumes [multi-hot ⊕ hand summary]: the
        # hand stripe becomes the encoder's N-wide output and the 10-dim
        # hand-summary stripe folds into its input (dropped from the trunk view).
        rules["hand_multihot"] = _EmbedRule(
            new_size=hand_width,
            encoding="card-set-embedding (hand encoder)",
            value_range="learned",
            notes=(
                f"My hand -> one {hand_width}-dim set embedding from the dedicated "
                f"hand encoder over [multi-hot ({hand}) ⊕ the redirected 10-dim "
                "hand summary]. Raw encoding is the multi-hot plus the (separate) "
                "hand_summary_me stripe."
            ),
        )
        rules["hand_summary_me"] = _EmbedRule(
            new_size=0,
            encoding="folded",
            value_range="-",
            notes=(
                "Redirected into the hand encoder's input (see hand_multihot); "
                "no longer a direct trunk input."
            ),
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
    return rules


def setup_embed_rules(
    card_embed_dim: int, hand_embed_width: int
) -> dict[str, _EmbedRule]:
    """The kept-cards multi-hot and tray integer-index stripes of the setup vector,
    shown at their post-embedding widths.

    The setup readout MLP receives ``kept_cards`` as one set vector from the
    frozen hand encoder, and each tray slot as a card-table embedding plus one
    tray-set embedding.  Everything else passes through unchanged.
    """
    tray = state.TRAY_SIZE
    kept_dim = layout.HAND_MULTIHOT_DIM
    _card_set_rule = _EmbedRule(
        new_size=hand_embed_width,
        encoding="card-set-embedding (hand encoder)",
        value_range="learned",
        notes=(
            f"180-dim multi-hot -> one {hand_embed_width}-dim set embedding via the "
            "frozen copy of the main net's hand encoder."
        ),
    )
    return {
        "kept_cards": _EmbedRule(
            new_size=hand_embed_width,
            encoding="card-set-embedding (hand encoder)",
            value_range="learned",
            notes=(
                f"Kept-card multi-hot ({kept_dim} dims) -> one {hand_embed_width}-dim "
                "set embedding via the frozen copy of the main net's hand encoder."
            ),
        ),
        "tray": _EmbedRule(
            new_size=tray * card_embed_dim + hand_embed_width,
            encoding="card-embedding + card-set-embedding",
            value_range="learned",
            notes=(
                f"{tray} tray slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({tray}×{card_embed_dim}) plus one {hand_embed_width}-dim "
                "tray-set embedding from the frozen hand encoder "
                f"({tray}×{card_embed_dim} + {hand_embed_width} = "
                f"{tray * card_embed_dim + hand_embed_width} total). "
                f"Raw encoding stores {tray} integer indices."
            ),
        ),
        # Appended card-set multi-hots: each 180-dim stripe is embedded as one
        # N-dim set vector.  Rules are looked up by name so they only fire when
        # the stripe is present (embed_layout skips names not in the layout).
        "turn1_playable": _card_set_rule,
        "playable_kept_cards": _card_set_rule,
    }


def choice_embed_rules(card_embed_dim: int) -> dict[str, _EmbedRule]:
    """The board-index / bird-index / kept-set stripes of the choice vector,
    embedded. The ``kept_multihot`` rule only fires when the stripe is present
    (``include_setup`` layouts) — ``embed_layout`` looks rules up by name."""
    slots = layout.CHOICE_BOARD_IDX_SLOTS
    kept = layout.CHOICE_KEPT_MULTIHOT_DIM
    return {
        "board_idx": _EmbedRule(
            new_size=slots * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{slots} board slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({slots}x{card_embed_dim}). Raw encoding stores {slots} integer "
                "indices (bird_index + 1; 0 = empty)."
            ),
        ),
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
            new_size=card_embed_dim,
            encoding="card-embedding (becomes-playable set, summed)",
            value_range="learned",
            notes=(
                f"The set of hand birds that would become playable by accepting this "
                f"choice -> one {card_embed_dim}-dim embedding, summed over the "
                f"birds' shared card vectors. Raw encoding is a "
                f"{layout.CHOICE_BECOMES_PLAYABLE_DIM}-wide multi-hot over all core birds."
            ),
        ),
    }
