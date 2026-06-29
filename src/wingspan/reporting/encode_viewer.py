# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — same intra-package
# coupling convention as card_feature.py and state_encode.py)
"""Extract non-zero stripe summaries from raw encoder vectors for HTML display.

Given a flat state vector or per-choice vector, walks the corresponding
:class:`~wingspan.encode.stripes.VectorLayout`, finds stripes whose slice is
non-zero (absolute value above a small threshold), and converts each active
entry into a compact :class:`~wingspan.reporting.gamelog_models.EncodedStripe`
record carrying only the non-zero sub-fields.  The result is embedded in the
HTML game log to power the encoding-viewer modal.

Stripes whose ``encoding == "complex"`` (learned card embeddings) are skipped
because individual dimensions have no human-readable meaning.  Normalized scalar
fields are shown as integers (the divisor is parsed from the ``notes`` field);
index and multi-hot fields are decoded to human-readable names.  Only
:func:`extract_state_stripes` and :func:`extract_choice_stripes` are public;
everything else is implementation detail.
"""

from __future__ import annotations

import re

import numpy as np

from wingspan import cards, decisions, setup_model
from wingspan.encode import layout, state_encode, stripes
from wingspan.gamelog import models as gamelog_models

# Values whose absolute magnitude is below this threshold are treated as zero.
_ZERO_THRESHOLD = 1e-6

# Extracts the divisor from notes like "Normalized ÷ 6." → 6.0.
_NORM_PATTERN = re.compile(r"÷\s*(\d+(?:\.\d+)?)")

# Stripe names that carry bird integer indices (value = bird_index + 1, 0 = empty slot).
_BIRD_INDEX_STRIPES: frozenset[str] = frozenset(
    {"card_idx_board", "card_idx_tray", "bird_id", "board_idx", "tray"}
)

# Stripe names that carry 180-dim bird multi-hot vectors.
_BIRD_MULTIHOT_STRIPES: frozenset[str] = frozenset(
    {
        "hand_multihot",
        "hand_playable_me",
        "hand_playable_eggs_me",
        "becomes_playable",
        "becomes_unplayable",
        "kept_multihot",
        "kept_cards",
        "turn1_playable",
        "playable_kept_cards",
    }
)

# Stripe names that carry 26-dim bonus-card multi-hot vectors.
_BONUS_MULTIHOT_STRIPES: frozenset[str] = frozenset(
    {"bonus_progress_held", "bonus_cards"}
)

# Stripe names that carry one-hot bonus-card indices.
_BONUS_ONEHOT_STRIPES: frozenset[str] = frozenset({"bonus_id", "kept_bonus"})

# Stripe names that carry one-hot decision-type indices.
_DECISION_TYPE_STRIPES: frozenset[str] = frozenset({"decision_type"})

# Setup stripe names shared across all candidates (deal context, not candidate-specific).
_SETUP_CONTEXT_STRIPES: frozenset[str] = frozenset(
    {"tray", "birdfeeder", "round_goals", "bonus_cards"}
)

# Bird-attribute sub-field names that are boolean (scalar 0 or 1, no divisor).
_BIRD_ATTR_BOOL_FIELDS: frozenset[str] = frozenset(
    {"flocking", "predator", "plays_another_bird", "caches_food", "or_cost"}
)


def extract_state_stripes(
    vector: list[float],
    include_setup: bool,
) -> list[gamelog_models.EncodedStripe]:
    """Return non-zero stripe summaries for a flat state feature vector.

    Uses :func:`~wingspan.encode.stripes.raw_state_stripe_layout` to get the
    raw (pre-embedding) layout whose offsets match the ``encode_state`` output
    directly, then delegates to :func:`_extract_nonzero_stripes`."""
    spec = layout.EncodingSpec(include_setup=include_setup)
    vector_layout = stripes.raw_state_stripe_layout(spec)
    return _extract_nonzero_stripes(
        np.array(vector, dtype=np.float32), vector_layout, include_setup
    )


def extract_choice_stripes(
    choice_vec: list[float],
    include_setup: bool,
) -> list[gamelog_models.EncodedStripe]:
    """Return non-zero stripe summaries for one row of the choice feature matrix.

    Uses :func:`~wingspan.encode.stripes.raw_choice_stripe_layout` to get the
    raw stripe layout, then delegates to :func:`_extract_nonzero_stripes`."""
    spec = layout.EncodingSpec(include_setup=include_setup)
    vector_layout = stripes.raw_choice_stripe_layout(spec)
    return _extract_nonzero_stripes(
        np.array(choice_vec, dtype=np.float32), vector_layout, include_setup
    )


def extract_setup_context_stripes(
    vector: list[float],
    encoding: setup_model.SetupEncoding,
) -> list[gamelog_models.EncodedStripe]:
    """Return non-zero stripe summaries for the shared deal-context portion of a
    setup candidate vector.

    The deal context (tray, birdfeeder, round goals, and bonus-cards when
    split_bonus is active) is identical across all 504 candidates; this function
    decodes it from the first candidate's vector for the 'Game State' panel.
    Stripes not in :data:`_SETUP_CONTEXT_STRIPES` are excluded."""
    vector_layout = setup_model.setup_stripe_layout(encoding)
    all_stripes = _extract_nonzero_stripes(
        np.array(vector, dtype=np.float32), vector_layout, include_setup=True
    )
    return [s for s in all_stripes if s.name in _SETUP_CONTEXT_STRIPES]


def extract_setup_candidate_stripes(
    vector: list[float],
    encoding: setup_model.SetupEncoding,
) -> list[gamelog_models.EncodedStripe]:
    """Return non-zero stripe summaries for the per-candidate portion of a setup
    candidate vector.

    Includes kept cards, foods, bonus, and the pricing blocks (kept_bonus_value,
    goal_affinity) — everything that varies across candidates. Stripes in
    :data:`_SETUP_CONTEXT_STRIPES` are excluded."""
    vector_layout = setup_model.setup_stripe_layout(encoding)
    all_stripes = _extract_nonzero_stripes(
        np.array(vector, dtype=np.float32), vector_layout, include_setup=True
    )
    return [s for s in all_stripes if s.name not in _SETUP_CONTEXT_STRIPES]


def extract_card_attr_stripes(bird: cards.Bird) -> list[gamelog_models.EncodedStripe]:
    """Return the non-zero bird-attribute sub-fields for ``bird``'s feature vector.

    Reads the bird's row from :func:`~wingspan.encode.state_encode.card_feature_matrix`,
    walks the ``bird_attrs`` sub-fields from ``card_feature_stripe_layout``, and
    decodes each non-zero sub-field to a human-readable ``decoded_label``.  The
    trailing ``bird_identity`` one-hot (stripe 1) is always set and carries no
    information worth showing in the viewer, so it is intentionally excluded."""
    feat_matrix = state_encode.card_feature_matrix()
    bird_row = feat_matrix[cards.bird_index(bird) + 1]

    # bird_attrs is stripe 0; bird_identity (stripe 1) is intentionally skipped.
    attrs_stripe = stripes.card_feature_stripe_layout().stripes[0]
    bird_attrs_slice = bird_row[
        attrs_stripe.offset : attrs_stripe.offset + attrs_stripe.size
    ]

    sub_fields = [
        encoded_sub
        for sub_field in (attrs_stripe.sub_fields or ())
        if (encoded_sub := _build_bird_attr_sub_field(sub_field, bird_attrs_slice))
        is not None
    ]
    if not sub_fields:
        return []
    return [
        gamelog_models.EncodedStripe(
            name=attrs_stripe.name,
            description=attrs_stripe.description,
            sub_fields=sub_fields,
        )
    ]


###### PRIVATE #######


#### Decode lookup helpers ####


def _bird_names() -> tuple[str, ...]:
    """Stable tuple of bird names indexed by bird_index (0-based)."""
    return tuple(bird.name for bird in cards.birds_ordered())


def _bonus_names() -> tuple[str, ...]:
    """Stable tuple of bonus card names indexed by bonus_index (0-based)."""
    return tuple(bonus.name for bonus in cards.bonus_cards_ordered())


def _denorm_scale(notes: str | None) -> float | None:
    """Return the divisor from a 'Normalized ÷ N' notes string, or None."""
    if not notes:
        return None
    match = _NORM_PATTERN.search(notes)
    return float(match.group(1)) if match else None


#### Stripe extraction ####


def _extract_nonzero_stripes(
    vector: np.ndarray,
    vector_layout: stripes.VectorLayout,
    include_setup: bool,
) -> list[gamelog_models.EncodedStripe]:
    """Walk a VectorLayout and return an EncodedStripe for each non-zero stripe.

    Skips stripes with ``encoding == "complex"`` (learned embeddings) and any
    stripe whose entire slice is within ``_ZERO_THRESHOLD`` of zero."""
    result: list[gamelog_models.EncodedStripe] = []
    for stripe_desc in vector_layout.stripes:
        if stripe_desc.encoding == "complex":
            continue
        stripe_slice = vector[
            stripe_desc.offset : stripe_desc.offset + stripe_desc.size
        ]
        if np.all(np.abs(stripe_slice) < _ZERO_THRESHOLD):
            continue
        encoded = _build_encoded_stripe(stripe_desc, stripe_slice, include_setup)
        if encoded.sub_fields:
            result.append(encoded)
    return result


def _build_encoded_stripe(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
    include_setup: bool,
) -> gamelog_models.EncodedStripe:
    """Build an EncodedStripe from one non-zero stripe slice.

    When the stripe has named sub-fields, delegates each element to
    :func:`_build_sub_field`.  When there are no sub-fields, delegates the whole
    stripe to :func:`_build_whole_stripe_sub_fields`."""
    if stripe_desc.sub_fields:
        sub_fields = [
            encoded_sub
            for sub_field in stripe_desc.sub_fields
            if (
                encoded_sub := _build_sub_field(
                    sub_field, stripe_slice, stripe_desc.encoding
                )
            )
            is not None
        ]
    else:
        sub_fields = _build_whole_stripe_sub_fields(
            stripe_desc, stripe_slice, include_setup
        )

    return gamelog_models.EncodedStripe(
        name=stripe_desc.name,
        description=stripe_desc.description,
        sub_fields=sub_fields,
    )


def _build_sub_field(
    sub_field: stripes.SubFieldDescriptor,
    stripe_slice: np.ndarray,
    stripe_encoding: str,
) -> gamelog_models.EncodedSubField | None:
    """Build one EncodedSubField, or None if this sub-field is zero.

    Denormalizes scalar values when notes contain a '÷ N' divisor.  Uses the
    sub-field's own ``encoding`` (falling back to the stripe encoding) to choose
    between one-hot, scalar, and multi-element representations."""
    sub_slice = stripe_slice[
        sub_field.relative_offset : sub_field.relative_offset + sub_field.size
    ]
    if np.all(np.abs(sub_slice) < _ZERO_THRESHOLD):
        return None

    effective_encoding = sub_field.encoding or stripe_encoding
    scale = _denorm_scale(sub_field.notes)

    # One-hot block: report the argmax position.
    if effective_encoding == "one-hot":
        return gamelog_models.EncodedSubField(
            name=sub_field.name,
            description=sub_field.description,
            encoding=effective_encoding,
            value_range=sub_field.value_range,
            notes=sub_field.notes,
            active_index=int(np.argmax(sub_slice)),
        )

    # Single scalar: denormalize to integer when a divisor is present.
    if sub_field.size == 1:
        raw_val = float(sub_slice[0])
        return gamelog_models.EncodedSubField(
            name=sub_field.name,
            description=sub_field.description,
            encoding=effective_encoding,
            value_range=sub_field.value_range,
            notes=sub_field.notes,
            raw_value=raw_val,
            decoded_label=str(round(raw_val * scale)) if scale is not None else None,
        )

    # Multi-element block: report non-zero positions, denormalized if possible.
    active_positions = np.where(np.abs(sub_slice) >= _ZERO_THRESHOLD)[0]
    raw_values = [float(sub_slice[pos]) for pos in active_positions]
    decoded_label: str | None = None
    if scale is not None:
        decoded_label = ", ".join(str(round(v * scale)) for v in raw_values)
    return gamelog_models.EncodedSubField(
        name=sub_field.name,
        description=sub_field.description,
        encoding=effective_encoding,
        value_range=sub_field.value_range,
        notes=sub_field.notes,
        raw_values=raw_values,
        decoded_label=decoded_label,
    )


def _build_whole_stripe_sub_fields(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
    include_setup: bool,
) -> list[gamelog_models.EncodedSubField]:
    """Represent a no-sub-field stripe as one or more EncodedSubField entries.

    Applies semantic decode for index, multi-hot, and one-hot stripes, and
    denormalizes scalar stripes when a '÷ N' divisor is present in the notes."""
    name = stripe_desc.name

    # Bird integer-index slots (one row per occupied position).
    if name in _BIRD_INDEX_STRIPES:
        return _decode_bird_index_stripe(stripe_desc, stripe_slice)

    # Bird multi-hot (collapse all active bits to one row).
    if name in _BIRD_MULTIHOT_STRIPES:
        return _decode_bird_multihot_stripe(stripe_desc, stripe_slice)

    # Bonus multi-hot (collapse to one row).
    if name in _BONUS_MULTIHOT_STRIPES:
        return _decode_bonus_multihot_stripe(stripe_desc, stripe_slice)

    # Bonus one-hot.
    if name in _BONUS_ONEHOT_STRIPES:
        active_idx = int(np.argmax(stripe_slice))
        bonus_name_tuple = _bonus_names()
        bonus_name = (
            bonus_name_tuple[active_idx]
            if active_idx < len(bonus_name_tuple)
            else str(active_idx)
        )
        return [
            gamelog_models.EncodedSubField(
                name=stripe_desc.name,
                description=stripe_desc.description,
                encoding=stripe_desc.encoding,
                value_range=stripe_desc.value_range,
                notes=stripe_desc.notes,
                active_index=active_idx,
                decoded_label=f"{active_idx} ({bonus_name})",
            )
        ]

    # Decision-type one-hot.
    if name in _DECISION_TYPE_STRIPES:
        active_idx = int(np.argmax(stripe_slice))
        active_classes = decisions.active_decision_classes(include_setup)
        class_name = (
            active_classes[active_idx].__name__
            if active_idx < len(active_classes)
            else str(active_idx)
        )
        return [
            gamelog_models.EncodedSubField(
                name=stripe_desc.name,
                description=stripe_desc.description,
                encoding=stripe_desc.encoding,
                value_range=stripe_desc.value_range,
                notes=stripe_desc.notes,
                active_index=active_idx,
                decoded_label=f"{active_idx} ({class_name})",
            )
        ]

    # Generic one-hot.
    if stripe_desc.encoding == "one-hot":
        return [
            gamelog_models.EncodedSubField(
                name=stripe_desc.name,
                description=stripe_desc.description,
                encoding=stripe_desc.encoding,
                value_range=stripe_desc.value_range,
                notes=stripe_desc.notes,
                active_index=int(np.argmax(stripe_slice)),
            )
        ]

    # Single scalar with optional denormalization.
    if stripe_desc.size == 1:
        raw_val = float(stripe_slice[0])
        scale = _denorm_scale(stripe_desc.notes)
        return [
            gamelog_models.EncodedSubField(
                name=stripe_desc.name,
                description=stripe_desc.description,
                encoding=stripe_desc.encoding,
                value_range=stripe_desc.value_range,
                notes=stripe_desc.notes,
                raw_value=raw_val,
                decoded_label=(
                    str(round(raw_val * scale)) if scale is not None else None
                ),
            )
        ]

    # Multi-element stripe: one row per non-zero index with optional denorm.
    active_indices = np.where(np.abs(stripe_slice) >= _ZERO_THRESHOLD)[0]
    scale = _denorm_scale(stripe_desc.notes)
    return [
        gamelog_models.EncodedSubField(
            name=f"{stripe_desc.name}[{idx}]",
            description=stripe_desc.description,
            encoding=stripe_desc.encoding,
            value_range=stripe_desc.value_range,
            notes=stripe_desc.notes,
            raw_value=float(stripe_slice[idx]),
            decoded_label=(
                str(round(float(stripe_slice[idx]) * scale))
                if scale is not None
                else None
            ),
        )
        for idx in active_indices
    ]


#### Stripe decode helpers ####


def _decode_bird_index_stripe(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
) -> list[gamelog_models.EncodedSubField]:
    """One sub-field per occupied slot in a bird integer-index stripe.

    Index 0 means empty (skipped).  Index N means bird_index N-1 (the encoder
    stores bird_index + 1 so that 0 is the unambiguous empty sentinel)."""
    bird_name_tuple = _bird_names()
    sub_fields: list[gamelog_models.EncodedSubField] = []
    for position, raw_val in enumerate(stripe_slice):
        int_idx = round(float(raw_val))
        if int_idx == 0:
            continue
        bird_name = (
            bird_name_tuple[int_idx - 1]
            if 0 < int_idx <= len(bird_name_tuple)
            else f"bird_{int_idx}"
        )
        sub_fields.append(
            gamelog_models.EncodedSubField(
                name=f"{stripe_desc.name}[{position}]",
                description=stripe_desc.description,
                encoding=stripe_desc.encoding,
                value_range=stripe_desc.value_range,
                notes=stripe_desc.notes,
                raw_value=float(raw_val),
                decoded_label=f"{int_idx} ({bird_name})",
            )
        )
    return sub_fields


def _decode_bird_multihot_stripe(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
) -> list[gamelog_models.EncodedSubField]:
    """Collapse an active bird multi-hot to one row listing all active birds."""
    bird_name_tuple = _bird_names()
    active = np.where(np.abs(stripe_slice) >= _ZERO_THRESHOLD)[0]
    if len(active) == 0:
        return []
    parts = [
        f"{idx} ({bird_name_tuple[int(idx)] if int(idx) < len(bird_name_tuple) else f'bird_{idx}'})"
        for idx in active
    ]
    return [
        gamelog_models.EncodedSubField(
            name=stripe_desc.name,
            description=stripe_desc.description,
            encoding=stripe_desc.encoding,
            value_range=stripe_desc.value_range,
            notes=stripe_desc.notes,
            decoded_label=", ".join(parts),
        )
    ]


def _decode_bonus_multihot_stripe(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
) -> list[gamelog_models.EncodedSubField]:
    """Collapse an active bonus multi-hot to one row listing all held bonuses."""
    bonus_name_tuple = _bonus_names()
    active = np.where(np.abs(stripe_slice) >= _ZERO_THRESHOLD)[0]
    if len(active) == 0:
        return []
    parts = [
        f"{idx} ({bonus_name_tuple[int(idx)] if int(idx) < len(bonus_name_tuple) else f'bonus_{idx}'})"
        for idx in active
    ]
    return [
        gamelog_models.EncodedSubField(
            name=stripe_desc.name,
            description=stripe_desc.description,
            encoding=stripe_desc.encoding,
            value_range=stripe_desc.value_range,
            notes=stripe_desc.notes,
            decoded_label=", ".join(parts),
        )
    ]


#### Bird-attribute stripe decoder ####


def _build_bird_attr_sub_field(
    sub_field: stripes.SubFieldDescriptor,
    attrs_slice: np.ndarray,
) -> gamelog_models.EncodedSubField | None:
    """Build one named EncodedSubField from a bird_attrs sub-field, or None if zero.

    Uses domain-specific decode logic per field name: multi-hot → active names,
    vector → count-per-slot labels, one-hot → name at argmax, boolean → 'yes',
    normalized scalar → denormalized integer.  Returns None when the sub-field
    slice is entirely within ``_ZERO_THRESHOLD`` of zero."""
    sub_slice = attrs_slice[
        sub_field.relative_offset : sub_field.relative_offset + sub_field.size
    ]
    if np.all(np.abs(sub_slice) < _ZERO_THRESHOLD):
        return None

    name = sub_field.name
    enc = sub_field.encoding or "scalar"
    vr = sub_field.value_range

    # Boolean flags: indicate the property is set.
    if name in _BIRD_ATTR_BOOL_FIELDS:
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            raw_value=float(sub_slice[0]),
            decoded_label="yes",
        )

    # Color: one-hot over the 4 power colors → name at argmax.
    if name == "color":
        active_idx = int(np.argmax(sub_slice))
        color_label = (
            layout._COLORS[active_idx].value
            if active_idx < len(layout._COLORS)
            else str(active_idx)
        )
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            active_index=active_idx,
            decoded_label=color_label,
        )

    # food_cost: 6-element vector, normalized ÷ 3 → count per food type.
    if name == "food_cost":
        scale = _denorm_scale(sub_field.notes) or 1.0
        food_labels = tuple(food.value for food in cards.ALL_FOODS) + ("wild",)
        parts = [
            f"{round(float(sub_slice[pos]) * scale)} {food_labels[pos]}"
            for pos in range(len(sub_slice))
            if abs(float(sub_slice[pos])) >= _ZERO_THRESHOLD and pos < len(food_labels)
        ]
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            decoded_label=", ".join(parts) if parts else None,
        )

    # nest: multi-hot over 4 concrete nest types; all four set = STAR wildcard.
    if name == "nest":
        active = np.where(np.abs(sub_slice) >= _ZERO_THRESHOLD)[0]
        if len(active) == len(layout._NEST_BASE_TYPES):
            decoded = "star (wildcard)"
        else:
            nest_labels = [nest_type.value for nest_type in layout._NEST_BASE_TYPES]
            decoded = ", ".join(
                nest_labels[int(idx)] for idx in active if int(idx) < len(nest_labels)
            )
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            decoded_label=decoded or None,
        )

    # habitats: multi-hot over the 3 habitat types.
    if name == "habitats":
        active = np.where(np.abs(sub_slice) >= _ZERO_THRESHOLD)[0]
        hab_labels = [hab.value for hab in cards.ALL_HABITATS]
        decoded = ", ".join(
            hab_labels[int(idx)] for idx in active if int(idx) < len(hab_labels)
        )
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            decoded_label=decoded or None,
        )

    # bonus_categories: multi-hot over 7 curated bonus-card categories.
    if name == "bonus_categories":
        active = np.where(np.abs(sub_slice) >= _ZERO_THRESHOLD)[0]
        cat_names = layout._KEPT_BONUS_NAMES
        decoded = ", ".join(
            cat_names[int(idx)] if int(idx) < len(cat_names) else f"cat_{idx}"
            for idx in active
        )
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            decoded_label=decoded or None,
        )

    # power_exchange: 13-element vector, normalized ÷ 3 → count per exchange slot.
    if name == "power_exchange":
        scale = _denorm_scale(sub_field.notes) or 1.0
        slot_names = layout._EXCHANGE_SLOT_NAMES
        parts = [
            f"{slot_names[pos]} ×{round(float(sub_slice[pos]) * scale)}"
            for pos in range(len(sub_slice))
            if abs(float(sub_slice[pos])) >= _ZERO_THRESHOLD and pos < len(slot_names)
        ]
        return gamelog_models.EncodedSubField(
            name=name,
            description=sub_field.description,
            encoding=enc,
            value_range=vr,
            notes=sub_field.notes,
            decoded_label=", ".join(parts) if parts else None,
        )

    # Fallback: normalized scalar (points, wingspan, egg_limit).
    scale = _denorm_scale(sub_field.notes)
    raw_val = float(sub_slice[0])
    return gamelog_models.EncodedSubField(
        name=name,
        description=sub_field.description,
        encoding=enc,
        value_range=vr,
        notes=sub_field.notes,
        raw_value=raw_val,
        decoded_label=str(round(raw_val * scale)) if scale is not None else None,
    )
