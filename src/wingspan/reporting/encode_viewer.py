"""Extract non-zero stripe summaries from raw encoder vectors for HTML display.

Given a flat state vector or per-choice vector, walks the corresponding
:class:`~wingspan.encode.stripes.VectorLayout`, finds stripes whose slice is
non-zero (absolute value above a small threshold), and converts each active
entry into a compact :class:`~wingspan.reporting.game_log_html.EncodedStripe`
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
from wingspan.encode import layout, stripes
from wingspan.reporting import game_log_html

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
        "kept_multihot",
        "kept_cards",
        "turn1_playable",
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


def extract_state_stripes(
    vector: list[float],
    include_setup: bool,
) -> list[game_log_html.EncodedStripe]:
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
) -> list[game_log_html.EncodedStripe]:
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
) -> list[game_log_html.EncodedStripe]:
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
) -> list[game_log_html.EncodedStripe]:
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
) -> list[game_log_html.EncodedStripe]:
    """Walk a VectorLayout and return an EncodedStripe for each non-zero stripe.

    Skips stripes with ``encoding == "complex"`` (learned embeddings) and any
    stripe whose entire slice is within ``_ZERO_THRESHOLD`` of zero."""
    result: list[game_log_html.EncodedStripe] = []
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
) -> game_log_html.EncodedStripe:
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

    return game_log_html.EncodedStripe(
        name=stripe_desc.name,
        description=stripe_desc.description,
        sub_fields=sub_fields,
    )


def _build_sub_field(
    sub_field: stripes.SubFieldDescriptor,
    stripe_slice: np.ndarray,
    stripe_encoding: str,
) -> game_log_html.EncodedSubField | None:
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
        return game_log_html.EncodedSubField(
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
        return game_log_html.EncodedSubField(
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
    return game_log_html.EncodedSubField(
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
) -> list[game_log_html.EncodedSubField]:
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
            game_log_html.EncodedSubField(
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
            game_log_html.EncodedSubField(
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
            game_log_html.EncodedSubField(
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
            game_log_html.EncodedSubField(
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
        game_log_html.EncodedSubField(
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
) -> list[game_log_html.EncodedSubField]:
    """One sub-field per occupied slot in a bird integer-index stripe.

    Index 0 means empty (skipped).  Index N means bird_index N-1 (the encoder
    stores bird_index + 1 so that 0 is the unambiguous empty sentinel)."""
    bird_name_tuple = _bird_names()
    sub_fields: list[game_log_html.EncodedSubField] = []
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
            game_log_html.EncodedSubField(
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
) -> list[game_log_html.EncodedSubField]:
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
        game_log_html.EncodedSubField(
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
) -> list[game_log_html.EncodedSubField]:
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
        game_log_html.EncodedSubField(
            name=stripe_desc.name,
            description=stripe_desc.description,
            encoding=stripe_desc.encoding,
            value_range=stripe_desc.value_range,
            notes=stripe_desc.notes,
            decoded_label=", ".join(parts),
        )
    ]
