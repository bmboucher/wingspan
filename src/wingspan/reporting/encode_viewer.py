"""Extract non-zero stripe summaries from raw encoder vectors for HTML display.

Given a flat state vector or per-choice vector, walks the corresponding
:class:`~wingspan.encode.stripes.VectorLayout`, finds stripes whose slice is
non-zero (absolute value above a small threshold), and converts each active
entry into a compact :class:`~wingspan.reporting.game_log_html.EncodedStripe`
record carrying only the non-zero sub-fields.  The result is embedded in the
HTML game log to power the encoding-viewer modal.

Stripes whose ``encoding == "complex"`` (learned card embeddings) are skipped
because individual dimensions have no human-readable meaning.  Only
:func:`extract_state_stripes` and :func:`extract_choice_stripes` are public;
everything else is implementation detail.
"""

from __future__ import annotations

import numpy as np

from wingspan.encode import layout, stripes
from wingspan.reporting import game_log_html

# Values whose absolute magnitude is below this threshold are treated as zero.
_ZERO_THRESHOLD = 1e-6


def extract_state_stripes(
    vector: list[float],
    include_setup: bool,
    card_embed_dim: int,
) -> list[game_log_html.EncodedStripe]:
    """Return non-zero stripe summaries for a flat state feature vector.

    Uses :func:`~wingspan.encode.stripes.state_stripe_layout` to get the
    canonical stripe layout for the given spec, then delegates to
    :func:`_extract_nonzero_stripes`."""
    spec = layout.EncodingSpec(include_setup=include_setup)
    vector_layout = stripes.state_stripe_layout(spec, card_embed_dim)
    return _extract_nonzero_stripes(np.array(vector, dtype=np.float32), vector_layout)


def extract_choice_stripes(
    choice_vec: list[float],
    include_setup: bool,
    card_embed_dim: int,
) -> list[game_log_html.EncodedStripe]:
    """Return non-zero stripe summaries for one row of the choice feature matrix.

    Uses :func:`~wingspan.encode.stripes.choice_stripe_layout` to get the
    canonical stripe layout, then delegates to :func:`_extract_nonzero_stripes`."""
    spec = layout.EncodingSpec(include_setup=include_setup)
    vector_layout = stripes.choice_stripe_layout(spec, card_embed_dim)
    return _extract_nonzero_stripes(
        np.array(choice_vec, dtype=np.float32), vector_layout
    )


###### PRIVATE #######


def _extract_nonzero_stripes(
    vector: np.ndarray,
    vector_layout: stripes.VectorLayout,
) -> list[game_log_html.EncodedStripe]:
    """Walk a VectorLayout and return an EncodedStripe for each non-zero stripe.

    Skips stripes with ``encoding == "complex"`` (learned embeddings) and any
    stripe whose entire slice is within ``_ZERO_THRESHOLD`` of zero.  For each
    kept stripe, delegates to :func:`_build_encoded_stripe`."""
    result: list[game_log_html.EncodedStripe] = []
    for stripe_desc in vector_layout.stripes:
        if stripe_desc.encoding == "complex":
            continue
        stripe_slice = vector[
            stripe_desc.offset : stripe_desc.offset + stripe_desc.size
        ]
        if np.all(np.abs(stripe_slice) < _ZERO_THRESHOLD):
            continue
        encoded = _build_encoded_stripe(stripe_desc, stripe_slice)
        if encoded.sub_fields:
            result.append(encoded)
    return result


def _build_encoded_stripe(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
) -> game_log_html.EncodedStripe:
    """Build an EncodedStripe from one non-zero stripe slice.

    When the stripe has named sub-fields, produces one EncodedSubField per
    non-zero sub-field element using :func:`_build_sub_field`.  When there are
    no sub-fields, treats the entire stripe as one synthetic sub-field using the
    stripe-level metadata."""
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
        # No sub-fields: represent the whole stripe as a single entry.
        sub_fields = _build_whole_stripe_sub_field(stripe_desc, stripe_slice)

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
    """Build one EncodedSubField, or None if this sub-field is effectively zero.

    Uses the sub-field's own ``encoding`` (falling back to the stripe encoding)
    to choose between one-hot active-index, scalar value, and multi-element
    raw-values representations."""
    sub_slice = stripe_slice[
        sub_field.relative_offset : sub_field.relative_offset + sub_field.size
    ]
    if np.all(np.abs(sub_slice) < _ZERO_THRESHOLD):
        return None

    effective_encoding = sub_field.encoding or stripe_encoding

    if effective_encoding == "one-hot":
        active_index = int(np.argmax(sub_slice))
        return game_log_html.EncodedSubField(
            name=sub_field.name,
            description=sub_field.description,
            encoding=effective_encoding,
            value_range=sub_field.value_range,
            notes=sub_field.notes,
            active_index=active_index,
        )

    if sub_field.size == 1:
        return game_log_html.EncodedSubField(
            name=sub_field.name,
            description=sub_field.description,
            encoding=effective_encoding,
            value_range=sub_field.value_range,
            notes=sub_field.notes,
            raw_value=float(sub_slice[0]),
        )

    # Multi-element: report only the non-zero positions.
    active_mask = np.abs(sub_slice) >= _ZERO_THRESHOLD
    raw_values = [float(sub_slice[idx]) for idx in np.where(active_mask)[0]]
    return game_log_html.EncodedSubField(
        name=sub_field.name,
        description=sub_field.description,
        encoding=effective_encoding,
        value_range=sub_field.value_range,
        notes=sub_field.notes,
        raw_values=raw_values,
    )


def _build_whole_stripe_sub_field(
    stripe_desc: stripes.StripeDescriptor,
    stripe_slice: np.ndarray,
) -> list[game_log_html.EncodedSubField]:
    """Represent a no-sub-field stripe as a single synthetic EncodedSubField.

    Uses the stripe-level ``encoding`` to pick the right value representation."""
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

    if stripe_desc.size == 1:
        return [
            game_log_html.EncodedSubField(
                name=stripe_desc.name,
                description=stripe_desc.description,
                encoding=stripe_desc.encoding,
                value_range=stripe_desc.value_range,
                notes=stripe_desc.notes,
                raw_value=float(stripe_slice[0]),
            )
        ]

    # Multi-element stripe: report non-zero values with their index.
    active_indices = np.where(np.abs(stripe_slice) >= _ZERO_THRESHOLD)[0]
    return [
        game_log_html.EncodedSubField(
            name=f"{stripe_desc.name}[{idx}]",
            description=stripe_desc.description,
            encoding=stripe_desc.encoding,
            value_range=stripe_desc.value_range,
            notes=stripe_desc.notes,
            raw_value=float(stripe_slice[idx]),
        )
        for idx in active_indices
    ]
