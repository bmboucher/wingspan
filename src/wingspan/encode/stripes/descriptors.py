"""Stripe and sub-field descriptor Pydantic models.

Provides :class:`SubFieldDescriptor`, :class:`StripeDescriptor`, and
:class:`VectorLayout` — the data classes that every stripe-registry function
returns.  Also houses ``_hand_summary_sub_fields``, which is shared between
the state layout and the hand-encoder-input layout.
"""

from __future__ import annotations

import pydantic

from wingspan import cards


class SubFieldDescriptor(pydantic.BaseModel):
    """One named element or logical sub-group within a complex stripe.

    Used by :class:`StripeDescriptor` to expose drill-down detail for stripes
    whose elements are semantically distinct from each other (e.g. the 7 scalars
    in ``misc_scalars``, the per-slot features in a board slot, …). Homogeneous
    stripes where every element has the same meaning (``hand_multihot``, bonus
    one-hots) do not carry sub-fields — the parent stripe's ``notes`` are
    sufficient there.
    """

    name: str
    """Dot-qualified sub-field name, e.g. ``forest_0.eggs``."""

    description: str
    """Human-readable sentence describing this specific element or block."""

    relative_offset: int
    """Index of the first element *within the parent stripe* (0-based)."""

    size: int = 1
    """Element count (1 for a scalar; >1 for a one-hot block treated as a unit)."""

    encoding: str
    """Encoding kind matching the parent stripe's vocabulary."""

    value_range: str
    """Typical element values."""

    notes: str | None = None
    """Additional normalization or sub-structure details."""

    group: str | None = None
    """Optional grouping label used to nest sub-fields in the HTML report
    (e.g. ``"slot_forest_0"`` groups a slot's per-slot elements together)."""


class StripeDescriptor(pydantic.BaseModel):
    """One named region of a flat feature vector."""

    name: str
    """Short reference name (snake_case, suitable for indexing or labelling)."""

    description: str
    """Human-readable sentence describing what this stripe encodes."""

    offset: int
    """Index of the first element in the flat vector."""

    size: int
    """Number of elements."""

    encoding: str
    """Encoding kind: ``scalar``, ``vector``, ``one-hot``, ``multi-hot``,
    ``integer-index``, or ``complex`` (structured block, see notes)."""

    value_range: str
    """Typical element values, e.g. ``[0, 1]``, ``{0, 1}``, ``int 0–180``."""

    notes: str | None = None
    """Sub-field layout, normalization constants, or other caveats."""

    sub_fields: tuple[SubFieldDescriptor, ...] = ()
    """Per-element drill-down for semantically distinct stripes. Empty for
    homogeneous stripes where every element has the same meaning."""


class VectorLayout(pydantic.BaseModel):
    """The complete named stripe breakdown of a flat feature vector."""

    total_size: int
    """Total element count (equals ``sum(stripes[i].size)``)."""

    stripes: tuple[StripeDescriptor, ...]


###### PRIVATE #######


def hand_summary_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """10 sub-fields for the hand-summary stripe: size, habitat counts, food multi-hot."""
    entries: list[tuple[str, str, str]] = [
        ("hand_size", "Total cards currently in hand.", "Normalized ÷ 10."),
        *[
            (
                f"{habitat.value}_count",
                f"Number of hand birds that live in {habitat.value} "
                "(a dual-habitat bird counts in each).",
                "Normalized ÷ 10.",
            )
            for habitat in cards.ALL_HABITATS
        ],
        *[
            (
                f"has_{food.value}_cost",
                f"1.0 if any hand bird has {food.value} in its food cost.",
                "{0, 1}.",
            )
            for food in cards.ALL_FOODS
        ],
        (
            "has_wild_cost",
            "1.0 if any hand bird has a wild token in its cost.",
            "{0, 1}.",
        ),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )
