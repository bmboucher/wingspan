"""Stripe and sub-field descriptor Pydantic models.

Provides :class:`StripeSpec`, :class:`SubFieldDescriptor`,
:class:`StripeDescriptor`, and :class:`VectorLayout` — the data classes
used throughout the encoding system.  :class:`StripeSpec` is the *input*
to layout computation (name + size only); :class:`VectorLayout` is the
*output* (names + auto-accumulated offsets + sizes).  Also houses
:func:`hand_summary_sub_fields`, shared between the state layout and the
hand-encoder-input layout.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import cards


class StripeSpec(pydantic.BaseModel):
    """Name and width of one stripe — the input to offset accumulation.

    Pass an ordered sequence of these to :meth:`VectorLayout.from_stripe_specs`
    to build a :class:`VectorLayout` whose offsets are guaranteed overlap-free
    by sequential accumulation.  Changing the order or sizes of specs changes
    the encoding and is a FRESH (checkpoint-invalidating) event."""

    name: str
    """Short reference name (snake_case).  Must be unique within a layout."""

    size: int
    """Number of elements this stripe occupies in the flat vector."""


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

    @pydantic.model_validator(mode="after")
    def _sub_fields_within_bounds(self) -> "StripeDescriptor":
        for sf in self.sub_fields:
            if sf.relative_offset + sf.size > self.size:
                raise ValueError(
                    f"Sub-field {sf.name!r} (relative_offset={sf.relative_offset}, "
                    f"size={sf.size}) extends past stripe {self.name!r} "
                    f"boundary (stripe size={self.size})"
                )
        return self


class VectorLayout(pydantic.BaseModel):
    """The complete named stripe breakdown of a flat feature vector."""

    total_size: int
    """Total element count (equals ``sum(stripes[i].size)``)."""

    stripes: tuple[StripeDescriptor, ...]

    @classmethod
    def from_stripe_specs(cls, specs: typing.Sequence[StripeSpec]) -> VectorLayout:
        """Build a VectorLayout by auto-accumulating offsets from ordered specs.

        Each stripe's offset equals the cumulative sum of all preceding sizes,
        so no manual arithmetic is required and overlaps are structurally
        impossible.  The resulting :class:`StripeDescriptor` objects contain the
        correct ``offset`` and ``size``; other metadata fields (description,
        encoding, value_range) are empty placeholders — enrich them if needed."""
        stripes: list[StripeDescriptor] = []
        running_offset = 0
        for spec in specs:
            stripes.append(
                StripeDescriptor(
                    name=spec.name,
                    description="",
                    offset=running_offset,
                    size=spec.size,
                    encoding="",
                    value_range="",
                )
            )
            running_offset += spec.size
        return cls(total_size=running_offset, stripes=tuple(stripes))

    def offset_of(self, name: str) -> int:
        """Offset of the named stripe in the flat vector.

        Raises :exc:`KeyError` if no stripe with that name exists."""
        for stripe in self.stripes:
            if stripe.name == name:
                return stripe.offset
        raise KeyError(f"No stripe named {name!r} in this VectorLayout")

    def size_of(self, name: str) -> int:
        """Width of the named stripe.

        Raises :exc:`KeyError` if no stripe with that name exists."""
        for stripe in self.stripes:
            if stripe.name == name:
                return stripe.size
        raise KeyError(f"No stripe named {name!r} in this VectorLayout")


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
