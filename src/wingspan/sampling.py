"""Seed-stable weighted sampling.

Stdlib-only (no torch, no numpy) so these helpers can be imported from the
torch-bearing training modules and from the torch-free setup-model helpers
without pulling in the full training stack.
"""

import random


def weighted_index(rng: random.Random, weights: list[float]) -> int:
    """Return an index sampled in proportion to ``weights`` using the seeded
    ``rng`` (not numpy's global state) so episodes stay reproducible.

    Falls through to the last index when floating-point accumulation leaves
    the roll un-claimed (guards against weights that sum to exactly 1.0 in
    IEEE 754 but whose partial sums never quite reach the roll value)."""
    roll = rng.random()
    acc = 0.0
    for index, weight in enumerate(weights):
        acc += weight
        if roll < acc:
            return index
    return len(weights) - 1
