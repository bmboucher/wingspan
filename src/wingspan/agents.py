"""Agents that resolve Decisions: random, CLI human, and a stub for an RL policy."""
from __future__ import annotations

import random
from typing import Optional

from .actions import Choice, Decision


def random_agent(rng: Optional[random.Random] = None):
    r = rng or random.Random()

    def agent(_engine, decision: Decision) -> Choice:
        return r.choice(decision.choices)

    return agent


def cli_agent():
    """Interactive human agent. Prints prompt and choices, reads index."""

    def agent(_engine, decision: Decision) -> Choice:
        print()
        print(decision.prompt)
        for i, c in enumerate(decision.choices):
            print(f"  [{i}] {c.label}")
        while True:
            raw = input("choice> ").strip()
            if raw == "" and len(decision.choices) == 1:
                return decision.choices[0]
            try:
                idx = int(raw)
            except ValueError:
                print("  enter a number")
                continue
            if 0 <= idx < len(decision.choices):
                return decision.choices[idx]
            print("  out of range")

    return agent


def mixed_agents(rng: random.Random, human_index: int) -> tuple:
    """Helper: one human at ``human_index``, opponent random."""
    a = cli_agent() if human_index == 0 else random_agent(rng)
    b = cli_agent() if human_index == 1 else random_agent(rng)
    return (a, b)
