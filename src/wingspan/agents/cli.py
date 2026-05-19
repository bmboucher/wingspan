"""Interactive human (stdin/stdout) agent and the mixed-hotseat helper.

The CLI agent walks a human through the offered choices index-by-index.
The combined ``SetupDecision`` is special-cased: enumerating all 504 legal
setup combinations on one screen is unusable, so the setup pick is broken
into three sub-dialogs (cards / foods / bonus) and the human's answer is
reassembled back into one of the offered ``SetupChoice`` instances.
"""

from __future__ import annotations

import itertools
import random
import typing

from wingspan import cards, decisions
from wingspan.agents import base
from wingspan.engine import core as engine_core


def cli_agent() -> engine_core.Agent:
    """Interactive human agent. Prints prompt and choices, reads index."""

    def agent[C: decisions.Choice](
        _engine: engine_core.Engine, decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.SetupDecision):
            # SetupDecision is Decision[SetupChoice], but the type checker
            # can't propagate that narrowing back onto the bound C — so the
            # SetupChoice return is cast through.
            return typing.cast(C, _cli_resolve_setup_choice(decision))
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


def mixed_agents(
    rng: random.Random, human_index: int,
) -> tuple[engine_core.Agent, engine_core.Agent]:
    """Two-player roster with one human at ``human_index`` and a random opponent."""
    a = cli_agent() if human_index == 0 else base.random_agent(rng)
    b = cli_agent() if human_index == 1 else base.random_agent(rng)
    return (a, b)


###### PRIVATE #######


def _cli_resolve_setup_choice(decision: decisions.SetupDecision) -> decisions.SetupChoice:
    """Three-step sub-dialog for the combined setup pick.

    Walks the human through cards → foods → bonus, then locates the matching
    fully-resolved ``SetupChoice`` in ``decision.choices`` and returns it.
    """
    dealt_cards = decision.dealt_cards
    dealt_bonus = decision.dealt_bonus

    print()
    print(decision.prompt)
    print("Cards dealt (each kept card costs 1 food):")
    for i, c in enumerate(dealt_cards):
        print(f"  [{i}] {c.name} ({c.points}vp, cost {c.food_cost.total})")
    if dealt_bonus:
        print("Bonus cards dealt:")
        for i, bc in enumerate(dealt_bonus):
            print(f"  [{i}] {bc.name}: {bc.vp_text}")

    kept_indices = _cli_pick_kept_cards(dealt_cards)
    kept_cards = tuple(dealt_cards[i] for i in kept_indices)
    kept_foods = _cli_pick_kept_foods(len(cards.ALL_FOODS) - len(kept_indices))
    bonus_card = _cli_pick_bonus(dealt_bonus)

    for c in decision.choices:
        if (
            c.kept_cards == kept_cards
            and c.kept_foods == tuple(kept_foods)
            and c.bonus_card == bonus_card
        ):
            return c
    raise AssertionError(
        "assembled setup answer did not match any offered SetupChoice: "
        f"keep={[b.name for b in kept_cards]} foods={kept_foods} bonus={bonus_card}"
    )


def _cli_pick_kept_cards(dealt_cards: list[cards.Bird]) -> tuple[int, ...]:
    """Step 1: pick a subset of the dealt cards to keep (2^n options)."""
    n = len(dealt_cards)
    options: list[tuple[int, ...]] = []
    for k in range(n + 1):
        for combo in itertools.combinations(range(n), k):
            options.append(combo)
    print()
    print("Step 1 — choose which cards to keep:")
    for idx, combo in enumerate(options):
        names = [dealt_cards[i].name for i in combo] if combo else ["(none)"]
        print(f"  [{idx}] keep {len(combo)}: {', '.join(names)}")
    return _read_index_choice(options, "keep> ")


def _cli_pick_kept_foods(
    k: int,
) -> tuple[cards.Food, ...]:
    """Step 2: pick which ``k`` distinct foods to keep (C(5,k) options).

    ``k`` is ``len(ALL_FOODS) - len(kept_cards)``, since each kept card costs
    one food off the player's one-of-each starting stash.
    """
    options: list[tuple[cards.Food, ...]] = list(
        itertools.combinations(cards.ALL_FOODS, k)
    )
    print()
    print(f"Step 2 — choose which {k} food(s) to keep:")
    for idx, combo in enumerate(options):
        labels = [f.value for f in combo] if combo else ["(none)"]
        print(f"  [{idx}] {', '.join(labels)}")
    return _read_index_choice(options, "keep foods> ")


def _cli_pick_bonus(
    dealt_bonus: list[cards.BonusCard],
) -> cards.BonusCard | None:
    """Step 3: pick which dealt bonus card to keep (1-2 options)."""
    if not dealt_bonus:
        return None
    print()
    print("Step 3 — choose which bonus card to keep:")
    for idx, bc in enumerate(dealt_bonus):
        print(f"  [{idx}] {bc.name}: {bc.vp_text}")
    return _read_index_choice(dealt_bonus, "bonus> ")


def _read_index_choice[T](options: typing.Sequence[T], prompt: str) -> T:
    """Read a 0-based index from stdin, looping on invalid input."""
    while True:
        raw = input(prompt).strip()
        if raw == "" and len(options) == 1:
            return options[0]
        try:
            i = int(raw)
        except ValueError:
            print("  enter a number")
            continue
        if 0 <= i < len(options):
            return options[i]
        print("  out of range")
