"""Human-readable formatters for cards and game state.

These helpers exist so the interactive CLI agent can present birds, bonus
cards, and board state with enough detail for a human to make an informed
choice -- the canonical engine-side ``Choice.label`` strings stay short for
the log, while this module produces the verbose rendering shown only to
the human at decision time.

All functions are pure (state-in, string-out) and have no side effects.
"""

from __future__ import annotations

from wingspan import cards, state

# ---------------------------------------------------------------------------
# Public formatters


def format_cost(cost: cards.BirdCost) -> str:
    """Compact rendering of a printed food cost.

    Examples: ``1seed``, ``1invertebrate+1*``, ``2*``, ``free``.
    The trailing ``*`` denotes wild slots (any food).
    """
    parts: list[str] = [
        f"{n}{f.value}" for f, n in zip(cards.ALL_FOODS, cost.specific) if n > 0
    ]
    if cost.wild:
        parts.append(f"{cost.wild}*")
    return "+".join(parts) if parts else "free"


def format_food_pool(pool: state.FoodPool) -> str:
    """Compact rendering of a food pool. Lists only non-zero foods."""
    parts = [f"{n}{f.value}" for f, n in pool.items() if n > 0]
    return "+".join(parts) if parts else "(empty)"


def format_bird(b: cards.Bird) -> str:
    """One-line bird summary with cost, stats, and habitats.

    Use :func:`format_bird_full` when the bird's power text should appear
    on a second line beneath this summary.
    """
    habs = "/".join(h.value for h in b.habitats)
    flags: list[str] = []
    if b.flocking:
        flags.append("flocking")
    if b.predator:
        flags.append("predator")
    flag_str = f" ({','.join(flags)})" if flags else ""
    return (
        f"{b.name}{flag_str} | {b.points}vp, cost:{format_cost(b.food_cost)}, "
        f"habitat:{habs}, nest:{b.nest.value}, eggs<={b.egg_limit}, "
        f"w={b.wingspan_cm}cm"
    )


def format_bird_full(b: cards.Bird, indent: str = "      ") -> str:
    """Multi-line bird summary: one-line stats then the printed power text."""
    head = format_bird(b)
    power_text = b.raw_power_text.strip()
    if not power_text:
        return head
    return f"{head}\n{indent}[{b.power.color.value}] {power_text}"


def format_played_bird(pb: state.PlayedBird) -> str:
    """Compact rendering of a bird in play (name + per-bird mutable state)."""
    parts = [f"eggs={pb.eggs}/{pb.bird.egg_limit}"]
    if pb.cached_food:
        parts.append(f"cached={pb.cached_food}")
    if pb.tucked_cards:
        parts.append(f"tucked={pb.tucked_cards}")
    return f"{pb.bird.name} ({', '.join(parts)})"


def format_bonus(bc: cards.BonusCard) -> str:
    """One-line bonus-card summary highlighting the scoring condition."""
    parts = [f"{bc.name}"]
    if bc.condition:
        parts.append(f"scores: {bc.condition}")
    if bc.vp_text:
        parts.append(f"VP: {bc.vp_text}")
    return " | ".join(parts)


def format_board(s: state.GameState, p: state.Player) -> str:
    """Multi-line summary of ``p``'s board and resources plus round context."""
    goal = s.round_goals[s.round_idx] if s.round_goals else None
    header = (
        f"=== [{p.name}] round {s.round_idx + 1}/4, "
        f"cubes={p.action_cubes_left}, food=[{format_food_pool(p.food)}], "
        f"eggs={p.total_eggs}, round-goal-pts={p.round_goal_points} ==="
    )
    lines = [header]
    if goal is not None:
        lines.append(f"  round goal: {goal.description} ({goal.category})")
    lines.append(f"  birdfeeder: [{format_food_pool(s.birdfeeder.counts)}]")
    lines.append(f"  tray: [{', '.join(b.name for b in s.tray) or '(empty)'}]")
    for h in cards.ALL_HABITATS:
        row = p.board[h]
        if row:
            birds = ", ".join(format_played_bird(pb) for pb in row)
        else:
            birds = "(empty)"
        lines.append(f"  {h.value}: {birds}")
    hand = ", ".join(b.name for b in p.hand) or "(empty)"
    lines.append(f"  hand: {hand}")
    bonus = ", ".join(format_bonus(bc) for bc in p.bonus_cards) or "(none)"
    lines.append(f"  bonus cards: {bonus}")
    return "\n".join(lines)
