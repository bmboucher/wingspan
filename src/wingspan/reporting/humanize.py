"""Humanize choices and log text for the HTML decision-log display.

All functions here depend only on ``decisions``, ``cards``, and ``state`` —
no engine or torch imports — so capture can import this module freely without
closing the ``engine`` ↔ ``instrumentation`` import cycle.

Public API: :func:`humanize_choice`, :func:`humanize_outcome`,
:func:`humanize_note`, :func:`humanize_forced`.
"""

from __future__ import annotations

import re
import typing

from wingspan import cards, decisions, state

# Habitat display labels matching the board panel.
_HABITAT_LABELS: dict[cards.Habitat, str] = {
    cards.Habitat.FOREST: "Forest",
    cards.Habitat.GRASSLAND: "Grassland",
    cards.Habitat.WETLAND: "Wetland",
}

_MAIN_ACTION_LABELS: dict[decisions.MainAction, str] = {
    decisions.MainAction.GAIN_FOOD: "Gain food",
    decisions.MainAction.LAY_EGGS: "Lay eggs",
    decisions.MainAction.DRAW_CARDS: "Draw cards",
    decisions.MainAction.PLAY_BIRD: "Play a bird",
}

_FOOD_VALUES: frozenset[str] = frozenset(food.value for food in cards.ALL_FOODS)

# Strips a leading "[Name] " player-name tag from a log line.
_PLAYER_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")

# Label for the birdfeeder's choice die (seed/invertebrate combo face).
_CHOICE_DIE_LABEL = "seed/invertebrate"

# Matches a row-count log line from any of the three main-action handlers.
_ROW_HAS_RE = re.compile(
    r"(gain food|lay eggs|draw cards): row has \d+ birds?",
    re.IGNORECASE,
)


def _food_subset_summary(choice: decisions.FoodSubsetChoice) -> str:
    """A readable multiset summary of a combined gain, e.g. ``2 fish + 1 seed``;
    choice-die resolutions are tagged ``(choice)``."""
    parts = [
        f"{amount} {food.value}" for food, amount in choice.plain.items() if amount > 0
    ]
    if choice.choice_inv:
        parts.append(f"{choice.choice_inv} {cards.Food.INVERTEBRATE.value} (choice)")
    if choice.choice_seed:
        parts.append(f"{choice.choice_seed} {cards.Food.SEED.value} (choice)")
    return " + ".join(parts) if parts else "nothing"


def humanize_choice(
    choice: decisions.Choice,
    gs: state.GameState,
    player_id: int | None = None,
    decision: decisions.Decision[typing.Any] | None = None,
) -> str:
    """A concise human label for one offered option.

    ``player_id`` is used to resolve ``BoardTargetChoice`` board lookups — pass
    ``decision.player_id`` when available.  ``decision`` is used to supply
    symmetric skip labels for ``AcceptExchangeDecision``."""
    if isinstance(choice, decisions.DrawSourceChoice):
        if choice.source == "tray" and choice.bird is not None:
            return f"{choice.bird.name} (tray)"
        return "Draw from the deck"
    if isinstance(choice, decisions.PlayBirdChoice):
        hab = _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
        return f"{choice.bird.name} → {hab}"
    if isinstance(choice, decisions.FoodChoice):
        # Item 7: choice-die gains show both possible foods.
        if choice.from_choice_die:
            return f"{choice.food.value} (from {_CHOICE_DIE_LABEL})"
        return choice.food.value  # renderer applies emoji
    if isinstance(choice, decisions.FoodSubsetChoice):
        return _food_subset_summary(choice)
    if isinstance(choice, decisions.FoodPaymentChoice):
        return choice.payment.format()
    if isinstance(choice, decisions.BoardTargetChoice):
        hab = _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
        if player_id is not None and 0 <= player_id < len(gs.players):
            row = gs.players[player_id].board[choice.habitat]
            if choice.slot < len(row):
                return f"{row[choice.slot].bird.name} ({hab})"
        return f"{hab} slot {choice.slot + 1}"
    if isinstance(choice, decisions.BonusCardChoice):
        return choice.bonus_card.name
    if isinstance(choice, decisions.BirdChoice):
        return choice.bird.name
    if isinstance(choice, decisions.HabitatChoice):
        return _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
    if isinstance(choice, decisions.PlayerIdChoice):
        return f"P{choice.player_id}"
    if isinstance(choice, decisions.SkipChoice):
        # Item 10: mirror the accept label so decline is as descriptive as accept.
        if isinstance(decision, decisions.AcceptExchangeDecision):
            pay_choices = [
                c for c in decision.choices if isinstance(c, decisions.PayCostChoice)
            ]
            if pay_choices:
                return f"Decline ({pay_choices[0].label})"
        return "Decline"
    if isinstance(choice, decisions.MainActionChoice):
        return _MAIN_ACTION_LABELS.get(choice.action, choice.action.value)
    if isinstance(choice, decisions.PlayedBirdChoice):
        return choice.played_bird.bird.name
    if isinstance(choice, decisions.TuckActivateChoice):
        plural = "cards" if choice.cards_to_tuck != 1 else "card"
        return f"Tuck {choice.cards_to_tuck} {plural}"
    if isinstance(choice, decisions.ResetBirdfeederChoice):
        return "Reset birdfeeder"
    if isinstance(choice, decisions.SetupChoice):
        if choice.kept_cards:
            names = [bird.name for bird in choice.kept_cards]
            shown = names[:3]
            suffix = "…" if len(names) > 3 else ""
            return f"Keep {', '.join(shown)}{suffix}"
        return "Keep no birds"
    return choice.display_label()


def humanize_outcome(
    decision: decisions.Decision[typing.Any],
    choice: decisions.Choice,
    gs: state.GameState,
) -> str:
    """A third-person present summary of what happened (no player prefix).

    Used as the collapsed-header text for decision boxes."""
    if isinstance(choice, decisions.MainActionChoice):
        return _MAIN_ACTION_LABELS.get(choice.action, choice.action.value)
    if isinstance(choice, decisions.PlayBirdChoice):
        hab = _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
        return f"Plays {choice.bird.name} in {hab}"
    if isinstance(choice, decisions.DrawSourceChoice):
        if choice.source == "tray" and choice.bird is not None:
            return f"Draws {choice.bird.name} from the tray"
        return "Draws from the deck"
    if isinstance(choice, decisions.FoodChoice):
        # Item 7: choice-die header names the die.
        if choice.from_choice_die:
            return f"Gains {choice.food.value} (choice die)"
        if isinstance(decision, decisions.SpendFoodDecision):
            return f"Discards {choice.food.value}"
        return f"Gains {choice.food.value}"
    if isinstance(choice, decisions.FoodSubsetChoice):
        return f"Gains {_food_subset_summary(choice)}"
    if isinstance(choice, decisions.SkipChoice):
        # Item 10: mirror accept label so decline reads symmetrically.
        if isinstance(decision, decisions.AcceptExchangeDecision):
            pay_choices = [
                c for c in decision.choices if isinstance(c, decisions.PayCostChoice)
            ]
            if pay_choices:
                return f"Declines: {pay_choices[0].label}"
        return "Declines"
    if isinstance(choice, decisions.BonusCardChoice):
        # Item 4: include "bonus card" in the label.
        return f"Keeps bonus card {choice.bonus_card.name}"
    if isinstance(choice, decisions.BirdChoice):
        # Item 8: discard vs pass depending on decision type and prompt.
        if isinstance(
            decision,
            (
                decisions.DiscardBirdForFoodDecision,
                decisions.BirdPowerDiscardFromHandDecision,
            ),
        ):
            verb = "Passes" if "pass" in decision.prompt.lower() else "Discards"
            return f"{verb} {choice.bird.name}"
        return f"Picks {choice.bird.name}"
    if isinstance(choice, decisions.SetupChoice):
        names = [bird.name for bird in choice.kept_cards] or ["no birds"]
        shown = names[:2]
        suffix = "…" if len(names) > 2 else ""
        return f"Sets up: {', '.join(shown)}{suffix}"
    if isinstance(choice, decisions.BoardTargetChoice):
        hab = _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
        player_id = decision.player_id
        if 0 <= player_id < len(gs.players):
            row = gs.players[player_id].board[choice.habitat]
            if choice.slot < len(row):
                bird_name = row[choice.slot].bird.name
                # Item 6: egg-cost removal gets a clearer header than "Targets".
                if isinstance(decision, decisions.RemoveEggDecision):
                    return f"Remove 1 egg from {bird_name}"
                return f"Targets {bird_name}"
        # Fallback: slot not found on the board.
        if isinstance(decision, decisions.RemoveEggDecision):
            return f"Remove 1 egg from {hab} slot {choice.slot + 1}"
        return f"Targets {hab}"
    if isinstance(choice, decisions.HabitatChoice):
        return _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
    if isinstance(choice, decisions.ResetBirdfeederChoice):
        return "Resets birdfeeder"
    if isinstance(choice, decisions.TuckActivateChoice):
        plural = "cards" if choice.cards_to_tuck != 1 else "card"
        return f"Tucks {choice.cards_to_tuck} {plural}"
    if isinstance(choice, decisions.PayCostChoice):
        return f"Accepts: {choice.label}"
    if isinstance(choice, decisions.FoodPaymentChoice):
        return f"Pays {choice.payment.format()}"
    if isinstance(choice, decisions.PlayedBirdChoice):
        return f"Uses {choice.played_bird.bird.name}"
    # Fallback: capitalize the choice label
    raw = humanize_choice(choice, gs, decision.player_id)
    return raw[:1].upper() + raw[1:] if raw else raw


def humanize_note(text: str) -> str:
    """Humanize a notification log line for the HTML decision panel.

    Strips the player-name prefix (``[Name] ``), applies pattern rewrites for
    common engine notifications, and sentence-cases anything unrecognized.
    Returns ``""`` for lines that are better omitted (row-count summaries,
    conversion echoes)."""

    # Strip leading [Name] prefix and whitespace.
    stripped = _PLAYER_PREFIX_RE.sub("", text).strip()

    # Bird play: "plays X into forest (paid seed, 0 eggs)"
    match = re.match(r"plays (.+?) into (\w+)\s*\(paid .+\)", stripped, re.IGNORECASE)
    if match:
        bird_name = match.group(1)
        hab_raw = match.group(2)
        hab_label = hab_raw[:1].upper() + hab_raw[1:]
        return f"Plays {bird_name} in {hab_label}"

    # Item 1: row-count lines — all three dropped (decisions already show outcome).
    if _ROW_HAS_RE.match(stripped):
        return ""

    # Drew specific card from deck: "drew from deck: Card Name"
    match = re.match(r"drew from deck: (.+)", stripped)
    if match:
        return f"Draws {match.group(1)} from the deck"

    # Gained food token: "+1 seed"
    match = re.match(r"\+1 (\w+)$", stripped)
    if match:
        return f"Gains {match.group(1)}"

    # Power activation: @ Bird - "power text"
    match = re.match(r'@ (.+?) - "(.+)"', stripped)
    if match:
        return f"{match.group(1)}: {match.group(2)}"

    # No-op power: "@ Bird - no brown power"
    match = re.match(r"@ (.+?) - no brown power", stripped)
    if match:
        return f"{match.group(1)}: no brown power"

    # Birdfeeder events
    stripped_lower = stripped.lower()
    if "birdfeeder" in stripped_lower and (
        "rerolled" in stripped_lower or "emptied" in stripped_lower
    ):
        return "Birdfeeder rerolled"
    if "resets the birdfeeder" in stripped_lower:
        return "Birdfeeder reset"

    # Item 9: conversion echoes — dropped, AcceptExchangeDecision "Accepts: …" already covers them.
    if re.match(r"convert: ", stripped):
        return ""

    # Extra play decisions
    if "declines the extra play" in stripped:
        return "Declines extra play"
    if "takes an EXTRA play" in stripped:
        return "Takes an extra play"
    if "has no playable bird" in stripped:
        return "No playable bird — wasted action"

    # Fallback: sentence-case the cleaned text.
    return stripped[:1].upper() + stripped[1:] if stripped else ""


def humanize_forced(label: str) -> str:
    """Humanize the ``display_label()`` substring from a forced single-choice decision.

    Common patterns from :class:`~wingspan.decisions.DrawSourceChoice`,
    :class:`~wingspan.decisions.BoardTargetChoice`, etc. are rewritten to
    human-friendly text; unrecognized labels are sentence-cased."""
    label = label.strip()

    # DrawSourceChoice deck: display_label() = "deck"
    if label == "deck":
        return "Draw from the deck"

    # DrawSourceChoice tray: display_label() = "tray[N]=Bird Name"
    match = re.match(r"tray\[\d+\]=(.+)", label)
    if match:
        return f"{match.group(1)} (tray)"

    # BoardTargetChoice: label built as "BirdName@habitat[slot]"
    match = re.match(r"(.+?)@(\w+)\[(\d+)\]", label)
    if match:
        bird_name = match.group(1)
        habitat = match.group(2)[:1].upper() + match.group(2)[1:]
        return f"{bird_name} ({habitat})"

    # Food value tokens (e.g. "seed", "fish") — renderer will apply emoji.
    if label in _FOOD_VALUES:
        return label

    # Generic fallback: sentence-case.
    return label[:1].upper() + label[1:] if label else label
