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


def humanize_choice(
    choice: decisions.Choice,
    gs: state.GameState,
    player_id: int | None = None,
) -> str:
    """A concise human label for one offered option.

    ``player_id`` is used to resolve ``BoardTargetChoice`` board lookups — pass
    ``decision.player_id`` when available."""
    if isinstance(choice, decisions.DrawSourceChoice):
        if choice.source == "tray" and choice.bird is not None:
            return f"{choice.bird.name} (tray)"
        return "Draw from the deck"
    if isinstance(choice, decisions.PlayBirdChoice):
        hab = _HABITAT_LABELS.get(choice.habitat, choice.habitat.value)
        return f"{choice.bird.name} → {hab}"
    if isinstance(choice, decisions.FoodChoice):
        return choice.food.value  # renderer applies emoji
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
        return f"Gains {choice.food.value}"
    if isinstance(choice, decisions.SkipChoice):
        return "Declines"
    if isinstance(choice, decisions.BonusCardChoice):
        return f"Keeps {choice.bonus_card.name}"
    if isinstance(choice, decisions.BirdChoice):
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
                return f"Targets {row[choice.slot].bird.name}"
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
    common engine notifications, and sentence-cases anything unrecognized."""

    # Strip leading [Name] prefix and whitespace.
    stripped = _PLAYER_PREFIX_RE.sub("", text).strip()

    # Bird play: "plays X into forest (paid seed, 0 eggs)"
    match = re.match(r"plays (.+?) into (\w+)\s*\(paid .+\)", stripped, re.IGNORECASE)
    if match:
        bird_name = match.group(1)
        hab_raw = match.group(2)
        hab_label = hab_raw[:1].upper() + hab_raw[1:]
        return f"Plays {bird_name} in {hab_label}"

    # Egg lay: "lay eggs: row has N birds, lay M eggs"
    match = re.match(r"lay eggs: row has \d+ birds?, lay (\d+) eggs?", stripped)
    if match:
        count = match.group(1)
        plural = "eggs" if count != "1" else "egg"
        return f"Lays {count} {plural}"

    # Draw cards count: "draw cards: row has N birds, draw N cards"
    match = re.match(r"draw cards: row has \d+ birds?, draw (\d+) cards?", stripped)
    if match:
        count = match.group(1)
        plural = "cards" if count != "1" else "card"
        return f"Draws {count} {plural}"

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
        return f"{match.group(1)}: no power"

    # Birdfeeder events
    stripped_lower = stripped.lower()
    if "birdfeeder" in stripped_lower and (
        "rerolled" in stripped_lower or "emptied" in stripped_lower
    ):
        return "Birdfeeder rerolled"
    if "resets the birdfeeder" in stripped_lower:
        return "Birdfeeder reset"

    # Conversions
    match = re.match(r"convert: discard (.+?) for \+1 food", stripped)
    if match:
        return f"Converted {match.group(1)} for food"
    match = re.match(r"convert: spend (.+?) for \+1 egg", stripped)
    if match:
        return f"Converted {match.group(1)} for egg"
    if re.match(r"convert: discard 1 egg for \+1 card", stripped):
        return "Converted egg for card"

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
