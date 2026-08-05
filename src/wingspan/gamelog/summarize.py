"""Fold an event's recorded effect ledger into one-line display text.

:mod:`wingspan.engine.ledger` records *what happened*, one entry per state
mutation.  This module turns a subtree of those records into the header line a
collapsed event shows, so both renderers — :mod:`wingspan.gamelog.render_text`
and :mod:`wingspan.reporting.game_log_capture` — say the same thing about the
same event.

Public API: :func:`summarize` (the typed aggregate), :func:`effect_phrase` (its
prose form), :func:`summary_text` (one event's header line), and the reveal pair
:func:`is_reveal` / :func:`reveal_text` — the hidden-information effects that
earn their own row in the log instead of folding silently into a header.

Food is rendered as repeated whole words (``"fish fish"``) rather than a count
prefix, because the HTML log's ``applyFoodEmoji`` substitutes on word
boundaries: ``"fish fish"`` becomes two fish glyphs, while ``"2fish"`` would
match nothing and print literally.

**Import discipline:** standard library, ``pydantic``, and
:mod:`wingspan.gamelog.models` only — the same rule the models module follows,
so the reporting layer can import this without closing an import cycle.
"""

from __future__ import annotations

import pydantic

from wingspan.gamelog import models

# Repeat a food word up to this many times ("fish fish fish") before switching
# to a multiplier ("4x fish"); repetition reads as tokens on the table, but
# stops being scannable past a handful.
_MAX_REPEATED_FOOD_WORDS = 3

# Shown in place of a phrase when an event recorded no effects at all.
_NO_EFFECT = "no effect"

# Header labels for the top-of-turn action pick, keyed by ``MainAction`` value.
# The three habitat actions name their habitat, since the row they activate is
# the thing a reader is looking for.
_MAIN_ACTION_LABELS: dict[str, str] = {
    "gain_food": "Forest (gain food)",
    "lay_eggs": "Grassland (lay eggs)",
    "draw_cards": "Wetland (draw cards)",
    "play_bird": "Play a bird",
}

# Fallback labels for a habitat activation that produced nothing, keyed by the
# same ``MainAction`` values (``ActivateBaseEvent.action`` shares the vocabulary).
_BASE_ACTION_LABELS: dict[str, str] = {
    "gain_food": "Gain food",
    "lay_eggs": "Lay eggs",
    "draw_cards": "Draw cards",
}

# Card sources whose arrival discloses a card nobody could see beforehand.
_HIDDEN_CARD_SOURCES = (models.CardSource.DECK, models.CardSource.DEAL)


class EventSummary(pydantic.BaseModel):
    """Everything one event's subtree did, aggregated by kind.

    Food maps are keyed by food-type name and preserve the order the tokens were
    recorded in, so a phrase built from one reads chronologically.  The three
    effect-typed lists keep whole records rather than bare names because their
    headers need a second field (a played bird's habitat, a refilled tray slot).
    """

    food_gained: dict[str, int] = {}
    food_spent: dict[str, int] = {}
    food_cached: dict[str, int] = {}
    food_uncached: dict[str, int] = {}
    eggs_laid: int = 0
    eggs_removed: int = 0
    cards_drawn: list[str] = []
    cards_tucked: list[str] = []
    cards_discarded: list[str] = []
    cards_passed: list[str] = []
    birds_played: list[models.PlayBirdEffect] = []
    birds_moved: list[models.MoveBirdEffect] = []
    tray_refills: list[models.TrayRefillEffect] = []
    dice_faces: list[str] = []

    @property
    def is_empty(self) -> bool:
        """True when the event's subtree recorded no effect at all."""
        return self == EventSummary()


def summarize(event: models.GameEvent) -> EventSummary:
    """Fold every effect in ``event``'s subtree that belongs to its own seat.

    A descendant event belonging to another seat — a pink reaction nested under
    the play that triggered it — is skipped along with its whole subtree: those
    effects are that seat's, and rolling them into this header would attribute
    another player's food gain to the acting player."""
    summary = EventSummary()
    _fold_event(event, event.player_id, summary)
    return summary


def summary_text(event: models.GameEvent) -> str:
    """The one-line header a collapsed ``event`` shows in either renderer.

    Derived from what the event's ledger records say happened, not from the
    card's printed power text — so a power that fizzled reads as having fizzled
    rather than repeating what it was supposed to do."""
    if isinstance(event, models.SetupEvent):
        return _setup_text(event)
    if isinstance(event, models.RoundGoalEvent):
        return _round_goal_text(event)
    if isinstance(event, models.FinalScoringEvent):
        return _final_scoring_text(event)

    summary = summarize(event)
    phrase = effect_phrase(summary) or _last_resolution(event)

    if isinstance(event, models.MainActionEvent):
        label = _MAIN_ACTION_LABELS.get(event.action or "")
        return f"Main action: {label}" if label else "Main action"
    if isinstance(event, models.ActivateBaseEvent):
        idle = _BASE_ACTION_LABELS.get(event.action, event.action)
        return phrase or f"{idle} — {_NO_EFFECT}"
    if isinstance(event, models.ActivateBrownEvent):
        if not event.is_brown:
            return f"{event.bird_name} — no brown power"
        return _power_text(event.bird_name, "brown", phrase)
    if isinstance(event, models.WhitePowerEvent):
        return _power_text(event.bird_name, "white", phrase)
    if isinstance(event, models.ReactionEvent):
        return _power_text(event.bird_name, "pink", phrase)
    if isinstance(event, models.PlayBirdEvent):
        return _play_bird_text(summary)
    if isinstance(event, models.ExtraPlayEvent):
        where = f" ({event.habitat.capitalize()})" if event.habitat else ""
        return f"Extra play{where}"
    if isinstance(event, models.TurnEndEvent):
        return f"End of turn: {phrase}" if phrase else "End of turn"
    if isinstance(event, models.RefillTrayEvent):
        names = [refill.card for refill in summary.tray_refills]
        return f"Tray refill: {', '.join(names)}" if names else "Tray refill"
    if isinstance(event, models.DealEvent):
        return f"Deal ({len(summary.cards_drawn)} cards)"
    return phrase or "Other"


def effect_phrase(summary: EventSummary) -> str:
    """Fold ``summary`` into a comma-joined clause list, first letter capitalized.

    Clause order follows the order the game does things — what was played and
    laid, then what moved between zones, then food, then dice — so a
    tuck-to-draw power reads ``"Tucks Black-Billed Magpie, draws Wood Stork"``
    rather than in reverse.  Returns ``""`` for an empty summary."""
    clauses: list[str] = []

    # Board changes first: what the player committed to the table.
    if summary.birds_played:
        clauses.append("plays " + ", ".join(bird.card for bird in summary.birds_played))
    clauses.extend(
        f"moves {move.card} to {move.to_habitat.capitalize()}"
        for move in summary.birds_moved
    )
    if summary.eggs_laid:
        clauses.append(f"lays {_count_noun(summary.eggs_laid, 'egg')}")
    if summary.eggs_removed:
        clauses.append(f"removes {_count_noun(summary.eggs_removed, 'egg')}")

    # Cards moving between zones, in the order a tuck-to-draw power does them.
    if summary.cards_tucked:
        clauses.append("tucks " + ", ".join(summary.cards_tucked))
    if summary.cards_drawn:
        clauses.append("draws " + ", ".join(summary.cards_drawn))
    if summary.cards_discarded:
        clauses.append("discards " + ", ".join(summary.cards_discarded))
    if summary.cards_passed:
        clauses.append("passes " + ", ".join(summary.cards_passed))

    # Food, then the reveals that are not cards.
    if summary.food_gained:
        clauses.append("gains " + food_words(summary.food_gained))
    if summary.food_spent:
        clauses.append("spends " + food_words(summary.food_spent))
    if summary.food_cached:
        clauses.append("caches " + food_words(summary.food_cached))
    if summary.food_uncached:
        clauses.append("takes " + food_words(summary.food_uncached) + " from cache")
    if summary.dice_faces:
        clauses.append("rolls " + " ".join(summary.dice_faces))
    if summary.tray_refills:
        clauses.append(
            "refills tray with " + ", ".join(item.card for item in summary.tray_refills)
        )

    if not clauses:
        return ""
    joined = ", ".join(clauses)
    return joined[0].upper() + joined[1:]


def food_words(counts: dict[str, int]) -> str:
    """Food tokens as space-separated whole words, e.g. ``"fish fish seed"``.

    Whole words (rather than the engine log's compact ``2fish``) are what the
    HTML log's word-boundary emoji substitution can match."""
    parts: list[str] = []
    for food, count in counts.items():
        if count <= 0:
            continue
        if count <= _MAX_REPEATED_FOOD_WORDS:
            parts.extend([food] * count)
        else:
            # The space after the multiplier keeps the food word on its own
            # boundary, so the emoji substitution still fires.
            parts.append(f"{count}x {food}")
    return " ".join(parts)


def is_reveal(effect: models.Effect) -> bool:
    """Whether ``effect`` disclosed information nobody could see beforehand.

    Reveals earn their own row in the decision log: they are the only record of
    a card that went deck-to-hand or deck-to-tuck, of fresh birdfeeder faces, of
    a predator's dice, or of what restocked a tray slot.  Every other effect
    folds silently into its event's header."""
    if isinstance(effect, models.DrawCardEffect):
        return effect.source in _HIDDEN_CARD_SOURCES
    if isinstance(effect, models.TuckCardEffect):
        return effect.source is models.CardSource.DECK
    return isinstance(
        effect,
        (
            models.FeederRerollEffect,
            models.DiceRollEffect,
            models.TrayRefillEffect,
        ),
    )


def reveal_text(effect: models.Effect) -> str:
    """The row text for a reveal effect; ``""`` for anything not a reveal."""
    if isinstance(effect, models.DrawCardEffect):
        origin = "the deck" if effect.source is models.CardSource.DECK else "the deal"
        return f"Draws {effect.card} from {origin}"
    if isinstance(effect, models.TuckCardEffect):
        return f"Tucks {effect.card} from the deck behind {effect.bird}"
    if isinstance(effect, models.FeederRerollEffect):
        return "Birdfeeder rerolled: " + " ".join(effect.faces)
    if isinstance(effect, models.DiceRollEffect):
        return f"{effect.bird} rolls: " + " ".join(effect.faces)
    if isinstance(effect, models.TrayRefillEffect):
        return f"Tray slot {effect.slot + 1}: {effect.card}"
    return ""


###### PRIVATE #######


#### Folding ####


def _fold_event(
    event: models.GameEvent, player_id: int | None, summary: EventSummary
) -> None:
    """Recursively accumulate ``player_id``'s effects from ``event``'s subtree."""
    for sub in event.sub_events:
        if isinstance(sub, models.Effect) and _same_seat(sub.player_id, player_id):
            _fold_effect(sub, summary)
    for child in event.children:
        if _same_seat(child.player_id, player_id):
            _fold_event(child, player_id, summary)


def _same_seat(node_player_id: int | None, player_id: int | None) -> bool:
    """Whether a node belongs to ``player_id``; a seatless node belongs to all."""
    return node_player_id is None or player_id is None or node_player_id == player_id


def _fold_effect(effect: models.Effect, summary: EventSummary) -> None:
    """Accumulate one recorded mutation into ``summary``."""
    if isinstance(effect, models.GainFoodEffect):
        _add_food(summary.food_gained, effect.food, effect.amount)
    elif isinstance(effect, models.SpendFoodEffect):
        _add_food(summary.food_spent, effect.food, effect.amount)
    elif isinstance(effect, models.CacheFoodEffect):
        _add_food(summary.food_cached, effect.food, effect.amount)
    elif isinstance(effect, models.UncacheFoodEffect):
        _add_food(summary.food_uncached, effect.food, effect.amount)
    elif isinstance(effect, models.LayEggEffect):
        summary.eggs_laid += effect.count
    elif isinstance(effect, models.RemoveEggEffect):
        summary.eggs_removed += effect.count
    elif isinstance(effect, models.DrawCardEffect):
        summary.cards_drawn.append(effect.card)
    elif isinstance(effect, models.TuckCardEffect):
        summary.cards_tucked.append(effect.card)
    elif isinstance(effect, models.DiscardCardEffect):
        summary.cards_discarded.append(effect.card)
    elif isinstance(effect, models.PassCardEffect):
        summary.cards_passed.append(effect.card)
    elif isinstance(effect, models.PlayBirdEffect):
        summary.birds_played.append(effect)
    elif isinstance(effect, models.MoveBirdEffect):
        summary.birds_moved.append(effect)
    elif isinstance(effect, models.TrayRefillEffect):
        summary.tray_refills.append(effect)
    elif isinstance(effect, (models.FeederRerollEffect, models.DiceRollEffect)):
        summary.dice_faces.extend(effect.faces)


def _add_food(pool: dict[str, int], food: str, amount: int) -> None:
    """Accumulate ``amount`` of ``food`` into a summary's food map."""
    pool[food] = pool.get(food, 0) + amount


def _last_resolution(event: models.GameEvent) -> str:
    """The outcome text of the last decision or forced move ``event`` resolved.

    The fallback for an event that changed nothing but was still *about*
    something — a power the player declined, a hunt that missed.  Without it
    such an event reads as ``no effect``, which hides that a choice was made."""
    for sub in reversed(event.sub_events):
        if isinstance(sub, models.ResolvedSubEvent):
            return sub.outcome_text
    return ""


#### Per-event-type headers ####


def _power_text(bird_name: str, color: str, phrase: str) -> str:
    """Header for a bird-power activation, tagged with its power color.

    The color is spelled out rather than left to styling because the plaintext
    log has no color, and brown / white / pink is how the game names the
    difference between an activated, a played, and a reactive power."""
    if not phrase:
        return f"{bird_name} ({color}) — {_NO_EFFECT}"
    return f"{bird_name} ({color}): {phrase}"


def _play_bird_text(summary: EventSummary) -> str:
    """Header for a bird being played: which bird, into which habitat."""
    if not summary.birds_played:
        return f"Plays a bird — {_NO_EFFECT}"
    placements = [
        f"{bird.card} in {bird.habitat.capitalize()}" for bird in summary.birds_played
    ]
    return "Plays " + ", ".join(placements)


def _setup_text(event: models.SetupEvent) -> str:
    """Header for a setup phase, listing kept cards and bonus."""
    parts: list[str] = []
    if event.kept_card_names:
        parts.append("kept: " + ", ".join(event.kept_card_names))
    if event.kept_bonus_name:
        parts.append("bonus: " + event.kept_bonus_name)
    detail = f" ({'; '.join(parts)})" if parts else ""
    return f"Setup{detail}"


def _round_goal_text(event: models.RoundGoalEvent) -> str:
    """Header for a round goal scoring, with per-seat counts and VP."""
    seat_parts = [
        f"P{seat_idx}: {count}/{vp}VP"
        for seat_idx, (count, vp) in enumerate(zip(event.counts, event.vps))
    ]
    detail = f" [{', '.join(seat_parts)}]" if seat_parts else ""
    return f"Round {event.round_idx + 1} goal — {event.description}{detail}"


def _final_scoring_text(event: models.FinalScoringEvent) -> str:
    """Header for the final scoring event, with per-seat totals."""
    totals = [str(score.total) for score in event.scores]
    return f"Final scoring [{', '.join(totals)}]"


def _count_noun(count: int, noun: str) -> str:
    """``"1 egg"`` / ``"2 eggs"`` — naive pluralization for the counted nouns here."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
