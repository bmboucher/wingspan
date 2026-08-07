"""Fuzzy name -> card resolution for interactive entry points.

Wraps the core-set catalog (:mod:`wingspan.cards.parse.catalog` for birds and
bonus cards, :mod:`wingspan.cards.parse.loader` for goals) with a three-tier
match: exact normalized name, then unique/ambiguous normalized-prefix, then
``difflib`` fuzzy candidates. Every lookup returns a list of candidates so a
caller can distinguish "no match" (empty), "resolved" (one element), and
"ambiguous" (multiple) without a separate sentinel.

Goals have no printed name, so :func:`find_goals` matches query tokens against
each goal's category tag and description instead. :func:`find_food` is a plain
alias table (no fuzzy tier) since the five foods and their short abbreviations
are a closed, tiny set.
"""

from __future__ import annotations

import difflib
import functools
import re

from wingspan.cards import schema
from wingspan.cards.parse import catalog, loader

# Minimum similarity ratio difflib.get_close_matches requires between the
# normalized query and a candidate name for a fuzzy match to count.
_FUZZY_CUTOFF = 0.6

# Maximum number of fuzzy candidates difflib.get_close_matches returns.
_FUZZY_MAX_CANDIDATES = 5

# Every character difflib.get_close_matches would otherwise stumble over
# (punctuation, apostrophes, hyphens) is folded to a space; runs of
# whitespace then collapse to one before casefolding.
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Goal descriptions spell out their category with bracketed icon words, e.g.
# "[egg] in [cavity]" -- the brackets are stripped to fold the description
# into the same bag-of-words search text as the category tag.
_BRACKET_PATTERN = re.compile(r"[\[\]]")

# Food name/alias -> Food. Deliberately excludes the bare "f", which is
# ambiguous between fish and fruit; callers must spell out enough to
# disambiguate.
_FOOD_ALIASES: dict[str, schema.Food] = {
    "invertebrate": schema.Food.INVERTEBRATE,
    "inv": schema.Food.INVERTEBRATE,
    "i": schema.Food.INVERTEBRATE,
    "seed": schema.Food.SEED,
    "s": schema.Food.SEED,
    "fish": schema.Food.FISH,
    "fi": schema.Food.FISH,
    "fruit": schema.Food.FRUIT,
    "fr": schema.Food.FRUIT,
    "berry": schema.Food.FRUIT,
    "rodent": schema.Food.RODENT,
    "r": schema.Food.RODENT,
    "rod": schema.Food.RODENT,
}


def find_birds(query: str) -> list[schema.Bird]:
    """Resolve ``query`` to bird candidates: exact normalized name match,
    else unique/ambiguous normalized-prefix matches, else fuzzy ``difflib``
    candidates over all 180 core-set bird names.

    Empty list means no match; a single element means resolved; multiple
    elements means ambiguous (the caller disambiguates)."""
    return _resolve_by_name(query, _bird_name_index())


def find_bonus_cards(query: str) -> list[schema.BonusCard]:
    """Resolve ``query`` to bonus-card candidates using the same three-tier
    scheme as :func:`find_birds`, over the 26 core-set bonus-card names."""
    return _resolve_by_name(query, _bonus_card_name_index())


def find_goals(query: str) -> list[schema.EndRoundGoal]:
    """Resolve ``query`` to end-of-round-goal candidates.

    Goals have no printed name, so the match target is each goal's category
    tag plus its description, folded into one bag of words (e.g. category
    ``eggs_cavity`` and description ``"[egg] in [cavity]"`` both contribute
    to matching "eggs cavity" or "egg cavity"). A goal matches when every
    normalized query token appears in its bag of words; ambiguous or
    unmatched queries fall back to fuzzy matching over the full search text.
    Same list-result contract as :func:`find_birds`."""
    query_tokens = _normalize(query).split()
    if not query_tokens:
        return []
    entries = _goal_search_index()
    token_matches = [
        goal
        for goal, search_text in entries
        if all(token in search_text.split() for token in query_tokens)
    ]
    if token_matches:
        return token_matches
    search_texts = [search_text for _, search_text in entries]
    fuzzy_texts = difflib.get_close_matches(
        _normalize(query), search_texts, n=_FUZZY_MAX_CANDIDATES, cutoff=_FUZZY_CUTOFF
    )
    return [goal for goal, search_text in entries if search_text in fuzzy_texts]


def find_food(query: str) -> schema.Food | None:
    """Resolve ``query`` to a single :class:`schema.Food` via the
    :data:`_FOOD_ALIASES` table (full names plus short aliases). Returns
    ``None`` for anything not in the table, including the bare "f", which is
    ambiguous between fish and fruit."""
    return _FOOD_ALIASES.get(_normalize(query))


def parse_bird_list(text: str) -> list[list[schema.Bird]]:
    """Split ``text`` on commas and resolve each token via :func:`find_birds`.

    Returns one candidate list per comma-delimited token, in order; the
    caller disambiguates any token whose candidate list is empty (no match)
    or has more than one element (ambiguous)."""
    return [find_birds(token) for token in text.split(",")]


###### PRIVATE #######


def _normalize(text: str) -> str:
    """Casefold ``text``, strip punctuation/apostrophes/hyphens, and collapse
    whitespace -- the normalization every lookup in this module matches on."""
    stripped = _PUNCTUATION_PATTERN.sub(" ", text)
    collapsed = _WHITESPACE_PATTERN.sub(" ", stripped)
    return collapsed.strip().casefold()


def _goal_search_text(goal: schema.EndRoundGoal) -> str:
    """Normalized bag-of-words search text for one goal: its category tag
    (underscores as spaces) plus its bracket-stripped description, e.g.
    category ``"eggs_cavity"`` + description ``"[egg] in [cavity]"``
    normalizes to ``"eggs cavity egg in cavity"``."""
    plain_description = _BRACKET_PATTERN.sub(" ", goal.description)
    return _normalize(f"{goal.category.replace('_', ' ')} {plain_description}")


def _resolve_by_name[T](query: str, index: dict[str, T]) -> list[T]:
    """Three-tier name resolution shared by :func:`find_birds` and
    :func:`find_bonus_cards`: exact normalized match, else normalized-prefix
    matches (returned whether unique or ambiguous), else fuzzy candidates
    over every normalized name in ``index``."""
    normalized_query = _normalize(query)
    if not normalized_query:
        return []
    exact = index.get(normalized_query)
    if exact is not None:
        return [exact]
    prefix_matches = [
        item for name, item in index.items() if name.startswith(normalized_query)
    ]
    if prefix_matches:
        return prefix_matches
    fuzzy_names = difflib.get_close_matches(
        normalized_query, index.keys(), n=_FUZZY_MAX_CANDIDATES, cutoff=_FUZZY_CUTOFF
    )
    return [index[name] for name in fuzzy_names]


@functools.lru_cache(maxsize=1)
def _bird_name_index() -> dict[str, schema.Bird]:
    """Normalized bird name -> Bird, built once from the full catalog."""
    return {_normalize(bird.name): bird for bird in catalog.birds_ordered()}


@functools.lru_cache(maxsize=1)
def _bonus_card_name_index() -> dict[str, schema.BonusCard]:
    """Normalized bonus-card name -> BonusCard, built once from the full
    catalog."""
    return {
        _normalize(bonus_card.name): bonus_card
        for bonus_card in catalog.bonus_cards_ordered()
    }


@functools.lru_cache(maxsize=1)
def _goal_search_index() -> tuple[tuple[schema.EndRoundGoal, str], ...]:
    """``(goal, normalized search text)`` pairs, built once from the full
    catalog."""
    _, _, goals = loader.load_all()
    return tuple((goal, _goal_search_text(goal)) for goal in goals)
