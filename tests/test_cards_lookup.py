"""Tests for wingspan.cards.lookup: fuzzy name -> card resolution.

Exercises the three-tier match (exact normalized name -> normalized-prefix
-> difflib fuzzy) over the real bundled catalog for birds and bonus cards,
the token-based goal match, the food alias table, and the comma-separated
``parse_bird_list`` helper.
"""

from __future__ import annotations

from wingspan.cards import lookup, schema


def _names(matches: list[schema.Bird] | list[schema.BonusCard]) -> list[str]:
    return [match.name for match in matches]


# ---------------------------------------------------------------------------
# find_birds


def test_find_birds_exact_match() -> None:
    matches = lookup.find_birds("Barn Owl")
    assert _names(matches) == ["Barn Owl"]


def test_find_birds_case_and_punctuation_insensitive() -> None:
    matches = lookup.find_birds("  BARN-owl!! ")
    assert _names(matches) == ["Barn Owl"]


def test_find_birds_unique_prefix_resolves() -> None:
    matches = lookup.find_birds("Barn O")
    assert _names(matches) == ["Barn Owl"]


def test_find_birds_ambiguous_prefix_returns_multiple() -> None:
    matches = lookup.find_birds("Barn")
    assert set(_names(matches)) == {"Barn Owl", "Barn Swallow"}


def test_find_birds_fuzzy_typo_resolution() -> None:
    # "Barn Owel" is not a prefix of any bird name, so this exercises the
    # fuzzy tier; "Barn Owl" (one edit away) must be among the candidates.
    matches = lookup.find_birds("Barn Owel")
    assert "Barn Owl" in _names(matches)


def test_find_birds_no_match_returns_empty() -> None:
    assert lookup.find_birds("Zzzznotabirdatall") == []


def test_find_birds_empty_query_returns_empty() -> None:
    assert lookup.find_birds("   ") == []


# ---------------------------------------------------------------------------
# find_bonus_cards


def test_find_bonus_cards_exact_match() -> None:
    matches = lookup.find_bonus_cards("Anatomist")
    assert _names(matches) == ["Anatomist"]


def test_find_bonus_cards_unique_prefix_resolves() -> None:
    matches = lookup.find_bonus_cards("Bird C")
    assert _names(matches) == ["Bird Counter"]


def test_find_bonus_cards_ambiguous_prefix_returns_multiple() -> None:
    matches = lookup.find_bonus_cards("Bird")
    assert set(_names(matches)) == {"Bird Counter", "Bird Feeder"}


def test_find_bonus_cards_fuzzy_typo_resolution() -> None:
    matches = lookup.find_bonus_cards("Antomist")
    assert "Anatomist" in _names(matches)


def test_find_bonus_cards_no_match_returns_empty() -> None:
    assert lookup.find_bonus_cards("Zzzznotabonuscard") == []


# ---------------------------------------------------------------------------
# find_goals


def test_find_goals_by_category_words() -> None:
    matches = lookup.find_goals("birds forest")
    assert len(matches) == 1
    assert matches[0].category == "birds_forest"


def test_find_goals_by_description_words() -> None:
    # "bird" and "egg" (singular) only appear together, alongside "bowl", in
    # the bowl_birds_with_eggs description -- exercises matching against the
    # description rather than the category tag.
    matches = lookup.find_goals("bowl bird egg")
    assert len(matches) == 1
    assert matches[0].category == "bowl_birds_with_eggs"


def test_find_goals_category_word_alone_is_ambiguous() -> None:
    # Every "eggs_<habitat>" goal's category contains the "eggs" token.
    matches = lookup.find_goals("eggs")
    assert len(matches) > 1


def test_find_goals_no_match_returns_empty() -> None:
    assert lookup.find_goals("zzzznotagoal") == []


def test_find_goals_empty_query_returns_empty() -> None:
    assert lookup.find_goals("   ") == []


# ---------------------------------------------------------------------------
# find_food


def test_find_food_full_names() -> None:
    assert lookup.find_food("invertebrate") == schema.Food.INVERTEBRATE
    assert lookup.find_food("seed") == schema.Food.SEED
    assert lookup.find_food("fish") == schema.Food.FISH
    assert lookup.find_food("fruit") == schema.Food.FRUIT
    assert lookup.find_food("rodent") == schema.Food.RODENT


def test_find_food_short_aliases() -> None:
    assert lookup.find_food("i") == schema.Food.INVERTEBRATE
    assert lookup.find_food("inv") == schema.Food.INVERTEBRATE
    assert lookup.find_food("s") == schema.Food.SEED
    assert lookup.find_food("fi") == schema.Food.FISH
    assert lookup.find_food("fr") == schema.Food.FRUIT
    assert lookup.find_food("berry") == schema.Food.FRUIT
    assert lookup.find_food("r") == schema.Food.RODENT
    assert lookup.find_food("rod") == schema.Food.RODENT


def test_find_food_bare_f_is_ambiguous() -> None:
    assert lookup.find_food("f") is None


def test_find_food_unknown_returns_none() -> None:
    assert lookup.find_food("nonsense") is None


def test_find_food_case_insensitive() -> None:
    assert lookup.find_food("  FISH  ") == schema.Food.FISH


# ---------------------------------------------------------------------------
# parse_bird_list


def test_parse_bird_list_resolves_ambiguous_and_misspelled_tokens() -> None:
    results = lookup.parse_bird_list(
        "Barn Owl, Blue Jay, Barn, Barn Owel, American Crow"
    )
    assert len(results) == 5
    assert _names(results[0]) == ["Barn Owl"]  # exact
    assert _names(results[1]) == ["Blue Jay"]  # exact
    assert set(_names(results[2])) == {"Barn Owl", "Barn Swallow"}  # ambiguous
    assert "Barn Owl" in _names(results[3])  # misspelled -> fuzzy candidates
    assert _names(results[4]) == ["American Crow"]  # exact
