"""The per-card HTML renderer (``wingspan.research.card_html``).

Black-box over the rendered documents: every catalog card is written under the
id the stats CSVs use, costs render as AND / OR / wild, power text is escaped
and its icon tags become labels, both bonus payout shapes render, and the
``wingspan research cards-html`` CLI writes the files.
"""

from __future__ import annotations

import pathlib

from wingspan.cards import lookup, schema
from wingspan.cards.parse import catalog
from wingspan.research import app, card_html, constants


def _bird(name: str) -> schema.Bird:
    (bird,) = lookup.find_birds(name)
    return bird


def _bonus(name: str) -> schema.BonusCard:
    (bonus,) = lookup.find_bonus_cards(name)
    return bonus


def test_write_cards_writes_every_card_under_its_csv_id(tmp_path: pathlib.Path) -> None:
    written = card_html.write_cards(tmp_path)

    assert len(written) == catalog.n_birds() + catalog.n_bonus_cards()
    assert (tmp_path / constants.CARD_HTML_BIRD_DIR / "91.html").is_file()
    assert (tmp_path / constants.CARD_HTML_BONUS_DIR / "anatomist.html").is_file()
    raven = (tmp_path / constants.CARD_HTML_BIRD_DIR / "91.html").read_text("utf-8")
    assert "Common Raven" in raven and "Corvus corax" in raven


def test_card_is_fixed_size_and_self_contained() -> None:
    document = card_html.bird_card_html(_bird("Common Raven"))

    assert f"width:{constants.CARD_WIDTH_PX}px" in document
    assert f"height:{constants.CARD_HEIGHT_PX}px" in document
    assert "http://" not in document and "https://" not in document


def test_and_cost_shows_counts_and_or_cost_shows_alternatives() -> None:
    raven = card_html.bird_card_html(_bird("Common Raven"))
    robin = card_html.bird_card_html(_bird("American Robin"))

    assert "1 rodent" in raven and "2 wild" in raven
    assert (
        'class="sep"' in robin and "1 " not in robin.split("Cost")[1].split("</div>")[0]
    )


def test_power_text_tags_become_labels() -> None:
    document = card_html.bird_card_html(_bird("Common Raven"))

    assert '<span class="ic egg">egg</span>' in document
    assert "[egg]" not in document


def test_card_icon_class_does_not_collide_with_the_card_frame() -> None:
    # The ``[card]`` power icon renders as class "ic card"; the fixed-size frame
    # must therefore not be styled by a bare ``.card`` rule.
    document = card_html.bird_card_html(_bird("American Oystercatcher"))

    assert '<span class="ic card">card</span>' in document
    assert ".card{" not in document


def test_names_are_html_escaped() -> None:
    document = card_html.bird_card_html(_bird("Franklin's Gull"))

    assert "Franklin&#x27;s Gull" in document


def test_bonus_tiers_and_per_bird_payout() -> None:
    tiered = card_html.bonus_card_html(_bonus("Anatomist"))
    per_bird = card_html.bonus_card_html(_bonus("Bird Counter"))

    assert "2 to 3 birds" in tiered and "4+ birds" in tiered and "7 pts" in tiered
    assert "2 pts per bird" in per_bird


def test_bonus_slug_collapses_punctuation() -> None:
    assert card_html.bonus_slug("Large Bird Specialist") == "large-bird-specialist"


def test_cli_writes_cards(tmp_path: pathlib.Path) -> None:
    assert app.main(["cards-html", "--out", str(tmp_path)]) == 0

    assert (tmp_path / constants.CARD_HTML_BIRD_DIR / "127.html").is_file()
