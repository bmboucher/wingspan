"""Tests for ``wingspan.aid.advisor`` -- the seat-0 agent.

No torch: the inner factory agent is a scripted stub that mimics only the
side effect the real model agent has on the ``DecisionProbe`` (writing a
value / policy annotation during its forward pass), never an actual network.
"""

from __future__ import annotations

import typing

import pytest

import aid_helpers
from wingspan import cards, decisions, state
from wingspan.agents import cli as agents_cli
from wingspan.aid import advisor, console, models, placeholders
from wingspan.aid import preview as preview_module
from wingspan.cards.parse import catalog
from wingspan.engine import core as engine_core
from wingspan.players import decision_probe

_SCORE_NORM = 50.0


def _make_inner(
    probe: decision_probe.DecisionProbe,
    value: float | None,
    annotation: decision_probe.PolicyAnnotation | None,
) -> engine_core.Agent:
    """A stub inner agent mimicking the factory model agent: writes to
    ``probe`` (as the real net would during its forward pass) and returns an
    arbitrary pick the advisor discards."""

    def inner[C: decisions.Choice](
        _engine: engine_core.Engine, decision: decisions.Decision[C]
    ) -> C:
        if value is not None:
            probe.record(value)
        if annotation is not None:
            probe.record_policy(annotation)
        return decision.choices[0]

    return inner


def _main_action_decision() -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=0,
        prompt="choose your main action",
        choices=[
            decisions.MainActionChoice(
                label="gain food", action=decisions.MainAction.GAIN_FOOD
            ),
            decisions.MainActionChoice(
                label="lay eggs", action=decisions.MainAction.LAY_EGGS
            ),
            decisions.MainActionChoice(
                label="draw cards", action=decisions.MainAction.DRAW_CARDS
            ),
        ],
    )


def _build(
    answers: list[str],
) -> tuple[
    engine_core.Engine,
    console.Console,
    list[str],
    decision_probe.DecisionProbe,
    placeholders.PlaceholderRegistry,
]:
    eng, *_ = engine_core.Engine.create(seed=11)
    con, transcript = aid_helpers.scripted_console(answers)
    probe = decision_probe.DecisionProbe()
    registry = placeholders.PlaceholderRegistry()
    return eng, con, transcript, probe, registry


# ---------------------------------------------------------------------------
# Ranked display, default/override, probe write-back, value readout


def test_ranked_display_shows_header_and_ranked_percentages() -> None:
    eng, con, transcript, probe, registry = _build([""])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    agent(eng, _main_action_decision())

    # Header: category (derived from the decision prompt) plus the top
    # pick's bare label, no bracket index.
    assert any(line == "Choose your main action: gain food" for line in transcript)
    # Ranked list: index-labeled so it stays typeable back into "your actual
    # move", percentage rounded to a whole number.
    assert any(line.strip() == "0: gain food    60%" for line in transcript)


def test_enter_resolves_to_argmax_choice() -> None:
    eng, con, _, probe, registry = _build([""])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)
    decision = _main_action_decision()

    chosen = agent(eng, decision)

    assert chosen == decision.choices[0]


def test_explicit_index_overrides_the_model_pick() -> None:
    eng, con, _, probe, registry = _build(["2"])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)
    decision = _main_action_decision()

    chosen = agent(eng, decision)

    assert chosen == decision.choices[2]


def test_probe_write_back_carries_the_corrected_chosen_idx() -> None:
    eng, con, _, probe, registry = _build(["2"])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)
    decision = _main_action_decision()

    agent(eng, decision)
    value, written_annotation = probe.take()

    assert value == 0.2
    assert written_annotation is not None
    assert written_annotation.chosen_idx == 2


def test_no_vp_expected_margin_line_is_printed() -> None:
    """The per-decision VP-margin readout was dropped from the condensed
    recommendation block -- guard against it creeping back in."""
    eng, con, transcript, probe, registry = _build([""])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    agent(eng, _main_action_decision())

    assert not any("VP expected margin" in line for line in transcript)


# ---------------------------------------------------------------------------
# Placeholder sweep


def test_placeholder_sweep_identifies_and_rewrites_choices() -> None:
    eng, con, _, probe, registry = _build([catalog.birds_ordered()[5].name, ""])
    echo = console.LogEcho(con)
    inner = _make_inner(probe, None, None)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    placeholder = registry.mint_bird()
    other_real_bird = catalog.birds_ordered()[7]
    eng.state.players[0].hand = [placeholder]
    decision = decisions.BirdPowerPickBirdFromHandDecision(
        player_id=0,
        prompt="pick a bird",
        choices=[
            decisions.BirdChoice(label="face-down", bird=placeholder),
            decisions.BirdChoice(label="other", bird=other_real_bird),
        ],
    )

    chosen = agent(eng, decision)

    real_bird = catalog.birds_ordered()[5]
    assert eng.state.players[0].hand == [real_bird]
    assert not registry.is_placeholder(eng.state.players[0].hand[0])
    assert decision.choices[0].bird is real_bird
    assert chosen.bird is real_bird


# ---------------------------------------------------------------------------
# Setup path


def test_setup_path_uses_resolve_setup_choice_dialog_and_writes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, con, _, probe, registry = _build([])
    echo = console.LogEcho(con)
    dealt_cards = list(catalog.birds_ordered()[0:2])
    # kept_foods populated to the real "food included" invariant (size
    # len(ALL_FOODS) - len(kept_cards)) on every choice, so ask_food reads
    # True and this combined (non-split) fixture never engages the
    # deferred-food preview path -- this test is about the dialog/write-back
    # plumbing, not split-axis rendering (see the split-axes tests below).
    choice_a = decisions.SetupChoice(
        label="a",
        kept_cards=(dealt_cards[0],),
        kept_foods=tuple(cards.ALL_FOODS[:4]),
        bonus_card=None,
    )
    choice_b = decisions.SetupChoice(
        label="b", kept_cards=(), kept_foods=tuple(cards.ALL_FOODS), bonus_card=None
    )
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose starting hand",
        choices=[choice_a, choice_b],
        dealt_cards=dealt_cards,
        dealt_bonus=[],
    )
    annotation = decision_probe.PolicyAnnotation(probs=[0.7, 0.3], chosen_idx=0)
    inner = _make_inner(probe, 0.4, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    def fake_dialog(
        given_decision: decisions.SetupDecision, tray: list[typing.Any]
    ) -> decisions.SetupChoice:
        assert given_decision is decision
        return choice_b

    monkeypatch.setattr(agents_cli, "resolve_setup_choice_dialog", fake_dialog)

    chosen = agent(eng, decision)
    value, written_annotation = probe.take()

    assert chosen == choice_b
    assert value == 0.4
    assert written_annotation is not None
    assert written_annotation.chosen_idx == 1


# ---------------------------------------------------------------------------
# preview_setup -- the combined upfront setup-keep recommendation


def _dealt_setup_state(
    seed: int = 3,
) -> tuple[engine_core.Engine, list[cards.Bird], list[cards.BonusCard]]:
    """A fresh engine with seat 0 in a faithful post-deal setup state: real
    birds popped off the shuffled deck into the hand, real bonus cards popped
    off the bonus deck, and one of each food -- mirroring what
    ``Engine._deal_setup_inputs`` leaves in place before a ``SetupDecision``
    is offered, without going through that private method."""
    eng, *_ = engine_core.Engine.create(seed=seed)
    player = eng.state.players[0]
    dealt_cards = [eng.state.bird_deck.pop() for _ in range(state.STARTING_HAND_SIZE)]
    player.hand = list(dealt_cards)
    dealt_bonus = [
        eng.state.bonus_deck.pop() for _ in range(state.STARTING_BONUS_CARDS_DEAL)
    ]
    for food in cards.ALL_FOODS:
        player.food[food] = 1
    return eng, dealt_cards, dealt_bonus


def _greedy_preview_inner(probe: decision_probe.DecisionProbe) -> engine_core.Agent:
    """Stub inner agent for ``preview_setup`` tests: always answers the first
    offered choice, writing a fixed value/policy annotation to ``probe`` on
    every call -- mimicking the real model agent's forward-pass side effect
    so there is something for the probe-drain assertion to drain."""
    annotation = decision_probe.PolicyAnnotation(probs=[1.0], chosen_idx=0)

    def inner[C: decisions.Choice](
        _engine: engine_core.Engine, decision: decisions.Decision[C]
    ) -> C:
        probe.record(0.1)
        probe.record_policy(annotation)
        return decision.choices[0]

    return inner


def test_preview_setup_returns_expected_keep_bonus_and_food() -> None:
    eng, dealt_cards, dealt_bonus = _dealt_setup_state()
    probe = decision_probe.DecisionProbe()
    inner = _greedy_preview_inner(probe)
    # Both axes deferred: bonus_card=None with bonus dealt, kept_foods=().
    preferred = decisions.SetupChoice(
        kept_cards=(dealt_cards[0], dealt_cards[1]), kept_foods=(), bonus_card=None
    )
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose starting hand",
        choices=[preferred],
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
    )

    result = preview_module.preview_setup(eng, inner, probe, decision, preferred)

    assert result.kept_cards == (dealt_cards[0], dealt_cards[1])
    # The bonus decision offers dealt_bonus in order; the greedy inner always
    # takes choices[0], i.e. the first dealt bonus card.
    assert result.bonus_card == dealt_bonus[0]
    # n_kept=2 -> the low-keep spend path discards 2 of the 5 dealt foods, in
    # ALL_FOODS order (invertebrate, then seed), leaving fish/fruit/rodent.
    assert result.kept_foods[cards.Food.FISH] == 1
    assert result.kept_foods[cards.Food.FRUIT] == 1
    assert result.kept_foods[cards.Food.RODENT] == 1
    assert result.kept_foods.total() == 3

    line = result.format_line()
    assert dealt_bonus[0].name in line
    assert "fish" in line


def test_preview_setup_does_not_mutate_the_real_state() -> None:
    """The real ``engine.state`` -- hand, food, deck/discard lengths -- is
    byte-identical after ``preview_setup``, even though the previewed keep
    would discard most of the hand and spend food: the preview must never
    leak a mutation (or a real draw/reroll) out of its cloned state."""
    eng, dealt_cards, dealt_bonus = _dealt_setup_state()
    probe = decision_probe.DecisionProbe()
    inner = _greedy_preview_inner(probe)
    player = eng.state.players[0]
    preferred = decisions.SetupChoice(
        kept_cards=(dealt_cards[0],), kept_foods=(), bonus_card=None
    )
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose starting hand",
        choices=[preferred],
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
    )
    hand_before = list(player.hand)
    food_before = player.food.model_copy(deep=True)
    deck_len_before = len(eng.state.bird_deck)
    bonus_deck_len_before = len(eng.state.bonus_deck)
    bonus_discard_len_before = len(eng.state.bonus_discard)
    bird_discard_len_before = len(eng.state.bird_discard)

    preview_module.preview_setup(eng, inner, probe, decision, preferred)

    assert player.hand == hand_before
    assert player.food == food_before
    assert len(eng.state.bird_deck) == deck_len_before
    assert len(eng.state.bonus_deck) == bonus_deck_len_before
    assert len(eng.state.bonus_discard) == bonus_discard_len_before
    assert len(eng.state.bird_discard) == bird_discard_len_before


def test_preview_setup_drains_the_probe() -> None:
    eng, dealt_cards, dealt_bonus = _dealt_setup_state()
    probe = decision_probe.DecisionProbe()
    inner = _greedy_preview_inner(probe)
    preferred = decisions.SetupChoice(
        kept_cards=(dealt_cards[0],), kept_foods=(), bonus_card=None
    )
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose starting hand",
        choices=[preferred],
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
    )

    preview_module.preview_setup(eng, inner, probe, decision, preferred)

    assert probe.take() == (None, None)


# ---------------------------------------------------------------------------
# Advisor rendering under split axes


def test_setup_move_under_split_axes_shows_compact_labels_and_preview_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a split-bonus regime the ranked lines use the compact
    ``keep:[...]`` label (not the misleading ``display_label``), and a
    combined ``model recommends:`` line follows the ranking."""
    eng, dealt_cards, dealt_bonus = _dealt_setup_state()
    con, transcript = aid_helpers.scripted_console([])
    echo = console.LogEcho(con)
    probe = decision_probe.DecisionProbe()
    registry = placeholders.PlaceholderRegistry()

    choice_keep_two = decisions.SetupChoice(
        kept_cards=(dealt_cards[0], dealt_cards[1]),
        kept_foods=tuple(cards.ALL_FOODS[:3]),
        bonus_card=None,
    )
    choice_keep_none = decisions.SetupChoice(
        kept_cards=(), kept_foods=tuple(cards.ALL_FOODS), bonus_card=None
    )
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="choose starting hand",
        choices=[choice_keep_two, choice_keep_none],
        dealt_cards=dealt_cards,
        dealt_bonus=dealt_bonus,
    )
    annotation = decision_probe.PolicyAnnotation(probs=[0.9, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.4, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    def fake_dialog(
        given_decision: decisions.SetupDecision, tray: list[typing.Any]
    ) -> decisions.SetupChoice:
        assert given_decision is decision
        return choice_keep_two

    monkeypatch.setattr(agents_cli, "resolve_setup_choice_dialog", fake_dialog)

    chosen = agent(eng, decision)

    assert chosen == choice_keep_two
    keep_two_label = f"keep:[{dealt_cards[0].name}, {dealt_cards[1].name}]"
    assert any(line == f"Setup: {keep_two_label}" for line in transcript)
    assert any(keep_two_label in line for line in transcript)
    assert any("keep:[none]" in line for line in transcript)
    assert any(line.startswith("model recommends:") for line in transcript)
    # display_label's "foods:[...] bonus:(none)" phrasing must not appear --
    # the compact label replaces it entirely under a deferred axis.
    assert not any("bonus:(none)" in line for line in transcript)


# ---------------------------------------------------------------------------
# SetupPreview.format_line -- edge cases


def test_format_line_empty_keep_omits_bonus_and_marks_multi_count_foods() -> None:
    pool = state.FoodPool()
    pool[cards.Food.FISH] = 1
    pool[cards.Food.SEED] = 2
    preview = models.SetupPreview(kept_cards=(), bonus_card=None, kept_foods=pool)

    line = preview.format_line()

    assert "keep [none]" in line
    assert "bonus [" not in line
    assert "fish" in line
    assert "seed x2" in line


def test_format_line_includes_bonus_segment_when_present() -> None:
    dealt_bonus = list(catalog.bonus_cards_ordered()[0:1])
    pool = state.FoodPool()
    preview = models.SetupPreview(
        kept_cards=(), bonus_card=dealt_bonus[0], kept_foods=pool
    )

    line = preview.format_line()

    assert f"bonus [{dealt_bonus[0].name}]" in line
    assert "foods [none]" in line
