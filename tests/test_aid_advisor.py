"""Tests for ``wingspan.aid.advisor`` -- the seat-0 agent.

No torch: the inner factory agent is a scripted stub that mimics only the
side effect the real model agent has on the ``DecisionProbe`` (writing a
value / policy annotation during its forward pass), never an actual network.
"""

from __future__ import annotations

import typing

import pytest

import aid_helpers
from wingspan import decisions
from wingspan.agents import cli as agents_cli
from wingspan.aid import advisor, console, placeholders
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


def test_ranked_display_shows_probabilities_and_model_pick_marker() -> None:
    eng, con, transcript, probe, registry = _build([""])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    agent(eng, _main_action_decision())

    assert any("60.0%" in line for line in transcript)
    assert any("← model pick" in line for line in transcript)


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


def test_value_readout_line_uses_score_norm() -> None:
    eng, con, transcript, probe, registry = _build([""])
    echo = console.LogEcho(con)
    annotation = decision_probe.PolicyAnnotation(probs=[0.6, 0.3, 0.1], chosen_idx=0)
    inner = _make_inner(probe, 0.2, annotation)
    agent = advisor.advisor_agent(inner, probe, con, echo, registry, _SCORE_NORM)

    agent(eng, _main_action_decision())

    assert any("+10.0 VP expected margin" in line for line in transcript)


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
    choice_a = decisions.SetupChoice(
        label="a", kept_cards=(dealt_cards[0],), kept_foods=(), bonus_card=None
    )
    choice_b = decisions.SetupChoice(
        label="b", kept_cards=(), kept_foods=(), bonus_card=None
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
