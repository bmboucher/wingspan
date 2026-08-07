"""The advisor agent -- the user's seat (seat 0).

Wraps a factory model agent so every genuine decision is displayed with the
model's ranked recommendation (and, at setup, the setup net's top picks)
before asking what was actually played at the physical table. Runs a
placeholder sweep on the deciding seat's hand first, since a draft power can
pass face-down cards into it that the model must never see un-identified,
and writes the corrected pick back onto the probe so an ``EventRecorder``
records what was actually played rather than the model's raw recommendation.
"""

from __future__ import annotations

import typing

from wingspan import decisions
from wingspan.agents import cli as agents_cli
from wingspan.agents import display
from wingspan.aid import console as console_module
from wingspan.aid import entry, placeholders
from wingspan.engine import core as engine_core
from wingspan.players import decision_probe

# How many of the model's top-ranked choices get a probability line shown.
_AID_TOP_K = 5


def advisor_agent(
    inner: engine_core.Agent,
    probe: decision_probe.DecisionProbe,
    con: console_module.Console,
    echo: console_module.LogEcho,
    registry: placeholders.PlaceholderRegistry,
    score_norm: float,
) -> engine_core.Agent:
    """Build the seat-0 agent: consult ``inner`` for a recommendation, show
    it to the user, then ask what was actually played and write the
    correction back onto ``probe``."""

    def agent[C: decisions.Choice](
        engine: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        echo.flush()
        _sweep_placeholder_hand(con, registry, engine, decision)

        if decisions.is_setup_decision(decision):
            chosen_idx, value, annotation = _resolve_setup_move(
                con, inner, probe, engine, decision
            )
        else:
            chosen_idx, value, annotation = _resolve_main_move(
                con, inner, probe, engine, decision, score_norm
            )

        if value is not None:
            probe.record(value)
        if annotation is not None:
            probe.record_policy(
                annotation.model_copy(update={"chosen_idx": chosen_idx})
            )
        return decision.choices[chosen_idx]

    return agent


###### PRIVATE #######


def _sweep_placeholder_hand(
    con: console_module.Console,
    registry: placeholders.PlaceholderRegistry,
    engine: engine_core.Engine,
    decision: decisions.Decision[typing.Any],
) -> None:
    """Identify and swap every placeholder still sitting in the deciding
    seat's hand before the model sees it -- a draft power can pass a
    face-down pile into our own hand mid-turn. Every bird-carrying choice
    referencing a just-swapped placeholder is rewritten in place so the
    offered options stay consistent with the identified hand."""
    hand = engine.state.players[decision.player_id].hand
    while registry.count_birds_in(hand) > 0:
        con.say("The opponent passed you a face-down card — identify it")
        placeholder = registry.first_bird_in(hand)
        assert placeholder is not None  # count_birds_in(hand) > 0 guarantees this
        real = entry.identify_bird(con, "Which bird was it? ")
        registry.swap_bird(hand, real)
        for choice in decision.choices:
            if (
                isinstance(choice, (decisions.BirdChoice, decisions.PlayBirdChoice))
                and choice.bird is placeholder
            ):
                choice.bird = real


def _resolve_setup_move(
    con: console_module.Console,
    inner: engine_core.Agent,
    probe: decision_probe.DecisionProbe,
    engine: engine_core.Engine,
    decision: decisions.Decision[typing.Any],
) -> tuple[int, float | None, decision_probe.PolicyAnnotation | None]:
    """The setup-decision branch: show the setup net's top-ranked keep
    recommendations (if any), then walk the user through the actual keep via
    the promoted CLI setup dialog and locate it among the offered choices."""
    setup_decision = typing.cast(decisions.SetupDecision, decision)
    inner(engine, decision)
    value, annotation = probe.take()

    if annotation is not None:
        for idx in _top_k_indices(annotation.probs, _AID_TOP_K):
            label = setup_decision.choices[idx].display_label()
            con.say(f"{annotation.probs[idx]:5.1%}  {label}")
    else:
        con.say("(no setup model — no recommendation)")

    tray_birds = [bird for bird in engine.state.tray if bird is not None]
    kept = agents_cli.resolve_setup_choice_dialog(setup_decision, tray_birds)
    chosen_idx = setup_decision.choices.index(kept)
    return chosen_idx, value, annotation


def _resolve_main_move(
    con: console_module.Console,
    inner: engine_core.Agent,
    probe: decision_probe.DecisionProbe,
    engine: engine_core.Engine,
    decision: decisions.Decision[typing.Any],
    score_norm: float,
) -> tuple[int, float | None, decision_probe.PolicyAnnotation | None]:
    """The general decision branch: show the board (for the two big
    decisions), the model's ranked recommendation, and the expected-margin
    readout, then ask what was actually played."""
    player = engine.state.players[decision.player_id]
    if isinstance(decision, (decisions.MainActionDecision, decisions.PlayBirdDecision)):
        con.say(display.format_board(engine.state, player))

    inner(engine, decision)
    value, annotation = probe.take()

    con.say(decision.prompt)
    top_indices = (
        _top_k_indices(annotation.probs, _AID_TOP_K) if annotation is not None else []
    )
    argmax_idx = top_indices[0] if top_indices else 0
    for idx, choice in enumerate(decision.choices):
        line = agents_cli.format_choice_line(idx, choice, player)
        if annotation is not None and idx in top_indices:
            line += f"  — {annotation.probs[idx]:.1%}"
        if annotation is not None and idx == argmax_idx:
            line += "  ← model pick"
        con.say(line)
    if value is not None:
        con.say(f"model eval: {value * score_norm:+.1f} VP expected margin")

    chosen_idx = _resolve_move_index(con, decision, argmax_idx)
    return chosen_idx, value, annotation


def _top_k_indices(probs: list[float], top_k: int) -> list[int]:
    """Indices of the ``top_k`` highest-probability entries in ``probs``,
    sorted descending by probability."""
    order = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)
    return order[:top_k]


def _resolve_move_index(
    con: console_module.Console,
    decision: decisions.Decision[typing.Any],
    default_idx: int,
) -> int:
    """Prompt for the actual move, defaulting to ``default_idx`` on a bare
    Enter; loops until a valid choice index is entered."""
    while True:
        raw = con.ask("your actual move [Enter = model pick]> ")
        if not raw:
            return default_idx
        if raw.isdigit() and int(raw) < len(decision.choices):
            return int(raw)
        con.say("enter a valid choice index")
