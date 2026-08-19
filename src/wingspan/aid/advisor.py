"""The advisor agent -- the user's seat (seat 0).

Wraps a factory model agent so every genuine decision is displayed with the
model's ranked recommendation (and, at setup, the setup net's top picks)
before asking what was actually played at the physical table -- or, under
``--trust-me``, auto-committing that top pick without asking. Runs a
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
from wingspan.aid import preview as preview_module
from wingspan.engine import core as engine_core
from wingspan.players import decision_probe

# How many of the model's top-ranked choices get a probability line shown.
_AID_TOP_K = 5

# Printed before every genuine decision made during the setup window (the
# deferred bonus/food picks resolve as in-game decisions routed through
# ``_resolve_main_move``, so without this framing they'd look like an
# ordinary turn).
_SETUP_FRAMING_LINE = "(still setup — this pick completes your opening)"

# Printed in place of the actual-move / setup-dialog prompt when
# ``trust_me`` is on, so the transcript records that the model's own pick
# was auto-committed rather than something the user selected.
_TRUST_LINE_PREFIX = "trusting model pick"


def advisor_agent(
    inner: engine_core.Agent,
    probe: decision_probe.DecisionProbe,
    con: console_module.Console,
    echo: console_module.LogEcho,
    registry: placeholders.PlaceholderRegistry,
    score_norm: float,
    *,
    trust_me: bool = False,
) -> engine_core.Agent:
    """Build the seat-0 agent: consult ``inner`` for a recommendation, show
    it to the user, then ask what was actually played and write the
    correction back onto ``probe`` -- or, when ``trust_me`` is set, skip the
    prompt and commit the model's top pick directly."""

    def agent[C: decisions.Choice](
        engine: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        echo.flush()
        _sweep_placeholder_hand(con, registry, engine, decision)

        if decisions.is_setup_decision(decision):
            chosen_idx, value, annotation = _resolve_setup_move(
                con, inner, probe, engine, decision, trust_me
            )
        else:
            chosen_idx, value, annotation = _resolve_main_move(
                con, inner, probe, engine, decision, score_norm, trust_me
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
    trust_me: bool,
) -> tuple[int, float | None, decision_probe.PolicyAnnotation | None]:
    """The setup-decision branch: show the setup net's top-ranked keep
    recommendations (if any), then walk the user through the actual keep via
    the promoted CLI setup dialog and locate it among the offered choices --
    or, under ``trust_me`` (with a recommendation on hand), auto-commit the
    top-ranked keep and skip the dialog entirely.

    Under a split-setup regime the offered ``SetupChoice``s pin the deferred
    axis (or axes) to their empty value, so ``display_label``'s
    ``foods:[none] bonus:(none)`` segments would misleadingly read as "keeps
    nothing" -- the ranked lines fall back to a compact ``keep:[...]`` label
    in that case, and a combined recommendation line (the model's preferred
    keep replayed through the real deferred-resolution steps, see
    ``preview.py``) is shown after them."""
    setup_decision = typing.cast(decisions.SetupDecision, decision)
    inner(engine, decision)
    value, annotation = probe.take()

    ask_bonus, ask_food = agents_cli.setup_dialog_axes(setup_decision)
    bonus_deferred = not ask_bonus and len(setup_decision.dealt_bonus) > 0
    food_deferred = not ask_food
    any_deferred = bonus_deferred or food_deferred

    top_indices: list[int] = []
    if annotation is not None:
        top_indices = _top_k_indices(annotation.probs, _AID_TOP_K)
        for idx in top_indices:
            choice = setup_decision.choices[idx]
            label = (
                _compact_keep_label(choice) if any_deferred else choice.display_label()
            )
            con.say(f"{annotation.probs[idx]:5.1%}  {label}")
    else:
        con.say("(no setup model — no recommendation)")

    if annotation is not None and any_deferred:
        preferred = setup_decision.choices[top_indices[0]]
        setup_preview = preview_module.preview_setup(
            engine, inner, probe, setup_decision, preferred
        )
        con.say(setup_preview.format_line())

    if trust_me and annotation is not None:
        chosen_idx = top_indices[0]
        choice = setup_decision.choices[chosen_idx]
        label = _compact_keep_label(choice) if any_deferred else choice.display_label()
        con.say(f"{_TRUST_LINE_PREFIX}: {label}")
    else:
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
    trust_me: bool,
) -> tuple[int, float | None, decision_probe.PolicyAnnotation | None]:
    """The general decision branch: show the board (for the two big
    decisions), the model's ranked recommendation, and the expected-margin
    readout, then ask what was actually played -- or, under ``trust_me``
    (with a recommendation on hand), auto-commit the model's top pick and
    skip the prompt.

    ``engine.state.turn_counter`` stays 0 for the entire setup window
    (including the deferred bonus/food picks a split-setup regime resolves
    through this same branch), so a framing line is printed first whenever a
    genuine decision reaches here before round 1 has properly begun."""
    if engine.state.turn_counter == 0:
        con.say(_SETUP_FRAMING_LINE)

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

    if trust_me and annotation is not None:
        chosen_idx = argmax_idx
        con.say(
            f"{_TRUST_LINE_PREFIX}: "
            f"{agents_cli.format_choice_line(argmax_idx, decision.choices[argmax_idx], player)}"
        )
    else:
        chosen_idx = _resolve_move_index(con, decision, argmax_idx)
    return chosen_idx, value, annotation


def _compact_keep_label(choice: decisions.SetupChoice) -> str:
    """Compact ``keep:[...]`` label for a setup choice, used in place of
    ``display_label`` whenever the bonus and/or food axes are deferred to
    later decisions -- ``display_label``'s ``foods:[none] bonus:(none)``
    segments would otherwise misread as "keeps nothing" for every option."""
    kept_names = [bird.name for bird in choice.kept_cards] or ["none"]
    return f"keep:[{', '.join(kept_names)}]"


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
