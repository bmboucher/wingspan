"""Headless end-to-end session test for ``wingspan aid`` (stage 4).

Drives a full, real game through ``engine.core.Engine.play_one_game`` over an
``oracle_state.OracleGameState`` built by ``build_state``, with the real
``advisor``/``relay``/``hooks`` stack wired exactly as ``app.py`` wires it --
only the model itself is stubbed out (torch-free), via a scripted inner agent
cast to the ``Agent`` protocol. A single fixed answer-queue console cannot
survive an arbitrary game flow (the exact decision sequence depends on which
actions the model picks), so the console here is a pattern-responding one
(``aid_helpers.responder_console``): it answers every prompt by matching the
most recently written line(s) against a table of substring rules, falling
back to a safe default otherwise.

Both seats deterministically pick their first offered option for every
genuine decision (the stub agent's fixed pick for seat 0, "opponent's
move> 0" for seat 1), so neither seat ever plays a bird -- the main-action
choices always list Gain Food first. This keeps the game's actual decision
surface small (repeated Gain Food actions, birdfeeder takes and rerolls)
while still exercising the full turn loop, the setup dialog, the advisor's
placeholder sweep, the relay's opponent-setup matching and hidden-card
auto-pick, and the instrumentation hooks -- for a genuine 52-turn game
(``sum(state.ROUND_CUBES) * 2`` seats).

**Setup-dialog strategy.** The advisor's seat-0 setup pick normally goes
through ``agents.cli.resolve_setup_choice_dialog``, which drives the arrow-key
``select_form`` widget; off a real terminal that falls back to reading
``builtins.input`` directly (bypassing the injected ``Console`` entirely).
Rather than reverse-engineer and monkeypatch that fallback's own prompt
sequence, this test monkeypatches ``resolve_setup_choice_dialog`` itself to
deterministically return a mid-range element of ``decision.choices`` -- a
smaller, more robust surface than scripting a second I/O channel, and every
*other* advisor/relay prompt still flows through the responder console
unchanged.
"""

from __future__ import annotations

import random
import re
import typing

import pytest

import aid_helpers
from wingspan import cards, decisions, state
from wingspan.agents import cli as agents_cli
from wingspan.aid import advisor
from wingspan.aid import console as console_module
from wingspan.aid import hooks, models
from wingspan.aid import oracle as oracle_module
from wingspan.aid import oracle_state, placeholders, relay
from wingspan.cards.parse import catalog
from wingspan.engine import core as engine_core
from wingspan.players import decision_probe

# players.build_agent's greedy score_norm default (training.config.TrainingConfig
# .score_norm) -- the advisor only uses it to scale a value readout that this
# stub never populates, so any positive value is equivalent here.
_STUB_SCORE_NORM = 50.0

# The birdfeeder/dice-predator die-face vocabulary the responder cycles
# through when asked what N dice showed -- five distinct valid faces (four
# foods plus the invertebrate/seed "choice" face) so no answer ever repeats a
# food unnecessarily.
_DIE_FACE_CYCLE = ["seed", "fish", "fruit", "rodent", "invertebrate"]

# The opponent-setup-keep answer pair the relay's ``_resolve_opponent_setup``
# dialog is fed: 2 birds kept implies STARTING_HAND_SIZE(5) - 2 = 3 distinct
# foods kept. This exact (count, foods) pairing is already proven to match a
# real enumerated SetupChoice by test_aid_relay.py's no-bonus-regime case.
_OPPONENT_KEPT_BIRD_COUNT = "2"
_OPPONENT_KEPT_FOODS = "rodent, invertebrate, fruit"

_DICE_COUNT_PATTERN = re.compile(r"^(\d+) dice were rolled")


def _feeder_faces(_line: str) -> str:
    """A full birdfeeder reroll answer: exactly ``BIRDFEEDER_DICE`` faces."""
    return " ".join(
        _DIE_FACE_CYCLE[index % len(_DIE_FACE_CYCLE)]
        for index in range(state.BIRDFEEDER_DICE)
    )


def _predator_dice_faces(line: str) -> str:
    """A dice-predator roll answer: the prompt's own leading die count, read
    back out of the matched line so the answer always has the right length."""
    match = _DICE_COUNT_PATTERN.match(line)
    count = int(match.group(1)) if match is not None else 1
    return " ".join(
        _DIE_FACE_CYCLE[index % len(_DIE_FACE_CYCLE)] for index in range(count)
    )


def _build_rules(identify_name: str) -> list[aid_helpers.ResponderRule]:
    """The full responder rule table for one session: every prompt substring
    the advisor/relay/hooks/oracle stack can present during actual gameplay
    (the pre-built setup entry means ``entry.run_setup_entry``'s own dialog
    never runs, so its prompts are not included here)."""
    return [
        ("Did the opponent play a bird", lambda _line: "n"),
        ("Did they play ANOTHER", lambda _line: "n"),
        ("your actual move", lambda _line: ""),
        ("opponent's move>", lambda _line: "0"),
        (
            "How many bird cards did the opponent keep",
            lambda _line: _OPPONENT_KEPT_BIRD_COUNT,
        ),
        ("food(s) did the opponent keep", lambda _line: _OPPONENT_KEPT_FOODS),
        ("enter its name", lambda _line: ""),
        ("Which bird was it", lambda _line: identify_name),
        ("Which bird did they play", lambda _line: identify_name),
        ("enter the opponent's bonus card", lambda _line: "n"),
        ("Correct?", lambda _line: ""),
        ("die faces", _feeder_faces),
        ("dice were rolled", _predator_dice_faces),
    ]


def _powerless_bird_name() -> str:
    """A real bird with no printed power -- the fixed answer for every
    "which bird was it / did they play" prompt, chosen powerless so
    identifying it never opens an unrelated side-decision."""
    return next(
        bird.name for bird in catalog.birds_ordered() if not bird.raw_power_text
    )


def _build_setup_entry() -> models.SetupEntry:
    """A schema-valid physical deal: distinct real birds/bonus cards for
    every slot (hand, tray, and the dealt bonus pair all disjoint), a
    balanced initial feeder roll, and the user going first."""
    birds = catalog.birds_ordered()
    bonus_cards = catalog.bonus_cards_ordered()
    _, _, goals = cards.load_all()
    feeder = models.FeederEntry(counts=[1, 1, 1, 1, 0], choice_dice=1)
    return models.SetupEntry(
        hand=tuple(birds[1:6]),
        bonus_pair=tuple(bonus_cards[0:2]),
        goals=tuple(goals[:4]),
        tray=tuple(birds[6:9]),
        feeder=feeder,
        start_player=0,
    )


def _stub_inner_agent(registry: placeholders.PlaceholderRegistry) -> engine_core.Agent:
    """A torch-free stand-in for the factory's trained inner agent.

    Asserts on every call that the advisor's placeholder sweep already ran --
    no placeholder bird sits in seat 0's hand -- then always returns the
    first offered choice without touching the ``DecisionProbe`` (mirroring
    the annotation-less path the advisor already handles via
    ``probe.take() -> (None, None)``, e.g. a setup-net-less setup pick)."""

    def stub_agent[C: decisions.Choice](
        engine: engine_core.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert registry.count_birds_in(engine.state.players[0].hand) == 0, (
            "the advisor's placeholder sweep must run before the inner "
            "agent ever sees seat 0's hand"
        )
        return decision.choices[0]

    return typing.cast(engine_core.Agent, stub_agent)


def _play_scripted_session(
    monkeypatch: pytest.MonkeyPatch, *, split_setup_bonus: bool
) -> tuple[
    engine_core.Engine,
    oracle_state.OracleGameState,
    list[str],
    placeholders.PlaceholderRegistry,
]:
    """Wire one full aid session -- the real oracle/advisor/relay/hooks stack,
    a stubbed model agent, and the pattern-responding console -- and play it
    to completion. Returns the finished engine, the game state (for the
    bird-deck invariant), the console's full transcript, and the placeholder
    registry (for the hand-composition invariant)."""
    registry = placeholders.PlaceholderRegistry()
    identify_name = _powerless_bird_name()
    con, transcript = aid_helpers.responder_console(_build_rules(identify_name))
    echo = console_module.LogEcho(con)
    oracle = oracle_module.SessionOracle(con, echo, registry)

    rng = random.Random(0)
    gs = oracle_state.build_state(rng, _build_setup_entry(), oracle)

    notes = models.TurnNotes()
    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=oracle, notes=notes, registry=registry
    )
    instrumentation = hooks.build_instrumentation(handler)

    probe = decision_probe.DecisionProbe()
    stub_inner = _stub_inner_agent(registry)
    advisor_seat = advisor.advisor_agent(
        stub_inner, probe, con, echo, registry, _STUB_SCORE_NORM
    )
    relay_seat = relay.relay_agent(con, echo, registry, notes)

    def _mid_range_setup_choice(
        decision: decisions.SetupDecision, tray: list[cards.Bird]
    ) -> decisions.SetupChoice:
        return decision.choices[len(decision.choices) // 2]

    monkeypatch.setattr(
        agents_cli, "resolve_setup_choice_dialog", _mid_range_setup_choice
    )

    eng = engine_core.Engine.play_one_game(
        gs,
        [advisor_seat, relay_seat],
        instrumentation=instrumentation,
        split_setup_bonus=split_setup_bonus,
        split_setup_food=False,
        combine_gain_food=False,
    )
    return eng, gs, transcript, registry


def _assert_session_invariants(
    eng: engine_core.Engine,
    gs: oracle_state.OracleGameState,
    transcript: list[str],
    registry: placeholders.PlaceholderRegistry,
) -> None:
    """The shared assertions both regime variants must satisfy: the game
    actually finished, every round scored, both seats have a final score, the
    bird deck shrank from its post-setup-reveal starting size without going
    negative, seat 1's hand holds only placeholders, and the advisor showed
    at least one ranked (or explicitly model-less) recommendation."""
    starting_deck_size = len(catalog.birds_ordered()) - state.TRAY_SIZE
    assert eng.state.game_over is True
    assert len(eng.state.scored_goals) == len(state.ROUND_CUBES)
    assert all(player.final_score is not None for player in eng.state.players)
    assert 0 <= len(gs.bird_deck) < starting_deck_size

    opponent_hand = eng.state.players[1].hand
    assert len(opponent_hand) >= 0
    assert all(registry.is_placeholder(bird) for bird in opponent_hand)

    assert any("model pick" in line or "no setup model" in line for line in transcript)


def test_full_session_completes_combined_setup_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opening bonus stays combined into the initial ``SetupDecision``
    (``split_setup_bonus=False``): a full game runs to completion."""
    eng, gs, transcript, registry = _play_scripted_session(
        monkeypatch, split_setup_bonus=False
    )
    _assert_session_invariants(eng, gs, transcript, registry)


def test_full_session_completes_split_setup_bonus_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opening bonus is deferred to an in-game ``CHOOSE_BONUS`` pick
    (``split_setup_bonus=True``): exercises the relay's placeholder-bonus
    auto-pick and the advisor's deferred-bonus path, end to end."""
    eng, gs, transcript, registry = _play_scripted_session(
        monkeypatch, split_setup_bonus=True
    )
    _assert_session_invariants(eng, gs, transcript, registry)
