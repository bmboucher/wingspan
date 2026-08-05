"""Event recorder: the structured game-event tree emitter.

:class:`EventRecorder` is a separate engine collaborator (not an
:class:`~wingspan.instrumentation.events.EventName` handler) that receives
``begin_*/end_event`` bracket calls and ``record_*`` calls from the engine and
its action modules.  It builds a :class:`~wingspan.gamelog.models.GameEventTree`
that both the HTML decision log and the plaintext detailed log consume as their
single source of truth.

**Import discipline:** top-level imports are limited to the standard library,
:mod:`wingspan.decisions`, and :mod:`wingspan.gamelog.models`.  Heavy deps
(:mod:`wingspan.reporting.encode_viewer`, :mod:`wingspan.reporting.humanize`,
:mod:`wingspan.engine.scoring`) are imported **lazily** inside the methods that
use them so that loading this module does not pull in torch or the full
reporting stack.  Engine types are guarded by :data:`typing.TYPE_CHECKING`.

This module must **not** import :mod:`wingspan.training.timestamps`; the
recorder stores raw ``turn_counter`` + ``setup_slot`` integers instead, and
the reporting layer reconstructs provisional timestamps from them.
"""

from __future__ import annotations

import typing

from wingspan import decisions as decisions_module
from wingspan.gamelog import models

if typing.TYPE_CHECKING:
    from wingspan.engine import core
    from wingspan.players import decision_probe

# Maximum options shown in a decision box (the chosen option is always included).
_MAX_DECISION_OPTIONS = 5

# Encoding for DecisionSubEvent.setup_slot (maps to provisional_timestamp offsets).
_SETUP_SLOT_KEEP = 0
_SETUP_SLOT_BONUS = 1
_SETUP_SLOT_FOOD = 2


class EventRecorder:
    """The structured event tree emitter, a separate engine collaborator.

    Maintains a phase list (mirroring the HTML handler's ``PhaseRecord`` list
    at 1-to-1 positional correspondence) and an open-event stack.

    Open-event stack rule:
      - ``begin_*`` pushes a new :class:`~models.GameEvent` onto the stack and
        attaches it to the stack-top's ``children`` (or the current phase's
        ``events`` when the stack is empty).
      - ``end_event`` pops the stack.
      - ``record_*`` appends to the stack-top's ``sub_events``; if the stack is
        empty, a :class:`~models.LooseEvent` is auto-created and appended to the
        current phase first.

    ``probes`` is a per-seat tuple, one entry per player in seat order — any
    table size the engine supports (2..5 seats).
    """

    def __init__(
        self,
        probes: tuple[decision_probe.DecisionProbe | None, ...],
    ) -> None:
        self._probes = probes
        self.root: models.GameEventTree = models.GameEventTree()
        self._current_phase: models.PhaseNode | None = None
        self._open_stack: list[models.AnyGameEvent] = []
        self._next_event_id: int = 0
        self._phase_loose_event: models.LooseEvent | None = None
        self._round_phase: models.PhaseNode | None = None

    # ---- Phase management ----

    def begin_game(self) -> None:
        """Reset the tree and push the ``game_start`` phase."""
        self.root = models.GameEventTree()
        self._next_event_id = 0
        self.begin_phase("game_start")

    def end_game(self, engine: core.Engine) -> None:
        """Push the ``game_end`` phase and append a :class:`~models.FinalScoringEvent`."""
        from wingspan.engine import scoring

        self.begin_phase("game_end")
        gs = engine.state
        scores: list[models.FinalScoreBreakdown] = []

        # Compute per-seat breakdowns using the same formula as scoring.final_scoring.
        for player in gs.players:
            bird_pts = sum(
                pb.bird.points for row in player.board.values() for pb in row
            )
            bonus_pts = sum(
                scoring.bonus_score(player, bc) for bc in player.bonus_cards
            )
            scores.append(
                models.FinalScoreBreakdown(
                    birds=bird_pts,
                    eggs=player.total_eggs,
                    tucked=player.total_tucked,
                    cached=player.total_cached,
                    bonus=bonus_pts,
                    goals=player.round_goal_points,
                    total=player.final_score or 0,
                )
            )

        self._append_phase_event(models.FinalScoringEvent(scores=scores))

    def begin_phase(
        self,
        kind: str,
        round_idx: int | None = None,
        turn_idx: int | None = None,
    ) -> None:
        """Push a new phase and clear the open-event stack.

        ``round_idx`` / ``turn_idx`` make the phase self-describing; the engine
        passes them from the round and turn loops so downstream consumers need
        not recover them positionally from the reporting layer."""
        self._current_phase = models.PhaseNode(
            kind=kind, round_idx=round_idx, turn_idx=turn_idx
        )
        self.root.phases.append(self._current_phase)
        self._open_stack.clear()
        self._phase_loose_event = None
        if kind == "round":
            self._round_phase = self._current_phase

    # ---- begin_* / end_event pair ----

    def begin_main_action(self, player_id: int) -> None:
        """Open a :class:`~models.MainActionEvent` on the stack."""
        self._push_event(models.MainActionEvent(player_id=player_id))

    def begin_play_bird(self, player_id: int) -> None:
        """Open a :class:`~models.PlayBirdEvent` on the stack."""
        self._push_event(models.PlayBirdEvent(player_id=player_id))

    def begin_white_power(self, player_id: int, bird_name: str) -> None:
        """Open a :class:`~models.WhitePowerEvent` on the stack."""
        self._push_event(
            models.WhitePowerEvent(player_id=player_id, bird_name=bird_name)
        )

    def begin_reaction(self, player_id: int, bird_name: str) -> None:
        """Open a :class:`~models.ReactionEvent` on the stack."""
        self._push_event(models.ReactionEvent(player_id=player_id, bird_name=bird_name))

    def begin_activate_base(self, player_id: int, habitat: str, action: str) -> None:
        """Open an :class:`~models.ActivateBaseEvent` on the stack."""
        self._push_event(
            models.ActivateBaseEvent(
                player_id=player_id, habitat=habitat, action=action
            )
        )

    def begin_activate_brown(
        self, player_id: int, bird_name: str, *, is_brown: bool
    ) -> None:
        """Open an :class:`~models.ActivateBrownEvent` on the stack."""
        self._push_event(
            models.ActivateBrownEvent(
                player_id=player_id, bird_name=bird_name, is_brown=is_brown
            )
        )

    def begin_extra_play(self, player_id: int, habitat: str | None) -> None:
        """Open an :class:`~models.ExtraPlayEvent` on the stack."""
        self._push_event(models.ExtraPlayEvent(player_id=player_id, habitat=habitat))

    def begin_turn_end(self, player_id: int) -> None:
        """Open a :class:`~models.TurnEndEvent` on the stack."""
        self._push_event(models.TurnEndEvent(player_id=player_id))

    def begin_refill_tray(self, player_id: int | None) -> None:
        """Open a :class:`~models.RefillTrayEvent` on the stack."""
        self._push_event(models.RefillTrayEvent(player_id=player_id))

    def begin_deal(self, player_id: int) -> None:
        """Open a :class:`~models.DealEvent` on the stack."""
        self._push_event(models.DealEvent(player_id=player_id))

    def begin_setup(self, player_id: int) -> None:
        """Open a :class:`~models.SetupEvent` on the stack."""
        self._push_event(models.SetupEvent(player_id=player_id))

    def end_event(self) -> None:
        """Pop the most recently opened event from the stack."""
        if self._open_stack:
            self._open_stack.pop()

    # ---- record_* ----

    def record_effect(self, effect: models.AnyEffect) -> None:
        """Append one recorded state mutation to the stack-top.

        Called only from :mod:`wingspan.engine.ledger`, which pairs every
        mutation with its effect in a single function so the two cannot
        drift."""
        self._stack_top().sub_events.append(effect)

    def record_forced(
        self,
        engine: core.Engine,
        decision: decisions_module.Decision[typing.Any],
        choice: decisions_module.Choice,
    ) -> None:
        """Append a :class:`~models.ForcedSubEvent` for a forced single-choice move.

        Carries the same clock/score fields as a genuine decision so forced
        moves are joinable to the timeline, but consumes no probe: the agent
        was never asked, so there is no policy annotation to attach."""
        from wingspan.reporting import humanize

        sub_event = models.ForcedSubEvent(
            player_id=decision.player_id,
            outcome_text=humanize.humanize_outcome(decision, choice, engine.state),
        )
        _fill_clock(sub_event, engine, decision)
        self._stack_top().sub_events.append(sub_event)

    def record_decision(
        self,
        engine: core.Engine,
        decision: decisions_module.Decision[typing.Any],
        choice: decisions_module.Choice,
    ) -> None:
        """Consume the probe and append a :class:`~models.DecisionSubEvent`.

        This is the single probe consumer — ``record_decision`` must be called
        exactly once per genuine decision, immediately after ``Engine.ask``
        fires the ``made_decision`` event."""
        from wingspan.reporting import humanize

        # Consume the probe; may be (None, None) for random / human seats.
        probe = self._probes[decision.player_id]
        value_pov, annotation = probe.take() if probe is not None else (None, None)

        gs = engine.state

        # Build option list + stripes when annotation is present.
        options: list[models.DecisionOption] = []
        state_stripes: list[models.EncodedStripe] | None = None
        if annotation is not None:
            options, state_stripes = _build_decision_options(
                engine, decision, annotation
            )

        # Stamp kept cards / bonus name on the enclosing SetupEvent.
        if isinstance(decision, decisions_module.SetupDecision) and isinstance(
            choice, decisions_module.SetupChoice
        ):
            _stamp_setup_kept_cards(self._open_stack, choice)
        elif isinstance(
            decision, decisions_module.BirdPowerPickBonusCardDecision
        ) and isinstance(choice, decisions_module.BonusCardChoice):
            _stamp_setup_bonus(self._open_stack, choice)

        sub_event = models.DecisionSubEvent(
            player_id=decision.player_id,
            outcome_text=humanize.humanize_outcome(decision, choice, gs),
            options=options,
            state_stripes=state_stripes,
            value=value_pov,
        )
        _fill_clock(sub_event, engine, decision)
        self._stack_top().sub_events.append(sub_event)

    def record_round_goal(
        self,
        engine: core.Engine,
        round_idx: int,
        description: str,
        counts: list[int],
        vps: list[int],
    ) -> None:
        """Append a :class:`~models.RoundGoalEvent` to the round it scores.

        Round-goal scoring runs after the round's last turn, when the current
        phase is that turn — so the event is routed back to the round phase it
        actually belongs to.  Appending to an earlier phase is safe for the
        reporting layer's positional ``zip``: it changes neither the number nor
        the order of phases."""
        event = models.RoundGoalEvent(
            round_idx=round_idx,
            description=description,
            counts=counts,
            vps=vps,
        )
        event.event_id = self._take_event_id()
        target_phase = self._round_phase or self._current_phase
        if target_phase is not None:
            target_phase.events.append(event)

    ###### PRIVATE #######

    def _push_event(self, event: models.AnyGameEvent) -> None:
        """Stamp an id, attach to the stack-top's children (or the phase), and push."""
        event.event_id = self._take_event_id()
        if self._open_stack:
            self._open_stack[-1].children.append(event)
        elif self._current_phase is not None:
            self._current_phase.events.append(event)
        self._open_stack.append(event)

    def _append_phase_event(self, event: models.AnyGameEvent) -> None:
        """Stamp an id and append ``event`` directly to the phase, bypassing the stack.

        Used by the scoring events, which are emitted between brackets rather
        than as an open/close pair."""
        event.event_id = self._take_event_id()
        if self._current_phase is not None:
            self._current_phase.events.append(event)

    def _take_event_id(self) -> int:
        """Return the next per-game monotonic event id."""
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    def _stack_top(self) -> models.AnyGameEvent:
        """Return the open event, or this phase's shared ``LooseEvent`` bucket.

        The bucket is created on first use and reused for the rest of the
        phase, so a run of unbracketed ``record_*`` calls collects into one
        event rather than one event apiece."""
        if self._open_stack:
            return self._open_stack[-1]
        if self._phase_loose_event is None:
            self._phase_loose_event = models.LooseEvent()
            self._append_phase_event(self._phase_loose_event)
        return self._phase_loose_event


#### Clock / score helpers ####


def _fill_clock(
    sub_event: models.ResolvedSubEvent,
    engine: core.Engine,
    decision: decisions_module.Decision[typing.Any],
) -> None:
    """Populate the shared clock and score fields on a resolved sub-event.

    Shared by ``record_decision`` and ``record_forced`` so both kinds of
    resolution land on the timeline identically."""
    from wingspan.engine import scoring

    gs = engine.state
    sub_event.turn_counter = gs.turn_counter
    sub_event.setup_slot = _setup_slot_for(gs.turn_counter, decision)
    sub_event.family_idx = decisions_module.family_index_for(type(decision))

    scores = [scoring.running_score(player) for player in gs.players]
    other_scores = [
        score for idx, score in enumerate(scores) if idx != decision.player_id
    ]
    sub_event.scores = scores
    sub_event.margin_before = float(scores[decision.player_id] - max(other_scores))


def _setup_slot_for(
    turn_counter: int,
    decision: decisions_module.Decision[typing.Any],
) -> int | None:
    """Encode the setup-window slot for timestamp reconstruction in reporting.

    ``None`` once the first turn has begun; otherwise which of the three
    opening sub-windows (keep / bonus / food) this decision belongs to."""
    if turn_counter >= 1:
        return None
    if decisions_module.is_setup_decision(decision):
        return _SETUP_SLOT_KEEP
    if isinstance(decision, decisions_module.BirdPowerPickBonusCardDecision):
        return _SETUP_SLOT_BONUS
    return _SETUP_SLOT_FOOD


#### Annotation helpers ####


def _build_decision_options(
    engine: core.Engine,
    decision: decisions_module.Decision[typing.Any],
    annotation: decision_probe.PolicyAnnotation,
) -> tuple[list[models.DecisionOption], list[models.EncodedStripe] | None]:
    """Build option list and state stripes from a policy annotation.

    Selects up to ``_MAX_DECISION_OPTIONS`` options by probability (always
    including the chosen option) and builds per-option choice stripes.
    The state-stripe panel is shared across all options for the decision."""
    from wingspan.reporting import encode_viewer, humanize

    gs = engine.state
    n_choices = len(decision.choices)
    ranked = sorted(
        range(n_choices), key=lambda idx: annotation.probs[idx], reverse=True
    )

    # Top-N by probability; force-include the chosen option if absent.
    shown_indices = ranked[:_MAX_DECISION_OPTIONS]
    if annotation.chosen_idx not in shown_indices:
        shown_indices = shown_indices[:-1] + [annotation.chosen_idx]

    # State stripes: setup path uses the deal context; main-net uses encode_state.
    state_stripes: list[models.EncodedStripe] | None = None
    if annotation.setup_feats is not None and annotation.setup_encoding is not None:
        if annotation.setup_feats:
            state_stripes = encode_viewer.extract_setup_context_stripes(
                annotation.setup_feats[0], annotation.setup_encoding
            )
    elif annotation.state_vec is not None:
        # The annotation's layout is the producing net's own (era-correct)
        # geometry; None falls back to the live layout inside the extractor.
        state_stripes = encode_viewer.extract_state_stripes(
            annotation.state_vec,
            include_setup=annotation.include_setup or False,
            vector_layout=annotation.state_layout,
        )

    # Per-option choice stripes and humanized labels.
    options: list[models.DecisionOption] = []
    for idx in shown_indices:
        idx_choice = decision.choices[idx]
        choice_stripes: list[models.EncodedStripe] | None = None
        if annotation.setup_feats is not None and annotation.setup_encoding is not None:
            if idx < len(annotation.setup_feats):
                choice_stripes = encode_viewer.extract_setup_candidate_stripes(
                    annotation.setup_feats[idx], annotation.setup_encoding
                )
        elif annotation.choice_feats is not None and idx < len(annotation.choice_feats):
            choice_stripes = encode_viewer.extract_choice_stripes(
                annotation.choice_feats[idx],
                include_setup=annotation.include_setup or False,
                vector_layout=annotation.choice_layout,
            )
        options.append(
            models.DecisionOption(
                label=humanize.humanize_choice(
                    idx_choice, gs, player_id=decision.player_id, decision=decision
                ),
                prob=annotation.probs[idx],
                score=annotation.scores[idx] if annotation.scores is not None else None,
                selected=(idx == annotation.chosen_idx),
                choice_stripes=choice_stripes,
            )
        )

    return options, state_stripes


def _stamp_setup_kept_cards(
    open_stack: typing.Sequence[models.AnyGameEvent],
    choice: decisions_module.SetupChoice,
) -> None:
    """Fill ``kept_card_names`` on the innermost open :class:`~models.SetupEvent`."""
    for event in reversed(open_stack):
        if isinstance(event, models.SetupEvent):
            event.kept_card_names = [bird.name for bird in choice.kept_cards]
            return


def _stamp_setup_bonus(
    open_stack: typing.Sequence[models.AnyGameEvent],
    choice: decisions_module.BonusCardChoice,
) -> None:
    """Fill ``kept_bonus_name`` on the innermost open :class:`~models.SetupEvent`."""
    for event in reversed(open_stack):
        if isinstance(event, models.SetupEvent):
            event.kept_bonus_name = choice.bonus_card.name
            return


#### Null recorder (no-op singleton for uninstrumented engines) ####


class _NullRecorder:
    """No-op drop-in for engines that run without the structured event log."""

    def __init__(self) -> None:
        self.root: models.GameEventTree = models.GameEventTree()

    def begin_game(self) -> None:
        pass

    def end_game(self, engine: core.Engine) -> None:
        pass

    def begin_phase(
        self,
        kind: str,
        round_idx: int | None = None,
        turn_idx: int | None = None,
    ) -> None:
        pass

    def begin_main_action(self, player_id: int) -> None:
        pass

    def begin_play_bird(self, player_id: int) -> None:
        pass

    def begin_white_power(self, player_id: int, bird_name: str) -> None:
        pass

    def begin_reaction(self, player_id: int, bird_name: str) -> None:
        pass

    def begin_activate_base(self, player_id: int, habitat: str, action: str) -> None:
        pass

    def begin_activate_brown(
        self, player_id: int, bird_name: str, *, is_brown: bool
    ) -> None:
        pass

    def begin_extra_play(self, player_id: int, habitat: str | None) -> None:
        pass

    def begin_turn_end(self, player_id: int) -> None:
        pass

    def begin_refill_tray(self, player_id: int | None) -> None:
        pass

    def begin_deal(self, player_id: int) -> None:
        pass

    def begin_setup(self, player_id: int) -> None:
        pass

    def end_event(self) -> None:
        pass

    def record_effect(self, effect: models.AnyEffect) -> None:
        pass

    def record_forced(
        self,
        engine: core.Engine,
        decision: decisions_module.Decision[typing.Any],
        choice: decisions_module.Choice,
    ) -> None:
        pass

    def record_decision(
        self,
        engine: core.Engine,
        decision: decisions_module.Decision[typing.Any],
        choice: decisions_module.Choice,
    ) -> None:
        pass

    def record_round_goal(
        self,
        engine: core.Engine,
        round_idx: int,
        description: str,
        counts: list[int],
        vps: list[int],
    ) -> None:
        pass


def null_recorder() -> _NullRecorder:
    """Build a fresh no-op recorder for an engine running without the event log.

    A factory rather than a shared constant: each engine gets its own instance,
    so nothing can reach through a module-level singleton and mutate a tree
    another game is nominally holding."""
    return _NullRecorder()


type AnyRecorder = EventRecorder | _NullRecorder
"""Union of the real and null recorder; the type of ``Engine.events``."""
