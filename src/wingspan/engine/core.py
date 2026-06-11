"""The Engine class: turn loop, setup, and the agent-decision plumbing.

The Engine drives the game forward by repeatedly asking the active player's
agent to resolve a ``Decision``. ``Engine.play_one_game(gs, agents)`` is the
top-level entry point — a static factory that constructs an Engine with
``agents`` already wired up, runs a full game on ``gs``, and returns the
engine for log/score inspection.

The bird/turn/scoring logic lives in sibling modules and is called as free
functions taking the Engine as the first argument:

- ``actions`` — play_bird / gain_food / lay_eggs / draw_cards, plus the
  per-card helpers (egg discard, feeder-die take, row activation, etc.)
- ``powers``  — bird-power dispatch (``dispatch_power``, ``apply_effect``)
- ``reactors`` — pink between-turn reactor hooks
- ``scoring`` — round and final scoring

Engine itself only owns the top-level turn loop, the setup phase, and the
``ask`` plumbing that routes a Decision through the active agent.
"""

from __future__ import annotations

import logging
import random
import typing

from wingspan import cards, decisions, state
from wingspan.engine import actions, log_format, scoring
from wingspan.instrumentation import dispatcher

logger = logging.getLogger(__name__)


# An agent is any callable that, given the engine and a Decision, returns
# the agent's chosen Choice. The return type tracks the Decision's Choice
# parameterization: calling an Agent with a ``Decision[C]`` returns a ``C``.
#
# Expressing that strongly requires a generic ``__call__`` rather than a
# generic callable alias: a plain ``Callable[[Engine, Decision[C]], C]``
# leaves ``C`` as a free TypeVar that has nowhere to bind at the use site
# (``list[Agent]``, ``agent: Agent`` parameters, etc.). The Protocol makes
# ``Agent`` itself non-generic — each agent instance is polymorphic across
# every Choice subtype.
class Agent(typing.Protocol):
    """A callable that resolves a Decision into one of its offered Choices.
    The return type is inferred from the Decision's Choice parameterization."""

    def __call__[C: decisions.Choice](
        self,
        engine: "Engine",
        decision: decisions.Decision[C],
        /,
    ) -> C: ...


class SetupKeep(typing.Protocol):
    """A pre-decided setup keep for one seat (e.g. a
    ``wingspan.setup_model.SetupCandidate``).

    The fixed-setup collection path only needs to convert a keep into a
    ``SetupChoice`` to apply it, so the engine depends on this minimal protocol
    rather than importing the setup-model package (and risking an import cycle)."""

    def to_setup_choice(self) -> decisions.SetupChoice: ...


# A callback that, given the engine and the per-seat dealt inputs
# ``((dealt_cards, dealt_bonus), …)``, returns each seat's decided setup keep.
# Supplied by the collector when setups are chosen externally — by the random
# generator or the setup model — instead of by asking an agent.
type SetupChooser = typing.Callable[
    ["Engine", tuple[tuple[list[cards.Bird], list[cards.BonusCard]], ...]],
    typing.Sequence[SetupKeep],
]


class Engine:
    """The game-driver. Holds the live ``GameState`` and the agents indexed
    by ``Player.id``.

    Agents are supplied at construction time; ``agent_for(player)`` resolves
    the controller for any player, with a clear error if one is missing.
    Tests that dispatch effects directly without driving a full game may
    omit ``agents`` and inject them later via the ``agents`` attribute."""

    def __init__(
        self,
        gs: state.GameState,
        agents: typing.Sequence[Agent] | None = None,
        instrumentation: dispatcher.Instrumentation | None = None,
    ):
        self.state = gs
        # ``agents`` is indexed by ``Player.id`` so opponent-prompting power
        # effects (pink reactors, "each player chooses" effects) can route to
        # the correct controller without threading agents through every method
        # signature.
        self.agents: list[Agent] = list(agents) if agents is not None else []
        if self.agents and len(self.agents) != len(gs.players):
            raise ValueError(
                f"agents count ({len(self.agents)}) does not match players "
                f"count ({len(gs.players)})"
            )
        # The event router fired at each instrumented site. Defaults to the
        # shared no-op ``EMPTY`` so an uninstrumented game pays nothing per event.
        self.instrumentation: dispatcher.Instrumentation = (
            instrumentation if instrumentation is not None else dispatcher.EMPTY
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        seed: int = 0,
    ) -> tuple[
        Engine, list[cards.Bird], list[cards.BonusCard], list[cards.EndRoundGoal]
    ]:
        """Construct a fresh ``Engine`` with seeded RNG, returning the engine
        plus the full card catalog so callers can introspect coverage,
        inspect specific cards, etc."""
        birds, bonuses, goals = cards.load_all()
        rng = random.Random(seed)
        gs = state.new_game(rng, birds, bonuses, goals)
        return Engine(gs), birds, bonuses, goals

    @staticmethod
    def play_one_game(
        gs: state.GameState,
        agents: tuple[Agent, Agent],
        instrumentation: dispatcher.Instrumentation | None = None,
        *,
        split_setup_bonus: bool = False,
        split_setup_food: bool = False,
    ) -> Engine:
        """Construct an Engine on ``gs`` with ``agents``, run a full game,
        and return the engine. The caller's ``gs`` is mutated in place, so
        either the returned engine or any pre-existing reference to ``gs``
        can be used to inspect the final log and scores.

        ``instrumentation`` attaches an event router for the duration of the
        game (default: the no-op ``EMPTY``).

        ``split_setup_bonus`` defers the opening bonus pick out of the combined
        ``SetupDecision`` to a follow-up in-game ``CHOOSE_BONUS`` decision (the
        ``split_setup_bonus`` regime); the dealt cards/food are still the
        ``SetupDecision``.

        ``split_setup_food`` defers the opening food pick to sequential in-game
        ``GainFoodDecision`` / ``SpendFoodDecision`` asks after the card-keep
        resolves (the ``split_setup_food`` regime). The number and kind of asks
        depends on how many birds the player kept (see
        ``_maybe_resolve_deferred_setup_food``)."""
        eng = Engine(gs, agents=agents, instrumentation=instrumentation)
        eng.log_section("=== GAME START ===", global_line=True)
        eng.instrumentation.game_start(engine=eng)
        log_format.log_game_setup(eng)
        eng._setup_phase(
            agents, defer_bonus=split_setup_bonus, defer_food=split_setup_food
        )
        for round_idx in range(4):
            eng._play_round(round_idx, agents)
        scoring.final_scoring(eng)
        eng.state.game_over = True
        eng.log_section("=== GAME END ===", global_line=True)
        eng.instrumentation.game_end(engine=eng)
        return eng

    @staticmethod
    def play_one_game_with_setups(
        gs: state.GameState,
        agents: tuple[Agent, Agent],
        choose_setups: SetupChooser,
        instrumentation: dispatcher.Instrumentation | None = None,
        *,
        split_setup_food: bool = False,
    ) -> Engine:
        """Like :meth:`play_one_game`, but the setup phase is resolved by
        ``choose_setups`` (the random generator or the setup model) instead of by
        asking each agent — the setup-model collection path.

        The engine still deals the starting hands / bonus / food, so the chooser
        decides over exactly the inputs an agent would have seen; everything after
        setup is identical to ``play_one_game``.

        ``split_setup_food`` defers food to sequential in-game food decisions after
        each seat's card-keep is applied (the ``split_setup_food`` regime)."""
        eng = Engine(gs, agents=agents, instrumentation=instrumentation)
        eng.log_section("=== GAME START ===", global_line=True)
        eng.instrumentation.game_start(engine=eng)
        log_format.log_game_setup(eng)
        eng._setup_phase_fixed(choose_setups, defer_food=split_setup_food)
        for round_idx in range(4):
            eng._play_round(round_idx, agents)
        scoring.final_scoring(eng)
        eng.state.game_over = True
        eng.log_section("=== GAME END ===", global_line=True)
        eng.instrumentation.game_end(engine=eng)
        return eng

    def agent_for(self, player: state.Player) -> Agent:
        """Return the agent controlling ``player``. Raises if unset — there is
        no silent fallback to the active player's agent. Public so power
        effects in other modules can dispatch opponent prompts."""
        if not self.agents or player.id >= len(self.agents):
            raise RuntimeError(
                f"No agent registered for player {player.id} ({player.name}). "
                f"Construct Engine(state, agents=[...])."
            )
        return self.agents[player.id]

    def ask[C: decisions.Choice](
        self,
        agent: Agent,
        decision: decisions.Decision[C],
    ) -> C:
        """Run ``agent`` against ``decision``, validate the answer, and return
        the matching Choice instance from ``decision.choices``.

        A decision offering a single legal choice is *forced* — there is
        nothing to decide — so it is resolved here without consulting the
        agent at all (the game log records the auto-pick so forced moves stay
        visible). This keeps every front-end consistent: the interactive
        CLI never prompts a human to "pick" the only option, and the RL
        collector only ever sees genuine forks, so no zero-signal steps are
        recorded (DECISIONS.md §0) and a forced move costs no forward pass
        during self-play. Because ``ask`` is the single choke point every
        agent call routes through, this one guard covers every decision type.

        Otherwise the answer is compared to ``decision.choices`` by Pydantic
        field equality, so an agent that constructs its own Choice with
        identical fields still resolves to the corresponding offered slot."""
        if len(decision.choices) == 1:
            self.log_skipped_decision(
                decision.player_id,
                f"only 1 choice: {decision.choices[0].display_label()}",
            )
            return decision.choices[0]
        self.instrumentation.making_decision(engine=self, decision=decision)
        choice: C = agent(self, decision)
        if choice not in decision.choices:
            raise ValueError(
                f"agent returned illegal choice {choice.display_label()!r} for "
                f"{type(decision).__name__}"
            )
        self.instrumentation.made_decision(
            engine=self, decision=decision, choice=choice
        )
        return choice

    def log(self, msg: str, player_id: int | None = None) -> None:
        """Append ``msg`` to both the plain-text log and the structured log.

        ``player_id`` tags the entry with the responsible player — used by
        the CLI to split per-player log files.  Pass ``None`` explicitly for
        truly global lines (section headers, game start/end).  The default
        (omitting the argument) uses ``state.current_player``, which is
        correct for the vast majority of per-turn log lines."""
        if player_id is None:
            resolved_id: int | None = self.state.current_player
        else:
            resolved_id = player_id
        self.state.log.append(msg)
        self.state.log_entries.append(state.LogEntry(player_id=resolved_id, text=msg))

    def log_global(self, msg: str) -> None:
        """Append a global line (no player attribution) to both logs.

        Use for section headers (``=== ROUND 1 ===``), game start/end banners,
        and any line that belongs to no single player's perspective."""
        self.state.log.append(msg)
        self.state.log_entries.append(state.LogEntry(player_id=None, text=msg))

    def log_section(self, msg: str, *, global_line: bool = False) -> None:
        """Log a ``===`` section header, ensuring exactly one blank line before it.

        Consecutive ``log_section`` calls share a single blank line between
        them (no double-blanks); the first call in an empty log skips the
        leading blank entirely.

        Pass ``global_line=True`` for banners that belong to no single player
        (round headers, game start/end).  The default uses the current player."""
        if self.state.log and self.state.log[-1] != "":
            if global_line:
                self.log_global("")
            else:
                self.log("")
        if global_line:
            self.log_global(msg)
        else:
            self.log(msg)

    def log_skipped_decision(self, player_id: int, reason: str) -> None:
        """Log that a decision point resolved without consulting the agent —
        either auto-picked (``ask``'s single-choice guard) or never built
        because no legal choice existed (handlers' empty-choices guards)."""
        self.log(
            f"[{self.state.players[player_id].name}] skipping decision, {reason}",
            player_id=player_id,
        )

    # ------------------------------------------------------------------
    # Decision plumbing
    # ------------------------------------------------------------------

    def _take_turn(self, agent: Agent) -> None:
        """Run one full turn for the active player: reset turn scratch, prompt
        for a main action, spend an action cube, dispatch the chosen action,
        resolve any extra plays accrued during it, then refill the bird tray.

        The tray refill happens once, at the very end of the turn — cards taken
        from the tray during the turn (by the draw action, conversions, or any
        non-Brant draw) leave their slots empty until now. Refilling here, after
        every effect has resolved, is what makes mid-turn draws see a shrinking
        tray instead of an ever-full one."""
        player = self.state.me()
        self.state.reset_turn_state()
        # Clear the once-between-turns cap for each of this player's birds so
        # they can fire again during the next between-turns window.
        for row in player.board.values():
            for pb in row:
                pb.pink_fired = False
        self.instrumentation.turn_start(engine=self, player=player)
        # Print the turn header first so it anchors the block, then the state
        # summary, then ask for the main action (which logs the AI distribution)
        # so the decision lines always follow their context header.
        turn_idx = state.ROUND_CUBES[self.state.round_idx] - player.action_cubes_left
        self.log_section(
            f"=== {player.name}, ROUND {self.state.round_idx + 1}, "
            f"TURN {turn_idx} ({player.action_cubes_left} CUBES LEFT) ==="
        )
        log_format.log_turn_summary(self)
        self.log("")
        choice = self.ask(agent, self._main_action_decision(player))
        player.action_cubes_left -= 1
        self._dispatch_main_action(agent, choice)
        actions.consume_extra_plays(self, player, agent)
        self._resolve_turn_end_discards(agent, player)
        self.state.refill_tray()
        self.instrumentation.turn_end(engine=self, player=player)

    @staticmethod
    def _main_action_label(choice: decisions.MainActionChoice) -> str:
        """Log-header text for the chosen main-action type. For ``PLAY_BIRD`` the
        specific bird / habitat / costs are logged later by ``do_play_bird``
        once the follow-up decisions resolve."""
        return choice.action.value.upper()

    def _dispatch_main_action(
        self,
        agent: Agent,
        choice: decisions.MainActionChoice,
    ) -> None:
        """Run the chosen main action. ``PLAY_BIRD`` opens the follow-up
        ``PlayBirdDecision`` (which bird, where; the costs are further
        follow-ups) and plays it; the other three run their habitat-row
        action."""
        if choice.action == decisions.MainAction.PLAY_BIRD:
            actions.do_play_bird_action(self, agent)
        elif choice.action == decisions.MainAction.GAIN_FOOD:
            actions.do_gain_food(self, agent)
        elif choice.action == decisions.MainAction.LAY_EGGS:
            actions.do_lay_eggs(self, agent)
        elif choice.action == decisions.MainAction.DRAW_CARDS:
            actions.do_draw_cards(self, agent)

    def _main_action_decision(
        self, player: state.Player
    ) -> decisions.MainActionDecision:
        """Build the main-action Decision for ``player`` — the action *type*.

        The three habitat-row actions are always offered (just inefficient if
        the row is empty); ``PLAY_BIRD`` is offered only when ``player`` has at
        least one legal play right now. Choosing ``PLAY_BIRD`` opens a follow-up
        ``PlayBirdDecision`` for the specific bird / habitat."""
        choices: list[decisions.MainActionChoice] = [
            decisions.MainActionChoice(
                label="gain food (forest)",
                action=decisions.MainAction.GAIN_FOOD,
            ),
            decisions.MainActionChoice(
                label="lay eggs (grassland)",
                action=decisions.MainAction.LAY_EGGS,
            ),
            decisions.MainActionChoice(
                label="draw cards (wetland)",
                action=decisions.MainAction.DRAW_CARDS,
            ),
        ]
        if actions.any_playable_bird_play(player):
            choices.append(
                decisions.MainActionChoice(
                    label="play a bird",
                    action=decisions.MainAction.PLAY_BIRD,
                )
            )
        return decisions.MainActionDecision(
            player_id=player.id,
            prompt=f"[{player.name}] choose a main action",
            choices=choices,
        )

    def _resolve_turn_end_discards(self, agent: Agent, player: state.Player) -> None:
        """Resolve end-of-turn discard obligations from 'draw N, then discard 1 at
        end of turn' birds (DRAW_CARDS_THEN_DISCARD_EOT).

        Each obligation is one mandatory discard from hand. A single card in hand
        auto-resolves via the ask() single-choice guard. Fizzles if hand is empty."""
        for _ in range(self.state.turn_end_discards):
            if not player.hand:
                self.log(f"  [{player.name}] end-of-turn discard: hand empty, skipped")
                break
            ch = self.ask(
                agent,
                decisions.BirdPowerDiscardFromHandDecision(
                    player_id=player.id,
                    prompt=f"[{player.name}] discard 1 card from hand (end-of-turn obligation)",
                    choices=[
                        decisions.BirdChoice(label=card.name, bird=card)
                        for card in player.hand
                    ],
                ),
            )
            assert isinstance(ch, decisions.BirdChoice)
            player.hand.remove(ch.bird)
            self.state.bird_discard.append(ch.bird)
            self.log(f"  [{player.name}] end-of-turn discard: {ch.bird.name}")

    # ------------------------------------------------------------------
    # Setup (kept on Engine because it depends heavily on _ask)
    # ------------------------------------------------------------------

    def _setup_phase(
        self,
        agents: typing.Sequence[Agent],
        *,
        defer_bonus: bool = False,
        defer_food: bool = False,
    ) -> None:
        """Pre-round-1 setup: deal each player a starting hand, prompt the
        combined keep-cards / discard-food / bonus-card pick, log the result.

        ``defer_bonus`` (the ``split_setup_bonus`` regime) drops the bonus from the
        ``SetupDecision`` and resolves it via a follow-up in-game ``CHOOSE_BONUS``
        pick instead.

        ``defer_food`` (the ``split_setup_food`` regime) drops food from the
        ``SetupDecision`` and resolves it via sequential in-game food decisions
        after the card-keep is applied."""
        for player in self.state.players:
            dealt_cards, dealt_bonus = self._deal_setup_inputs(player)
            # Set current_player before any logging so all setup lines for this
            # player are attributed to them in the structured log.
            self.state.current_player = player.id
            if defer_bonus and defer_food:
                self.log_section(f"=== SETUP: {player.name} CHOOSING BIRDS ===")
            elif defer_bonus:
                self.log_section(
                    f"=== SETUP: {player.name} CHOOSING BIRDS AND FOOD ==="
                )
            elif defer_food:
                self.log_section(
                    f"=== SETUP: {player.name} CHOOSING BIRDS AND BONUS CARD ==="
                )
            else:
                self.log_section(
                    f"=== SETUP: {player.name} CHOOSING BIRDS, FOOD, AND BONUS CARD ==="
                )
            log_format.log_dealt_hand(self, player, dealt_cards)
            log_format.log_dealt_bonus(self, dealt_cards, dealt_bonus, player)
            self._resolve_setup_choice(
                player,
                agents,
                dealt_cards,
                dealt_bonus,
                defer_bonus=defer_bonus,
                defer_food=defer_food,
            )

    def _setup_phase_fixed(
        self, choose_setups: SetupChooser, *, defer_food: bool = False
    ) -> None:
        """Resolve setup from a chooser callback rather than agent prompts (the
        setup-model path): deal each player's inputs, ask the chooser for both
        seats' keeps over those inputs, then apply them. Skips ``Engine.ask``
        because both seats' setups are decided together up front.

        A keep whose ``bonus_card`` is ``None`` while bonus cards were dealt has
        deferred its bonus (the ``split_setup_bonus`` regime): the bonus is then
        picked via the in-game ``CHOOSE_BONUS`` head over the already-applied
        cards/food, recorded like any other in-game decision.

        ``defer_food`` (the ``split_setup_food`` regime) skips the food update in
        ``_apply_setup_choice`` and instead resolves food via sequential in-game
        food decisions immediately after the card-keep and bonus are applied."""
        dealt = tuple(self._deal_setup_inputs(player) for player in self.state.players)
        keeps = choose_setups(self, dealt)
        for player in self.state.players:
            dealt_cards, dealt_bonus = dealt[player.id]
            sc = keeps[player.id].to_setup_choice()
            self.state.current_player = player.id
            self.log_section(f"=== SETUP: {player.name} CHOOSING BIRDS AND FOOD ===")
            log_format.log_dealt_hand(self, player, dealt_cards)
            log_format.log_dealt_bonus(self, dealt_cards, dealt_bonus, player)
            self._apply_setup_choice(
                player, dealt_cards, dealt_bonus, sc, defer_food=defer_food
            )
            self._maybe_resolve_deferred_setup_bonus(player, dealt_bonus, sc)
            self._maybe_resolve_deferred_setup_food(
                player,
                self.agent_for(player),
                len(sc.kept_cards),
                defer_food=defer_food,
            )
            self._log_setup_result(player)

    def _log_setup_result(self, player: state.Player) -> None:
        """Log a player's post-setup starting hand and retained food."""
        self.log(
            f"[{player.name}] starts with "
            f"hand=[{', '.join(bird.name for bird in player.hand)}] "
            f"food={player.food.format()}"
        )

    # ------------------------------------------------------------------
    # Round / extra-plays helpers
    # ------------------------------------------------------------------

    def _play_round(self, round_idx: int, agents: typing.Sequence[Agent]) -> None:
        """Reset per-round state, log the goal, then alternate turns until
        both players have spent every action cube."""
        self.state.round_idx = round_idx
        for player in self.state.players:
            player.action_cubes_left = state.ROUND_CUBES[round_idx]
            for row in player.board.values():
                for pb in row:
                    pb.activations = 0
        self.log_section(
            f"=== ROUND {round_idx + 1} "
            f"({state.ROUND_CUBES[round_idx]} ACTIONS EACH) ===",
            global_line=True,
        )
        self.log_global(f"Round goal: {self.state.round_goals[round_idx].description}")
        self.instrumentation.round_start(engine=self, round_num=round_idx)
        # Turn order rotates each round off the randomly-chosen first player;
        # both players hold equal cubes, so a strict alternation drains them
        # evenly. ``current_player`` is set immediately before each turn so the
        # acting player and ``agents[idx]`` never desync.
        first = (self.state.start_player + round_idx) % len(self.state.players)
        order = (first, 1 - first)
        while any(player.action_cubes_left > 0 for player in self.state.players):
            for idx in order:
                if self.state.players[idx].action_cubes_left > 0:
                    self.state.current_player = idx
                    self._take_turn(agents[idx])
        scoring.score_round_goal(self, round_idx)
        # Discard the face-up tray and replenish it with fresh cards so every
        # round opens on three new options. This matches the rulebook's
        # end-of-round setup step; the last turn's refill_tray() still runs
        # (restoring any slots emptied during that turn) and those cards are
        # then discarded here before the new cards go face-up.
        self.state.reset_tray()
        self.instrumentation.round_end(engine=self, round_num=round_idx)

    # Power-granted extra plays are resolved by ``actions.consume_extra_plays``
    # (a free function, like the other action logic), called from ``_take_turn``.

    # ------------------------------------------------------------------
    # Setup sub-helpers
    # ------------------------------------------------------------------

    def _deal_starting_hand(self, player: state.Player) -> None:
        """Draw ``STARTING_HAND_SIZE`` birds from the top of the deck into
        ``player``'s hand. Silently deals fewer if the deck is short."""
        for _ in range(state.STARTING_HAND_SIZE):
            drawn = self.state.draw_bird()
            if drawn:
                player.hand.append(drawn)

    def _deal_setup_inputs(
        self, player: state.Player
    ) -> tuple[list[cards.Bird], list[cards.BonusCard]]:
        """Deal ``player``'s starting hand and bonus cards and give one of each
        food, returning the dealt cards and dealt bonus the setup pick is made
        over. The shared dealing prefix of both setup paths (the ask-the-agent
        ``_resolve_setup_choice`` and the fixed-setup ``_setup_phase_fixed``), so
        a chooser decides over exactly the inputs an agent would see."""
        self._deal_starting_hand(player)
        dealt_bonus = self._deal_starting_bonus()
        for food in cards.ALL_FOODS:
            player.food[food] = 1
        return list(player.hand), dealt_bonus

    def _resolve_setup_choice(
        self,
        player: state.Player,
        agents: typing.Sequence[Agent],
        dealt_cards: list[cards.Bird],
        dealt_bonus: list[cards.BonusCard],
        *,
        defer_bonus: bool = False,
        defer_food: bool = False,
    ) -> None:
        """Present the combined hand / food / bonus pick as a single Decision over
        the already-dealt inputs.

        Enumerates every legal ``SetupChoice`` (kept-card subset × retained-food
        subset of matching size × bonus card). For the default 5-card / 2-bonus
        deal that produces 2 * sum_k C(5,k)^2 = 504 choices, which matches the RL
        action space.

        ``defer_bonus`` drops the bonus axis from the enumeration (each
        ``SetupChoice`` carries ``bonus_card=None``) and resolves the bonus through
        a follow-up in-game ``CHOOSE_BONUS`` pick — the ``split_setup_bonus``
        regime.

        ``defer_food`` drops the food axis (each ``SetupChoice`` carries
        ``kept_foods=()``) and resolves food through sequential in-game food
        decisions after the card-keep applies — the ``split_setup_food`` regime."""
        choices = self._build_setup_choices(
            dealt_cards,
            dealt_bonus,
            include_bonus=not defer_bonus,
            include_food=not defer_food,
        )
        self.state.current_player = player.id
        decision = decisions.SetupDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] choose starting hand (kept cards cost 1 food each)"
                f"{'' if defer_bonus else ' and bonus card'}"
            ),
            choices=choices,
            dealt_cards=dealt_cards,
            dealt_bonus=dealt_bonus,
        )
        self.instrumentation.setup_start(
            engine=self, player=player, dealt_bonus=dealt_bonus
        )
        chosen = self.ask(agents[player.id], decision)
        self._apply_setup_choice(
            player, dealt_cards, dealt_bonus, chosen, defer_food=defer_food
        )
        kept_names = ", ".join(bird.name for bird in chosen.kept_cards) or "(none)"
        # Omit foods and bonus when deferred — they haven't been chosen yet and
        # printing them as "(none)" is misleading.
        suffix_parts: list[str] = []
        if not defer_food:
            foods_str = ", ".join(food.value for food in chosen.kept_foods) or "none"
            suffix_parts.append(f"foods [{foods_str}]")
        if not defer_bonus and chosen.bonus_card is not None:
            suffix_parts.append(f"bonus: {chosen.bonus_card.name}")
        suffix = (", " + ", ".join(suffix_parts)) if suffix_parts else ""
        self.log(f"[{player.name}] keeps {kept_names}{suffix}")
        self._maybe_resolve_deferred_setup_bonus(player, dealt_bonus, chosen)
        self._maybe_resolve_deferred_setup_food(
            player, agents[player.id], len(chosen.kept_cards), defer_food=defer_food
        )

    def _deal_starting_bonus(self) -> list[cards.BonusCard]:
        """Pop ``STARTING_BONUS_CARDS_DEAL`` bonus cards from the deck (or as
        many as remain)."""
        dealt: list[cards.BonusCard] = []
        for _ in range(state.STARTING_BONUS_CARDS_DEAL):
            if self.state.bonus_deck:
                dealt.append(self.state.bonus_deck.pop())
        return dealt

    @staticmethod
    def _build_setup_choices(
        dealt_cards: list[cards.Bird],
        dealt_bonus: list[cards.BonusCard],
        *,
        include_bonus: bool = True,
        include_food: bool = True,
    ) -> list[decisions.SetupChoice]:
        """Enumerate every legal ``SetupChoice``.

        Delegates to ``setup_model.candidates.enumerate_setup_candidates`` so the
        504-candidate set (and its ``(kept_mask, kept_food_combo, bonus)`` order)
        has a single source of truth shared with the setup model — which scores
        exactly these candidates. Imported lazily to avoid an import cycle (the
        setup-model package transitively imports the engine). Each candidate
        renders to a ``SetupChoice`` whose label is built lazily by
        ``display_label`` (only the chosen / displayed option ever needs it).

        ``include_bonus=False`` (the ``split_setup_bonus`` regime) drops the bonus
        axis — each ``SetupChoice`` carries ``bonus_card=None`` — so the bonus is
        instead picked by the in-game ``CHOOSE_BONUS`` head.

        ``include_food=False`` (the ``split_setup_food`` regime) drops the food
        axis — each ``SetupChoice`` carries ``kept_foods=()`` — so food is instead
        resolved by sequential in-game food decisions after the card-keep applies."""
        from wingspan.setup_model import candidates as setup_candidates

        return [
            candidate.to_setup_choice()
            for candidate in setup_candidates.enumerate_setup_candidates(
                dealt_cards,
                dealt_bonus,
                include_bonus=include_bonus,
                include_food=include_food,
            )
        ]

    def _maybe_resolve_deferred_setup_bonus(
        self,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
        sc: decisions.SetupChoice,
    ) -> None:
        """Pick ``player``'s bonus card via the in-game ``CHOOSE_BONUS`` head when
        the setup keep deferred it (the ``split_setup_bonus`` regime).

        A keep with ``bonus_card is None`` while bonus cards were dealt is the
        deferral signal — it never arises in the combined-keep regime, where the
        enumerator always assigns one of the dealt bonuses. The pick is asked over a
        minimal "start of round 1" snapshot (round 0, full action cubes; the tray /
        birdfeeder / round goals were set by ``new_game``) so the in-game head scores
        the bonus over a faithful opening, and it routes through ``Engine.ask`` so a
        collecting agent records it like any other in-game decision."""
        if not dealt_bonus or sc.bonus_card is not None:
            # Bonus was included in the combined setup choice (non-deferred path).
            # Fire setup_applied now so all listeners receive it; the HTML log
            # handler checks choice.bonus_card to decide whether to open a phase.
            self.instrumentation.setup_applied(engine=self, player=player, choice=sc)
            return
        self.log_section(f"=== SETUP: {player.name} CHOOSING BONUS CARD ===")
        # Fire setup_applied at the start of this segment so the HTML log's
        # zip(phases, segments) pairs this phase with the bonus-choosing narration.
        self.instrumentation.setup_applied(engine=self, player=player, choice=sc)
        # ``_play_round`` resets the cubes again before real play, so pre-loading
        # them here only shapes the encoded snapshot the bonus pick is scored over.
        for seat in self.state.players:
            seat.action_cubes_left = state.ROUND_CUBES[0]
        self.state.current_player = player.id
        decision = decisions.BirdPowerPickBonusCardDecision(
            player_id=player.id,
            prompt=f"[{player.name}] keep a bonus card",
            choices=[
                decisions.BonusCardChoice(label=bonus.name, bonus_card=bonus)
                for bonus in dealt_bonus
            ],
        )
        chosen = self.ask(self.agent_for(player), decision)
        player.bonus_cards.append(chosen.bonus_card)
        for bonus in dealt_bonus:
            if bonus is not chosen.bonus_card:
                self.state.bonus_discard.append(bonus)
        self.log(f"[{player.name}] keeps bonus: {chosen.bonus_card.name}")

    def _apply_setup_choice(
        self,
        player: state.Player,
        dealt_cards: list[cards.Bird],
        dealt_bonus: list[cards.BonusCard],
        sc: decisions.SetupChoice,
        *,
        defer_food: bool = False,
    ) -> None:
        """Mutate ``player`` / ``self.state`` to reflect the chosen setup combination.

        When ``defer_food`` is True (the ``split_setup_food`` regime), the food
        update is skipped here — food is instead resolved by a subsequent call to
        ``_maybe_resolve_deferred_setup_food``."""
        kept = list(sc.kept_cards)
        player.hand = kept
        for card in dealt_cards:
            if card not in kept:
                self.state.bird_discard.append(card)
        if not defer_food:
            for food in cards.ALL_FOODS:
                if food not in sc.kept_foods:
                    player.food[food] -= 1
        if sc.bonus_card is not None:
            player.bonus_cards.append(sc.bonus_card)
            for bonus in dealt_bonus:
                if bonus is not sc.bonus_card:
                    self.state.bonus_discard.append(bonus)

    def _maybe_resolve_deferred_setup_food(
        self,
        player: state.Player,
        agent: Agent,
        n_kept: int,
        *,
        defer_food: bool,
    ) -> None:
        """Resolve the opening food pick via in-game decisions when the setup keep
        deferred it (the ``split_setup_food`` regime).

        Food decisions depend on how many birds were kept, splitting evenly around
        the "pay-for-cards" midpoint so both spend and gain sides of food valuation
        get training signal:

          0 kept → keep all 5 food (no decision needed)
          1 kept → 5 food dealt, 1 × SpendFoodDecision (discard 1)
          2 kept → 5 food dealt, 2 × SpendFoodDecision (no repeat)
          3 kept → zero food,    2 × GainFoodDecision  (no repeat)
          4 kept → zero food,    1 × GainFoodDecision
          5 kept → keep no food (no decision needed)

        All decisions are mandatory (no SkipChoice offered) and route through
        ``Engine.ask`` so collecting agents record them like any in-game decision.
        The encoder's action-cubes snapshot is pre-loaded to round-1 values (same
        as the bonus-deferral path) so the heads score food over a faithful opening.
        """
        if not defer_food:
            return

        # Pre-load round-1 cubes so the in-game encoder sees a faithful opening.
        # ``_play_round`` resets them again before real play.
        for seat in self.state.players:
            seat.action_cubes_left = state.ROUND_CUBES[0]
        self.state.current_player = player.id

        if n_kept >= 3:
            # High-keep: player would have no food left after paying for birds.
            # Zero out the post-deal food pool and grant food via gain decisions.
            for food in cards.ALL_FOODS:
                player.food[food] = 0
            n_gains = min(5 - n_kept, 2)  # 3 kept → 2, 4 kept → 1, 5 kept → 0
            gained: set[cards.Food] = set()
            for gain_num in range(n_gains):
                available = [food for food in cards.ALL_FOODS if food not in gained]
                decision = decisions.GainFoodDecision(
                    player_id=player.id,
                    prompt=(
                        f"[{player.name}] setup: choose food to gain "
                        f"({gain_num + 1}/{n_gains})"
                    ),
                    choices=[
                        decisions.FoodChoice(label=food.value, food=food)
                        for food in available
                    ],
                )
                chosen = self.ask(agent, decision)
                assert isinstance(chosen, decisions.FoodChoice)
                player.food[chosen.food] += 1
                gained.add(chosen.food)
        else:
            # Low-keep: player has 5 food from the deal and discards down.
            n_spends = n_kept  # 0 kept → 0, 1 kept → 1, 2 kept → 2
            for spend_num in range(n_spends):
                held = [food for food in cards.ALL_FOODS if player.food[food] > 0]
                decision = decisions.SpendFoodDecision(
                    player_id=player.id,
                    prompt=(
                        f"[{player.name}] setup: choose food to discard "
                        f"({spend_num + 1}/{n_spends})"
                    ),
                    choices=[
                        decisions.FoodChoice(label=food.value, food=food)
                        for food in held
                    ],
                )
                chosen = self.ask(agent, decision)
                assert isinstance(chosen, decisions.FoodChoice)
                player.food[chosen.food] -= 1


# ---------------------------------------------------------------------------
# Module-level conveniences


def print_coverage_report() -> None:
    """Log how many of the loaded bird powers are modelled by the parser."""
    birds, _, _ = cards.load_all()
    impl, total = cards.power_coverage(birds)
    logger.info(f"Bird power coverage: {impl}/{total} ({impl*100//total}%) modelled")
