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
        ``SetupDecision``."""
        eng = Engine(gs, agents=agents, instrumentation=instrumentation)
        eng.log_section("=== GAME START ===")
        eng.instrumentation.game_start(engine=eng)
        eng._setup_phase(agents, defer_bonus=split_setup_bonus)
        for round_idx in range(4):
            eng._play_round(round_idx, agents)
        scoring.final_scoring(eng)
        eng.state.game_over = True
        eng.log_section("=== GAME END ===")
        eng.instrumentation.game_end(engine=eng)
        return eng

    @staticmethod
    def play_one_game_with_setups(
        gs: state.GameState,
        agents: tuple[Agent, Agent],
        choose_setups: SetupChooser,
        instrumentation: dispatcher.Instrumentation | None = None,
    ) -> Engine:
        """Like :meth:`play_one_game`, but the setup phase is resolved by
        ``choose_setups`` (the random generator or the setup model) instead of by
        asking each agent — the setup-model collection path.

        The engine still deals the starting hands / bonus / food, so the chooser
        decides over exactly the inputs an agent would have seen; everything after
        setup is identical to ``play_one_game``."""
        eng = Engine(gs, agents=agents, instrumentation=instrumentation)
        eng.log_section("=== GAME START ===")
        eng.instrumentation.game_start(engine=eng)
        eng._setup_phase_fixed(choose_setups)
        for round_idx in range(4):
            eng._play_round(round_idx, agents)
        scoring.final_scoring(eng)
        eng.state.game_over = True
        eng.log_section("=== GAME END ===")
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

    def log(self, msg: str) -> None:
        self.state.log.append(msg)

    def log_section(self, msg: str) -> None:
        """Log a ``===`` section header, ensuring exactly one blank line before it.

        Consecutive ``log_section`` calls share a single blank line between
        them (no double-blanks); the first call in an empty log skips the
        leading blank entirely."""
        if self.state.log and self.state.log[-1] != "":
            self.log("")
        self.log(msg)

    def log_skipped_decision(self, player_id: int, reason: str) -> None:
        """Log that a decision point resolved without consulting the agent —
        either auto-picked (``ask``'s single-choice guard) or never built
        because no legal choice existed (handlers' empty-choices guards)."""
        self.log(f"[{self.state.players[player_id].name}] skipping decision, {reason}")

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
        self.log(f"--> {self._main_action_label(choice)}")
        player.action_cubes_left -= 1
        self._dispatch_main_action(agent, choice)
        actions.consume_extra_plays(self, player, agent)
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

    # ------------------------------------------------------------------
    # Setup (kept on Engine because it depends heavily on _ask)
    # ------------------------------------------------------------------

    def _setup_phase(
        self, agents: typing.Sequence[Agent], *, defer_bonus: bool = False
    ) -> None:
        """Pre-round-1 setup: deal each player a starting hand, prompt the
        combined keep-cards / discard-food / bonus-card pick, log the result.

        ``defer_bonus`` (the ``split_setup_bonus`` regime) drops the bonus from the
        ``SetupDecision`` and resolves it via a follow-up in-game ``CHOOSE_BONUS``
        pick instead."""
        for player in self.state.players:
            dealt_cards, dealt_bonus = self._deal_setup_inputs(player)
            if defer_bonus:
                self.log_section(
                    f"=== SETUP: {player.name} CHOOSING BIRDS AND FOOD ==="
                )
            else:
                self.log_section(
                    f"=== SETUP: {player.name} CHOOSING BIRDS, FOOD, AND BONUS CARD ==="
                )
            self._resolve_setup_choice(
                player, agents, dealt_cards, dealt_bonus, defer_bonus=defer_bonus
            )
            self._log_setup_result(player)

    def _setup_phase_fixed(self, choose_setups: SetupChooser) -> None:
        """Resolve setup from a chooser callback rather than agent prompts (the
        setup-model path): deal each player's inputs, ask the chooser for both
        seats' keeps over those inputs, then apply them. Skips ``Engine.ask``
        because both seats' setups are decided together up front.

        A keep whose ``bonus_card`` is ``None`` while bonus cards were dealt has
        deferred its bonus (the ``split_setup_bonus`` regime): the bonus is then
        picked via the in-game ``CHOOSE_BONUS`` head over the already-applied
        cards/food, recorded like any other in-game decision."""
        dealt = tuple(self._deal_setup_inputs(player) for player in self.state.players)
        keeps = choose_setups(self, dealt)
        for player in self.state.players:
            dealt_cards, dealt_bonus = dealt[player.id]
            sc = keeps[player.id].to_setup_choice()
            self.log_section(f"=== SETUP: {player.name} CHOOSING BIRDS AND FOOD ===")
            self._apply_setup_choice(player, dealt_cards, dealt_bonus, sc)
            self._maybe_resolve_deferred_setup_bonus(player, dealt_bonus, sc)
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
            f"({state.ROUND_CUBES[round_idx]} ACTIONS EACH) ==="
        )
        self.log(f"Round goal: {self.state.round_goals[round_idx].description}")
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
        regime."""
        choices = self._build_setup_choices(
            dealt_cards, dealt_bonus, include_bonus=not defer_bonus
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
        chosen = self.ask(agents[player.id], decision)
        self._apply_setup_choice(player, dealt_cards, dealt_bonus, chosen)
        kept_names = ", ".join(bird.name for bird in chosen.kept_cards) or "(none)"
        foods_str = ", ".join(food.value for food in chosen.kept_foods) or "none"
        bonus_part = (
            f" | bonus: {chosen.bonus_card.name}"
            if chosen.bonus_card is not None
            else ""
        )
        self.log(f"[{player.name}] keeps {kept_names}, foods [{foods_str}]{bonus_part}")
        self._maybe_resolve_deferred_setup_bonus(player, dealt_bonus, chosen)

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
        instead picked by the in-game ``CHOOSE_BONUS`` head."""
        from wingspan.setup_model import candidates as setup_candidates

        return [
            candidate.to_setup_choice()
            for candidate in setup_candidates.enumerate_setup_candidates(
                dealt_cards, dealt_bonus, include_bonus=include_bonus
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
            return
        self.log_section(f"=== SETUP: {player.name} CHOOSING BONUS CARD ===")
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
    ) -> None:
        """Mutate ``player`` / ``self.state`` to reflect the chosen setup combination."""
        kept = list(sc.kept_cards)
        player.hand = kept
        for card in dealt_cards:
            if card not in kept:
                self.state.bird_discard.append(card)
        for food in cards.ALL_FOODS:
            if food not in sc.kept_foods:
                player.food[food] -= 1
        if sc.bonus_card is not None:
            player.bonus_cards.append(sc.bonus_card)
            for bonus in dealt_bonus:
                if bonus is not sc.bonus_card:
                    self.state.bonus_discard.append(bonus)
        self.instrumentation.setup_applied(engine=self, player=player, choice=sc)


# ---------------------------------------------------------------------------
# Module-level conveniences


def print_coverage_report() -> None:
    """Log how many of the loaded bird powers are modelled by the parser."""
    birds, _, _ = cards.load_all()
    impl, total = cards.power_coverage(birds)
    logger.info(f"Bird power coverage: {impl}/{total} ({impl*100//total}%) modelled")
