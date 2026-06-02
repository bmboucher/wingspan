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
from wingspan.engine import actions, scoring

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
    ) -> Engine:
        """Construct an Engine on ``gs`` with ``agents``, run a full game,
        and return the engine. The caller's ``gs`` is mutated in place, so
        either the returned engine or any pre-existing reference to ``gs``
        can be used to inspect the final log and scores."""
        eng = Engine(gs, agents=agents)
        eng.log("=== Wingspan game start ===")
        eng._setup_phase(agents)
        for round_idx in range(4):
            eng._play_round(round_idx, agents)
        scoring.final_scoring(eng)
        eng.state.game_over = True
        eng.log("=== Wingspan game end ===")
        return eng

    @staticmethod
    def play_one_game_with_setups(
        gs: state.GameState,
        agents: tuple[Agent, Agent],
        choose_setups: SetupChooser,
    ) -> Engine:
        """Like :meth:`play_one_game`, but the setup phase is resolved by
        ``choose_setups`` (the random generator or the setup model) instead of by
        asking each agent — the setup-model collection path.

        The engine still deals the starting hands / bonus / food, so the chooser
        decides over exactly the inputs an agent would have seen; everything after
        setup is identical to ``play_one_game``."""
        eng = Engine(gs, agents=agents)
        eng.log("=== Wingspan game start ===")
        eng._setup_phase_fixed(choose_setups)
        for round_idx in range(4):
            eng._play_round(round_idx, agents)
        scoring.final_scoring(eng)
        eng.state.game_over = True
        eng.log("=== Wingspan game end ===")
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
        agent at all. This keeps every front-end consistent: the interactive
        CLI never prompts a human to "pick" the only option, and the RL
        collector only ever sees genuine forks, so no zero-signal steps are
        recorded (DECISIONS.md §1.4) and a forced move costs no forward pass
        during self-play. Because ``ask`` is the single choke point every
        agent call routes through, this one guard covers every decision type.

        Otherwise the answer is compared to ``decision.choices`` by Pydantic
        field equality, so an agent that constructs its own Choice with
        identical fields still resolves to the corresponding offered slot."""
        if len(decision.choices) == 1:
            return decision.choices[0]
        choice: C = agent(self, decision)
        if choice not in decision.choices:
            raise ValueError(
                f"agent returned illegal choice {choice.display_label()!r} for "
                f"{type(decision).__name__}"
            )
        return choice

    def log(self, msg: str) -> None:
        self.state.log.append(msg)

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
        # Blank separator + combined turn/decision header so each turn is one
        # visually scannable block: `[Pn] turn (X cubes) --> ACTION` followed
        # by the indented action result and any sub-events.
        self.log("")
        choice = self.ask(agent, self._main_action_decision(player))
        self.log(
            f"[{player.name}] turn ({player.action_cubes_left} cubes left) "
            f"--> {self._main_action_label(choice)}"
        )
        player.action_cubes_left -= 1
        self._dispatch_main_action(agent, choice)
        actions.consume_extra_plays(self, player, agent)
        self.state.refill_tray()

    @staticmethod
    def _main_action_label(choice: decisions.MainActionChoice) -> str:
        """Log-header text for the chosen main-action type. For ``PLAY_BIRD`` the
        specific bird / habitat / payment is logged later by ``do_play_bird``
        once the follow-up ``PlayBirdDecision`` resolves."""
        return choice.action.value.upper()

    def _dispatch_main_action(
        self,
        agent: Agent,
        choice: decisions.MainActionChoice,
    ) -> None:
        """Run the chosen main action. ``PLAY_BIRD`` opens the follow-up
        ``PlayBirdDecision`` (which bird, where, paid how) and plays it; the
        other three run their habitat-row action."""
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
        ``PlayBirdDecision`` for the specific bird / habitat / payment."""
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

    def _setup_phase(self, agents: typing.Sequence[Agent]) -> None:
        """Pre-round-1 setup: deal each player a starting hand, prompt the
        combined keep-cards / discard-food / bonus-card pick, log the result."""
        for player in self.state.players:
            dealt_cards, dealt_bonus = self._deal_setup_inputs(player)
            self._resolve_setup_choice(player, agents, dealt_cards, dealt_bonus)
            self._log_setup_result(player)

    def _setup_phase_fixed(self, choose_setups: SetupChooser) -> None:
        """Resolve setup from a chooser callback rather than agent prompts (the
        setup-model path): deal each player's inputs, ask the chooser for both
        seats' keeps over those inputs, then apply them. Skips ``Engine.ask``
        because both seats' setups are decided together up front."""
        dealt = tuple(self._deal_setup_inputs(player) for player in self.state.players)
        keeps = choose_setups(self, dealt)
        for player in self.state.players:
            dealt_cards, dealt_bonus = dealt[player.id]
            self._apply_setup_choice(
                player, dealt_cards, dealt_bonus, keeps[player.id].to_setup_choice()
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
        self.log(
            f"--- Round {round_idx + 1} "
            f"(each player gets {state.ROUND_CUBES[round_idx]} actions) ---"
        )
        self.log(
            f"Round goal: {self.state.round_goals[round_idx].description} "
            f"({self.state.round_goals[round_idx].category})"
        )
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
    ) -> None:
        """Present the combined hand / food / bonus pick as a single Decision over
        the already-dealt inputs.

        Enumerates every legal ``SetupChoice`` (kept-card subset × retained-food
        subset of matching size × bonus card). For the default 5-card / 2-bonus
        deal that produces 2 * sum_k C(5,k)^2 = 504 choices, which matches the RL
        action space."""
        choices = self._build_setup_choices(dealt_cards, dealt_bonus)
        self.state.current_player = player.id
        decision = decisions.SetupDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] choose starting hand (kept cards cost 1 food each) "
                f"and bonus card"
            ),
            choices=choices,
            dealt_cards=dealt_cards,
            dealt_bonus=dealt_bonus,
        )
        chosen = self.ask(agents[player.id], decision)
        self._apply_setup_choice(player, dealt_cards, dealt_bonus, chosen)
        bonus_name = chosen.bonus_card.name if chosen.bonus_card else "(none)"
        self.log(
            f"[{player.name}] keeps {len(chosen.kept_cards)} card(s), "
            f"foods [{','.join(food.value for food in chosen.kept_foods) or 'none'}], "
            f"bonus '{bonus_name}'"
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
    ) -> list[decisions.SetupChoice]:
        """Enumerate every legal ``SetupChoice``.

        Delegates to ``setup_model.candidates.enumerate_setup_candidates`` so the
        504-candidate set (and its ``(kept_mask, kept_food_combo, bonus)`` order)
        has a single source of truth shared with the setup model — which scores
        exactly these candidates. Imported lazily to avoid an import cycle (the
        setup-model package transitively imports the engine). Each candidate
        renders to a ``SetupChoice`` whose label is built lazily by
        ``display_label`` (only the chosen / displayed option ever needs it)."""
        from wingspan.setup_model import candidates as setup_candidates

        return [
            candidate.to_setup_choice()
            for candidate in setup_candidates.enumerate_setup_candidates(
                dealt_cards, dealt_bonus
            )
        ]

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


# ---------------------------------------------------------------------------
# Module-level conveniences


def print_coverage_report() -> None:
    """Log how many of the loaded bird powers are modelled by the parser."""
    birds, _, _ = cards.load_all()
    impl, total = cards.power_coverage(birds)
    logger.info(f"Bird power coverage: {impl}/{total} ({impl*100//total}%) modelled")
