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

import itertools
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

        Agents may return either a ``Choice`` instance (preferred) or a raw
        integer index (CLI fallback). The returned Choice is compared by
        Pydantic field equality, so an agent that constructs its own Choice
        with identical fields still resolves to the corresponding slot in
        ``decision.choices``."""
        choice: C = agent(self, decision)
        if choice not in decision.choices:
            raise ValueError(
                f"agent returned illegal choice {choice.label!r} for "
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
        self._consume_extra_plays(player, agent)
        self.state.refill_tray()

    @staticmethod
    def _main_action_label(
        choice: decisions.MainActionChoice | decisions.PlayBirdChoice,
    ) -> str:
        """Log-header text for the chosen main action."""
        if isinstance(choice, decisions.PlayBirdChoice):
            return (
                f"PLAY_BIRD ({choice.bird.name} -> {choice.habitat.value}, "
                f"pay {choice.payment.format()})"
            )
        return choice.action.value.upper()

    def _dispatch_main_action(
        self,
        agent: Agent,
        choice: decisions.MainActionChoice | decisions.PlayBirdChoice,
    ) -> None:
        """Run the chosen main action. A ``PlayBirdChoice`` plays that specific
        bird in the chosen habitat for the chosen payment; a ``MainActionChoice``
        runs its habitat-row action."""
        if isinstance(choice, decisions.PlayBirdChoice):
            actions.do_play_bird(
                self, agent, choice.bird, choice.habitat, choice.payment
            )
        elif choice.action == decisions.MainAction.GAIN_FOOD:
            actions.do_gain_food(self, agent)
        elif choice.action == decisions.MainAction.LAY_EGGS:
            actions.do_lay_eggs(self, agent)
        elif choice.action == decisions.MainAction.DRAW_CARDS:
            actions.do_draw_cards(self, agent)

    def _main_action_decision(
        self, player: state.Player
    ) -> decisions.MainActionDecision:
        """Build the main-action Decision for ``player``.

        The three habitat-row actions are always offered (just inefficient if
        the row is empty); each legal way ``player`` can play a bird right now is
        also offered as its own ``PlayBirdChoice``, so the menu is "3 habitat
        actions + one option per (bird, habitat, food payment)". The egg cost
        is still resolved as a follow-up decision when the play runs."""
        choices: list[decisions.MainActionChoice | decisions.PlayBirdChoice] = [
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
        for bird, habitat, payment in actions.playable_bird_plays(
            player, habitat_filter=None
        ):
            choices.append(
                decisions.PlayBirdChoice(
                    label=(
                        f"play {bird.name} in {habitat.value} "
                        f"for {payment.format()}"
                    ),
                    bird=bird,
                    habitat=habitat,
                    payment=payment,
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
            self._deal_starting_hand(player)
            self._resolve_setup_choice(player, agents)
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

    def _consume_extra_plays(self, player: state.Player, agent: Agent) -> None:
        """Resolve any +extra-play credits accrued during the turn."""
        while self.state.turn_extra_plays > 0:
            self.state.turn_extra_plays -= 1
            if not actions.can_play_bird(self, player):
                self.state.turn_extra_play_habitat = None
                break
            habitat_filter = self.state.turn_extra_play_habitat
            if habitat_filter is not None:
                self.log(
                    f"[{player.name}] takes an EXTRA play in [{habitat_filter.value}]"
                )
            else:
                self.log(f"[{player.name}] takes an EXTRA play")
            actions.do_play_bird(self, agent)
            # Habitat lock applies to a single extra play only.
            self.state.turn_extra_play_habitat = None

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

    def _resolve_setup_choice(
        self,
        player: state.Player,
        agents: typing.Sequence[Agent],
    ) -> None:
        """Present the combined hand / food / bonus pick as a single Decision.

        We deal bonus cards, give the player one of each food, then enumerate
        every legal ``SetupChoice`` (kept-card subset × discarded-food subset
        of matching size × bonus card). For the default 5-card / 2-bonus deal
        that produces 2 * sum_k C(5,k)^2 = 504 choices, which matches the RL
        action space."""
        dealt_bonus = self._deal_starting_bonus()
        for food in cards.ALL_FOODS:
            player.food[food] = 1
        dealt_cards = list(player.hand)
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

        Iteration order is ``(kept_mask, kept_food_combo, bonus)`` so the list
        is deterministic for a given deal — useful when matching a CLI-assembled
        answer back to a Choice instance."""
        num_cards = len(dealt_cards)
        all_foods = list(cards.ALL_FOODS)
        bonuses: list[cards.BonusCard | None] = (
            list(dealt_bonus) if dealt_bonus else [None]
        )
        out: list[decisions.SetupChoice] = []
        for mask in range(1 << num_cards):
            kept = tuple(dealt_cards[i] for i in range(num_cards) if mask & (1 << i))
            kept_food_size = len(all_foods) - len(kept)
            for food_combo in itertools.combinations(all_foods, kept_food_size):
                for bc in bonuses:
                    out.append(
                        decisions.SetupChoice(
                            label=Engine._setup_choice_label(kept, food_combo, bc),
                            kept_cards=kept,
                            kept_foods=tuple(food_combo),
                            bonus_card=bc,
                        )
                    )
        return out

    @staticmethod
    def _setup_choice_label(
        kept_cards: tuple[cards.Bird, ...],
        kept_foods: tuple[cards.Food, ...],
        bonus_card: cards.BonusCard | None,
    ) -> str:
        kept_names = [bird.name for bird in kept_cards] or ["none"]
        food_names = [food.value for food in kept_foods] or ["none"]
        bonus = bonus_card.name if bonus_card is not None else "(none)"
        return (
            f"keep:[{','.join(kept_names)}] foods:[{','.join(food_names)}] "
            f"bonus:{bonus}"
        )

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
