"""Self-play data collection.

Plays one full self-play game where both seats consult the same network, and
records every multi-option decision as a :class:`wingspan.training.steps.Step` (state
features, candidate features, chosen index, player id, judgment-family head
index). After the game it reads each player's final board into a
:class:`metrics.ScoreBreakdown`, so the loop can both train on the trajectory
and report the score's six-way split live.

Single-option forced moves are not recorded — the trainable surface is the
moments with a genuine fork (DECISIONS.md §0).

The bundled card catalog is parsed once and reused across games (the card
models are immutable and ``state.new_game`` copies the deck lists before
shuffling), which avoids re-reading the JSON on every game — the dominant
fixed cost of ``Engine.create``.

This single-game collector is the baseline the two scaled collectors relate to:
``mp_collect`` runs :func:`play_game` itself inside each worker process, while
``batched_collect`` reimplements the per-game loop around one shared forward
pass. Which one the training loop uses is decided per device in
``loop._collect`` (CPU → ``mp_collect``; CUDA → ``batched_collect``). See
``training/COLLECTORS.md`` for the side-by-side.
"""

from __future__ import annotations

import enum
import functools
import random
import typing

import numpy as np
import pydantic
import torch

from wingspan import cards, decisions, engine, model, setup_model, state
from wingspan.engine import scoring

# ``steps`` is aliased because ``GameRecord.steps`` (and the local recording
# lists) would shadow the bare module name (the ``core as engine_core`` rule).
from wingspan.training import config, metrics, policy, setup_net
from wingspan.training import steps as training_steps
from wingspan.training import timestamps

# Distinct salt for the setup-selection RNG (random generator + setup-net sampling),
# kept separate from the in-game sampling and opponent salts so a seed reproduces
# the setup, the game, and the opponent without the streams sharing a sequence.
_SETUP_RNG_SALT = 0xC2B2AE35
# Salt for the post-setup continuation reseed, distinct from the (unsalted)
# main-net sampling stream so a game's dice/draw rerolls and its policy sampling
# draw from independent sequences off the same continuation seed.
_CONTINUATION_SALT = 0x27D4EB2F
# Offset that separates a batch's shared deal seed from the per-game seeds within
# the same iteration (game seeds occupy ``base + [0, games_per_iter)``; batch deal
# seeds occupy ``base + OFFSET + [0, n_batches)`` — disjoint, same iteration stride).
_BATCH_SEED_OFFSET = 5000


class GameRecord(pydantic.BaseModel):
    """One finished self-play game: its recorded steps plus the per-player
    final score breakdown, the winner (0, 1, or -1 for a tie), and the
    board-shuffle ``seed`` that produced it (carried so the persisted per-game
    history row stays independently reproducible)."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    steps: list[training_steps.Step]
    breakdowns: tuple[metrics.ScoreBreakdown, metrics.ScoreBreakdown]
    winner: int
    seed: int
    # Setup-model samples recorded this game (one per net-controlled seat, only
    # in the recording / model-driven phases); empty when the setup model is off
    # or in the unrecorded random phase.
    setup_samples: list[setup_model.SetupSample] = pydantic.Field(
        default_factory=list[setup_model.SetupSample]
    )
    # Game-clock time of the terminal margin checkpoint: the end of the final
    # turn's window (``timestamps.final_timestamp``). Consumed only by the
    # ``decision_delta`` reward mode's λ^Δt discounting; the default keeps
    # hand-built fixtures that omit it valid.
    final_timestamp: float = 0.0

    @property
    def scores(self) -> tuple[int, int]:
        return (round(self.breakdowns[0].total), round(self.breakdowns[1].total))


class SetupPhase(enum.IntEnum):
    """Which setup regime a game is collected under (a pure function of the
    lifetime iteration vs the configured thresholds; computed by the loop)."""

    RANDOM_NO_RECORD = 0  # iter < record_start: random setups, not recorded
    RANDOM_RECORD = 1  # record_start <= iter < train: random setups, recorded
    MODEL_DRIVEN = 2  # iter >= train: the setup net chooses + records on-policy

    @property
    def records(self) -> bool:
        """Whether games in this phase contribute setup-model training samples."""
        return self is not SetupPhase.RANDOM_NO_RECORD


class SetupGameSpec(pydantic.BaseModel):
    """How one game's setups are decided — the picklable per-game directive the
    loop computes and the collectors (sequential or process-parallel) act on.

    ``deal_seed`` produces the deal (shared across a batch in the random phases so
    its games compare different keeps over one deal); ``continuation_seed`` reseeds
    the post-setup game so those games still diverge. ``tuple_index`` picks this
    game's joint setup out of the batch's generated set (random phases only)."""

    model_config = pydantic.ConfigDict(frozen=True)

    phase: SetupPhase
    deal_seed: int
    continuation_seed: int
    tuple_index: int
    iteration: int


def build_setup_specs(
    cfg: config.RunConfig, iteration: int, phase: SetupPhase
) -> list[SetupGameSpec]:
    """One :class:`SetupGameSpec` per game this iteration.

    In the random phases, games are grouped into shared-deal batches of
    ``setup_tuples_per_batch``: the games of a batch share a deal seed (so they
    explore that deal's keeps) but keep distinct continuation seeds. In the
    model-driven phase each game deals independently and the setup net chooses per
    seat, so deal and continuation seeds coincide."""
    base = cfg.misc.seed * 1_000_000 + iteration * 10_000
    tuples_per_batch = cfg.training.setup.tuples_per_batch
    specs: list[SetupGameSpec] = []
    for game_idx in range(cfg.run.games_per_iter):
        game_seed = base + game_idx
        if phase is SetupPhase.MODEL_DRIVEN:
            specs.append(
                SetupGameSpec(
                    phase=phase,
                    deal_seed=game_seed,
                    continuation_seed=game_seed,
                    tuple_index=0,
                    iteration=iteration,
                )
            )
        else:
            batch_seed = base + _BATCH_SEED_OFFSET + game_idx // tuples_per_batch
            specs.append(
                SetupGameSpec(
                    phase=phase,
                    deal_seed=batch_seed,
                    continuation_seed=game_seed,
                    tuple_index=game_idx % tuples_per_batch,
                    iteration=iteration,
                )
            )
    return specs


def play_game(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    seed: int,
    opponent_agent: engine.Agent | None = None,
) -> GameRecord:
    """Play one game and return its recorded transitions + scores.

    With ``opponent_agent`` omitted this is ordinary self-play: both seats
    consult the policy and every multi-option decision is recorded. With an
    ``opponent_agent`` (the random-opponent bootstrap phase), the net plays
    seat 0 and ``opponent_agent`` plays seat 1; only the net's decisions are
    recorded, since the opponent's off-policy moves are not trained on.
    """
    eng = new_engine(seed)
    recorded: list[training_steps.Step] = []
    net_agent = _recording_agent(net, device, rng, recorded)
    agent_a, agent_b = (
        (net_agent, net_agent)
        if opponent_agent is None
        else (net_agent, opponent_agent)
    )
    engine.Engine.play_one_game(eng.state, (agent_a, agent_b))
    timestamps.finalize_timestamps(recorded)

    breakdowns = (
        player_breakdown(eng.state.players[0]),
        player_breakdown(eng.state.players[1]),
    )
    score_0, score_1 = breakdowns[0].total, breakdowns[1].total
    winner = 0 if score_0 > score_1 else (1 if score_1 > score_0 else -1)
    return GameRecord(
        steps=recorded,
        breakdowns=breakdowns,
        winner=winner,
        seed=seed,
        final_timestamp=timestamps.final_timestamp(eng.state.turn_counter),
    )


def play_game_with_setup(
    net: model.PolicyValueNet,
    device: torch.device,
    spec: SetupGameSpec,
    generator: setup_model.RandomSetupGenerator,
    setup_policy_net: setup_net.SetupNet | None,
    setup_temperature: float,
    opponent_agent: engine.Agent | None = None,
    split_setup_bonus: bool = False,
    split_setup_food: bool = False,
    setup_greedy: bool = False,
    use_actor_critic: bool = False,
) -> GameRecord:
    """Play one game whose setups are chosen externally (the setup-model path).

    The in-game decisions are still recorded for the main net exactly as in
    :func:`play_game`; the setup phase is bypassed (no ``SetupDecision`` is ever
    asked) and resolved by the random generator or the setup net per ``spec``.
    Per net-controlled seat (seat 0 always; seat 1 too in self-play) a
    ``SetupSample`` is recorded in the recording / model-driven phases, with its
    realized margin filled in once the game's scores are known.

    ``split_setup_bonus`` defers each net seat's bonus pick out of the setup keep
    (its candidate carries ``bonus_card=None``) to the engine's in-game
    ``CHOOSE_BONUS`` head; a random-opponent seat keeps its generator-chosen
    bonus. The deferred pick is then a recorded in-game step like any other.

    ``split_setup_food`` defers each net seat's food pick to sequential in-game
    GAIN_FOOD/SPEND_FOOD decisions after the card-keep is applied; a
    random-opponent seat likewise has its food resolved by the engine.

    ``use_actor_critic`` enables actor-critic collection: selection uses the
    policy head's logits, and each recorded ``SetupSample`` carries ``chosen_idx``
    and ``all_candidates`` so the learner can compute a REINFORCE gradient."""
    eng = new_engine(spec.deal_seed)
    main_rng = random.Random(spec.continuation_seed)
    recorded: list[training_steps.Step] = []
    net_agent = _recording_agent(net, device, main_rng, recorded)
    if opponent_agent is None:
        agent_a, agent_b = net_agent, net_agent
        net_seats = (0, 1)
    else:
        agent_a, agent_b = net_agent, opponent_agent
        net_seats = (0,)

    setup_rng = random.Random(spec.deal_seed ^ _SETUP_RNG_SALT)
    # Pending entries: (seat, chosen_features, chosen_idx | None, all_candidates | None)
    pending_setups: list[tuple[int, np.ndarray, int | None, np.ndarray | None]] = []

    # Encoding derived from the net (when available) or from the active flags.
    # Used consistently for both candidate scoring and feature recording so the
    # stored samples always match what the net expects.
    setup_enc = (
        setup_policy_net.encoding
        if setup_policy_net is not None
        else setup_model.SetupEncoding(
            split_food=split_setup_food, split_bonus=split_setup_bonus
        )
    )

    def choose_setups(
        chooser_engine: engine.Engine,
        dealt: tuple[tuple[list[cards.Bird], list[cards.BonusCard]], ...],
    ) -> list[setup_model.SetupCandidate]:
        # Shared context: tray / birdfeeder / round goals — no per-seat bonus
        # cards here; those are attached per-seat inside _choose_setups and in
        # the recording branch below.
        base_context = setup_model.SetupContext.from_state(chooser_engine.state)
        keep_results = _choose_setups(
            spec,
            dealt,
            base_context,
            setup_enc=setup_enc,
            net_seats=net_seats,
            generator=generator,
            setup_policy_net=setup_policy_net,
            setup_temperature=setup_temperature,
            setup_rng=setup_rng,
            device=device,
            defer_bonus=split_setup_bonus,
            defer_food=split_setup_food,
            setup_greedy=setup_greedy,
            use_actor_critic=use_actor_critic,
        )
        if spec.phase.records:
            for seat in net_seats:
                result = keep_results[seat]
                _, seat_dealt_bonus = dealt[seat]
                seat_context = base_context.model_copy(
                    update={"dealt_bonus_cards": tuple(seat_dealt_bonus)}
                )
                chosen_features = setup_model.encode_setup_candidate(
                    result.candidate, seat_context, setup_enc
                )
                pending_setups.append(
                    (seat, chosen_features, result.chosen_idx, result.all_candidates)
                )
        # Shared deal, independent continuation: reseed the post-setup game (and
        # reshuffle the undealt deck) so a batch's games — which share ``deal_seed``
        # and thus the deal — still diverge through the rest of play.
        chooser_engine.state.rng = random.Random(
            spec.continuation_seed ^ _CONTINUATION_SALT
        )
        chooser_engine.state.rng.shuffle(chooser_engine.state.bird_deck)
        return [result.candidate for result in keep_results]

    engine.Engine.play_one_game_with_setups(
        eng.state, (agent_a, agent_b), choose_setups, split_setup_food=split_setup_food
    )
    timestamps.finalize_timestamps(recorded)

    breakdowns = (
        player_breakdown(eng.state.players[0]),
        player_breakdown(eng.state.players[1]),
    )
    score_0, score_1 = breakdowns[0].total, breakdowns[1].total
    winner = 0 if score_0 > score_1 else (1 if score_1 > score_0 else -1)
    totals = (score_0, score_1)
    setup_samples = [
        setup_model.SetupSample(
            features=chosen_features,
            margin=totals[seat] - totals[1 - seat],
            iteration=spec.iteration,
            chosen_idx=chosen_idx,
            all_candidates=all_candidates,
        )
        for seat, chosen_features, chosen_idx, all_candidates in pending_setups
    ]
    return GameRecord(
        steps=recorded,
        breakdowns=breakdowns,
        winner=winner,
        seed=spec.continuation_seed,
        setup_samples=setup_samples,
        final_timestamp=timestamps.final_timestamp(eng.state.turn_counter),
    )


def running_margin(game: state.GameState, player_id: int) -> float:
    """``player_id``'s live score margin (own − opponent) if the game ended now.

    Recorded as each :class:`training_steps.Step`'s ``margin_before`` and
    differenced into the per-decision ``decision_delta`` return with
    ``MARGIN`` basis (``learner._flatten``). Shared by both recording agents
    so the sequential and batched collectors snapshot the margin identically."""
    own = scoring.running_score(game.players[player_id])
    opponent = scoring.running_score(game.players[1 - player_id])
    return float(own - opponent)


def running_own_score(game: state.GameState, player_id: int) -> float:
    """``player_id``'s own live score if the game ended now.

    Recorded as each :class:`training_steps.Step`'s ``score_before`` and
    differenced into the per-decision ``decision_delta`` return with
    ``OWN_SCORE`` basis (``learner._flatten``)."""
    return float(scoring.running_score(game.players[player_id]))


def player_breakdown(player: state.Player) -> metrics.ScoreBreakdown:
    """Split ``player``'s final score into its six sources — the exact terms
    ``engine.scoring.final_scoring`` sums (birds + bonus + eggs + tucked +
    cached-food + round-goal)."""
    bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
    bonus_pts = sum(scoring.bonus_score(player, bc) for bc in player.bonus_cards)
    return metrics.ScoreBreakdown(
        birds=float(bird_pts),
        eggs=float(player.total_eggs),
        cached=float(player.total_cached),
        tucked=float(player.total_tucked),
        goals=float(player.round_goal_points),
        bonus=float(bonus_pts),
    )


def new_engine(seed: int) -> engine.Engine:
    """Construct a fresh game engine on a seeded shuffle of the cached catalog."""
    birds, bonuses, goals = _catalog()
    game = state.new_game(random.Random(seed), list(birds), list(bonuses), list(goals))
    return engine.Engine(game)


###### PRIVATE #######


@functools.lru_cache(maxsize=1)
def _catalog() -> (
    tuple[list[cards.Bird], list[cards.BonusCard], list[cards.EndRoundGoal]]
):
    """Parse the bundled card catalog once and reuse it across every game."""
    return cards.load_all()


def _recording_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    record_into: list[training_steps.Step],
) -> engine.Agent:
    """An agent that samples from the policy and appends every multi-option
    decision it makes to ``record_into`` (both seats share the buffer, tagged
    by ``player_id``)."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            # The net's encoding excludes setup (the opening is the separate setup
            # model's job); resolve a SetupDecision reached via ``play_game`` with a
            # random keep rather than scoring it. The setup-model path bypasses the
            # opening in the engine, so this only fires for a setup-excluded net
            # collected through the plain (non-setup-model) game path.
            return decisions.random_choice(decision, eng.state.rng)
        family_idx = decisions.family_index_for(type(decision))
        state_vec = net.encode_state(eng.state, decision)
        choice_feats = net.encode_choices(decision, eng.state)
        chosen_idx = policy.sample_action(
            net, device, state_vec, choice_feats, family_idx, rng
        )
        record_into.append(
            training_steps.Step(
                state=state_vec,
                choices=choice_feats,
                chosen_idx=chosen_idx,
                player_id=decision.player_id,
                family_idx=family_idx,
                margin_before=running_margin(eng.state, decision.player_id),
                score_before=running_own_score(eng.state, decision.player_id),
                timestamp=timestamps.provisional_timestamp(
                    decision, eng.state.turn_counter
                ),
            )
        )
        return decision.choices[chosen_idx]

    return agent


class _KeepResult(typing.NamedTuple):
    """The result of choosing one seat's setup: the chosen candidate plus
    optional actor-critic data (populated only when ``use_actor_critic=True``
    and the phase is MODEL_DRIVEN for a net-controlled seat)."""

    candidate: setup_model.SetupCandidate
    # Index into all_candidates that was selected; None outside actor-critic mode.
    chosen_idx: int | None
    # Full (K, feature_dim) matrix of every candidate's features; None outside
    # actor-critic mode. Compressed to float16 by _compact before IPC.
    all_candidates: np.ndarray | None


def _choose_setups(
    spec: SetupGameSpec,
    dealt: tuple[tuple[list[cards.Bird], list[cards.BonusCard]], ...],
    context: setup_model.SetupContext,
    *,
    setup_enc: setup_model.SetupEncoding,
    net_seats: tuple[int, ...],
    generator: setup_model.RandomSetupGenerator,
    setup_policy_net: setup_net.SetupNet | None,
    setup_temperature: float,
    setup_rng: random.Random,
    device: torch.device,
    defer_bonus: bool = False,
    defer_food: bool = False,
    setup_greedy: bool = False,
    use_actor_critic: bool = False,
) -> list[_KeepResult]:
    """Decide both seats' setups for one game.

    Random phases draw a joint setup from the generator (the batch's ``deal_seed``
    seeds it, so a batch's games share the generated set and ``tuple_index`` picks
    this game's); the model-driven phase scores each net seat's 504 candidates
    with the setup net and samples (softmax over predicted margins or policy
    logits), while any random-opponent seat keeps a food-aware random keep.

    ``defer_bonus`` (the ``split_setup_bonus`` regime) removes the bonus from each
    net seat's keep so the engine's in-game ``CHOOSE_BONUS`` head picks it instead:
    the random generator still produces full 3-tuples (its tuple/games-per-batch
    counts are unchanged) but a net seat's chosen candidate is bonus-stripped here,
    and the model-driven path scores bonus-free candidates directly. A
    random-opponent seat is never touched, so it keeps its generator-chosen bonus.

    ``defer_food`` (the ``split_setup_food`` regime) analogously strips food from
    net seat candidates (``kept_foods=()``). The generator already produces food-
    free candidates when ``split_food`` is set; the strip here covers the MODEL_DRIVEN
    path's enumerated candidates and the RANDOM path's net-seat candidates.

    When ``use_actor_critic=True`` and the phase is MODEL_DRIVEN, net-seat results
    carry ``chosen_idx`` and ``all_candidates`` so the learner can compute a
    REINFORCE gradient at training time."""
    if spec.phase is not SetupPhase.MODEL_DRIVEN:
        joint = generator.generate(setup_rng, (dealt[0], dealt[1]), context)
        chosen = joint[spec.tuple_index % len(joint)]
        results: list[_KeepResult] = []
        for seat, candidate in enumerate((chosen[0], chosen[1])):
            if seat in net_seats:
                if defer_bonus:
                    candidate = _strip_bonus(candidate)
                if defer_food:
                    candidate = _strip_food(candidate)
            results.append(_KeepResult(candidate, None, None))
        return results

    assert setup_policy_net is not None, "model-driven setup needs a setup net"
    keeps: list[_KeepResult] = []
    for seat in (0, 1):
        dealt_cards, dealt_bonus = dealt[seat]
        if seat in net_seats:
            # Build a seat-specific context with this seat's bonus cards so the
            # bonus_cards multi-hot and affinity stripes are filled correctly.
            seat_context = context.model_copy(
                update={"dealt_bonus_cards": tuple(dealt_bonus)}
            )
            candidates = setup_model.enumerate_setup_candidates(
                dealt_cards,
                dealt_bonus,
                include_bonus=not defer_bonus,
                include_food=not defer_food,
            )
            features = np.stack(
                [
                    setup_model.encode_setup_candidate(c, seat_context, setup_enc)
                    for c in candidates
                ]
            )

            # Use policy logits for selection when actor-critic is enabled,
            # otherwise fall back to predicted value margins.
            if use_actor_critic:
                scores = _setup_policy_logits(setup_policy_net, device, features)
            else:
                scores = _setup_predict(setup_policy_net, device, features)
            sample_rng = None if setup_greedy else setup_rng
            index = setup_model.select_by_margins(scores, setup_temperature, sample_rng)
            keeps.append(
                _KeepResult(
                    candidates[index],
                    index if use_actor_critic else None,
                    features if use_actor_critic else None,
                )
            )
        else:
            keeps.append(
                _KeepResult(
                    generator.generate_one(
                        setup_rng, (dealt_cards, dealt_bonus), context
                    ),
                    None,
                    None,
                )
            )
    return keeps


def _strip_bonus(
    candidate: setup_model.SetupCandidate,
) -> setup_model.SetupCandidate:
    """A copy of ``candidate`` with its bonus pick dropped (``bonus_card=None``),
    so the engine defers it to the in-game ``CHOOSE_BONUS`` head. ``SetupCandidate``
    is frozen, so this returns a new instance."""
    return candidate.model_copy(update={"bonus_card": None})


def _strip_food(
    candidate: setup_model.SetupCandidate,
) -> setup_model.SetupCandidate:
    """A copy of ``candidate`` with food deferred (``kept_foods=()``), so the
    engine resolves food via in-game GAIN_FOOD/SPEND_FOOD decisions instead.
    ``SetupCandidate`` is frozen, so this returns a new instance."""
    return candidate.model_copy(update={"kept_foods": ()})


def _setup_predict(
    setup_policy_net: setup_net.SetupNet, device: torch.device, features: np.ndarray
) -> np.ndarray:
    """Forward a candidate feature matrix ``(K, feature_dim)`` through the setup
    net and return the ``(K,)`` predicted margins (value head)."""
    with torch.no_grad():
        feats_t = torch.tensor(features, dtype=torch.float32, device=device)
        return setup_policy_net(feats_t).cpu().numpy()


def _setup_policy_logits(
    setup_policy_net: setup_net.SetupNet, device: torch.device, features: np.ndarray
) -> np.ndarray:
    """Forward ``(K, feature_dim)`` features through the policy head and return
    the ``(K,)`` logits used for softmax candidate selection (actor-critic mode)."""
    with torch.no_grad():
        feats_t = torch.tensor(features, dtype=torch.float32, device=device)
        policy_logits, _ = setup_policy_net.policy_and_value(feats_t)
        return policy_logits.cpu().numpy()
