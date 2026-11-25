"""The forced-card bias study (``docs/RESEARCH.md``, "Forced-card bias").

Measures whether pushing the model to play one specific bird (default Cedar
Waxwing) lowers its win rate, as the push scales from ``x = 0`` to ``x = 1``, and
what steering it away from the bird does (``x`` down to ``-1``). Each game is
sampled self-play (every decision, the setup keep included, is drawn from the
softmax, never the argmax) between a *biased* seat and unbiased
opponents running the same nets. At a decision that matches the profile the biased
seat's probabilities are reweighted: every conformant option's probability is
multiplied by ``1 + x`` and every non-conformant option's by ``1 - x``, then
renormalised; ``x`` lies in ``[-1, 1]`` and a negative ``x`` avoids the bird
(``x = -1`` never picks it while an alternative exists). The profile is two
decisions:

* ``MainActionDecision``; the conformant option is ``PLAY_BIRD``. For ``x >= 0``
  it matches when the seat can legally play the forced bird now. For ``x < 0`` it
  matches only when the forced bird is the *sole* playable bird, so avoiding the
  card does not also suppress the seat's other bird plays; with other birds
  playable the main action is left unbiased.
* ``PlayBirdDecision`` when any option plays the forced bird (this covers
  power-granted extra plays); the conformant options are those plays, in any
  habitat.

Later decisions (habitat, eggs, food payment) are not biased. By default the
biased seat alternates by game parity (``game_index % num_players``); with
``BiasedSeats.ALL`` every seat is biased. Games use the same seeds
``spec.game_seed(index)`` at every factor, so factors are paired.

To raise how often the forced bird is dealt early, :func:`place_forced_card`
swaps it into a uniformly random position among the first ``forced_card_found_in``
cards (the tray, then the deck from the top) before play starts. The position is
drawn from an RNG seeded by the game seed alone, so every factor and seat sees the
same deck.

Net loading, sharding, and the process pool are shared with the other studies
(:mod:`wingspan.research.game_stats`, :mod:`wingspan.research.shard_runner`).
:mod:`wingspan.research.forced_card_agg` turns each factor's raw table into the
combined summary.
"""

from __future__ import annotations

import functools
import pathlib
import random
import time
import typing

import numpy as np
import torch

from wingspan import cards, decisions, engine, model, state
from wingspan.engine import actions
from wingspan.instrumentation.handlers import card_stats
from wingspan.research import constants, game_stats, models, shard_runner
from wingspan.training import collect, policy
from wingspan.training import setup_net as setup_net_module


def run_forced_card_study(
    spec: models.GameStatsSpec,
    forced: models.ForcedCardSpec,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    out_path: pathlib.Path,
    *,
    progress_stream: typing.TextIO | None = None,
) -> models.GameStatsSummary:
    """Play the study at one factor against the run in ``checkpoint_dir`` and
    write the raw per-seat table to ``out_path``.

    Raises before the output file is created when the run cannot be loaded."""
    started = time.monotonic()
    nets = game_stats.load_game_nets(checkpoint_dir, device)
    rows = shard_runner.stream_study(
        spec=spec,
        shards=game_stats.plan_shards(spec, constants.CARD_STATS_GAMES_PER_SHARD),
        nets=nets,
        load_nets=game_stats.load_game_nets,
        study_shard=functools.partial(study_shard, forced=forced),
        checkpoint_dir=checkpoint_dir,
        device=device,
        out_path=out_path,
        columns=models.ForcedCardSeatRecord.csv_columns(),
        to_row=models.ForcedCardSeatRecord.to_csv_row,
        progress_stream=progress_stream,
    )
    return models.GameStatsSummary(
        out_path=str(out_path),
        games=spec.games,
        rows=rows,
        workers=spec.workers,
        elapsed_seconds=time.monotonic() - started,
    )


def resolve_forced_card_id(requested: int | None) -> int:
    """The ``cards.Bird.id`` to force: ``requested`` when it names a catalog
    bird, else the id of ``constants.DEFAULT_FORCED_CARD_NAME`` looked up by name.

    Raises ``ValueError`` for an id absent from the catalog."""
    catalog = cards.birds_ordered()
    if requested is None:
        (default,) = [
            bird for bird in catalog if bird.name == constants.DEFAULT_FORCED_CARD_NAME
        ]
        return default.id
    if requested not in {bird.id for bird in catalog}:
        raise ValueError(
            f"forced card id {requested} is not a bird in the card database"
        )
    return requested


def bird_name(forced_card_id: int) -> str:
    """The catalog name of the bird with id ``forced_card_id``."""
    (bird,) = [bird for bird in cards.birds_ordered() if bird.id == forced_card_id]
    return bird.name


def bias_probs(
    probs: np.ndarray, conformant: np.ndarray, force_card_factor: float
) -> np.ndarray:
    """Reweight ``probs``: conformant options by ``1 + x``, the rest by ``1 - x``,
    renormalised to sum to 1. ``x = 0`` is the identity.

    ``conformant`` is a boolean mask aligned to ``probs``. When the reweighted
    total is zero or non-finite the result falls back to: for ``x > 0`` (no
    conformant mass at ``x = 1``) uniform over the conformant options; for
    ``x < 0`` uniform over the non-conformant options, or the input unchanged when
    every option is conformant (avoiding is impossible). With no conformant option
    at all the input is returned unchanged. Raises ``ValueError`` for ``x``
    outside ``[-1, 1]``."""
    if not (
        constants.MIN_FORCE_CARD_FACTOR
        <= force_card_factor
        <= constants.MAX_FORCE_CARD_FACTOR
    ):
        raise ValueError(
            f"force_card_factor {force_card_factor} is outside "
            f"[{constants.MIN_FORCE_CARD_FACTOR:g}, {constants.MAX_FORCE_CARD_FACTOR:g}]"
        )
    if not conformant.any():
        return probs
    weights = np.where(conformant, 1.0 + force_card_factor, 1.0 - force_card_factor)
    reweighted = probs.astype(np.float64) * weights
    total = float(reweighted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _degenerate_fallback(probs, conformant, force_card_factor)
    return reweighted / total


def conformant_mask[C: decisions.Choice](
    eng: engine.Engine,
    decision: decisions.Decision[C],
    forced_card_id: int,
    force_card_factor: float,
) -> np.ndarray | None:
    """The boolean mask of ``decision``'s options that play the forced bird, or
    ``None`` when the decision is outside the bias profile (another decision
    class, a main action the sign of ``force_card_factor`` does not match, or a
    play menu without the bird)."""
    if isinstance(decision, decisions.MainActionDecision):
        return main_action_mask(eng, decision, forced_card_id, force_card_factor)
    if isinstance(decision, decisions.PlayBirdDecision):
        return play_bird_mask(decision, forced_card_id)
    return None


def main_action_mask(
    eng: engine.Engine,
    decision: decisions.MainActionDecision,
    forced_card_id: int,
    force_card_factor: float,
) -> np.ndarray | None:
    """``PLAY_BIRD`` is the conformant option when the deciding player's legal
    plays include the forced bird (``force_card_factor >= 0``) or consist only of
    it (``force_card_factor < 0``, so avoiding the card leaves other bird plays
    unbiased); ``None`` otherwise."""
    player = eng.state.players[decision.player_id]
    plays = actions.playable_bird_plays(player, None)
    forced_plays = [bird for bird, _ in plays if bird.id == forced_card_id]
    if not forced_plays:
        return None
    if force_card_factor < 0.0 and len(forced_plays) != len(plays):
        return None
    mask = np.array(
        [choice.action == decisions.MainAction.PLAY_BIRD for choice in decision.choices]
    )
    return mask if mask.any() else None


def play_bird_mask(
    decision: decisions.PlayBirdDecision, forced_card_id: int
) -> np.ndarray | None:
    """Every option playing the forced bird (any habitat); ``None`` when the menu
    has none."""
    mask = np.array([choice.bird.id == forced_card_id for choice in decision.choices])
    return mask if mask.any() else None


def sampling_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    *,
    setup_net: setup_net_module.SetupNet | None = None,
    forced: models.ForcedCardSpec | None = None,
) -> engine.Agent:
    """An agent that samples every decision from the softmax with ``rng`` — the
    setup keep included — and, when ``forced`` carries a nonzero factor, applies
    the forced-card bias at matching decisions.

    Single-option decisions short-circuit. A setup-excluded net resolves the
    opening with ``setup_net``'s softmax (uniform-random without one)."""
    policy_net = net
    bias = forced if forced is not None and forced.force_card_factor != 0.0 else None

    def agent[C: decisions.Choice](
        eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not policy_net.include_setup and decisions.is_setup_decision(decision):
            if setup_net is None:
                return decisions.random_choice(decision, rng)
            setup_decision = typing.cast(decisions.SetupDecision, decision)
            _, setup_probs, _ = policy.setup_scores_and_probs(
                setup_net, setup_decision, eng.state, device
            )
            keep = policy.sample_index_from_probs(
                setup_probs, len(decision.choices), rng
            )
            return decision.choices[keep]
        family_idx = decisions.family_index_for(type(decision))
        state_vec = policy_net.encode_state(eng.state, decision)
        choice_feats = policy_net.encode_choices(decision, eng.state)
        probs = policy.policy_probs(
            policy_net, device, state_vec, choice_feats, family_idx
        )
        if bias is not None:
            mask = conformant_mask(
                eng, decision, bias.forced_card_id, bias.force_card_factor
            )
            if mask is not None:
                probs = bias_probs(probs, mask, bias.force_card_factor)
        return decision.choices[
            policy.sample_index_from_probs(probs, len(decision.choices), rng)
        ]

    return agent


def biased_seat(game_index: int, num_players: int) -> int:
    """The seat whose policy is biased in game ``game_index`` (alternates)."""
    return game_index % num_players


def seat_is_biased(
    seat: int, game_index: int, num_players: int, biased_seats: models.BiasedSeats
) -> bool:
    """Whether ``seat`` uses the biased policy in game ``game_index``."""
    if biased_seats is models.BiasedSeats.ALL:
        return True
    return seat == biased_seat(game_index, num_players)


def place_forced_card(
    game: state.GameState, forced_card_id: int, found_in: int, rng: random.Random
) -> None:
    """Swap the forced bird to a uniform position among the first ``found_in``
    cards of ``game``: the tray slots in order, then the deck from its top (the
    end of ``bird_deck``). The bird is uniform over that window and the rest of
    the order stays uniformly random, which equals biasing each draw to
    ``1 / (found_in - drawn)``. ``found_in == 0`` is a no-op.

    Raises ``ValueError`` when ``found_in`` is negative or exceeds the cards
    available, the tray is not fully dealt, or the bird is not exactly once in
    tray plus deck."""
    if found_in < 0:
        raise ValueError(f"forced_card_found_in {found_in} is negative")
    if found_in == 0:
        return
    tray_birds = [bird for bird in game.tray if bird is not None]
    if len(tray_birds) != state.TRAY_SIZE:
        raise ValueError("the tray must be fully dealt to place the forced card")
    order = [*tray_birds, *reversed(game.bird_deck)]
    if found_in > len(order):
        raise ValueError(
            f"forced_card_found_in {found_in} exceeds the {len(order)} cards "
            "in the tray and deck"
        )
    holders = [index for index, bird in enumerate(order) if bird.id == forced_card_id]
    if len(holders) != 1:
        raise ValueError(
            f"forced card {forced_card_id} appears {len(holders)} times in the "
            "tray and deck, expected 1"
        )
    target = rng.randrange(found_in)
    order[holders[0]], order[target] = order[target], order[holders[0]]
    game.tray[:] = order[: state.TRAY_SIZE]
    game.bird_deck[:] = reversed(order[state.TRAY_SIZE :])


def study_shard(
    spec: models.GameStatsSpec,
    shard: models.GameStatsShard,
    nets: game_stats.GameNets,
    *,
    forced: models.ForcedCardSpec,
) -> list[models.ForcedCardSeatRecord]:
    """Play every game of ``shard`` and return its per-seat records in game
    order, then seat order."""
    records: list[models.ForcedCardSeatRecord] = []
    for game_index in shard.game_indices:
        records.extend(play_game(spec, game_index, nets, forced))
    return records


def play_game(
    spec: models.GameStatsSpec,
    game_index: int,
    nets: game_stats.GameNets,
    forced: models.ForcedCardSpec,
) -> list[models.ForcedCardSeatRecord]:
    """Play one seeded sampled game — the biased seat(s) against unbiased copies
    of the same agent — and return one record per seat. The forced bird is first
    placed among the opening cards per ``forced.forced_card_found_in``."""
    seed = spec.game_seed(game_index)
    eng = collect.new_engine(seed, nets.num_players)
    placement_rng = random.Random(
        constants.FORCED_CARD_PLACEMENT_RNG_SEED_TEMPLATE.format(seed=seed)
    )
    place_forced_card(
        eng.state, forced.forced_card_id, forced.forced_card_found_in, placement_rng
    )
    biased = [
        seat_is_biased(seat, game_index, nets.num_players, forced.biased_seats)
        for seat in range(nets.num_players)
    ]
    agents = [
        sampling_agent(
            nets.policy_net,
            nets.device,
            random.Random(
                constants.FORCED_CARD_RNG_SEED_TEMPLATE.format(seed=seed, seat=seat)
            ),
            setup_net=nets.setup_net,
            forced=forced if biased[seat] else None,
        )
        for seat in range(nets.num_players)
    ]
    capture = card_stats.CardStatsCapture()
    engine.Engine.play_one_game(
        eng.state,
        agents,
        instrumentation=capture.instrumentation(),
        verbose_turn_log=False,
        split_setup_bonus=nets.split_setup_bonus,
        split_setup_food=nets.split_setup_food,
        combine_gain_food=nets.combine_gain_food,
    )
    name = bird_name(forced.forced_card_id)
    return [
        models.ForcedCardSeatRecord(
            game_index=game_index,
            seed=seed,
            seat=seat.seat,
            biased=biased[seat.seat],
            won=seat.won,
            tie=capture.tie,
            final_score=seat.final_score,
            forced_card_in_hand=_held_after_setup(seat, name),
            forced_card_played=name in seat.played_birds,
        )
        for seat in capture.seats
    ]


###### PRIVATE #######


def _degenerate_fallback(
    probs: np.ndarray, conformant: np.ndarray, force_card_factor: float
) -> np.ndarray:
    """Distribution when the reweighted total vanished: uniform over the
    conformant options for a positive factor; for a negative one, uniform over
    the non-conformant options, or ``probs`` unchanged when there are none."""
    if force_card_factor >= 0.0:
        pushed = conformant.astype(np.float64)
        return pushed / float(pushed.sum())
    avoided = (~conformant).astype(np.float64)
    if not avoided.any():
        return probs
    return avoided / float(avoided.sum())


def _held_after_setup(seat: card_stats.CardSeatCapture, name: str) -> bool:
    """Whether the bird was in the seat's hand at any point after setup: kept as
    the opener, in any round's hand, or placed on the board."""
    return (
        name in seat.kept_birds
        or name in seat.played_birds
        or any(name in names for names in seat.round_birds)
    )
