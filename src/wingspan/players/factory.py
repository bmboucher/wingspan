"""Seat agents from parsed player specs — shared by the game-running CLIs.

:func:`build_agent` turns a :class:`spec.PlayerSpec` into an Agent: the
interactive CLI human, the uniform-random agent, or a trained
``PolicyValueNet`` wrapped in the log-annotating policy agent. When a seat is
AI-driven, the agent annotates the game log at every genuine decision with the
policy's top-``_MAX_LOGGED_OPTIONS`` options sorted best-first, each showing
its raw pre-softmax score and softmax probability. Forced moves (a single
legal option) are not annotated. When sampling (not greedy) an
``[<player> chose: ...]`` line follows the ranked list to record which option
was actually sampled.

:func:`resolve_split_setup_bonus` derives the opening-bonus regime from the
``TrainConfig`` stored in each loaded checkpoint so games mirror how the nets
were trained: a checkpoint trained under the ``split_setup_bonus`` regime gets
a bonus-free ``SetupDecision`` with the bonus deferred to the in-game
``CHOOSE_BONUS`` pick, exactly as in training. Config-free (human/random)
matchups keep the engine's combined default.
"""

from __future__ import annotations

import random
import typing

import numpy as np
import torch

from wingspan import agents, decisions, engine, model, setup_model
from wingspan.players import loaders, spec
from wingspan.training import config, policy
from wingspan.training import setup_net as setup_net_module

# Cap and floor on the per-decision annotation: never list more than this many
# options, and never list one the policy assigns less than this probability to
# (a percent). Together they keep the log readable when a decision has hundreds
# of legal options while the policy spreads only a little mass across most of
# them. The ``SetupDecision`` is exempt from the floor (its near-uniform opening
# distribution would otherwise print nothing) and always shows its top picks.
_MAX_LOGGED_OPTIONS = 5
_MIN_PROB_PCT = 1.0
_SMALL_DECISION_THRESHOLD = 5


def build_agent(
    player_spec: spec.PlayerSpec,
    device: torch.device,
    rng: random.Random,
    greedy: bool,
) -> tuple[engine.Agent, config.TrainConfig | None]:
    """Build the Agent for one seat, plus the ``TrainConfig`` its checkpoint
    was trained under (``None`` for the config-free human and random agents),
    so regime flags like ``split_setup_bonus`` can mirror the training run.
    ``greedy`` applies only to ``MODEL`` seats — argmax instead of on-policy
    sampling — and is irrelevant and ignored for ``HUMAN`` / ``RANDOM``. A
    ``MODEL`` seat loads the spec's checkpoint (plus the optional setup net
    from its run directory) and wraps it in the log-annotating policy agent."""
    if player_spec.kind is spec.PlayerKind.HUMAN:
        return agents.cli_agent(), None
    if player_spec.kind is spec.PlayerKind.RANDOM:
        return agents.random_agent(rng), None
    assert (
        player_spec.checkpoint_path is not None and player_spec.run_dir is not None
    ), "a MODEL spec carries its checkpoint path and run dir"
    net, train_config = loaders.load_policy_net(player_spec.checkpoint_path, device)
    setup_net_instance = loaders.load_setup_net(player_spec.run_dir, device)
    agent = _logged_policy_agent(net, device, rng, greedy, setup_net=setup_net_instance)
    return agent, train_config


def resolve_split_setup_bonus(
    configs: typing.Sequence[config.TrainConfig | None],
) -> bool:
    """Whether the games should run the ``split_setup_bonus`` regime (the opening
    bonus deferred out of the ``SetupDecision`` to the in-game ``CHOOSE_BONUS``
    pick), derived from the loaded checkpoints' configs so play mirrors how
    the nets were trained. Config-free (human/random) seats express no
    preference; with no AI seat at all the engine's combined default applies.
    Two checkpoints trained under different regimes cannot share a faithful
    game, so a disagreement raises rather than silently mis-modelling one
    seat's opening."""
    flags = {cfg.split_setup_bonus_active for cfg in configs if cfg is not None}
    if len(flags) > 1:
        raise ValueError(
            "Checkpoints disagree on the split_setup_bonus regime: one was "
            "trained with the opening bonus deferred to the in-game CHOOSE_BONUS "
            "pick, the other with it baked into the setup keep. The seats cannot "
            "mirror both in one game — pick checkpoints from the same regime."
        )
    return next(iter(flags), False)


###### PRIVATE #######


#### The log-annotating policy agent ####


def _logged_policy_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    greedy: bool,
    setup_net: setup_net_module.SetupNet | None = None,
) -> engine.Agent:
    """An AI agent that, for every genuine (multi-option) decision, writes the
    policy's ranked softmax distribution into the game log before picking — by
    argmax when ``greedy``, else by sampling on-policy.

    When ``net.include_setup`` is ``False`` (the default) and a ``SetupDecision``
    is encountered, ``setup_net`` is used instead of the main net to score the
    504 keep-combinations and log their probability distribution. Falls back to a
    random pick when ``setup_net`` is ``None`` (e.g. no ``setup.pt`` found)."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            return _handle_setup_decision(eng, decision, setup_net, device, greedy, rng)
        return _handle_main_decision(eng, decision, net, device, greedy, rng)

    return agent


def _handle_setup_decision[C: decisions.Choice](
    eng: engine.Engine,
    decision: decisions.Decision[C],
    setup_net: setup_net_module.SetupNet | None,
    device: torch.device,
    greedy: bool,
    rng: random.Random,
) -> C:
    """Score a SetupDecision with the setup net (or randomly if unavailable)."""
    if setup_net is None:
        chosen = decisions.random_choice(decision, eng.state.rng)
        eng.log(f"[{eng.state.me().name}] setup chosen at random (no setup model)")
        return chosen
    setup_decision = typing.cast(decisions.SetupDecision, decision)
    scores, probs = _compute_setup_scores_and_probs(
        setup_net, setup_decision, eng, device
    )
    _log_distribution(eng, decision, probs, greedy, scores=scores)
    n_choices = len(decision.choices)
    if greedy:
        chosen_idx = int(np.argmax(probs))
    else:
        chosen_idx = policy.sample_index_from_probs(probs, n_choices, rng)
    chosen = decision.choices[chosen_idx]
    if not greedy:
        eng.log(
            f"[{eng.state.me().name} chose: {chosen.display_label()} "
            f"({float(probs[chosen_idx]) * 100.0:.3f}%)]"
        )
    return chosen


def _handle_main_decision[C: decisions.Choice](
    eng: engine.Engine,
    decision: decisions.Decision[C],
    net: model.PolicyValueNet,
    device: torch.device,
    greedy: bool,
    rng: random.Random,
) -> C:
    """Score a regular decision with one forward pass through the main policy net."""
    # One forward pass gives the full distribution over the legal options. The
    # net owns its encoding (a compat-era net encodes in its frozen geometry).
    family_idx = decisions.family_index_for(type(decision))
    state_vec = net.encode_state(eng.state, decision)
    choice_feats = net.encode_choices(decision, eng.state)
    logits, probs = policy.policy_logits_and_probs(
        net, device, state_vec, choice_feats, family_idx
    )
    _log_distribution(eng, decision, probs, greedy, scores=logits)

    # Pick from the same probs already in hand: argmax for greedy strength play,
    # otherwise the on-policy sampling rule. Calling np.argmax directly (rather
    # than policy.greedy_action) avoids a redundant forward pass.
    n_choices = len(decision.choices)
    if greedy:
        chosen_idx = int(np.argmax(probs))
    else:
        chosen_idx = policy.sample_index_from_probs(probs, n_choices, rng)
    chosen = decision.choices[chosen_idx]
    if not greedy:
        eng.log(
            f"[{eng.state.me().name} chose: {chosen.display_label()} "
            f"({float(probs[chosen_idx]) * 100.0:.3f}%)]"
        )
    return chosen


def _log_distribution[C: decisions.Choice](
    eng: engine.Engine,
    decision: decisions.Decision[C],
    probs: np.ndarray,
    greedy: bool,
    scores: np.ndarray | None = None,
) -> None:
    """Append the ranked option list for one decision to the game log: a header
    line, then one line per shown option (rank, softmax probability, optional raw
    score, label). At most ``_MAX_LOGGED_OPTIONS`` options are shown; options below
    ``_MIN_PROB_PCT`` are suppressed for large decisions — except the
    ``SetupDecision``, whose distribution over hundreds of keeps is near-uniform
    early in training, so the floor would print nothing; its top picks are always
    documented."""
    n_choices = len(decision.choices)
    ranked = sorted(range(n_choices), key=lambda idx: float(probs[idx]), reverse=True)
    floor_exempt = n_choices < _SMALL_DECISION_THRESHOLD or decisions.is_setup_decision(
        decision
    )
    min_prob = 0.0 if floor_exempt else _MIN_PROB_PCT / 100.0
    shown = [idx for idx in ranked if float(probs[idx]) >= min_prob][
        :_MAX_LOGGED_OPTIONS
    ]

    player_name = eng.state.me().name
    mode = " | greedy" if greedy else ""
    eng.log(f"[{player_name}: {type(decision).__name__} | {n_choices} choices{mode}]")
    for rank, option_idx in enumerate(shown, start=1):
        prob_pct = float(probs[option_idx]) * 100.0
        score_str = (
            f"  ({float(scores[option_idx]):+6.2f})" if scores is not None else ""
        )
        label = decision.choices[option_idx].display_label()
        eng.log(f"{rank}. {label}")
        eng.log(f"    {prob_pct:6.3f}%{score_str}")


#### Setup model scoring ####


def _compute_setup_scores_and_probs(
    net_instance: setup_net_module.SetupNet,
    decision: decisions.SetupDecision,
    eng: engine.Engine,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every choice in ``decision`` through the setup net.

    Returns ``(margins, probs)`` where ``margins`` is the raw per-choice score
    vector and ``probs`` is the softmax distribution, both aligned to
    ``decision.choices``."""
    context = setup_model.SetupContext.from_state(eng.state)

    # Encode each choice using the same candidate → feature-vector path the
    # training pipeline uses, which guarantees alignment with the saved weights.
    vecs = np.stack(
        [
            setup_model.encode_setup_candidate(
                setup_model.SetupCandidate.from_setup_choice(choice), context
            )
            for choice in decision.choices
        ]
    )
    feats = torch.tensor(vecs, dtype=torch.float32, device=device)
    with torch.no_grad():
        margins = net_instance(feats).cpu().numpy()

    shifted = margins - margins.max()
    weights = np.exp(shifted)
    return margins, weights / weights.sum()
