"""Selfplay CLI: run Wingspan games with configurable per-seat agent matchups.

Extends the plain ``random`` self-play command to mixed matchups — any seat can
be the uniform-random agent or a trained ``PolicyValueNet`` loaded from a
training checkpoint, so the three modes random/random, random/AI and AI/AI all
run through one command.

When a seat is AI-driven, the agent annotates the game log at every genuine
decision with the policy's top-``_MAX_LOGGED_OPTIONS`` options sorted best-first,
each showing its raw pre-softmax score and softmax probability. Forced moves (a
single legal option) are not annotated. When running with temperature sampling
(not ``--greedy``) an ``[AI chose: ...]`` line follows the ranked list to record
which option was actually sampled.

The opening-bonus regime is auto-derived from the ``TrainConfig`` stored in each
loaded checkpoint so games mirror how the nets were trained: a checkpoint trained
under the ``split_setup_bonus`` regime gets a bonus-free ``SetupDecision`` with
the bonus deferred to the in-game ``CHOOSE_BONUS`` pick, exactly as in training.
Config-free (random-only) matchups keep the engine's combined default.

Usage: ``wingspan selfplay --help`` or ``python -m wingspan.cli selfplay --help``.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import typing

import numpy as np
import torch
import yaml

from wingspan import agents, decisions, encode, engine, model, setup_model
from wingspan.agents import display
from wingspan.instrumentation import config as instrumentation_config
from wingspan.instrumentation import dispatcher
from wingspan.training import artifacts, config, policy
from wingspan.training import setup_net as setup_net_module
from wingspan.training import setup_runmeta

# Cap and floor on the per-decision annotation: never list more than this many
# options, and never list one the policy assigns less than this probability to
# (a percent). Together they keep the log readable when a decision has hundreds
# of legal options while the policy spreads only a little mass across most of
# them. The ``SetupDecision`` is exempt from the floor (its near-uniform opening
# distribution would otherwise print nothing) and always shows its top picks.
_MAX_LOGGED_OPTIONS = 5
_MIN_PROB_PCT = 1.0
_SMALL_DECISION_THRESHOLD = 5

# The named checkpoint specs ``--p0`` / ``--p1`` accept, mapped to the on-disk
# artifact filenames inside ``--checkpoint-dir``. Any spec not in this table is
# treated as a direct path to a ``.pt`` file.
_NAMED_SPECS: dict[str, str] = {
    "last": artifacts.LAST_CKPT,
    "best": artifacts.BEST_CKPT,
    "opponent": artifacts.OPPONENT_CKPT,
}


def main_selfplay(argv: list[str] | None = None) -> int:
    """Run one or more selfplay games, optionally writing annotated game logs.

    Returns a process exit code: 0 on success, 1 if an AI agent's checkpoint
    cannot be loaded (missing file or an encoding-incompatible network)."""
    args = _build_parser().parse_args(argv)

    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    rng = random.Random(seed)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    device = torch.device(args.device)

    # Resolve both agents up front so a bad checkpoint (or a regime mismatch
    # between two checkpoints) fails before any game runs, with a clean message
    # rather than a mid-game traceback.
    try:
        agent_a, config_a = _make_agent(
            args.p0, checkpoint_dir, device, rng, args.greedy
        )
        agent_b, config_b = _make_agent(
            args.p1, checkpoint_dir, device, rng, args.greedy
        )
        split_setup_bonus = _resolve_split_setup_bonus((config_a, config_b))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading agent: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        regime = "  |  opening bonus: split (CHOOSE_BONUS)" if split_setup_bonus else ""
        print(f"Seed: {seed}  |  P0: {args.p0}  vs  P1: {args.p1}{regime}")

    instrumentation = _open_instrumentation(args, seed)
    try:
        for game_idx in range(args.games):
            eng, _, _, _ = engine.Engine.create(seed=seed + game_idx)
            engine.Engine.play_one_game(
                eng.state,
                (agent_a, agent_b),
                instrumentation=instrumentation,
                split_setup_bonus=split_setup_bonus,
            )
            scores = [player.final_score for player in eng.state.players]
            if not args.quiet:
                print(
                    f"Game {game_idx + 1}: scores={scores}, "
                    f"log lines={len(eng.state.log)}"
                )
            if args.log:
                log_path = args.log if args.games == 1 else f"{args.log}.{game_idx}"
                _write_log(log_path, eng.state.log)
                if not args.quiet:
                    print(f"  log -> {log_path}")
    finally:
        instrumentation.close()
    return 0


###### PRIVATE #######


#### Argument parsing ####


def _build_parser() -> argparse.ArgumentParser:
    """The ``selfplay`` argument parser. ``--p0`` / ``--p1`` each take an agent
    spec: ``random``, a named checkpoint (``last`` / ``best`` / ``opponent``),
    or a direct path to a ``.pt`` file."""
    parser = argparse.ArgumentParser(
        prog="wingspan selfplay",
        description="Run Wingspan selfplay games with configurable agent matchups.",
    )
    spec_help = (
        "Agent for player %s: 'random', a named checkpoint "
        "('last'/'best'/'opponent'), or a path to a .pt file (default: random)."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument(
        "--log", type=str, default=None, help="Path to write detailed game log(s)."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--p0", type=str, default="random", help=spec_help % "0")
    parser.add_argument("--p1", type=str, default="random", help=spec_help % "1")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        dest="checkpoint_dir",
        help="Directory to resolve named checkpoint specs against.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device for AI inference."
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="AI agents pick the argmax option instead of sampling.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Path to an instrumentation config (YAML/JSON): event handlers to "
        "attach to every game.",
    )
    parser.add_argument(
        "--instrument-out",
        type=str,
        default=None,
        dest="instrument_out",
        help="Directory the instrumentation handlers write their output under "
        "(default: current directory).",
    )
    return parser


#### Instrumentation ####


def _open_instrumentation(
    args: argparse.Namespace, seed: int
) -> dispatcher.Instrumentation:
    """Build and open the event-callback router from ``--instrument`` — the
    standalone instrumentation config (same shape as ``TrainConfig.instrumentation``).
    Returns the no-op ``EMPTY`` router when the flag is absent. The caller must
    ``close`` whatever this returns when the run ends."""
    if args.instrument is None:
        return dispatcher.EMPTY
    text = pathlib.Path(args.instrument).read_text(encoding="utf-8")
    cfg = instrumentation_config.InstrumentationConfig.model_validate(
        yaml.safe_load(text)
    )
    out_dir = (
        pathlib.Path(args.instrument_out)
        if args.instrument_out is not None
        else pathlib.Path(".")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumentation = cfg.build()
    instrumentation.open(
        instrumentation_config.RunContext(
            output_dir=out_dir,
            run_name="selfplay",
            seed=seed,
            matchup=(str(args.p0), str(args.p1)),
        )
    )
    return instrumentation


#### Agent construction ####


def _make_agent(
    spec: str,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    rng: random.Random,
    greedy: bool,
) -> tuple[engine.Agent, config.TrainConfig | None]:
    """Resolve an agent spec to a callable Agent plus the ``TrainConfig`` its
    checkpoint was trained under (``None`` for the config-free ``random`` agent,
    so regime flags like ``split_setup_bonus`` can mirror the training run).
    ``random`` yields the uniform agent (``greedy`` is irrelevant and ignored);
    any other spec loads the named or path checkpoint and wraps it in the
    log-annotating policy agent."""
    if spec == "random":
        return agents.random_agent(rng), None
    checkpoint_path = _resolve_checkpoint_path(spec, checkpoint_dir)
    net, train_config = _load_policy_net(checkpoint_path, device)
    setup_net_instance = _load_setup_net(checkpoint_dir, device)
    agent = _logged_policy_agent(net, device, rng, greedy, setup_net=setup_net_instance)
    return agent, train_config


def _resolve_split_setup_bonus(
    configs: typing.Sequence[config.TrainConfig | None],
) -> bool:
    """Whether the games should run the ``split_setup_bonus`` regime (the opening
    bonus deferred out of the ``SetupDecision`` to the in-game ``CHOOSE_BONUS``
    pick), derived from the loaded checkpoints' configs so selfplay mirrors how
    the nets were trained. Config-free (random) seats express no preference; with
    no AI seat at all the engine's combined default applies. Two checkpoints
    trained under different regimes cannot share a faithful game, so a
    disagreement raises rather than silently mis-modelling one seat's opening."""
    flags = {cfg.split_setup_bonus_active for cfg in configs if cfg is not None}
    if len(flags) > 1:
        raise ValueError(
            "Checkpoints disagree on the split_setup_bonus regime: one was "
            "trained with the opening bonus deferred to the in-game CHOOSE_BONUS "
            "pick, the other with it baked into the setup keep. Selfplay cannot "
            "mirror both in one game — pick checkpoints from the same regime."
        )
    return next(iter(flags), False)


def _resolve_checkpoint_path(spec: str, checkpoint_dir: pathlib.Path) -> pathlib.Path:
    """Map a named spec to its artifact under ``checkpoint_dir``; treat anything
    else as a direct path to a checkpoint file."""
    if spec in _NAMED_SPECS:
        return checkpoint_dir / _NAMED_SPECS[spec]
    return pathlib.Path(spec)


def _load_policy_net(
    checkpoint_path: pathlib.Path, device: torch.device
) -> tuple[model.PolicyValueNet, config.TrainConfig]:
    """Load a ``PolicyValueNet`` from a training checkpoint, rebuilding it from
    the ``TrainConfig`` stored alongside the weights so the caller need not know
    the network's layer widths; the parsed config is returned with the net so
    regime flags (e.g. ``split_setup_bonus``) can mirror the training run. Raises
    with a clear message when the file is missing, lacks a config, or was trained
    against an incompatible encoding layout."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model with "
            "`wingspan-dashboard` first, or pass a direct .pt path."
        )

    # Our own trusted checkpoint carries a config dict + metrics, not just
    # tensors, so the full (non weights-only) unpickler is required.
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )
    if "config" not in payload:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has no embedded 'config' — it is "
            "not a valid self-describing training checkpoint."
        )

    # The net is rebuilt from the checkpoint's own topology, so its layer widths
    # always match its weights; what must match the *current* code is the
    # encoding layout (state/choice feature dims and the family head order),
    # since freshly-encoded states are fed into the net at inference. A net
    # trained with a different topology is still perfectly usable here.
    saved = config.TrainConfig.model_validate(payload["config"])
    current = config.TrainConfig()
    if _encoding_key(saved) != _encoding_key(current):
        raise ValueError(
            "Checkpoint encoding layout is incompatible with the current code:\n"
            f"  saved:   {_encoding_key(saved)}\n"
            f"  current: {_encoding_key(current)}\n"
            "It was trained against a different encode.py / decisions.py layout."
        )

    net = model.PolicyValueNet(arch=saved.arch, spec=saved.encoding_spec).to(device)
    net.load_state_dict(payload["model"])
    net.eval()
    return net, saved


def _encoding_key(cfg: config.TrainConfig) -> tuple[int, int, tuple[str, ...]]:
    """The encoding-compatibility signature: the parts of the architecture that
    must agree with the live ``encode`` / ``decisions`` modules for a checkpoint
    to consume freshly-encoded inputs (the layer widths are excluded — they are
    self-consistent with the loaded weights)."""
    return (cfg.state_dim, cfg.choice_dim, cfg.family_order)


#### Setup model helpers ####


def _load_setup_net(
    checkpoint_dir: pathlib.Path, device: torch.device
) -> setup_net_module.SetupNet | None:
    """Load the separately-trained ``SetupNet`` from ``checkpoint_dir``.

    Returns ``None`` — degrading to random setup picks — only when the setup
    artifacts are absent (the run trained without a setup model). Artifacts that
    exist but fail to load raise: a present-but-broken ``setup.pt`` is an error,
    not something to silently paper over."""
    ckpt_path = checkpoint_dir / artifacts.SETUP_CKPT
    config_path = checkpoint_dir / artifacts.SETUP_CONFIG_JSON
    if not ckpt_path.exists() or not config_path.exists():
        return None
    descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(ckpt_path, map_location=device, weights_only=False),
    )
    net_instance = setup_net_module.SetupNet.from_setup_config(descriptor)
    net_instance.load_state_dict(payload["setup_model"])
    net_instance.eval()
    return net_instance.to(device)


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
    # One forward pass gives the full distribution over the legal options.
    family_idx = decisions.family_index_for(type(decision))
    state_vec = encode.encode_state(eng.state, decision, net.spec)
    choice_feats = encode.encode_choices(decision, eng.state, net.spec)
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


#### Log file output ####


def _write_log(path: str, lines: list[str]) -> None:
    """Write the game log line-by-line to ``path`` (UTF-8, newline-terminated)."""
    with open(path, "w", encoding="utf-8") as log_file:
        for line in lines:
            log_file.write(display.strip_ansi(line) + "\n")
