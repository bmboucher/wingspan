"""Selfplay CLI: run Wingspan games with configurable per-seat agent matchups.

Extends the plain ``random`` self-play command to mixed matchups — any seat can
be the uniform-random agent or a trained ``PolicyValueNet`` loaded from a
training checkpoint, so the three modes random/random, random/AI and AI/AI all
run through one command.

When a seat is AI-driven, the agent annotates the game log at every genuine
decision with the policy's softmax distribution over the legal options — sorted
best-first, filtered to the top ``_MAX_LOGGED_OPTIONS`` options at or above
``_MIN_PROB_PCT`` — turning the log into a readable move-by-move analysis of what
the network was thinking. Forced moves (a single legal option) are not annotated,
matching the engine's rule that they are not real decisions.

Usage: ``python -m wingspan.cli selfplay --help`` or ``wingspan-selfplay --help``.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import typing

import numpy as np
import torch

from wingspan import agents, decisions, encode, engine, model
from wingspan.training import artifacts, config, policy

# Cap and floor on the per-decision annotation: never list more than this many
# options, and never list one the policy assigns less than this probability to
# (a percent). Together they keep the log readable when a decision has hundreds
# of legal options (e.g. the setup deal) while the policy spreads only a little
# mass across most of them.
_MAX_LOGGED_OPTIONS = 30
_MIN_PROB_PCT = 1.0

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

    # Resolve both agents up front so a bad checkpoint fails before any game
    # runs, with a clean message rather than a mid-game traceback.
    try:
        agent_a = _make_agent(args.p0, checkpoint_dir, device, rng, args.greedy)
        agent_b = _make_agent(args.p1, checkpoint_dir, device, rng, args.greedy)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading agent: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Seed: {seed}  |  P0: {args.p0}  vs  P1: {args.p1}")

    for game_idx in range(args.games):
        eng, _, _, _ = engine.Engine.create(seed=seed + game_idx)
        engine.Engine.play_one_game(eng.state, (agent_a, agent_b))
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
    return 0


###### PRIVATE #######


#### Argument parsing ####


def _build_parser() -> argparse.ArgumentParser:
    """The ``selfplay`` argument parser. ``--p0`` / ``--p1`` each take an agent
    spec: ``random``, a named checkpoint (``last`` / ``best`` / ``opponent``),
    or a direct path to a ``.pt`` file."""
    parser = argparse.ArgumentParser(
        description="Run Wingspan selfplay games with configurable agent matchups."
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
    return parser


#### Agent construction ####


def _make_agent(
    spec: str,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    rng: random.Random,
    greedy: bool,
) -> engine.Agent:
    """Resolve an agent spec to a callable Agent. ``random`` yields the uniform
    agent (``greedy`` is irrelevant and ignored); any other spec loads the named
    or path checkpoint and wraps it in the log-annotating policy agent."""
    if spec == "random":
        return agents.random_agent(rng)
    checkpoint_path = _resolve_checkpoint_path(spec, checkpoint_dir)
    net = _load_policy_net(checkpoint_path, device)
    return _logged_policy_agent(net, device, rng, greedy)


def _resolve_checkpoint_path(spec: str, checkpoint_dir: pathlib.Path) -> pathlib.Path:
    """Map a named spec to its artifact under ``checkpoint_dir``; treat anything
    else as a direct path to a checkpoint file."""
    if spec in _NAMED_SPECS:
        return checkpoint_dir / _NAMED_SPECS[spec]
    return pathlib.Path(spec)


def _load_policy_net(
    checkpoint_path: pathlib.Path, device: torch.device
) -> model.PolicyValueNet:
    """Load a ``PolicyValueNet`` from a training checkpoint, rebuilding it from
    the ``TrainConfig`` stored alongside the weights so the caller need not know
    the network's layer widths. Raises with a clear message when
    the file is missing, lacks a config, or was trained against an incompatible
    encoding layout."""
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
            f"Checkpoint at {checkpoint_path} has no 'config' — it predates the "
            "self-describing checkpoint format and cannot be loaded here."
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
    return net


def _encoding_key(cfg: config.TrainConfig) -> tuple[int, int, tuple[str, ...]]:
    """The encoding-compatibility signature: the parts of the architecture that
    must agree with the live ``encode`` / ``decisions`` modules for a checkpoint
    to consume freshly-encoded inputs (the layer widths are excluded — they are
    self-consistent with the loaded weights)."""
    return (cfg.state_dim, cfg.choice_dim, cfg.family_order)


#### The log-annotating policy agent ####


def _logged_policy_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    greedy: bool,
) -> engine.Agent:
    """An AI agent that, for every genuine (multi-option) decision, writes the
    policy's ranked softmax distribution into the game log before picking — by
    argmax when ``greedy``, else by sampling on-policy."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            return decisions.random_choice(decision, eng.state.rng)

        # One forward pass gives the full distribution over the legal options.
        family_idx = decisions.family_index_for(type(decision))
        state_vec = encode.encode_state(eng.state, decision, net.spec)
        choice_feats = encode.encode_choices(decision, eng.state, net.spec)
        probs = policy.policy_probs(net, device, state_vec, choice_feats, family_idx)

        _log_distribution(eng, decision, probs, greedy)

        # Pick from the same probs already in hand: argmax for greedy strength
        # play, otherwise the on-policy sampling rule. Calling np.argmax directly
        # (rather than policy.greedy_action) avoids a redundant forward pass.
        n_choices = len(decision.choices)
        if greedy:
            chosen_idx = int(np.argmax(probs))
        else:
            chosen_idx = policy.sample_index_from_probs(probs, n_choices, rng)

        chosen = decision.choices[chosen_idx]
        eng.log(
            f"[AI chose: {chosen.display_label()} "
            f"({float(probs[chosen_idx]) * 100.0:.1f}%)]"
        )
        return chosen

    return agent


def _log_distribution[C: decisions.Choice](
    eng: engine.Engine,
    decision: decisions.Decision[C],
    probs: np.ndarray,
    greedy: bool,
) -> None:
    """Append the ranked, filtered option list for one decision to the game log:
    a header line, then one line per shown option (rank, probability, label)."""
    n_choices = len(decision.choices)
    ranked = sorted(range(n_choices), key=lambda idx: float(probs[idx]), reverse=True)
    min_prob = _MIN_PROB_PCT / 100.0
    shown = [idx for idx in ranked if float(probs[idx]) >= min_prob][
        :_MAX_LOGGED_OPTIONS
    ]

    mode = " | greedy" if greedy else ""
    eng.log(f"[AI: {type(decision).__name__} | {n_choices} choices{mode}]")
    for rank, option_idx in enumerate(shown, start=1):
        prob_pct = float(probs[option_idx]) * 100.0
        label = decision.choices[option_idx].display_label()
        eng.log(f"  {rank:2d}. {prob_pct:5.1f}%  {label}")


#### Log file output ####


def _write_log(path: str, lines: list[str]) -> None:
    """Write the game log line-by-line to ``path`` (UTF-8, newline-terminated)."""
    with open(path, "w", encoding="utf-8") as log_file:
        for line in lines:
            log_file.write(line + "\n")
