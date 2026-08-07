"""The ``wingspan aid`` CLI entry point.

Wires the pieces the rest of the package builds -- the setup dialog
(``entry``), the interactive oracle (``oracle`` / ``oracle_state``), the
advisor and relay seats (``advisor`` / ``relay``), and the instrumentation
hooks (``hooks``) -- into one live-assisted session over the real
``engine.core.Engine``. The session is entirely interactive: everything
between ``main``'s argument resolution and the final report runs off
``builtins.input``/``print`` (or, in tests, a scripted stand-in), so there is
no batch mode and no mid-game resume.

``--html`` is deliberately not offered: wiring the navigable HTML game-log
viewer needs the training-config timeline plumbing ``cli._open_instrumentation``
carries for ``wingspan play`` (seat configs, the YAML instrumentation layer),
which does not compose cleanly with ``AidHandler``'s own six-event router.
``--log`` / ``--jsonl`` cover the aid session's structured-output needs.
"""

from __future__ import annotations

import argparse
import collections.abc
import pathlib
import random
import sys

import torch

from wingspan import players
from wingspan.agents import display
from wingspan.aid import advisor
from wingspan.aid import console as console_module
from wingspan.aid import entry, hooks, models
from wingspan.aid import oracle as oracle_module
from wingspan.aid import oracle_state, placeholders, relay
from wingspan.engine import core as engine_core
from wingspan.engine import scoring
from wingspan.gamelog import models as gamelog_models
from wingspan.gamelog import recorder as gamelog_recorder
from wingspan.gamelog import render_jsonl as gamelog_render_jsonl
from wingspan.players import decision_probe
from wingspan.training import config

# The aid feature advises exactly one physical seat against one relayed
# opponent seat -- see the plan's "user-approved scope" (2-player checkpoints
# only, no N-player support).
_NUM_PLAYERS = 2


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``wingspan aid``: run one live-assisted physical-game
    session end to end and report the final scores.

    Returns a process exit code: 0 on success, 1 if the model spec cannot be
    resolved (missing/incompatible checkpoint, a human/random spec, a
    checkpoint not trained at 2 players) or the session was aborted."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    seed = args.seed if args.seed is not None else random.randint(0, 1 << 30)
    rng = random.Random(seed)
    device = torch.device(args.device)

    loaded = _load_advisor(args, device, rng)
    if loaded is None:
        return 1
    inner_agent, cfg, probe = loaded

    regimes = _resolve_regimes((cfg,))
    if regimes is None:
        return 1

    rec = (
        gamelog_recorder.EventRecorder(probes=(probe, None))
        if (args.log or args.jsonl)
        else gamelog_recorder.null_recorder()
    )
    con = console_module.Console()
    try:
        eng, handler = _run_interactive_session(
            con, rng, inner_agent, probe, cfg, rec, regimes
        )
    except KeyboardInterrupt:
        print("\nsession aborted (not resumable)", file=sys.stderr)
        return 1

    report = _build_report(eng, handler)
    _print_report(con, report)
    _write_outputs(args, seed, eng, rec, report, regimes)
    return 0


###### PRIVATE #######

#### Argument parsing ####


def _build_parser() -> argparse.ArgumentParser:
    """The ``aid`` argument parser: a checkpoint spec plus the shared
    device/seed/output flags mirrored from ``wingspan play``."""
    parser = argparse.ArgumentParser(
        prog="wingspan aid",
        description="Live assistant for a physical 2-player Wingspan game: "
        "the model ranks each of your decisions and tracks the board as you "
        "relay the opponent's visible moves.",
        epilog="The session starts at game setup and tracks through to game "
        "end -- there is no mid-game entry. A confirmed move cannot be "
        "undone. Only 2-player checkpoints can drive the advisor seat.",
    )
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="last",
        help="Checkpoint to advise with: 'last' / 'best' / 'opponent', a run "
        "directory, or a path to a .pt file (default: last).",
    )
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to write the plaintext session log (the engine's own log, "
        "ANSI-stripped).",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to write the flat structured game log (see `wingspan play "
        "--jsonl`); appends one game keyed by seed.",
    )
    return parser


#### Advisor / regime resolution ####


def _load_advisor(
    args: argparse.Namespace, device: torch.device, rng: random.Random
) -> tuple[engine_core.Agent, config.TrainConfig, decision_probe.DecisionProbe] | None:
    """Resolve ``args.model`` into the advisor's inner agent, refusing a
    human/random spec and printing a clean message on any load failure."""
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    spec = players.parse_player_spec(args.model, checkpoint_dir)
    if spec.kind is not players.PlayerKind.MODEL:
        print(
            "wingspan aid needs a trained model checkpoint to advise",
            file=sys.stderr,
        )
        return None
    probe = decision_probe.DecisionProbe()
    try:
        inner_agent, cfg = players.build_agent(
            spec, device, rng, True, value_probe=probe
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading agent: {exc}", file=sys.stderr)
        return None
    assert cfg is not None, "a MODEL spec always returns its TrainConfig"
    return inner_agent, cfg, probe


def _resolve_regimes(
    configs: collections.abc.Sequence[config.TrainConfig | None],
) -> tuple[bool, bool, bool] | None:
    """Derive the opening-setup regime flags from the advisor's checkpoint
    config, and confirm it was trained at the aid feature's fixed 2-player
    table. Prints a clean message and returns ``None`` on either failure."""
    try:
        split_setup_bonus = players.resolve_split_setup_bonus(configs)
        split_setup_food = players.resolve_split_setup_food(configs)
        combine_gain_food = players.resolve_combine_gain_food(configs)
        players.resolve_num_players(configs, _NUM_PLAYERS)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    return split_setup_bonus, split_setup_food, combine_gain_food


#### The interactive session ####


def _run_interactive_session(
    con: console_module.Console,
    rng: random.Random,
    inner_agent: engine_core.Agent,
    probe: decision_probe.DecisionProbe,
    cfg: config.TrainConfig,
    rec: gamelog_recorder.AnyRecorder,
    regimes: tuple[bool, bool, bool],
) -> tuple[engine_core.Engine, hooks.AidHandler]:
    """Run the setup dialog through game end: build the oracle-backed state
    from the entered deal, wire the advisor/relay seats and the physical-table
    hooks, then drive a full game on the real ``Engine``."""
    split_setup_bonus, split_setup_food, combine_gain_food = regimes
    setup_entry = entry.run_setup_entry(con)

    registry = placeholders.PlaceholderRegistry()
    echo = console_module.LogEcho(con)
    oracle = oracle_module.SessionOracle(con, echo, registry)
    gs = oracle_state.build_state(rng, setup_entry, oracle)
    notes = models.TurnNotes()
    handler = hooks.AidHandler(
        con=con, echo=echo, oracle=oracle, notes=notes, registry=registry
    )
    instrumentation = hooks.build_instrumentation(handler)
    advisor_seat = advisor.advisor_agent(
        inner_agent, probe, con, echo, registry, cfg.training.score_norm
    )
    relay_seat = relay.relay_agent(con, echo, registry, notes)

    eng = engine_core.Engine.play_one_game(
        gs,
        [advisor_seat, relay_seat],
        instrumentation=instrumentation,
        event_recorder=rec,
        split_setup_bonus=split_setup_bonus,
        split_setup_food=split_setup_food,
        combine_gain_food=combine_gain_food,
    )
    return eng, handler


#### Reporting ####


def _build_report(
    eng: engine_core.Engine, handler: hooks.AidHandler
) -> models.SessionReport:
    """The end-of-session summary: final scores in seat order, the winning
    seat (``None`` for a tie), and whether the opponent's bonus card(s) were
    entered for exact final scoring."""
    scores = [player.final_score or 0 for player in eng.state.players]
    winner_seat = scoring.determine_winner(eng.state.players)
    return models.SessionReport(
        scores=scores,
        winner=None if winner_seat < 0 else winner_seat,
        opponent_bonus_entered=handler.opponent_bonus_entered,
    )


def _print_report(con: console_module.Console, report: models.SessionReport) -> None:
    """Announce the final scores, the winner, and -- when the final-round
    bonus entry was skipped -- the opponent-bonus caveat."""
    con.say(f"Final scores -- you: {report.scores[0]}, opponent: {report.scores[1]}")
    if report.winner is None:
        con.say("Result: tie")
    else:
        winner_label = "You win" if report.winner == 0 else "Opponent wins"
        con.say(f"Result: {winner_label}")
    if not report.opponent_bonus_entered:
        con.say("opponent bonus VP is a placeholder — count that category manually")


def _write_outputs(
    args: argparse.Namespace,
    seed: int,
    eng: engine_core.Engine,
    rec: gamelog_recorder.AnyRecorder,
    report: models.SessionReport,
    regimes: tuple[bool, bool, bool],
) -> None:
    """Write the optional ``--log`` / ``--jsonl`` outputs, mirroring
    ``cli.main_play``'s equivalent file writes."""
    if args.log:
        content = "\n".join(display.strip_ansi(line) for line in eng.state.log)
        pathlib.Path(args.log).write_text(content + "\n", encoding="utf-8")
    if args.jsonl:
        assert isinstance(rec, gamelog_recorder.EventRecorder)
        gamelog_render_jsonl.append_game(
            pathlib.Path(args.jsonl),
            rec.root,
            _session_meta(seed, args.model, regimes, report),
        )


def _session_meta(
    seed: int,
    model_spec: str,
    regimes: tuple[bool, bool, bool],
    report: models.SessionReport,
) -> gamelog_models.GameMeta:
    """The flat log's header row for this aid session."""
    split_setup_bonus, split_setup_food, combine_gain_food = regimes
    return gamelog_models.GameMeta(
        game_id=str(seed),
        source=gamelog_models.LogSource.AID,
        seed=seed,
        num_players=_NUM_PLAYERS,
        seats=[f"aid:{model_spec}", "relay"],
        split_setup_bonus=split_setup_bonus,
        split_setup_food=split_setup_food,
        combine_gain_food=combine_gain_food,
        scores=report.scores,
        winner=report.winner,
    )
