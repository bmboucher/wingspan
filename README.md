# Wingspan

A simulator and reinforcement-learning training pipeline for the **core-set,
two-player** board game [Wingspan](https://stonemaiergames.com/games/wingspan/).
You can play a full game from the terminal, run quick automated games for logs or
debugging, watch a trained network play (against itself, a frozen past self, or a
random agent), and train a neural-network agent by self-play while watching it
improve on a live dashboard.

## Setup

Requires **Python 3.12+**. Install the package (editable) into your environment:

```
pip install -e .
```

This pulls in everything needed to play and to train (PyTorch, NumPy, Pydantic,
rich). For the test suite and developer tooling (pyright, black, isort, pytest):

```
pip install -e ".[dev]"
```

Training runs on **CPU**: self-play collection is CPU-bound and fans out across
worker processes, and the gradient update is small, so the whole pipeline is
designed for CPU. (A CUDA PyTorch build still works for one-off experiments, but
CPU is the supported path.)

## Play games

One command runs every matchup — interactive play, quick automated games, and
trained-AI matches. Set each seat with `--p0` / `--p1`; the value is `human`
(interactive play in the terminal), `random` (the uniform-random agent), a
named checkpoint (`last`, `best`, or `opponent`, resolved against
`--checkpoint-dir`), a path to a `.pt` file, or a run directory (its `last.pt`
is seated). The default matchup is `last` vs `last` — the most recent trained
model playing itself — so the bare command needs a trained model (see **Train
an agent**); use `--p0 human --p1 random` to play without one:

```
wingspan play                                 # latest trained model vs itself
wingspan play --p0 human --p1 random          # play interactively vs a random opponent
wingspan play --p0 human --p1 human           # two humans, hotseat on one keyboard
wingspan play --p0 human --p1 best --seed 42  # take on the strongest checkpoint, reproducible deal
wingspan play --p0 random --p1 random --games 5 --log games.log  # quick automated games -> games.log.0 ..
wingspan play --p0 best --p1 opponent --games 10        # best vs the frozen ladder rung
wingspan play --p0 best --p1 best --greedy --log ai.log # best vs itself, argmax play
wingspan play --p0 runs/exp/last.pt --p1 random         # a checkpoint by path
```

A `human` seat gets a numbered menu for every choice the rules require — pick a
bird, choose a habitat, pay food, lay eggs, and so on. When a seat is
AI-driven, every genuine decision is annotated in the game log with the
policy's ranked probability distribution over the legal options, turning the
log into a move-by-move readout of what the network was "thinking", and
`--greedy` makes AI seats take the argmax option instead of sampling. `--games
N` plays a series (game *i* deals with `--seed + i`), `--log` writes the full
action-by-action log per game, and `--quiet` suppresses the per-game summary.

## Train an agent

The training app is **FLIGHT PLAN** — a full-screen TUI that opens on a
configuration screen first, then transitions to a live `top`-style dashboard
once a run is started or resumed:

```
wingspan dashboard                    # open FLIGHT PLAN (always starts in config)
wingspan dashboard --device cpu       # force CPU (the supported path)
wingspan dashboard --games-per-iter 256 --eval-every 5 --eval-games 128
```

The config screen lets you tune every hyperparameter (learning rate,
games/iteration, evaluation cadence, target-iteration milestone, network
width, …) and manage the runs already in the checkpoint directory. Arrow keys
move between fields, ←/→ nudge a value (or cycle a choice), and Enter edits
one directly; every value is validated as you type. The screen shows whether
the directory holds a resumable run and flags which edits would force a fresh
start (changing the network width can't reuse old weights). From there:

- **Start** resumes a compatible run, or starts a fresh one in an empty
  directory.
- **New run** (or starting after an incompatible edit) prompts you to
  **archive** the existing run to `checkpoints/archive/<label>/` — preserving
  its checkpoints, metrics, and log — before the new run begins, so a long
  training run is never silently overwritten.

Once started, the dashboard runs until you press **Ctrl+C**, which asks it to
finish the current game, save a final checkpoint, and print a summary. As
strength improves the evaluation opponent advances from the random agent to
frozen past selves (a self-play ladder). Checkpoints (`last.pt`, `best.pt`, the
frozen `opponent.pt`), a per-iteration `metrics.jsonl`, and a per-game
`games.jsonl` are written to the checkpoint directory (`checkpoints/` by
default; change with `--checkpoint-dir`), and runs are resumable. Pass
`--iterations N` to stop automatically after N rounds instead.

You can also set a **target-iteration** milestone (in the config screen):
the run pauses there, runs a large fixed-model self-play evaluation, and waits
for you to **[C]ontinue** — optionally entering a new target — or **[E]nd** the
run and return to the config screen.

See [docs/TRAINING.md](docs/TRAINING.md) for the training program and
[docs/DECISIONS.md](docs/DECISIONS.md) for the per-decision modelling direction.

## Run a tournament

`wingspan tournament` runs a round-robin between any mix of trained AIs and the
random agent, tracks Elo ratings, and writes a JSON report.

```
wingspan tournament                                        # interactive picker (default)
wingspan tournament --base-dir checkpoints                 # picker scans a different dir
wingspan tournament --no-picker --ai runs/exp1 --ai runs/exp2        # skip picker; explicit dirs
wingspan tournament --no-picker --ai runs/exp1 --include-random      # add the random agent
wingspan tournament --no-picker --ai runs/exp1 --ai runs/exp2 \
    --games-per-pair 64 --out results.json                 # 64 mirrored games per pair
wingspan tournament --quiet                                # no live UI; periodic progress only
```

By default an interactive picker scans `checkpoints/` and lets you choose
competitors with arrow keys. Pass `--no-picker` to specify them via `--ai
<checkpoint-dir>` flags (repeatable) and optionally `--include-random`. Each
pair plays `--games-per-pair` games (default 32) as mirrored deals. The live
dashboard (skipped with `--quiet`) shows Elo ratings and W-L-T updating as
games finish; **q** / Ctrl+C stops after the current games complete. The final
standings and the full JSON report are written to `--out`
(`tournament_report.json` by default).

## Installed commands

After `pip install -e .` all tools are available through a single `wingspan`
command with subcommands:

| Command                 | What it does                                            |
| ----------------------- | ------------------------------------------------------- |
| `wingspan play`         | Games between any mix of human, random, and AI seats    |
| `wingspan dashboard`    | FLIGHT PLAN: config screen → live training dashboard    |
| `wingspan tournament`   | Round-robin tournament between trained AIs              |
| `wingspan inspect`      | Model introspection report (vectors, architecture, params) |
| `wingspan cloud`        | Headless S3-persisted training (container use)          |
| `wingspan monitor`      | FLOCK WATCH: live roster of cloud runs                  |

Run `wingspan --help` for the full list, or `wingspan <command> --help` for
per-command usage.

## Tests

Run the full quality gate (pyright + isort + black + pytest):

```
bash scripts/quality_gate.sh
```

See `CLAUDE.md` for the gate's section flags and pass-through argument reference.

## Further reading

| File | What it covers |
| ---- | -------------- |
| [docs/PROJECT.md](docs/PROJECT.md) | Full package layout and what's modelled |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision/choice taxonomy, `ALL_DECISION_CLASSES` ordering, featurization |
| [docs/BIRDS.md](docs/BIRDS.md) | Every bird power: `EffectKind` patterns, handler mappings, implementation gaps |
| [docs/TRAINING.md](docs/TRAINING.md) | Training program, hyperparameter guidance, eval ladder |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Research agenda, per-project feasibility analysis |
