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

## Play a game interactively

```
wingspan play                 # you are player 0 vs a random opponent
wingspan play --you 1         # control player 1 instead
wingspan play --both-human    # two humans, hotseat on one keyboard
wingspan play --seed 42       # reproducible deal
```

The game presents a numbered menu for every choice the rules require — pick a
bird, choose a habitat, pay food, lay eggs, and so on. A short game log and the
final scores are printed at the end.

## Run fast automated games

For quick games with no prompts — useful for logs, debugging, or sanity checks —
two **random** agents can play to completion:

```
wingspan random                            # one game, prints the result
wingspan random --games 5                  # five games back to back
wingspan random --log game.log             # write the full action-by-action log
wingspan random --games 5 --log games.log  # writes games.log.0 .. games.log.4
```

To pit a *trained* network against itself, a frozen past self, or the random
agent, use the `selfplay` command below.

## Watch trained agents play

Once you have a checkpoint (see **Train an agent**), `selfplay` runs games with
any agent in either seat, so the random/random, random/AI, and AI/AI matchups all
go through one command. Set each seat with `--p0` / `--p1`; the value is
`random`, a named checkpoint (`last`, `best`, or `opponent`, resolved against
`--checkpoint-dir`), or a path to a `.pt` file:

```
wingspan selfplay --p0 best                            # trained "best" vs random
wingspan selfplay --p0 best --p1 opponent --games 10   # best vs the frozen ladder rung
wingspan selfplay --p0 best --p1 best --greedy --log ai.log  # best vs itself, argmax play
wingspan selfplay --p0 runs/exp/last.pt --p1 random    # a checkpoint by path
```

When a seat is AI-driven, every genuine decision is annotated in the game log
with the policy's ranked probability distribution over the legal options, turning
the log into a move-by-move readout of what the network was "thinking".
`--greedy` makes AI agents take the argmax option instead of sampling; `--games`,
`--log`, `--seed`, and `--quiet` behave as in the `random` command.

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

See [TRAINING.md](TRAINING.md) for the training program and
[DECISIONS.md](DECISIONS.md) for the per-decision modelling direction.

## Installed commands

After `pip install -e .` all tools are available through a single `wingspan`
command with subcommands:

| Command                 | What it does                                            |
| ----------------------- | ------------------------------------------------------- |
| `wingspan play`         | Interactive game against a random opponent              |
| `wingspan random`       | Random-vs-random automated games                        |
| `wingspan selfplay`     | Configurable per-seat agent matchups (random or AI)     |
| `wingspan dashboard`    | FLIGHT PLAN: config screen → live training dashboard    |
| `wingspan tournament`   | Round-robin tournament between trained AIs              |
| `wingspan inspect`      | Model introspection report (vectors, architecture, params) |
| `wingspan cloud`        | Headless S3-persisted training (container use)          |
| `wingspan monitor`      | FLOCK WATCH: live roster of cloud runs                  |

Run `wingspan --help` for the full list, or `wingspan <command> --help` for
per-command usage.

## Tests

```
python -m pytest tests/
```

## What's modelled

- **Core set, two players, no automa:** 180 birds, 26 bonus cards, 16
  end-of-round goals.
- Every bird's "when played / when activated / between turns" power is handled
  by a small library of generic power patterns. All core-set birds are covered;
  anything a future pattern doesn't yet recognise falls back to a logged no-op
  so a game never crashes, and the interactive game prints a power-coverage
  report at startup so you can see what's modelled.

## How it's organized

All card data and game state are Pydantic models, and the engine drives every
state change through them.

- `src/wingspan/cards/` — bird / bonus / goal definitions, the power-text
  parser, and the JSON card loader (card data is bundled in `src/wingspan/data/`,
  from the [wingsearch](https://github.com/navarog/wingsearch) project).
- `src/wingspan/engine/` — the game engine: turn loop, the four main actions,
  bird-power dispatch, between-turn reactors, and scoring.
- `src/wingspan/agents/` — the random agent and the interactive human (CLI)
  agent.
- `src/wingspan/encode/`, `model.py`, `architecture.py` — the RL feature encoder
  (state + per-choice), the policy/value network, and its torch-free topology
  descriptor.
- `src/wingspan/selfplay.py` — the configurable-matchup self-play runner
  (trained checkpoints and/or the random agent in any seat).
- `src/wingspan/training/` — the live training-and-monitoring dashboard
  (self-play, learning, evaluation, checkpointing) and its interactive
  configurator.
- `src/wingspan/setup_model/` — an optional, separate model for the
  start-of-game hand/food keep (a value-regression bandit, off by default).
