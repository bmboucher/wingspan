# Wingspan

A simulator and reinforcement-learning training pipeline for the board game
[Wingspan](https://stonemaiergames.com/games/wingspan/). You can play a full
game from the terminal, run quick automated games for logs or debugging, and
train a neural-network agent by self-play while watching it improve on a live
dashboard.

## What's modelled

- **Core set, two players, no automa:** 180 birds, 26 bonus cards, 16
  end-of-round goals.
- Every bird's "when played / when activated / between turns" power is handled
  by a small library of generic power patterns. All core-set birds are covered;
  anything a future pattern doesn't yet recognise falls back to a logged no-op
  so a game never crashes, and the interactive game prints a power-coverage
  report at startup so you can see what's modelled.

## Install

```
pip install -e .
```

This pulls in everything needed to play and to train (PyTorch, NumPy, Pydantic,
rich). For the test suite and developer tooling:

```
pip install -e ".[dev]"
```

Everything runs on CPU out of the box; a CUDA-capable PyTorch build is picked up
automatically if one is present.

## Play a game interactively

```
python -m wingspan.cli manual                 # you are player 0 vs a random opponent
python -m wingspan.cli manual --you 1         # control player 1 instead
python -m wingspan.cli manual --both-human    # two humans, hotseat on one keyboard
python -m wingspan.cli manual --seed 42       # reproducible deal
```

The game presents a numbered menu for every choice the rules require — pick a
bird, choose a habitat, pay food, lay eggs, and so on. A short game log and the
final scores are printed at the end.

## Run fast automated games

For quick games with no prompts — useful for logs, debugging, or sanity checks —
two **random** agents can play to completion:

```
python -m wingspan.cli random                            # one game, prints the result
python -m wingspan.cli random --games 5                  # five games back to back
python -m wingspan.cli random --log game.log             # write the full action-by-action log
python -m wingspan.cli random --games 5 --log games.log  # writes games.log.0 .. games.log.4
```

Games between **AI players** — the policy/value network playing against itself —
are run by the training pipeline below: self-play generates the training data,
and the agent is periodically evaluated head-to-head against a random opponent.

## Train an agent

The main training app is a live, `top`-style dashboard ("FLYWAY CONTROL") that
runs self-play, learns from it, evaluates against a random opponent, and
checkpoints as it goes:

```
python -m wingspan.training                    # auto-detects CUDA, falls back to CPU
python -m wingspan.training --device cpu       # self-play collection is often fastest on CPU
python -m wingspan.training --games-per-iter 64 --eval-every 2 --eval-games 32
```

It runs until you press **Ctrl+C**, which asks it to finish the current game,
save a final checkpoint, and print a summary. Checkpoints (`last.pt`, `best.pt`)
and a metrics log are written to the checkpoint directory (`checkpoints/` by
default; change it with `--checkpoint-dir`), and runs are resumable. Pass
`--iterations N` to stop automatically after N rounds instead.

A simpler one-shot cycle is also available — it plays a batch of self-play
games, runs a single training update starting from random weights, and saves a
checkpoint:

```
python -m wingspan.train --device cuda --episodes 32
```

See [TRAINING.md](TRAINING.md) for the training program and
[DECISIONS.md](DECISIONS.md) for the per-decision modelling direction.

## Installed commands

After `pip install -e .` the same entry points are available as plain commands:

| Command             | Equivalent to                   |
| ------------------- | ------------------------------- |
| `wingspan-play`     | `python -m wingspan.cli manual` |
| `wingspan-random`   | `python -m wingspan.cli random` |
| `wingspan-dashboard`| `python -m wingspan.training`   |
| `wingspan-train`    | `python -m wingspan.train`      |

## Tests

```
python -m pytest tests/
```

## How it's organized

All card data and game state are Pydantic models, and the engine drives every
state change through them.

- `src/wingspan/cards/` — bird / bonus / goal definitions, the power-text
  parser, and the JSON card loader (card data is bundled in `src/wingspan/data/`,
  from the [wingsearch](https://github.com/navarog/wingsearch) project).
- `src/wingspan/engine/` — the game engine: turn loop, the four main actions,
  bird-power dispatch, between-turn reactors, and scoring.
- `src/wingspan/agents/` — the random agent and the interactive human agent.
- `src/wingspan/encode.py`, `model.py`, `train.py` — the RL feature encoder, the
  policy/value network, and the self-play training loop.
- `src/wingspan/training/` — the live training-and-monitoring dashboard.
