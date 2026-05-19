# Wingspan

A simulator and RL training pipeline for the board game [Wingspan](https://stonemaiergames.com/games/wingspan/).

## Scope

- **Core set only** (180 birds, 26 bonus cards, 16 end-of-round goals).
- 2-player automa-free games.
- Many bird "when played" / "when activated" / "between turns" powers are implemented via a small set of generic power patterns. Birds whose powers don't map onto the supported patterns fall back to a logged no-op so simulation always runs; the simulator prints a coverage report on startup so you can see which birds are fully modelled.

## Install

```
pip install -e .
```

PyTorch with CUDA must be available for the GPU training cycle (any 2.x build works).

## Run a game manually (criterion 1)

```
python -m wingspan.cli manual          # you control player 0, opponent random
python -m wingspan.cli manual --both-human
```

The CLI presents numbered menus for every choice the rules require.

## Watch a random self-play game (criterion 2)

```
python -m wingspan.cli random --log game.log
python -m wingspan.cli random --games 5 --log games.log     # writes games.log.0 ..
```

Two random agents play; the action-by-action log is written to disk.

## Run one training cycle (criterion 3)

```
python -m wingspan.train --device cuda --episodes 32 --epochs 1
```

Collects self-play data (policy net + epsilon exploration vs. random opponent) and runs a REINFORCE-with-value-baseline update.

## Tests

```
python -m pytest tests/
```

## Layout

Domain models are Pydantic v2 `BaseModel`s; the engine drives state mutation through them.

- `src/wingspan/data/` — card data (downloaded from the [wingsearch](https://github.com/navarog/wingsearch) project).
- `src/wingspan/cards/` — bird/bonus card schema + power-text parser + JSON loader.
    - `schema.py` — enums, `Effect`/`Power` IR, `Bird`/`BonusCard`/`EndRoundGoal` models.
    - `parse.py` — power-text → structured `Effect` parser.
    - `load.py` — JSON loaders + power-coverage report.
- `src/wingspan/state.py` — `GameState`, `Player`, `PlayedBird`, `Birdfeeder`.
- `src/wingspan/actions.py` — `Decision`/`Choice` interface for agent prompts.
- `src/wingspan/engine/` — game engine.
    - `core.py` — `Engine` class, turn loop, setup, decision plumbing.
    - `main_actions.py` — play_bird / gain_food / lay_eggs / draw_cards.
    - `powers.py` — bird-power dispatch (`apply_effect` switch).
    - `reactors.py` — pink between-turn reactor hooks.
    - `scoring.py` — round-goal + final scoring.
    - `helpers.py` — pure helpers (food enumeration, egg ladders, etc.).
- `src/wingspan/agents/` — agent implementations.
    - `base.py` — random-policy agent.
    - `cli.py` — interactive human (stdin/stdout) agent plus the hotseat `mixed_agents` helper.
- `src/wingspan/encode.py` — state/action tensor encoders for RL.
- `src/wingspan/model.py` — PyTorch policy/value net.
- `src/wingspan/train.py` — self-play data collection + training loop.
- `src/wingspan/cli.py` — entry points.
