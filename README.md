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

## Run a game manually (criterion 1)

```
wingspan-play
```

The CLI presents numbered menus for every choice the rules require.

## Watch a random self-play game (criterion 2)

```
wingspan-random --log game.log
```

This runs two random agents and writes a detailed action-by-action log.

## Run one training cycle (criterion 3)

```
wingspan-train --device cuda --episodes 32 --epochs 1
```

This collects self-play data with random+exploration policy, then performs one DQN-style update on the GPU.

## Layout

- `src/wingspan/data/` — card data (downloaded from the [wingsearch](https://github.com/navarog/wingsearch) project).
- `src/wingspan/cards.py` — bird/bonus card schemas + loader.
- `src/wingspan/state.py` — core game state types.
- `src/wingspan/actions.py` — action interface + decision points.
- `src/wingspan/game.py` — turn / round / scoring engine.
- `src/wingspan/powers.py` — bird power dispatch.
- `src/wingspan/agents.py` — random + human agents.
- `src/wingspan/encode.py` — state/action tensor encoders for RL.
- `src/wingspan/model.py` — PyTorch policy/value net.
- `src/wingspan/train.py` — self-play data collection + training loop.
- `src/wingspan/cli.py` — entry points.
