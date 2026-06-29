# Project layout

What the simulator covers and the top-level package map. Each subpackage has its
own `INDEX.md`; load only the one(s) relevant to the area you're working in.

## What's modelled

- **Core set, two players, no automa:** 180 birds, 26 bonus cards, 16 end-of-round goals.
- Every bird's "when played / when activated / between turns" power is handled
  by a small library of generic power patterns. All core-set birds are covered;
  anything a future pattern doesn't yet recognise falls back to a logged no-op
  so a game never crashes (`cards.power_coverage` reports what's modelled).

## How it's organized

All card data and game state are Pydantic models, and the engine drives every
state change through them. Card data is bundled in `src/wingspan/data/`, from
the [wingsearch](https://github.com/navarog/wingsearch) project.

## Root-level modules

```
src/wingspan/
  __init__.py      # package release version only
  __main__.py
  cli.py           # the unified `wingspan play` entry point (argparse + series runner)
  state.py         # GameState, Player, Board, FoodPool, PlayedBird, Birdfeeder, new_game
  decisions.py     # Decision[C] hierarchy + Choice hierarchy + MainAction + judgment families
  architecture.py  # ModelArchitecture + ActivationName (torch-free network topology descriptor)
  version.py       # MODEL_VERSION artifact-compat version + load-time check (torch-free)
  sampling.py      # weighted_index(...) — seed-stable weighted sampling (torch-free)
  data/*.json      # wingsearch card data (bundled)
```

## Subpackages

| Package | Purpose | Detail |
|---------|---------|--------|
| `agents/` | Interactive + random agents; human-readable formatters | [`agents/INDEX.md`](../src/wingspan/agents/INDEX.md) |
| `cards/` | Immutable card definitions + power-text parser | [`cards/INDEX.md`](../src/wingspan/cards/INDEX.md) |
| `cloud/` | Containerized, S3-persisted training runs + monitor | [`cloud/INDEX.md`](../src/wingspan/cloud/INDEX.md) |
| `compat/` | Version-specific artifact shims (cleared at a MAJOR bump) | [`compat/INDEX.md`](../src/wingspan/compat/INDEX.md) |
| `encode/` | State/choice tensor encoders for RL | [`encode/INDEX.md`](../src/wingspan/encode/INDEX.md) |
| `engine/` | Turn loop, action dispatch, pink reactors, scoring | [`engine/INDEX.md`](../src/wingspan/engine/INDEX.md) |
| `gamelog/` | Structured game-event tree: models, recorder, plaintext renderer | [`gamelog/INDEX.md`](../src/wingspan/gamelog/INDEX.md) |
| `instrumentation/` | General-purpose event-callback instrumentation for games | [`instrumentation/INDEX.md`](../src/wingspan/instrumentation/INDEX.md) |
| `model/` | PyTorch policy-value network | [`model/INDEX.md`](../src/wingspan/model/INDEX.md) |
| `players/` | Seat players from CLI specs (shared by play + tournament) | [`players/INDEX.md`](../src/wingspan/players/INDEX.md) |
| `reporting/` | Model introspection and HTML report generation | [`reporting/INDEX.md`](../src/wingspan/reporting/INDEX.md) |
| `setup_model/` | Separately-trained setup model (value-regression bandit) | [`setup_model/INDEX.md`](../src/wingspan/setup_model/INDEX.md) |
| `tournament/` | Round-robin tournament between trained AIs | [`tournament/INDEX.md`](../src/wingspan/tournament/INDEX.md) |
| `training/` | Live training + monitoring dashboard ("FLIGHT PLAN") | [`training/INDEX.md`](../src/wingspan/training/INDEX.md) |
