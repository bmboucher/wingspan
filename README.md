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
state change through them. Card data is bundled in `src/wingspan/data/`, from
the [wingsearch](https://github.com/navarog/wingsearch) project.

```
src/wingspan/
  __init__.py            # version only
  cli.py                 # argparse entry points (manual / random)
  state.py               # GameState, Player, Board, FoodPool, PlayedBird, Birdfeeder, new_game
  decisions.py           # Decision[C] hierarchy + Choice hierarchy + MainAction + judgment families
  architecture.py        # ModelArchitecture + ActivationName (torch-free network topology descriptor)
  model/                 # PyTorch network (package)
    __init__.py          #   re-exports PolicyValueNet
    core.py              #   PolicyValueNet actor-critic (built from a ModelArchitecture)
    mlp.py               #   shared MLP body/readout builders (policy net + setup net build identical stacks)
    hand_model.py        #   stateless multi-card set-embedder helpers (hand / tray / setup kept-set)
  selfplay.py            # selfplay CLI: per-seat agent matchups over trained checkpoints
  reporting/             # model introspection and HTML report generation (package)
    __init__.py          #   re-exports generate_html_report, main_inspect
    html.py              #   standalone HTML model-summary report generator
    svg.py               #   SVG architecture-diagram builder (embedded in html.py)
    inspect_cli.py       #   model introspection CLI (vector layout, architecture, parameters)
  data/*.json            # wingsearch card data (bundled)

  encode/                # state/choice tensor encoders for RL (package)
    layout.py            # feature dims, stripe offsets, normalization scales (the chain)
    stripes/             # programmatic stripe registry for the state/choice vectors
      __init__.py        #   re-exports SubFieldDescriptor, StripeDescriptor, VectorLayout,
                         #   state_stripe_layout, choice_stripe_layout, card_feature_stripe_layout
      descriptors.py     #   SubFieldDescriptor / StripeDescriptor / VectorLayout models
      embed_rules.py     #   post-embedding rewrite logic shared by state and choice
      state.py           #   state_stripe_layout + state sub-field builders
      choice.py          #   choice_stripe_layout + choice sub-field builders
      card_feature.py    #   card_feature_stripe_layout + hand_encoder_input_stripe_layout
    state_encode.py      # encode_state / state_size + per-aspect state summaries
    choice_encode.py     # encode_choices + per-Choice featurizers + stripe fillers

  cards/                 # immutable card definitions
    __init__.py          # re-exports the public surface (Bird, Food, parse_power, load_all, ...)
    schema.py            # enums, Effect/Power IR, Bird/BonusCard/EndRoundGoal models,
                         #   BirdRecord/BonusRecord/GoalRecord raw-JSON record models
    parse/               # JSON loader + power-text parser (package)
      tags.py            # inline-icon tag tables + number-word parsing
      registry.py        # ordered @pattern / @pink_pattern matcher registries
      power.py           # parse_power + normalization + dispatch
      matchers.py        # general power-text matchers (pink_matchers.py: reactive ones)
      loader.py          # load_all / power_coverage (the JSON loader)
      catalog.py         # stable card -> dense-index maps for the encoder
      fields.py          # record-field parsers (parse_*, goal_category)

  engine/                # mutation logic
    __init__.py          # re-exports Engine, Agent, print_coverage_report
    core.py              # Engine class, Agent protocol, turn loop, setup, ask plumbing
    actions.py           # do_play_bird / do_gain_food / do_lay_eggs / do_draw_cards
    powers/              # bird-power dispatch (package)
      registry.py        # _HANDLERS table + @registry.handles decorator + handler_for
      dispatch.py        # dispatch_power / apply_effect (registry lookup) / lay_one_egg_on_nest
      grants.py egg_trade.py multi_actor.py tray_trade.py drafting.py
      nest_aggregate.py predator_repeat.py   # @handles handlers grouped by family
    reactors.py          # pink between-turn reactor hooks
    scoring.py           # score_round_goal, final_scoring
    helpers.py           # cost_meets, enumerate_payments — pure functions

  agents/
    __init__.py          # re-exports random_agent, cli_agent, mixed_agents
    base.py              # random_agent
    cli.py               # cli_agent + mixed_agents (hotseat helper)
    display.py           # human-readable formatters for cards and game state
    interactive.py       # terminal selection-form widget for the interactive CLI

  setup_model/           # the separately-trained setup model (value-regression bandit)
    architecture.py      # SetupArchitecture topology descriptor (+ its shape_key)
    candidates.py        # the keep options the setup model scores + selection
    encode.py            # per-candidate feature encoder
    stripes.py           # programmatic stripe registry for the setup input vector
    generate.py          # random-setup generation (the pre-model training phase)
    record.py            # the setup training sample + its on-disk store

  instrumentation/       # general-purpose event-callback instrumentation for games
    config.py            # serializable instrumentation config + per-run context
    dispatcher.py        # the live event router an Engine holds
    events.py            # event taxonomy + per-shape handler base classes
    registry.py          # config-class-name <-> handler bijection
    handlers/            # card_visits (per-bird play tallies), decision_logger (JSONL rows)

  training/              # live training + monitoring dashboard ("FLIGHT PLAN")
    __main__.py / app.py # entry point: argparse (+ --config) -> worker thread + rich.Live loop
    config.py            # TrainConfig (self-describing hyperparameters, §5.1)
    artifacts.py         # shared on-disk filenames (LAST/BEST/OPPONENT ckpt, metrics+games logs, model_config/process json)
    runmeta.py           # model_config.json (full topology, reconstitutable) + dated process_<stamp>.json sidecars (torch-free); read_model_config reader
    metrics.py           # ScoreBreakdown / FamilyCounts / EvalResult / IterationMetrics / GameOutcome (games.jsonl row)
    metrics_log.py       # cached reader for the append-only metrics.jsonl history
    runstate.py          # RunState: the shared live snapshot the dashboard reads (+ RunProgress)
    steps.py             # Step: the recorded self-play transition the learner consumes
    policy.py            # single-decision sample (collect) + greedy (eval)
    collect.py           # baseline single-game collector -> recorded steps + score breakdown
    mp_collect.py        # process-parallel collection (the CPU path; see COLLECTORS.md)
    batched_collect.py   # batched-forward collection (the CUDA path; see COLLECTORS.md)
    learner.py           # length-bucketed REINFORCE + advantage norm (§3.3, §4.2a)
    setup_net.py         # SetupNet: the setup model's MLP value-regressor
    setup_learner.py     # setup-model updates: offline fit + on-policy MSE
    setup_runmeta.py     # setup_config.json descriptor sidecar
    evaluate.py          # paired-game strength vs the reference opponent + 95% CI (§7)
    convergence.py       # series + axis-window math for the convergence charts
    sysmon.py            # host telemetry sampling for the SYSTEM band
    loop.py              # TrainingLoop orchestrator (run loop, __init__, system monitor)
    loop_resume.py       # resume from checkpoint, init phase/target, metadata writes
    loop_collect.py      # batched / multiprocess self-play collection
    loop_setup.py        # setup-model lifecycle (fit, update, sync, resume)
    loop_eval.py         # paired-game eval, opponent graduation / advancement
    loop_target.py       # target-milestone sequence (checkpoint → eval → pause)
    loop_checkpoint.py   # commit, checkpoint write, finish; seed + atomic-I/O helpers
    loop_metrics.py      # pure metrics aggregation (no loop state)
    theme.py             # palette + glyph constants ("wetland dawn")
    charts/              # custom rich renderables (package)
      geometry.py        # layout constants (gutter, inset width, ...)
      braille.py         # the 2x4-dot braille bitmap canvas
      text_helpers.py    # sparkline / eighth-block bar / human-count
      convergence_chart.py  # GettingBetterChart + its drawing helpers
      histogram.py       # FamilyHistogram
      insets.py          # the docked eval inset + narrow-panel strip
    dashboard.py         # the five-band Layout + per-region renderers
    configure/           # interactive "FLIGHT PLAN" configurator (python -m wingspan.training --config)
      fields.py          # FieldSpec hierarchy + FIELD_SPECS + read/format/commit/nudge
      runs.py            # RunSummary + inspect_run / archive_run / clear_run / list_archives
      state.py           # ConfiguratorState + Mode/Outcome/ConfirmPrompt value-objects
      keys.py            # cross-platform raw single-key reader (msvcrt / termios), non-blocking
      screen.py          # the rich Layout + per-region renderers + the modal
      controller.py      # run_configurator Live loop + console-free build_initial_state / dispatch
      arch_diagram.py    # the live ARCHITECTURE diagram

  tournament/            # round-robin tournament between trained AIs (wingspan-tournament)
    app.py               # entry point: pick competitors, play live, write the report
    participants.py      # competitor specs, on-disk run discovery, agent loading
    schedule.py          # the round-robin game schedule
    runner.py            # plays the scheduled games (process-parallel, sequential fallback)
    elo.py results.py state.py dashboard.py picker.py config.py

  cloud/                 # containerized, S3-persisted training runs + monitor
    runner.py            # headless supervisor (wingspan-cloud)
    runfile.py           # the single YAML run-file configuring one cloud run
    s3sync.py            # the S3 persistence sidecar around the loop
    status.py            # the compact monitoring snapshot of a run
    monitor.py           # "FLOCK WATCH" read-only roster of cloud runs (wingspan-monitor)

tests/                   # pytest; tests prepend src/ to sys.path themselves
```
