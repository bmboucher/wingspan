# Project layout

Full annotated package layout and what the simulator covers.

## What's modelled

- **Core set, two players, no automa:** 180 birds, 26 bonus cards, 16
  end-of-round goals.
- Every bird's "when played / when activated / between turns" power is handled
  by a small library of generic power patterns. All core-set birds are covered;
  anything a future pattern doesn't yet recognise falls back to a logged no-op
  so a game never crashes (`cards.power_coverage` reports what's modelled).

## How it's organized

All card data and game state are Pydantic models, and the engine drives every
state change through them. Card data is bundled in `src/wingspan/data/`, from
the [wingsearch](https://github.com/navarog/wingsearch) project.

```
src/wingspan/
  __init__.py            # package release version only
  cli.py                 # the unified `wingspan play` entry point (argparse + series runner)
  state.py               # GameState, Player, Board, FoodPool, PlayedBird, Birdfeeder, new_game
  decisions.py           # Decision[C] hierarchy + Choice hierarchy + MainAction + judgment families
  architecture.py        # ModelArchitecture + ActivationName (torch-free network topology descriptor)
  version.py             # MODEL_VERSION artifact-compat version + load-time check (torch-free)
  compat/                # version-specific artifact shims (deleted wholesale at a MAJOR bump)
    v0_0.py              #   pre-0.1 choice geometry: frozen row transform + PolicyValueNetV00
  model/                 # PyTorch network (package)
    __init__.py          #   re-exports PolicyValueNet
    core.py              #   PolicyValueNet actor-critic (built from a ModelArchitecture)
    mlp.py               #   shared MLP body/readout builders (policy net + setup net build identical stacks)
    hand_model.py        #   stateless multi-card set-embedder helpers (hand / tray / setup kept-set)
  players/               # seat players from CLI specs (shared by play + tournament)
    spec.py              #   the player-spec grammar (human / random / named / .pt path / run dir)
    loaders.py           #   both self-describing checkpoint load paths + encoding-compat keys
    factory.py           #   spec -> Agent (log-annotating policy agent) + regime resolution
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
    __init__.py          # re-exports random_agent, cli_agent
    base.py              # random_agent
    cli.py               # cli_agent (the interactive human agent)
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
    runmeta.py           # model_config.json (full topology, reconstitutable) + dated process_<stamp>.json sidecars (torch-free); read_model_config reader + the era-routed descriptor reporting seam (*_for / build_*)
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
