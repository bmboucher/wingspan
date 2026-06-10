# training.configure — Interactive "FLIGHT PLAN" configurator

Terminal UI for editing `TrainConfig`, browsing runs, and starting or archiving
training. Launched via `python -m wingspan.training --config`. Built on `rich`
for rendering and a cross-platform raw-key reader for input; no curses.

## Modules

**`__init__.py`**

**`fields.py`** — The field specification system:
- `FieldSpec` — abstract base with `attr`, `label`, `section`, `help`, and
  optional `group`, `visible_when`, `impact`.
- Concrete subclasses: `IntField`, `OptionalIntField`, `FloatField`,
  `ChoiceField`, `TextField`, `PathField`, `LayersField`,
  `OptionalPathField` (nullable filesystem-path field; displays `none_label`
  when `None`, parses back to `None` on empty or `"none"` input; used for
  `bootstrap_opponent_checkpoint` in the EVAL/bootstrap group).
- `FIELD_SPECS: list[FieldSpec]` — ordered list of all editable fields shown
  in the configurator. Includes `bootstrap_opponent_checkpoint`
  (`OptionalPathField`, group `"bootstrap"`, visible only when
  `initial_vs_random=True`, `ChangeImpact.REGIME`), and the OPTIM-section reward
  controls `reward_mode` (`ChoiceField` over `RewardMode`) and `reward_discount`
  (`FloatField`, visible only when `reward_mode == decision_delta`), both REGIME.
- `read_field(cfg, spec) -> FieldValue`, `format_value(cfg, spec) -> str`,
  `commit(cfg, spec, raw) -> (TrainConfig, str | None)`,
  `nudge(cfg, spec, direction) -> (TrainConfig, str | None)`.

**`runs.py`** — Run management:
- `RunSummary(run_name, iteration, best_win_rate, last_updated)` — compact
  snapshot of a run read from `status.json` / `model_config.json`.
- `inspect_run(run_dir) -> RunSummary` — reads the run directory.
- `archive_run(run_dir, archive_dir)` — moves a run to an archive subdirectory.
- `clear_run(run_dir)` — deletes checkpoints but keeps logs.
- `list_archives(archive_dir) -> list[RunSummary]`.

**`state.py`** — Configurator data model:
- `Mode` StrEnum: `CONFIG`, `RUNS`, `ARCH`, `CONFIRM`, `RUNNING`.
- `Outcome` StrEnum: `START`, `RESUME`, `QUIT`.
- `ConfirmPrompt(message, yes_outcome, no_outcome)` — modal prompt descriptor.
- `ConfiguratorState(config, mode, selected_field, runs, ...)` — the full
  immutable snapshot the screen renders from; mutations return a new instance.

**`keys.py`** — Cross-platform raw single-key reader:
- `read_key() -> str | None` — non-blocking; returns a key name (`"UP"`,
  `"ENTER"`, `"q"`, etc.) or `None` if no key is ready.
- Uses `msvcrt` on Windows, `termios`+`tty` on POSIX. Called in a tight
  loop by `controller.py`.

**`screen.py`** — `rich` layout and rendering:
- `build_layout() -> Layout` — the two-panel (field list / help + arch diagram)
  layout structure.
- `render_config_panel(state)`, `render_runs_panel(state)`,
  `render_arch_panel(state)` — per-mode panel renderers.
- `render_modal(state)` — overlays the confirm prompt when `state.mode == CONFIRM`.

**`controller.py`** — Main loop and event dispatch:
- `run_configurator(config) -> (Outcome, TrainConfig)` — starts `rich.Live`,
  reads keys via `keys.py`, dispatches to `dispatch(state, key) -> ConfiguratorState`,
  and loops until an outcome is reached.
- `build_initial_state(config) -> ConfiguratorState` — console-free constructor
  for testing.
- `dispatch(state, key)` — pure state-transition function; no I/O.

**`arch_diagram.py`** — `ArchDiagram(arch: ModelArchitecture)`: a `rich`
renderable that draws the live architecture as a text-art block diagram,
updated in real time as the user nudges width fields in the configurator.
