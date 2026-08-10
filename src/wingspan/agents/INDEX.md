# agents — Agent implementations

Agents that implement the `Agent` protocol defined in `engine.core`. All concrete
agents are generic callables: `def __call__[C: Choice](self, engine, decision, /) -> C`.

## Modules

**`__init__.py`** — re-exports `random_agent`, `cli_agent`.

**`base.py`** — `random_agent`: the reference uniform-random policy. Selects uniformly
from `decision.choices`; used as the baseline opponent during training and evaluation.
No state; implemented as a plain function matching the `Agent` protocol.

**`cli.py`** — `cli_agent`: the interactive human agent. For a `SetupDecision`,
delegates to the terminal selection widget in `interactive.py` (its two
sub-dialogs, keep-cards-and-bonus then keep-foods); every other decision runs
its own inline `input("choice> ")` loop. Uses `display.py` to render
the current game state before prompting. Suitable for human-vs-AI play via
`wingspan play`. Public helpers for reuse by other interactive entry points:
`format_choice_line(idx, choice, player)` renders one offered-choice line with
type-aware extra context; `setup_dialog_axes(decision)` inspects the offered
`SetupChoice`s and returns `(ask_bonus, ask_food)` — under the split-setup
regimes the engine defers the bonus and/or food pick to a later decision, so
every offered choice carries that axis at its empty value (`bonus_card=None`
/ `kept_foods=()`); `resolve_setup_choice_dialog(decision, tray)` runs the
split-aware cards (+ bonus, + foods as applicable) sub-dialog for a
`SetupDecision` and returns the matching `SetupChoice`, asking only for the
axes `setup_dialog_axes` says are actually offered.

**`display.py`** — Human-readable formatters for cards and game state. Key functions:
`format_bird(bird)`, `format_bonus(bc)`, `format_board(game_state, player)`.
Emits 24-bit ANSI truecolor when stdout is a TTY (`sys.stdout.isatty`); plain text is only the
non-tty fallback.

**`interactive.py`** — Terminal selection-form widget.
`select_form(sections: Sequence[Section], *, header, instructions=..., live_options=None,
live_footer=None) -> list[list[int]]` is a multi-section arrow-key checkbox/radio
widget with in-place ANSI redraw, returning one ascending index list per section.
`cli_agent` calls it only from the two `SetupDecision` sub-dialogs (kept
cards + bonus, then kept foods); the numbered-list-plus-integer behavior is
just its non-tty fallback, `_select_form_fallback(sections, header)`.
`draw_frame(lines, prev_line_count)` (public
— promoted from a private helper) is the shared in-place ANSI frame redraw both this
widget and `wingspan.aid.widgets`' typeahead/counts tty shells draw through.
