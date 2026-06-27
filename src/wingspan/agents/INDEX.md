# agents — Agent implementations

Agents that implement the `Agent` protocol defined in `engine.core`. All concrete
agents are generic callables: `def __call__[C: Choice](self, engine, decision, /) -> C`.

## Modules

**`__init__.py`** — re-exports `random_agent`, `cli_agent`.

**`base.py`** — `random_agent`: the reference uniform-random policy. Selects uniformly
from `decision.choices`; used as the baseline opponent during training and evaluation.
No state; implemented as a plain function matching the `Agent` protocol.

**`cli.py`** — `cli_agent`: the interactive human agent. Delegates to the terminal
selection widget in `interactive.py` for each decision; uses `display.py` to render
the current game state before prompting. Suitable for human-vs-AI play via
`wingspan play`.

**`display.py`** — Human-readable formatters for cards and game state. Key functions:
`format_bird(bird)`, `format_bonus(bc)`, `format_board(gs)`. Output is plain text
for terminal display.

**`interactive.py`** — Terminal selection-form widget. `select_form(choices, prompt)`
renders a numbered list and reads a validated integer from stdin. Used by `cli_agent`
to present each `Decision`'s `choices` list.
