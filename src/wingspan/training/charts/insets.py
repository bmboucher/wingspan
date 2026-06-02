"""The docked EVAL/COLLECT inset and its narrow-panel one-line strip fallback.
``eval_inset`` / ``collect_inset`` / ``eval_strip`` / ``collect_strip`` are
the public entry points the convergence chart docks; the rest are private
layout helpers (the cinematic hero number, the key/value rows, etc.)."""

from __future__ import annotations

from rich import text

from wingspan.training import metrics, runstate, theme
from wingspan.training.charts import geometry


def eval_inset(state: runstate.RunState, height: int) -> list[text.Text]:
    """The left-docked eval box: the cinematic hero win-rate, then the most
    recent win-rate / margin, then an identical EWMA section, then the eval
    sample size, the challenger (``random`` or ``gen N @ iter`` — the frozen
    generation and the iteration it was frozen at), and how many iterations have
    passed since the last upgrade. Padded with blank lines to ``height`` so it
    aligns with the plots."""
    last_eval = _latest_eval(state)
    body: list[text.Text] = [_inset_title(last_eval)]

    if last_eval is None:
        body.append(_inset_text("  awaiting first eval…", theme.TEXT_MUTED))
    else:
        _, result = last_eval
        ewma = state.eval_ewma()
        body.extend(_hero_block(result.win_rate * 100.0, state.best_win_rate))
        # The challenger identity rides right under the hero number so it
        # survives even when a short panel truncates the LAST / EWMA detail.
        body.append(_inset_kv("challenger", _inset_opponent(state), theme.TEXT_DIM2))
        body.append(_inset_kv("since adv", _inset_since(state), theme.TEXT_DIM2))
        body.append(_inset_blank())
        body.append(_inset_section("LAST"))
        body.append(
            _inset_kv("win rate", f"{result.win_rate * 100:.1f}%", theme.WIN_COLOR)
        )
        body.append(
            _inset_kv("margin", f"{result.mean_margin:+.1f} pts", theme.MARGIN_COLOR)
        )
        if ewma is not None:
            body.append(_inset_blank())
            body.append(_inset_section("EWMA"))
            body.append(
                _inset_kv("win rate", f"{ewma.win_rate * 100:.1f}%", theme.WIN_COLOR)
            )
            body.append(
                _inset_kv("margin", f"{ewma.mean_margin:+.1f} pts", theme.MARGIN_COLOR)
            )
        body.append(_inset_blank())
        body.append(_inset_kv("eval games", f"{result.n_games}", theme.TEXT_DIM2))
        if state.best_win_rate is not None:
            body.append(
                _inset_kv(
                    "best so far", f"{state.best_win_rate * 100:.1f}%", theme.HIST_MID
                )
            )

    while len(body) < height - 1:
        body.append(_inset_blank())
    # Reserve the final row for the bottom border so it survives even when the
    # body (hero + both sections + the readouts) would otherwise overflow the
    # available height and push the border off the bottom.
    footer = text.Text(no_wrap=True, end="")
    footer.append("└" + "─" * (geometry.INSET_W - 2) + "┘", style=theme.BORDER_EVAL)
    return body[: height - 1] + [footer]


def _hero_block(win_pct: float, best: float | None) -> list[text.Text]:
    """The oversized, value-recolored win-rate hero number in a heavy box."""
    color = theme.hero_color(win_pct)
    digits = " ".join(f"{win_pct:.1f}%")
    inner = geometry.INSET_W - 4
    top = text.Text(no_wrap=True, end="")
    top.append("│ ", style=theme.BORDER_EVAL)
    top.append("╔" + "═" * (inner - 2) + "╗", style=color)
    top.append(" │", style=theme.BORDER_EVAL)
    mid = text.Text(no_wrap=True, end="")
    mid.append("│ ", style=theme.BORDER_EVAL)
    mid.append("║", style=color)
    mid.append(digits.center(inner - 2), style=f"bold {color}")
    mid.append("║", style=color)
    mid.append(" │", style=theme.BORDER_EVAL)
    bot = text.Text(no_wrap=True, end="")
    bot.append("│ ", style=theme.BORDER_EVAL)
    bot.append("╚" + "═" * (inner - 2) + "╝", style=color)
    bot.append(" │", style=theme.BORDER_EVAL)
    return [top, mid, bot]


def _inset_kv(label: str, value: str, value_color: str) -> text.Text:
    line = text.Text(no_wrap=True, end="")
    line.append("│ ", style=theme.BORDER_EVAL)
    line.append(f"{label:<13}", style=theme.TEXT_MUTED)
    line.append(f"{value:>{geometry.INSET_W - 4 - 13}}", style=value_color)
    line.append(" │", style=theme.BORDER_EVAL)
    return line


def _inset_text(content: str, color: str) -> text.Text:
    line = text.Text(no_wrap=True, end="")
    line.append("│ ", style=theme.BORDER_EVAL)
    line.append(content.ljust(geometry.INSET_W - 4), style=color)
    line.append(" │", style=theme.BORDER_EVAL)
    return line


def _inset_blank() -> text.Text:
    """An empty inset row that keeps the box's side borders (used as a spacer
    between sections and to pad the inset down to the chart height)."""
    return _inset_text("", theme.BORDER_EVAL)


def _inset_title(last_eval: tuple[int, metrics.EvalResult] | None) -> text.Text:
    """The top border of the eval inset, naming the most recent eval iteration."""
    label = "EVAL" if last_eval is None else f"EVAL · iter {last_eval[0]:04d}"
    return _inset_box_title(label)


def _inset_box_title(label: str) -> text.Text:
    """A docked-inset top border carrying ``label`` (shared by the EVAL and
    COLLECT insets)."""
    title = text.Text(no_wrap=True, end="")
    title.append("┌─ ", style=theme.BORDER_EVAL)
    title.append(label + " ", style=theme.BORDER_EVAL)
    title.append(
        "─" * max(0, geometry.INSET_W - title.cell_len - 1) + "┐",
        style=theme.BORDER_EVAL,
    )
    return title


def _inset_section(label: str) -> text.Text:
    """A dim, centered section header (``-- LAST --`` / ``-- EWMA --``) inside the
    eval inset."""
    line = text.Text(no_wrap=True, end="")
    line.append("│ ", style=theme.BORDER_EVAL)
    line.append(
        f"-- {label} --".center(geometry.INSET_W - 4), style=f"bold {theme.TEXT_MUTED}"
    )
    line.append(" │", style=theme.BORDER_EVAL)
    return line


def _inset_opponent(state: runstate.RunState) -> str:
    """The current reference opponent (the "challenger"): ``random`` while still
    evaluating against the random agent, otherwise the frozen self generation and
    the iteration it was frozen at (``gen N @ iter``)."""
    gen = state.opponent_generation
    if gen == 0:
        return "random"
    return f"gen{gen} @ {state.opponent_since_iteration:04d}"


def _inset_since(state: runstate.RunState) -> str:
    """How many iterations since the frozen self model was last advanced (a dash
    while still evaluating against the random agent, where no frozen self
    exists)."""
    if state.opponent_generation == 0:
        return "—"
    return f"{state.iteration - state.opponent_since_iteration} iters"


def eval_strip(state: runstate.RunState) -> list[text.Text]:
    """Compact one-line eval readout when the panel is too narrow for the inset."""
    last_eval = _latest_eval(state)
    line = text.Text(no_wrap=True, end="")
    line.append(" " * geometry.GUTTER_W)
    if last_eval is None:
        line.append("eval: awaiting first evaluation…", style=theme.TEXT_MUTED)
        return [line]
    _, result = last_eval
    ewma = state.eval_ewma()
    line.append("eval ", style=theme.TEXT_MUTED)
    line.append(
        f"{result.win_rate * 100:.1f}%", style=theme.hero_color(result.win_rate * 100)
    )
    line.append(f" ±{result.ci95 * 100:.1f}%  ", style=theme.TEXT_DIM2)
    line.append(f"margin {result.mean_margin:+.1f}", style=theme.MARGIN_COLOR)
    if ewma is not None:
        line.append(
            f"  ewma {ewma.win_rate * 100:.1f}% / {ewma.mean_margin:+.1f}",
            style=theme.TEXT_DIM2,
        )
    return [line]


def collect_inset(state: runstate.RunState, height: int) -> list[text.Text]:
    """The left-docked bootstrap-phase box — the random-phase twin of
    :func:`eval_inset`: the cinematic hero collection win-rate (vs random), its
    last / EWMA readouts, and the graduation target. Padded to ``height`` so it
    aligns with the plots."""
    last_iter = state.last_iter
    label = (
        "COLLECT" if last_iter is None else f"COLLECT · iter {last_iter.iteration:04d}"
    )
    body: list[text.Text] = [_inset_box_title(label)]

    if last_iter is None or last_iter.collection_win_rate is None:
        body.append(_inset_text("  collecting vs random…", theme.TEXT_MUTED))
    else:
        last = last_iter.collection_win_rate
        ewma = state.collection_win_rate_ewma()
        hero_pct = (ewma if ewma is not None else last) * 100.0
        body.extend(_hero_block(hero_pct, None))
        body.append(_inset_section("LAST"))
        body.append(_inset_kv("win rate", f"{last * 100:.1f}%", theme.WIN_COLOR))
        body.append(
            _inset_kv("margin", f"{last_iter.avg_margin:+.1f} pts", theme.MARGIN_COLOR)
        )
        if ewma is not None:
            body.append(_inset_blank())
            body.append(_inset_section("EWMA"))
            body.append(_inset_kv("win rate", f"{ewma * 100:.1f}%", theme.WIN_COLOR))
            margin_ewma = state.collection_margin_ewma()
            if margin_ewma is not None:
                body.append(
                    _inset_kv("margin", f"{margin_ewma:+.1f} pts", theme.MARGIN_COLOR)
                )
        body.append(_inset_blank())
        body.append(_inset_kv("opponent", "random", theme.TEXT_DIM2))
        body.append(
            _inset_kv(
                "graduate @",
                f"{state.config.random_phase_win_rate * 100:.0f}%",
                theme.TEXT_DIM2,
            )
        )

    while len(body) < height - 1:
        body.append(_inset_blank())
    footer = text.Text(no_wrap=True, end="")
    footer.append("└" + "─" * (geometry.INSET_W - 2) + "┘", style=theme.BORDER_EVAL)
    return body[: height - 1] + [footer]


def collect_strip(state: runstate.RunState) -> list[text.Text]:
    """Compact one-line bootstrap readout when the panel is too narrow for the
    inset (the random-phase twin of :func:`eval_strip`)."""
    line = text.Text(no_wrap=True, end="")
    line.append(" " * geometry.GUTTER_W)
    last = None if state.last_iter is None else state.last_iter.collection_win_rate
    if last is None:
        line.append(
            "collect: vs random, awaiting first iteration…", style=theme.TEXT_MUTED
        )
        return [line]
    ewma = state.collection_win_rate_ewma()
    line.append("collect ", style=theme.TEXT_MUTED)
    line.append(f"{last * 100:.1f}%", style=theme.hero_color(last * 100))
    line.append(" vs random", style=theme.TEXT_DIM2)
    if ewma is not None:
        margin_ewma = state.collection_margin_ewma()
        suffix = (
            f"  ewma {ewma * 100:.1f}% / {margin_ewma:+.1f}"
            if margin_ewma is not None
            else f"  ewma {ewma * 100:.1f}%"
        )
        line.append(suffix, style=theme.TEXT_DIM2)
    return [line]


def _latest_eval(
    state: runstate.RunState,
) -> tuple[int, metrics.EvalResult] | None:
    for item in reversed(state.history):
        if item.eval is not None:
            return (item.iteration, item.eval)
    return None
