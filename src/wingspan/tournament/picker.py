"""The interactive competitor picker — a full-screen checklist of trained runs.

``run_picker`` discovers every selectable run under a base checkpoint dir (the
active run plus its archives), shows them as a togglable list, and returns the
chosen competitor specs. Input uses the configurator's cross-platform
:class:`keys.KeyReader`; the screen is a single ``rich`` panel repainted on every
keypress. Returns ``None`` if the user cancels (Esc / Ctrl-C).
"""

from __future__ import annotations

import pydantic
from rich import box, console, live, panel, table, text

from wingspan.tournament import participants
from wingspan.training import theme
from wingspan.training.configure import keys

# Minimum competitors a tournament needs before the picker will start.
_MIN_COMPETITORS = 2


class _Item(pydantic.BaseModel):
    """One selectable competitor row: its spec plus a one-line description."""

    spec: participants.ParticipantSpec
    subtitle: str


def run_picker(
    base_dir: str,
    term: console.Console,
    include_random_option: bool,
    games_per_pair: int,
) -> list[participants.ParticipantSpec] | None:
    """Pick the tournament's competitors interactively. ``None`` if cancelled."""
    items = _build_items(base_dir, include_random_option)
    selected = {index for index, item in enumerate(items) if _is_model(item)}
    cursor = 0
    warning = ""

    with live.Live(
        _render(items, selected, cursor, base_dir, games_per_pair, warning),
        console=term,
        screen=True,
        auto_refresh=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as display:
        with keys.KeyReader() as reader:
            while True:
                display.update(
                    _render(items, selected, cursor, base_dir, games_per_pair, warning),
                    refresh=True,
                )
                event = reader.poll()
                if event is None:
                    continue
                if event.kind in (keys.KeyKind.ESCAPE, keys.KeyKind.INTERRUPT):
                    return None
                if event.kind is keys.KeyKind.ENTER:
                    if len(selected) >= _MIN_COMPETITORS:
                        return [items[index].spec for index in sorted(selected)]
                    warning = f"select at least {_MIN_COMPETITORS} competitors"
                    continue
                cursor, warning = _handle_key(event, items, selected, cursor)


###### PRIVATE #######


def _handle_key(
    event: keys.KeyEvent,
    items: list[_Item],
    selected: set[int],
    cursor: int,
) -> tuple[int, str]:
    """Apply a navigation / toggle key, returning the new cursor and any warning."""
    if event.kind is keys.KeyKind.UP:
        return (cursor - 1) % len(items), ""
    if event.kind is keys.KeyKind.DOWN:
        return (cursor + 1) % len(items), ""
    if event.char == " ":
        selected.symmetric_difference_update({cursor})
        return cursor, ""
    if event.char in ("a", "A"):
        selected.update(range(len(items)))
        return cursor, ""
    if event.char in ("n", "N"):
        selected.clear()
        return cursor, ""
    return cursor, ""


def _build_items(base_dir: str, include_random_option: bool) -> list[_Item]:
    """Every selectable competitor: discovered runs, then the random agent."""
    items = [
        _Item(spec=option.to_spec(), subtitle=_run_subtitle(option))
        for option in participants.discover_runs(base_dir)
    ]
    if include_random_option:
        items.append(
            _Item(
                spec=participants.random_spec(),
                subtitle="uniform-random baseline",
            )
        )
    return items


def _is_model(item: _Item) -> bool:
    return item.spec.kind is participants.ParticipantKind.MODEL


def _run_subtitle(option: participants.RunOption) -> str:
    """A compact ``iter N · best W%`` line for a discovered run."""
    parts: list[str] = []
    if option.iteration is not None:
        parts.append(f"iter {option.iteration}")
    if option.best_win_rate is not None:
        parts.append(f"best {option.best_win_rate * 100:.0f}%")
    return " · ".join(parts)


def _render(
    items: list[_Item],
    selected: set[int],
    cursor: int,
    base_dir: str,
    games_per_pair: int,
    warning: str,
) -> panel.Panel:
    """The full-screen picker panel: header, the checklist, and a help footer."""
    grid = table.Table.grid(padding=(0, 1))
    grid.add_column(width=1)
    grid.add_column(width=3)
    grid.add_column(justify="left")
    grid.add_column(justify="left", style=theme.TEXT_MUTED)
    if not items:
        grid.add_row(
            "", "", text.Text(f"no runs found under {base_dir}/", theme.BAD), ""
        )
    for index, item in enumerate(items):
        marker = "›" if index == cursor else " "
        checkbox = "[x]" if index in selected else "[ ]"
        name_style = theme.TEXT_BRIGHT if index == cursor else theme.TEXT_PRIMARY
        grid.add_row(
            text.Text(marker, style=theme.GOOD),
            text.Text(
                checkbox, style=theme.GOOD if index in selected else theme.TEXT_MUTED
            ),
            text.Text(item.spec.display_name, style=name_style),
            item.subtitle,
        )
    footer = text.Text(no_wrap=True, end="")
    footer.append(f"{len(selected)} selected", style=theme.TEXT_DIM2)
    footer.append(f"   games/pair: {games_per_pair}", style=theme.TEXT_MUTED)
    footer.append(
        "   ↑↓ move · space toggle · a all · n none · enter start · esc cancel",
        style=theme.TEXT_MUTED,
    )
    if warning:
        footer.append(f"   {warning}", style=theme.BAD)
    body = console.Group(
        theme.gradient_text("WINGSPAN // SELECT COMPETITORS"), text.Text(""), grid
    )
    return panel.Panel(
        console.Group(body, text.Text(""), footer),
        border_style=theme.BORDER_HEADLINE,
        box=box.ROUNDED,
        padding=(1, 2),
    )
