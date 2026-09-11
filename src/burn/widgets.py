"""Small rich renderables shared by the views."""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table
from rich.text import Text

Column = tuple[str, int]


def meter(fraction: float, width: int, style: str = "cyan") -> Text:
    """An htop bracket meter."""
    filled = max(0, min(width, round(fraction * width)))
    return Text.assemble(
        ("[", "dim"), ("|" * filled, style), (" " * (width - filled), ""), ("]", "dim")
    )


def bar(percent: float, width: int = 8) -> str:
    """A solid proportion bar, for table cells."""
    filled = min(width, max(0, round(percent / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def grid(rows: Iterable[tuple[str, object]]) -> Table:
    """A two-column label/value block, the shape every detail pane uses."""
    layout = Table.grid(padding=(0, 2))
    layout.add_column(style="dim")
    layout.add_column()
    for name, value in rows:
        layout.add_row(name, value)
    return layout


def columns(left: Iterable[Column], right: Iterable[Column] = ()) -> Table:
    """A borderless table: named left-aligned columns, then right-aligned ones."""
    table = Table(box=None, pad_edge=False, header_style="bold", title_justify="left")
    for name, width in left:
        table.add_column(name, width=width, no_wrap=True)
    for name, width in right:
        table.add_column(name, width=width, justify="right", no_wrap=True)
    return table
