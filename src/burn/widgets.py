"""Small rich renderables shared by the views."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

from rich.table import Table
from rich.text import Text

from .format import percent_tint

__all__ = []

Column = tuple[str, int]


def widget_meter(fraction: float, width: int, style: str | None = None) -> Text:
    """Render a bracket meter.

    Without a style, each filled cell takes the heat color of the band its
    midpoint falls in, so a meter at 90% runs cool to hot and shows red only
    on the cells past 85%.
    """
    filled = _clamp(fraction * width, width)
    cells = [style or percent_tint(100 * (i + 0.5) / width) for i in range(filled)]
    runs = [("|" * len(list(run)), tint) for tint, run in itertools.groupby(cells)]
    return Text.assemble(("[", "dim"), *runs, (" " * (width - filled), ""), ("]", "dim"))


def widget_bar(percent: float, width: int = 8) -> str:
    """Render a proportion bar for a table cell."""
    filled = _clamp(percent / 100 * width, width)
    return "█" * filled + "░" * (width - filled)


def widget_grid(rows: Iterable[tuple[str, object]]) -> Table:
    """Render a two-column label and value block."""
    layout = Table.grid(padding=(0, 2))
    layout.add_column(style="dim")
    layout.add_column()
    for name, value in rows:
        layout.add_row(name, value)
    return layout


def widget_columns(left: Iterable[Column], right: Iterable[Column] = ()) -> Table:
    """Create a borderless table with left and right columns."""
    table = Table(box=None, pad_edge=False, header_style="bold", title_justify="left")
    for name, width in left:
        table.add_column(name, width=width, no_wrap=True)
    for name, width in right:
        table.add_column(name, width=width, justify="right", no_wrap=True)
    return table


# -------------------------------------------------------------------- internals


def _clamp(value: float, width: int) -> int:
    """Round a fill amount into [0, width]; non-finite input draws empty (round() would raise)."""
    if not math.isfinite(value):
        return 0
    return max(0, min(width, round(value)))
