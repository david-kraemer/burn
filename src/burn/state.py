"""Interactive state and table rows.

:class:`View` is frozen; a keypress produces a new one. Selection is keyed
by row key, not index, so sorting/filtering/window changes never need to
clamp or slide the cursor onto a neighbouring row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from operator import attrgetter

from .analysis import tabulate
from .model import Row, Snapshot

TABLE, HELP, FILTER = "table", "help", "filter"
TOOLS, TURNS = "tools", "turns"

# Sortable columns, in the order used by `s`: field name, column heading.
SORTS = (
    ("weight", "WEIGHT"),
    ("rate", "RATE"),
    ("cost", "$"),
    ("ctx", "CTX"),
    ("cache", "HIT"),
    ("think", "THINK"),
    ("session", "SESSION"),
)

AGENTS = (None, "cc", "cx")

MIN_WINDOW = 5
MAX_WINDOW = 7 * 24 * 60


@dataclass(frozen=True, slots=True)
class View:
    """State controlled by the keyboard."""

    window: int = 300  # minutes of history
    interval: float = 2.0  # seconds between samples
    limit: float = 0.0  # the 5h weighted-token allowance, if known
    sort: str = "weight"
    reverse: bool = True
    source: str | None = None
    needle: str = ""
    selected: str | None = None
    paused: bool = False
    zoomed: bool = False
    panel: str = TOOLS
    mode: str = TABLE
    draft: str = ""
    started: float = field(default_factory=time.time)

    @property
    def seconds(self) -> float:
        return self.window * 60


def rows(view: View, snapshot: Snapshot) -> list[Row]:
    """The filtered and sorted table."""
    narrowed = snapshot.since(view.seconds).from_agent(view.source)
    return sorted(
        tabulate(narrowed, view.needle), key=attrgetter(view.sort), reverse=view.reverse
    )


def cursor(view: View, table: list[Row]) -> int:
    """The current selection, starting at the first row."""
    for index, row in enumerate(table):
        if row.key == view.selected:
            return index
    return 0


def viewport(table: list[Row], at: int, capacity: int) -> tuple[list[Row], int]:
    """Rows to draw while keeping the cursor visible."""
    if len(table) <= capacity:
        return table, 0
    offset = max(0, min(at - capacity // 2, len(table) - capacity))
    return table[offset : offset + capacity], offset
