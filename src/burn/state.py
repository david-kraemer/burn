"""Interactive state and table rows.

:class:`View` is frozen; a keypress produces a new one. Selection is keyed
by row key, not index, so sorting/filtering/window changes never need to
clamp or slide the cursor onto a neighbouring row.
"""

from __future__ import annotations

import dataclasses
import enum
import operator
import time

from .analysis import snapshot_table
from .model import Row, Snapshot, Source

__all__ = ["View"]



class Mode(enum.StrEnum):
    """What the keyboard is driving: the table, the help screen, or the filter prompt."""

    TABLE = "table"
    HELP = "help"
    FILTER = "filter"


class Panel(enum.StrEnum):
    """The detail shown beside a zoomed session."""

    TOOLS = "tools"
    TURNS = "turns"


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

AGENTS = (None, *Source)

MIN_WINDOW = 5
MAX_WINDOW = 7 * 24 * 60


@dataclasses.dataclass(frozen=True, slots=True)
class View:
    """State controlled by the keyboard."""

    window: int = 300  # minutes of history
    interval: float = 2.0  # seconds between samples
    limit: float = 0.0  # the 5h weighted-token allowance, if known
    sort: str = "weight"
    reverse: bool = True
    source: Source | None = None
    needle: str = ""
    selected: str | None = None
    paused: bool = False
    zoomed: bool = False
    panel: Panel = Panel.TOOLS
    mode: Mode = Mode.TABLE
    draft: str = ""
    started: float = dataclasses.field(default_factory=time.time)

    @property
    def seconds(self) -> float:
        return self.window * 60

    @property
    def filtering(self) -> bool:
        """Keys are typing into the filter prompt."""
        return self.mode is Mode.FILTER

    @property
    def helping(self) -> bool:
        """The help screen covers the table."""
        return self.mode is Mode.HELP


def view_rows(view: View, snapshot: Snapshot) -> list[Row]:
    """The filtered and sorted table."""
    narrowed = snapshot.since(view.seconds).from_agent(view.source)
    return sorted(
        snapshot_table(narrowed, view.needle),
        key=operator.attrgetter(view.sort),
        reverse=view.reverse,
    )


def view_cursor(view: View, table: list[Row]) -> int:
    """The current selection, starting at the first row."""
    for index, row in enumerate(table):
        if row.key == view.selected:
            return index
    return 0


def table_viewport(table: list[Row], at: int, capacity: int) -> tuple[list[Row], int]:
    """Rows to draw while keeping the cursor visible."""
    if len(table) <= capacity:
        return table, 0
    offset = max(0, min(at - capacity // 2, len(table) - capacity))
    return table[offset : offset + capacity], offset
