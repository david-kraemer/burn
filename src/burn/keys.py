"""Map keypresses to view state.

``apply`` is total and pure. It never touches the terminal. It never
reads the disk. It returns ``None`` only to mean quit. That is what lets
a test exercise the whole interactive surface as a list of keystrokes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .model import Row
from .state import (
    AGENTS,
    FILTER,
    HELP,
    MAX_WINDOW,
    MIN_WINDOW,
    SORTS,
    TABLE,
    TOOLS,
    TURNS,
    View,
    cursor,
)

UP = ("k", "\x1b[A")
DOWN = ("j", "\x1b[B")
ENTER = ("\r", "\n")
ESCAPE = "\x1b"
BACKSPACE = ("\x7f", "\b")
QUIT = ("q", "\x03")

# Shown in the key bar, in this order.
BINDINGS = (
    ("h", "Help"),
    ("s", "Sort"),
    ("/", "Filter"),
    ("a", "Agent"),
    ("↵", "Zoom"),
    ("t", "Tools"),
    ("+-", "Window"),
    ("q", "Quit"),
)

HELP_TEXT = (
    ("↑ ↓  j k", "move the cursor"),
    ("↵  enter", "zoom the selected session into the pane below"),
    ("t", "swap the zoom pane between tools and turns"),
    ("s  S", "cycle the sort column forwards / backwards"),
    ("r", "reverse the sort order"),
    ("/", "filter by session, project, model or agent; ↵ accepts, esc clears"),
    ("a", "cycle agent: both → claude → codex"),
    ("+  -", "widen / narrow the time window"),
    ("space", "pause and resume sampling"),
    ("c", "clear the filter, zoom and agent selection"),
    ("h  ?", "this screen"),
    ("q", "quit"),
)


def apply(key: str, view: View, table: Sequence[Row]) -> View | None:
    """Return the next view, or None to quit."""
    if view.mode == FILTER:
        return typing(key, view)
    if view.mode == HELP:
        return replace(view, mode=TABLE)
    return command(key, view, table)


def typing(key: str, view: View) -> View:
    """Filter-entry state."""
    if key in ENTER:
        return replace(view, needle=view.draft, mode=TABLE)
    if key == ESCAPE:
        return replace(view, draft="", mode=TABLE)
    if key in BACKSPACE:
        return replace(view, draft=view.draft[:-1])
    return replace(view, draft=view.draft + key) if key.isprintable() else view


def command(key: str, view: View, table: Sequence[Row]) -> View | None:
    """A keypress while the table is active."""
    match key:
        case k if k in QUIT:
            return None
        case "h" | "?":
            return replace(view, mode=HELP)
        case " ":
            return replace(view, paused=not view.paused)
        case k if k in DOWN:
            return replace(view, selected=step(view, table, +1))
        case k if k in UP:
            return replace(view, selected=step(view, table, -1))
        case k if k in ENTER:
            return replace(view, zoomed=not view.zoomed, selected=step(view, table, 0))
        case "t":
            return replace(view, panel=TURNS if view.panel == TOOLS else TOOLS)
        case "s" | "S":
            return replace(view, sort=cycle(view.sort, +1 if key == "s" else -1))
        case "r":
            return replace(view, reverse=not view.reverse)
        case "/":
            return replace(view, mode=FILTER, draft=view.needle)
        case "a":
            return replace(view, source=AGENTS[(AGENTS.index(view.source) + 1) % len(AGENTS)])
        case "+" | "=":
            return replace(view, window=min(view.window * 2, MAX_WINDOW))
        case "-" | "_":
            return replace(view, window=max(view.window // 2, MIN_WINDOW))
        case "c":
            return replace(view, needle="", zoomed=False, source=None)
        case _:
            return view


def step(view: View, table: Sequence[Row], by: int) -> str | None:
    """Return the selected row's key, clamped to the table."""
    if not table:
        return None
    at = min(max(cursor(view, list(table)) + by, 0), len(table) - 1)
    return table[at].key


def cycle(sort: str, by: int) -> str:
    names = [field for field, _ in SORTS]
    return names[(names.index(sort) + by) % len(names)]
