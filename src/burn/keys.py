"""Map keypresses to view state.

``key_apply`` is total, pure, and returns None only to mean quit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .model import Row
from .state import (
    AGENTS,
    MAX_WINDOW,
    MIN_WINDOW,
    SORTS,
    Mode,
    Panel,
    View,
    view_cursor,
)

__all__ = []

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


def key_apply(key: str, view: View, table: Sequence[Row]) -> View | None:
    """Return the next view, or None to quit."""
    if view.filtering:
        return _typing(key, view)
    if view.helping:
        return dataclasses.replace(view, mode=Mode.TABLE)
    return _command(key, view, table)


# -------------------------------------------------------------------- internals


def _typing(key: str, view: View) -> View:
    """Filter-entry state."""
    if key in ENTER:
        return dataclasses.replace(view, needle=view.draft, mode=Mode.TABLE)
    if key == ESCAPE:
        return dataclasses.replace(view, draft="", mode=Mode.TABLE)
    if key in BACKSPACE:
        return dataclasses.replace(view, draft=view.draft[:-1])
    return dataclasses.replace(view, draft=view.draft + key) if key.isprintable() else view


def _command(key: str, view: View, table: Sequence[Row]) -> View | None:
    """A keypress while the table is active."""
    match key:
        case k if k in QUIT:
            return None
        case "h" | "?":
            return dataclasses.replace(view, mode=Mode.HELP)
        case " ":
            return dataclasses.replace(view, paused=not view.paused)
        case k if k in DOWN:
            return dataclasses.replace(view, selected=_step(view, table, +1))
        case k if k in UP:
            return dataclasses.replace(view, selected=_step(view, table, -1))
        case k if k in ENTER:
            return dataclasses.replace(view, zoomed=not view.zoomed, selected=_step(view, table, 0))
        case "t":
            flipped = Panel.TURNS if view.panel is Panel.TOOLS else Panel.TOOLS
            return dataclasses.replace(view, panel=flipped)
        case "s" | "S":
            return dataclasses.replace(view, sort=_cycle(view.sort, +1 if key == "s" else -1))
        case "r":
            return dataclasses.replace(view, reverse=not view.reverse)
        case "/":
            return dataclasses.replace(view, mode=Mode.FILTER, draft=view.needle)
        case "a":
            following = AGENTS[(AGENTS.index(view.source) + 1) % len(AGENTS)]
            return dataclasses.replace(view, source=following)
        case "+" | "=":
            return dataclasses.replace(view, window=min(view.window * 2, MAX_WINDOW))
        case "-" | "_":
            return dataclasses.replace(view, window=max(view.window // 2, MIN_WINDOW))
        case "c":
            return dataclasses.replace(view, needle="", zoomed=False, source=None)
        case _:
            return view


def _step(view: View, table: Sequence[Row], by: int) -> str | None:
    """Return the selected row's key, clamped to the table."""
    if not table:
        return None
    at = min(max(view_cursor(view, list(table)) + by, 0), len(table) - 1)
    return table[at].key


def _cycle(sort: str, by: int) -> str:
    names = [field for field, _ in SORTS]
    return names[(names.index(sort) + by) % len(names)]
