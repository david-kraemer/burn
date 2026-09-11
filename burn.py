#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["rich>=13.7"]
# ///
"""Token burn inspection for Claude Code and Codex.

Both agents append JSONL transcripts to disk as they work, so nothing needs to
be instrumented. Every view here is built from one unified record -- a single
API call, normalised across the two very different transcript formats.

    burn                 live per-session monitor
    burn tools           which tools are inflating context
    burn turns           the most expensive prompts
    burn session <id>    one session in detail
    burn cost            spend, at rates calibrated from your own billing
    burn waste           cache re-creation you are paying for
    burn verify          audit against Claude's own totals
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import sys
import termios
import time
import tty
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

CLAUDE_ROOT = Path.home() / ".claude" / "projects"
CODEX_ROOT = Path.home() / ".codex" / "sessions"
RATES_CACHE = Path.home() / ".cache" / "burn" / "rates.json"

BLOCK = 300 * 60  # Claude Code's rolling quota block, in seconds
CACHE_TTL = 5 * 60  # the short ephemeral cache tier expires this fast

# Relative to one input token, at Anthropic's published ratios: cache writes
# cost 1.25x, cache reads 0.1x, output 5x. A raw token sum is ~95% cache reads
# and badly overstates what a session is actually consuming.
WEIGHTS = (1.0, 1.25, 0.1, 5.0)

SPARK = "▁▂▃▄▅▆▇█"
FANOUT_TOOLS = {"Task", "Agent", "Workflow"}


@dataclass(slots=True)
class Call:
    """One API request, whichever agent made it."""

    at: float
    source: str
    session: str
    project: str
    model: str
    counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    thinking: int = 0
    cache_5m: int = 0
    cache_1h: int = 0
    prompt: str = ""
    fanout: bool = False

    @property
    def prefix(self) -> int:
        """Everything the model was fed: the whole conversation so far."""
        return self.counts[0] + self.counts[1] + self.counts[2]

    @property
    def weight(self) -> float:
        return weighted(self.counts)


@dataclass(slots=True)
class Tooling:
    """A tool result landing back in the conversation."""

    at: float
    session: str
    name: str
    size: int


@dataclass
class Ledger:
    """Calls and tool results harvested from both transcript trees."""

    calls: dict[str, Call] = field(default_factory=dict)
    tools: list[Tooling] = field(default_factory=list)
    offsets: dict[Path, int] = field(default_factory=dict)
    carry: dict[Path, dict] = field(default_factory=dict)
    limits: dict | None = None
    limits_at: str = ""

    def ordered(self) -> list[Call]:
        return sorted(self.calls.values(), key=lambda c: c.at)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("view", nargs="?", default="live",
                        choices=["live", "tools", "turns", "session", "cost", "waste", "verify"])
    parser.add_argument("target", nargs="?", help="session id prefix, for the session view")
    parser.add_argument("--window", type=int, default=300, help="minutes of history (default 300)")
    parser.add_argument("--interval", type=float, default=2.0, help="live refresh seconds")
    parser.add_argument("--limit", type=float, default=0.0,
                        help="your 5h weighted-token allowance, to turn the Claude meter "
                             "into a real gauge (accepts 40e6)")
    parser.add_argument("--sort", default="weight", choices=[f for f, _ in SORTS])
    parser.add_argument("--once", action="store_true", help="one frame of the live view, then exit")
    parser.add_argument("--source", choices=["cc", "cx"], help="restrict to one agent")
    parser.add_argument("--top", type=int, default=15, help="rows in the drill-down views")
    args = parser.parse_args()

    console = Console()
    view = View(window=args.window, interval=args.interval, limit=args.limit,
                sort=args.sort, source=args.source)
    if args.view == "live" and not args.once:
        monitor(console, view)
        return

    ledger = Ledger()
    harvest(ledger, args.window)
    calls = [c for c in ledger.ordered() if not args.source or c.source == args.source]

    match args.view:
        case "live":
            console.print(dashboard(ledger, view, console.size.height))
        case "tools":
            console.print(tools_view(calls, ledger.tools, args.top))
        case "turns":
            console.print(turns_view(calls, args.top))
        case "session":
            console.print(session_view(calls, ledger.tools, args.target))
        case "cost":
            console.print(cost_view(calls, args.window))
        case "waste":
            console.print(waste_view(calls))
        case "verify":
            console.print(verify_view())


# ---------------------------------------------------------------- live view


SORTS = (
    ("weight", "WEIGHT"), ("rate", "RATE"), ("cost", "$"),
    ("ctx", "CTX"), ("cache", "HIT"), ("think", "THINK"), ("session", "SESSION"),
)

ZOOM_LINES = 13

KEYS = (
    ("h", "Help"), ("s", "Sort"), ("/", "Filter"), ("a", "Agent"),
    ("\u21b5", "Zoom"), ("t", "Tools"), ("+-", "Window"), ("q", "Quit"),
)


@dataclass
class View:
    """Everything the user can change from the keyboard."""

    window: int = 300
    interval: float = 2.0
    limit: float = 0.0
    sort: str = "weight"
    reverse: bool = True
    source: str | None = None
    needle: str = ""
    cursor: int = 0
    paused: bool = False
    zoom: str | None = None
    panel: str = "tools"
    mode: str = "table"
    draft: str = ""
    started: float = field(default_factory=time.time)
    first_seen: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Row:
    """One session, reduced to the columns the table sorts on."""

    source: str
    session: str
    project: str
    model: str
    ctx: int
    cache: float
    think: int
    weight: float
    rate: float
    cost: float
    calls: int
    last: float
    fanout: bool


def monitor(console: Console, view: View) -> None:
    """The interactive loop: redraw on a timer, and at once on a keypress."""
    ledger = Ledger()
    with cbreak() as interactive, Live(console=console, screen=True, auto_refresh=False) as live:
        due = 0.0
        while True:
            if not view.paused and time.time() >= due:
                harvest(ledger, view.window)
                due = time.time() + view.interval
            live.update(dashboard(ledger, view, console.size.height), refresh=True)
            if not interactive:
                time.sleep(view.interval)
                continue
            key = keypress(max(0.05, due - time.time()) if not view.paused else 0.5)
            if key and not handle(key, view, ledger):
                return


def dashboard(ledger: Ledger, view: View, height: int = 24) -> Group:
    """Meters, burn history, the session table, and a key bar."""
    now = time.time()
    calls = [c for c in ledger.ordered() if c.at >= now - view.window * 60]
    rows = tabulate(calls, now, view)

    if view.mode == "help":
        return Group(masthead(ledger, view, rows, now), Text(""), helpscreen(), keybar(view))

    view.cursor = max(0, min(view.cursor, len(rows) - 1)) if rows else 0

    # Chrome: masthead, meters, burn lanes, the table's own header, the scroll
    # indicator, key bar and the blank lines between them. The rest is table.
    chrome = 13 + (ZOOM_LINES if view.zoom else 0)
    shown, offset = viewport(rows, view.cursor, max(3, height - chrome))

    body: list[RenderableType] = [
        masthead(ledger, view, rows, now),
        Text(""),
        meters(ledger, view, calls, now),
        Text(""),
        burn_spark(calls, now, view.window),
        Text(""),
        sessions_table(shown, view, now, offset),
        scrollbar(rows, shown, offset),
    ]
    if view.zoom:
        chosen = [c for c in calls if c.session == view.zoom]
        body += [Text(""), zoom_panel(chosen, ledger.tools, view)]
    body += [Text(""), keybar(view)]
    return Group(*body)


def viewport(rows: list[Row], cursor: int, capacity: int) -> tuple[list[Row], int]:
    """The slice of rows to draw, keeping the cursor comfortably inside it."""
    if len(rows) <= capacity:
        return rows, 0
    offset = max(0, min(cursor - capacity // 2, len(rows) - capacity))
    return rows[offset:offset + capacity], offset


def scrollbar(rows: list[Row], shown: list[Row], offset: int) -> Text:
    """Say what is off-screen, the way a pager does."""
    hidden_above, hidden_below = offset, len(rows) - offset - len(shown)
    if not hidden_above and not hidden_below:
        return Text("")
    parts = []
    if hidden_above:
        parts.append(f"▲ {hidden_above} above")
    if hidden_below:
        parts.append(f"▼ {hidden_below} below")
    return Text("  " + "   ".join(parts), style="dim")


def masthead(ledger: Ledger, view: View, rows: list[Row], now: float) -> Text:
    """The one-line summary: uptime, how many sessions, and what is filtered."""
    live_now = sum(1 for r in rows if now - r.last <= 120)
    line = Text.assemble(
        ("burn", "bold"), f"  {datetime.now(UTC).astimezone():%H:%M:%S}  ",
        ("up ", "dim"), span(now - view.started), "   ",
        (f"{len(rows)} sessions", "bold"), f", {live_now} active   ",
        ("window ", "dim"), span(view.window * 60),
    )
    if view.source:
        line.append(f"   agent {view.source}", style="yellow")
    if view.needle:
        line.append(f"   /{view.needle}", style="yellow")
    if view.paused:
        line.append("   PAUSED", style="black on yellow")
    return line


def meters(ledger: Ledger, view: View, calls: list[Call], now: float) -> Group:
    """htop-style bracket meters, one per quota window."""
    lines: list[RenderableType] = []
    stamps = sorted(c.at for c in ledger.calls.values() if c.source == "cc")
    block = current_block(stamps, now)
    if block is None:
        lines.append(Text.assemble(("Claude 5h ", "cyan"), ("  idle", "dim")))
    else:
        start, _ = block
        inside = [c for c in ledger.calls.values() if c.source == "cc" and c.at >= start]
        spent = sum(c.weight for c in inside)
        money = sum(dollars(c) for c in inside)
        elapsed, left = now - start, max(BLOCK - (now - start), 0.0)
        # With no quota recorded on disk there is no denominator, so the meter
        # tracks the block's clock unless a limit was supplied.
        if view.limit:
            fraction, tail = spent / view.limit, f"{quantity(spent)}/{quantity(view.limit)}"
        else:
            fraction, tail = elapsed / BLOCK, quantity(spent)
        projected = spent + spent / max(elapsed, 60) * left
        lines.append(Text.assemble(
            ("Claude 5h ", "cyan"), meter(fraction, 24), f" {tail}",
            (f" →{quantity(projected)}", "dim"),
            (f"  ${money:.2f}→${money + money / max(elapsed, 60) * left:.2f}" if money else ""),
            (f"  {span(left)} left", "dim"),
        ))

    limits = ledger.limits or {}
    for name in ("primary", "secondary"):
        gauge = limits.get(name)
        if not gauge:
            continue
        # Codex reports each window's length explicitly, and does not promise
        # that "primary" means five hours -- for six weeks of this history the
        # only gauge present was the seven-day one, under that same key.
        used = gauge.get("used_percent") or 0.0
        tag = f"Codex {span((gauge.get('window_minutes') or 0) * 60):>4} "
        lines.append(Text.assemble(
            (tag, "magenta"), meter(used / 100, 24, tint(used)),
            f" {used:5.1f}%", (f"  resets {clock(gauge.get('resets_at'))}", "dim"),
        ))
    if not limits:
        lines.append(Text.assemble(("Codex     ", "magenta"), ("  no rate-limit event seen", "dim")))
    return Group(*lines)


def burn_spark(calls: list[Call], now: float, window: int) -> RenderableType:
    """Weighted burn per bucket across the window, one lane per agent."""
    width = 48
    step = window * 60 / width
    lines: list[RenderableType] = []
    for source, style in (("cc", "cyan"), ("cx", "magenta")):
        series = [0.0] * width
        for call in calls:
            if call.source != source:
                continue
            slot = int((call.at - (now - window * 60)) / step)
            if 0 <= slot < width:
                series[slot] += call.weight
        if not any(series):
            continue
        lines.append(Text.assemble(
            (f"burn {source} ", "dim"), (sparkline(series, max(series)), style),
            (f"  peak {quantity(max(series))}/{span(step)}", "dim"),
        ))
    return Group(*lines) if lines else Text("burn      (idle)", style="dim")


def tabulate(calls: list[Call], now: float, view: View) -> list[Row]:
    """Collapse calls into one sorted, filtered row per session."""
    groups: dict[tuple[str, str, str], list[Call]] = defaultdict(list)
    for call in calls:
        if view.source and call.source != view.source:
            continue
        groups[(call.source, call.session, call.project)].append(call)

    rows = []
    for (source, session, project), items in groups.items():
        hits = sum(c.counts[2] for c in items)
        misses = sum(c.counts[1] for c in items)
        rows.append(Row(
            source=source, session=session, project=project, model=items[-1].model,
            ctx=items[-1].prefix, cache=100 * hits / max(hits + misses, 1),
            think=sum(c.thinking for c in items),
            weight=sum(c.weight for c in items),
            rate=sum(c.weight for c in items if c.at >= now - 300) / 5,
            cost=sum(dollars(c) for c in items), calls=len(items),
            last=max(c.at for c in items), fanout=any(c.fanout for c in items),
        ))
    if view.needle:
        needle = view.needle.lower()
        rows = [r for r in rows
                if needle in f"{r.session}{r.project}{r.model}{r.source}".lower()]
    for row in rows:
        view.first_seen.setdefault(row.session, now)
    return sorted(rows, key=lambda r: getattr(r, view.sort), reverse=view.reverse)


def sessions_table(rows: list[Row], view: View, now: float, offset: int = 0) -> Table:
    """The process table, with a cursor and a marked sort column."""
    table = Table(box=None, pad_edge=False, collapse_padding=True, header_style="bold")
    table.add_column(" ", width=1, no_wrap=True)
    for name, width in (("SRC", 3), ("SESSION", 9), ("PROJECT", 10), ("MODEL", 9)):
        table.add_column(heading(name, view), width=width, no_wrap=True)
    for name, width in (("CTX", 6), ("HIT", 5), ("THINK", 5),
                        ("WEIGHT", 7), ("RATE", 6), ("$", 5)):
        table.add_column(heading(name, view), width=width, justify="right", no_wrap=True)

    for index, row in enumerate(rows, start=offset):
        idle = now - row.last > 120
        fresh_row = now - view.first_seen.get(row.session, 0) < 20
        style = "on grey23" if index == view.cursor else ("dim" if idle else "")
        table.add_row(
            "▸" if index == view.cursor else " ",
            Text(row.source, style="cyan" if row.source == "cc" else "magenta"),
            Text(row.session + ("⑂" if row.fanout else ""),
                 style="green" if fresh_row else ""),
            trim(row.project, 10), trim(label(row.model), 9),
            quantity(row.ctx),
            Text(f"{row.cache:.0f}%", style=tint(100 - row.cache)),
            quantity(row.think),
            Text(quantity(row.weight), style="bold"),
            Text(quantity(row.rate) if row.rate else "·",
                 style="green" if row.rate else "dim"),
            f"{row.cost:.2f}" if row.cost else "·",
            style=style,
        )
    if not rows:
        table.add_row(" ", *["·"] * 10)
    return table


def zoom_panel(calls: list[Call], tooling: list[Tooling], view: View) -> RenderableType:
    """The detail pane under the table, for whichever session is selected."""
    if not calls:
        return Text(f"  {view.zoom}: no calls in this window", style="dim")
    if view.panel == "turns":
        return turns_view(calls, 6)
    return tools_view(calls, tooling, 6)


def helpscreen() -> RenderableType:
    """Every binding, since a top-alike is worthless if you must read the source."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column()
    for key, what in (
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
    ):
        grid.add_row(key, what)
    return Group(Text("  Keys", style="bold"), Text(""), grid, Text(""),
                 Text("  Columns: CTX is the current context size, HIT the share of input\n"
                      "  served from cache, THINK reasoning tokens, WEIGHT tokens in\n"
                      "  input-token equivalents, RATE the weighted burn over five minutes.\n"
                      "  ⑂ marks a session that fanned out to subagents.",
                      style="dim"))


def keybar(view: View) -> Text:
    """htop's function-key strip, in the letters this program actually reads."""
    if view.mode == "filter":
        return Text.assemble(("filter", "black on yellow"), " ", view.draft,
                             ("▏", "bold"), ("   ↵ accept   esc cancel", "dim"))
    bar = Text()
    for key, name in KEYS:
        bar.append(key, style="dim")
        bar.append(name, style="black on cyan")
        bar.append(" ")
    return bar


def heading(name: str, view: View) -> Text:
    """Mark the column the table is sorted on, the way top does."""
    for field_, column in SORTS:
        if column == name and view.sort == field_:
            return Text(name + ("▼" if view.reverse else "▲"), style="black on cyan")
    return Text(name)


def meter(fraction: float, width: int, style: str = "cyan") -> Text:
    """An htop bracket meter."""
    filled = max(0, min(width, round(fraction * width)))
    return Text.assemble(
        ("[", "dim"), ("|" * filled, style), (" " * (width - filled), ""), ("]", "dim")
    )


# ------------------------------------------------------------------ keyboard


def handle(key: str, view: View, ledger: Ledger) -> bool:
    """Apply one keypress. Returns False to quit."""
    if view.mode == "filter":
        match key:
            case "\r" | "\n":
                view.needle, view.mode = view.draft, "table"
            case "\x1b":
                view.draft, view.mode = "", "table"
            case "\x7f" | "\b":
                view.draft = view.draft[:-1]
            case _ if key.isprintable():
                view.draft += key
        return True
    if view.mode == "help":
        view.mode = "table"
        return True

    rows = tabulate([c for c in ledger.ordered() if c.at >= time.time() - view.window * 60],
                    time.time(), view)
    # A filter or an expiring session can shrink the table under the cursor.
    view.cursor = min(view.cursor, len(rows) - 1) if rows else 0
    match key:
        case "q" | "\x03":
            return False
        case "h" | "?":
            view.mode = "help"
        case " ":
            view.paused = not view.paused
        case "j" | "\x1b[B":
            view.cursor = min(view.cursor + 1, max(len(rows) - 1, 0))
        case "k" | "\x1b[A":
            view.cursor = max(0, view.cursor - 1)
        case "\r" | "\n":
            view.zoom = None if view.zoom else (rows[view.cursor].session if rows else None)
        case "t":
            view.panel = "turns" if view.panel == "tools" else "tools"
        case "s" | "S":
            names = [f for f, _ in SORTS]
            step = 1 if key == "s" else -1
            view.sort = names[(names.index(view.sort) + step) % len(names)]
        case "r":
            view.reverse = not view.reverse
        case "/":
            view.mode, view.draft = "filter", view.needle
        case "a":
            order = [None, "cc", "cx"]
            view.source = order[(order.index(view.source) + 1) % len(order)]
        case "+" | "=":
            view.window = min(view.window * 2, 10080)
        case "-" | "_":
            view.window = max(view.window // 2, 5)
        case "c":
            view.needle, view.zoom, view.source = "", None, None
    return True


@contextmanager
def cbreak() -> Iterator[bool]:
    """Put the terminal in character-at-a-time mode, if there is one."""
    if not sys.stdin.isatty():
        yield False
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def keypress(timeout: float) -> str | None:
    """One key, or None if nothing arrived before the timeout."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    key = sys.stdin.read(1)
    if key == "\x1b" and select.select([sys.stdin], [], [], 0.02)[0]:
        key += sys.stdin.read(2)  # an arrow key, not a bare escape
    return key


# ----------------------------------------------------------- drill-down views


def tools_view(calls: list[Call], tooling: list[Tooling], top: int) -> RenderableType:
    """What is actually inflating the context, tool by tool.

    Context growth between two consecutive calls in a session, less the
    assistant's own output, is what tool results and user text injected. That
    growth is real billed tokens, split across the tools that ran in between in
    proportion to the size of what they returned.
    """
    blame = attribution(calls, tooling)
    if not blame:
        return Text("No attributable context growth in this window.", style="dim")

    table = Table(box=None, pad_edge=False, header_style="bold", title_justify="left")
    table.add_column("TOOL", width=20, no_wrap=True)
    for name, width in (("CALLS", 5), ("ADDED", 7), ("PER CALL", 8), ("CARRIED", 8), ("COST", 7)):
        table.add_column(name, width=width, justify="right", no_wrap=True)
    table.add_column("", width=10, no_wrap=True)

    grand = sum(v[2] for v in blame.values())
    ranked = sorted(blame.items(), key=lambda kv: -kv[1][2])[:top]
    for name, (grown, count, carried, cost) in ranked:
        share = carried / max(grand, 1)
        table.add_row(
            trim(name, 20), f"{count:,}", quantity(grown), quantity(grown / max(count, 1)),
            Text(quantity(carried), style="bold"), f"${cost:.2f}" if cost else "·",
            Text(bar(100 * share, 10), style=tint(100 * share * 2)),
        )
    added = sum(v[0] for v in blame.values())
    return Group(
        Text.assemble(("Context burn by tool", "bold"),
                      f"   {quantity(added)} added → {quantity(grand)} weighted with re-reads"),
        Text(""), table, Text(""),
        Text("ADDED is what the tool injected; CARRIED is that re-read by every later call.",
             style="dim"),
    )


def turns_view(calls: list[Call], top: int) -> RenderableType:
    """The most expensive prompts, so the costly asks are identifiable."""
    turns: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for call in calls:
        if call.prompt:
            turns[(call.session, call.prompt)].append(call)
    if not turns:
        return Text("No prompts captured in this window.", style="dim")

    table = Table(box=None, pad_edge=False, header_style="bold")
    for name, width in (("WHEN", 5), ("SESSION", 8)):
        table.add_column(name, width=width, no_wrap=True)
    for name, width in (("CALLS", 5), ("WEIGHT", 7), ("COST", 6)):
        table.add_column(name, width=width, justify="right", no_wrap=True)
    table.add_column("PROMPT", width=38, no_wrap=True)

    ranked = sorted(turns.items(), key=lambda kv: -sum(c.weight for c in kv[1]))
    for (session, prompt), items in ranked[:top]:
        cost = sum(dollars(c) for c in items)
        table.add_row(
            f"{datetime.fromtimestamp(items[0].at, UTC).astimezone():%H:%M}",
            session, str(len(items)),
            Text(quantity(sum(c.weight for c in items)), style="bold"),
            f"${cost:.2f}" if cost else "·",
            trim(" ".join(prompt.split()), 38),
        )
    return Group(Text("Most expensive turns", style="bold"), Text(""), table)


def session_view(calls: list[Call], tooling: list[Tooling], target: str | None) -> RenderableType:
    """One session end to end: context curve, cache behaviour, tool mix."""
    if not target:
        return Text("Usage: burn session <session-id-prefix>", style="dim")
    items = [c for c in calls if c.session.startswith(target)]
    if not items:
        return Text(f"No calls for session {target!r} in this window.", style="dim")

    session = items[0].session
    weight = sum(c.weight for c in items)
    hits = sum(c.counts[2] for c in items)
    misses = sum(c.counts[1] for c in items)
    peak = max(c.prefix for c in items)
    collapses = len(compactions(items))
    models = defaultdict(float)
    for call in items:
        models[label(call.model)] += call.weight

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim")
    facts.add_column()
    facts.add_row("session", f"{session}  ·  {items[0].project}  ·  {items[0].source}")
    facts.add_row("span", f"{span(items[-1].at - items[0].at)} over {len(items)} calls")
    facts.add_row("weighted", f"{quantity(weight)}  (~${sum(dollars(c) for c in items):.2f})")
    facts.add_row("context", f"{quantity(items[-1].prefix)} now, peak {quantity(peak)}")
    facts.add_row("cache", f"{100 * hits / max(hits + misses, 1):.0f}% hits, "
                           f"{quantity(misses)} re-created")
    facts.add_row("thinking", quantity(sum(c.thinking for c in items)))
    facts.add_row("models", ", ".join(f"{m} {quantity(w)}" for m, w in
                                      sorted(models.items(), key=lambda kv: -kv[1])))
    if collapses:
        facts.add_row("compactions", f"{collapses} (context discarded and rebuilt)")
    if any(c.fanout for c in items):
        facts.add_row("fan-out", Text("yes — subagent usage is billed but unrecorded", "yellow"))

    curve = resample([float(c.prefix) for c in items], 56)
    low, high = min(curve), max(curve)
    return Group(
        Text("Session detail", style="bold"), Text(""), facts, Text(""),
        Text.assemble("context  ", (sparkline(curve, high, low * 0.98), "cyan"),
                      f"  {quantity(low)} → {quantity(high)}"),
        Text(""),
        tools_view(items, tooling, 10),
    )


def cost_view(calls: list[Call], window: int) -> RenderableType:
    """Spend by model and project, at rates calibrated from your own billing."""
    rates = calibrated_rates()
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("MODEL", max_width=26, no_wrap=True)
    for name in ("$/WEIGHTED Mtok", "SAMPLES"):
        table.add_column(name, justify="right")
    for model, (rate, samples) in sorted(rates.items()):
        table.add_row(label(model), f"{rate:.2f}", str(samples))

    by_project: dict[tuple[str, str], float] = defaultdict(float)
    for call in calls:
        by_project[(call.source, call.project)] += dollars(call)
    spend = Table(box=None, pad_edge=False, header_style="bold")
    spend.add_column("PROJECT", max_width=24, no_wrap=True)
    for name in ("WEIGHT", "COST"):
        spend.add_column(name, justify="right")
    weights: dict[tuple[str, str], float] = defaultdict(float)
    for call in calls:
        weights[(call.source, call.project)] += call.weight
    for key, cost in sorted(by_project.items(), key=lambda kv: -kv[1])[:15]:
        spend.add_row(f"{key[0]}  {trim(key[1], 20)}", quantity(weights[key]),
                      f"${cost:.2f}" if cost else "·")

    return Group(
        Text("Effective rates", style="bold"),
        Text("Solved from Claude's own cost-state records: dollars billed over weighted\n"
             "tokens observed, using only sessions whose tokens reconcile. Codex records\n"
             "no cost at all, so its rows below show tokens only.", style="dim"),
        Text(""), table, Text(""),
        Text(f"Spend over the last {span(window * 60)}", style="bold"), Text(""), spend,
    )


def waste_view(calls: list[Call]) -> RenderableType:
    """Cache re-creation: the same prefix, paid for again at 12.5x the read price.

    The short cache tier expires after five minutes. Step away for longer and
    the next call re-creates the whole prefix, which is why an idle session can
    cost more than a busy one.
    """
    by_session: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_session[call.session].append(call)

    idle_tokens = idle_events = 0
    churn_tokens = 0
    for items in by_session.values():
        for previous, call in pairwise(items):
            if not call.counts[1]:
                continue
            if call.at - previous.at > CACHE_TTL:
                idle_tokens += call.counts[1]
                idle_events += 1
            else:
                churn_tokens += call.counts[1]

    total = idle_tokens + churn_tokens
    if not total:
        return Text("No cache re-creation in this window.", style="dim")

    # A re-created token costs 1.25 units where a read costs 0.1: 12.5x.
    premium = idle_tokens * (WEIGHTS[1] - WEIGHTS[2])
    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim")
    facts.add_column()
    facts.add_row("re-created after idle", f"{quantity(idle_tokens)} over {idle_events} resumes")
    facts.add_row("re-created in flow", f"{quantity(churn_tokens)} (new turns, unavoidable)")
    facts.add_row("idle premium", f"{quantity(premium)} weighted tokens, "
                                  f"~${premium * blended_rate() / 1e6:.2f}")
    facts.add_row("", Text(f"{100 * idle_tokens / total:.0f}% of all cache creation followed "
                           f"a gap longer than {span(CACHE_TTL)}", style="dim"))

    tiers = Table(box=None, pad_edge=False, header_style="bold")
    tiers.add_column("SESSION", max_width=10, no_wrap=True)
    for name in ("5m TIER", "1h TIER", "IDLE RESUMES"):
        tiers.add_column(name, justify="right")
    ranked = sorted(((k, v) for k, v in by_session.items() if sum(c.counts[1] for c in v)),
                    key=lambda kv: -sum(c.counts[1] for c in kv[1]))[:10]
    for session, items in ranked:
        resumes = sum(1 for a, b in pairwise(items)
                      if b.counts[1] and b.at - a.at > CACHE_TTL)
        tiers.add_row(session, quantity(sum(c.cache_5m for c in items)),
                      quantity(sum(c.cache_1h for c in items)), str(resumes))

    return Group(
        Text("Cache re-creation", style="bold"), Text(""), facts, Text(""), tiers, Text(""),
        Text("Sessions leaning on the 5m tier pay this every time you step away; the 1h "
             "tier costs more to write but survives the gap.", style="dim"),
    )


def verify_view() -> RenderableType:
    """Compare observed tokens with the totals Claude Code computed itself.

    cost-state records land when a session closes and cover everything it was
    billed for, subagents included, so the shortfall against them is exactly the
    traffic this tool cannot see.
    """
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("SESSION")
    for name in ("OBSERVED", "BILLED", "UNSEEN"):
        table.add_column(name, justify="right")
    table.add_column("LIKELY CAUSE")

    gaps: list[float] = []
    for path in sorted(CLAUDE_ROOT.glob("*/*.jsonl")):
        observed, billed, fanout = reconcile(path)
        if billed <= 0:
            continue
        gap = 100 * (billed - observed) / billed
        gaps.append(gap)
        if gap >= 15:
            table.add_row(
                short(path.stem), quantity(observed), quantity(billed),
                Text(f"{gap:.0f}%", style=tint(gap)),
                "fan-out (Task/Workflow)" if fanout else "background calls",
            )
    if not gaps:
        return Text("No closed sessions with cost-state records yet.", style="dim")
    gaps.sort()
    return Group(table, Text(""), Text(
        f"{len(gaps)} closed sessions · median unseen {gaps[len(gaps) // 2]:.1f}% "
        f"· worst {gaps[-1]:.0f}%", style="bold"))


# -------------------------------------------------------------------- analysis


def attribution(
    calls: Iterable[Call], tooling: list[Tooling]
) -> dict[str, tuple[float, int, float, float]]:
    """Split each session's context growth across the tools that caused it.

    Also charges each tool for what it keeps costing. Tokens a tool puts into
    the conversation are written to cache once and then re-read by every later
    call in the session, so a big result early in a long session is far more
    expensive than the same result at the end.
    """
    events: dict[str, list[Tooling]] = defaultdict(list)
    for event in tooling:
        events[event.session].append(event)
    for series in events.values():
        series.sort(key=lambda e: e.at)

    sessions: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        sessions[call.session].append(call)

    blame: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0.0, 0.0])
    for session, items in sessions.items():
        series = events.get(session, [])
        stamps = [e.at for e in series]
        for strand in threads(items):
            for index, (previous, call) in enumerate(pairwise(strand)):
                # Written to cache once, then re-read by every later call in
                # this thread. It stops there: a compaction starts a new
                # thread, and what the summary replaced is not re-read again.
                carry = WEIGHTS[1] + WEIGHTS[2] * max(len(strand) - index - 2, 0)
                # Growth is what entered the conversation, less what the model
                # itself wrote.
                growth = call.prefix - previous.prefix - previous.counts[3]
                if growth <= 0:
                    continue
                between = series[bisect_right(stamps, previous.at):bisect_right(stamps, call.at)]
                rate = per_token_cost(call)
                if not between:
                    row = blame["(prompt / system)"]
                    row[0] += growth
                    row[1] += 1
                    row[2] += growth * carry
                    row[3] += growth * carry * rate
                    continue
                total = sum(max(e.size, 1) for e in between)
                for event in between:
                    portion = growth * max(event.size, 1) / total
                    row = blame[event.name]
                    row[0] += portion
                    row[1] += 1
                    row[2] += portion * carry
                    row[3] += portion * carry * rate
    return {name: (row[0], row[1], row[2], row[3]) for name, row in blame.items()}


def compactions(items: list[Call]) -> list[int]:
    """Indices where the context was discarded and rebuilt from a summary."""
    return [i for i, (a, b) in enumerate(pairwise(items), start=1)
            if b.prefix < a.prefix * 0.7]


def threads(items: list[Call]) -> list[list[Call]]:
    """Split one session's calls into separately growing conversations.

    A session id does not always mean a single linear conversation. Codex runs
    side threads under the same id, so its calls arrive interleaved -- a real
    session here bounces between a 162k prefix and an 88k one. Read as one
    conversation, every switch back up looks like 74k of fresh context, and the
    session accumulates 3.6M of "growth" against a context that never exceeded
    168k. A conversation only ever grows, so each call belongs to the open
    thread whose last prefix sits closest below it; a call that undercuts every
    open thread has been compacted, and starts a new one.
    """
    open_: list[list[Call]] = []
    for call in items:
        fits = [t for t in open_ if t[-1].prefix <= call.prefix]
        if fits:
            max(fits, key=lambda t: t[-1].prefix).append(call)
        else:
            open_.append([call])
    return open_


def current_block(stamps: list[float], now: float) -> tuple[float, float] | None:
    """Start and last-activity time of the five-hour block covering now.

    A block opens on the hour containing the first request after a five-hour
    lull, and closes five hours later or after five idle hours.
    """
    start = last = None
    for stamp in stamps:
        if start is None or stamp - start >= BLOCK or stamp - last >= BLOCK:
            start = stamp - stamp % 3600
        last = stamp
    if start is None or now - start >= BLOCK:
        return None
    return start, last


def calibrated_rates() -> dict[str, tuple[float, int]]:
    """Dollars per weighted megatoken, per model, from your own billing records.

    Claude writes a cost-state record carrying its own dollar figure when a
    session closes. Dividing that by the weighted tokens observed in the same
    session gives an effective rate with nothing hardcoded -- but only for
    sessions whose tokens reconcile, since a session that fanned out to
    subagents is billed for traffic no transcript contains.
    """
    global _RATES
    if _RATES is not None:
        return _RATES
    if fresh(RATES_CACHE):
        try:
            cached = json.loads(RATES_CACHE.read_text())
            _RATES = {k: (float(v[0]), int(v[1])) for k, v in cached.items()}
        except (OSError, ValueError, TypeError, IndexError, KeyError):
            _RATES = None  # truncated or hand-edited; fall through and rebuild
        if _RATES is not None:
            return _RATES

    pools: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for path in CLAUDE_ROOT.glob("*/*.jsonl"):
        observed, billed, fanout = reconcile(path)
        if fanout or billed <= 0 or observed <= 0 or abs(billed - observed) / billed > 0.10:
            continue
        for model, weight, cost in session_models(path):
            pool = pools[model]
            pool[0] += cost
            pool[1] += weight
            pool[2] += 1
    _RATES = {m: (1e6 * p[0] / p[1], int(p[2])) for m, p in pools.items() if p[1] > 0}
    # Written via a temporary file so two concurrent runs cannot leave a
    # half-written cache behind for the next 24 hours.
    try:
        RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        scratch = RATES_CACHE.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps(_RATES))
        scratch.replace(RATES_CACHE)
    except OSError:
        pass  # a cache is an optimisation, not a requirement
    return _RATES


_RATES: dict | None = None


def per_token_cost(call: Call) -> float:
    rate, _ = calibrated_rates().get(call.model, (0.0, 0))
    return rate / 1e6


def dollars(call: Call) -> float:
    return call.weight * per_token_cost(call)


def blended_rate() -> float:
    rates = [r for r, _ in calibrated_rates().values()]
    return sum(rates) / len(rates) if rates else 0.0


def reconcile(path: Path) -> tuple[float, float, bool]:
    """Weighted tokens this tool sees, what Claude billed, and whether it fanned out."""
    observed = billed = 0.0
    fanout = False
    seen: set[str] = set()
    for record in records(path):
        if record.get("type") == "cost-state":
            billed = sum(
                weighted((u["inputTokens"], u["cacheCreationInputTokens"],
                          u["cacheReadInputTokens"], u["outputTokens"]))
                for u in (record.get("modelUsage") or {}).values()
            )
            continue
        if record.get("type") != "assistant":
            continue
        message = record.get("message") or {}
        usage = message.get("usage")
        if not usage or message.get("model") in (None, "<synthetic>"):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("name") in FANOUT_TOOLS:
                fanout = True
        tag = f"{message.get('id')}|{record.get('requestId')}"
        if tag in seen:
            continue
        seen.add(tag)
        observed += weighted(usage_counts(usage))
    return observed, billed, fanout


def session_models(path: Path) -> Iterator[tuple[str, float, float]]:
    """Per-model weighted tokens and dollars from a session's cost-state record."""
    for record in records(path):
        if record.get("type") != "cost-state":
            continue
        for model, u in (record.get("modelUsage") or {}).items():
            weight = weighted((u["inputTokens"], u["cacheCreationInputTokens"],
                               u["cacheReadInputTokens"], u["outputTokens"]))
            if weight > 0 and u["costUSD"] > 0:
                yield model, weight, u["costUSD"]


# ------------------------------------------------------------------ ingestion


def harvest(ledger: Ledger, window: int) -> None:
    """Read whatever is new in both transcript trees into the ledger."""
    horizon = time.time() - (max(window, 300) + 60) * 60
    for path, source in transcripts(horizon):
        lines, offset = new_lines(path, ledger.offsets.get(path, 0))
        ledger.offsets[path] = offset
        if lines:
            reader = read_claude if source == "cc" else read_codex
            reader(lines, ledger, path)

    cutoff = time.time() - max(window, 300) * 60 - 3600
    for tag in [t for t, c in ledger.calls.items() if c.at < cutoff]:
        del ledger.calls[tag]
    ledger.tools = [e for e in ledger.tools if e.at >= cutoff]


def read_claude(lines: list[str], ledger: Ledger, path: Path) -> None:
    """Claude Code assistant records, merged across their content blocks.

    One API call is written as several records, one per content block, each
    repeating the same usage. Keying on (message id, request id) merges them
    without counting the tokens more than once -- and without losing the
    tool_use blocks that live in the later records.
    """
    state = ledger.carry.setdefault(path, {"prompt": "", "pending": {}})
    for record in decode_all(lines):
        kind = record.get("type")
        message = record.get("message") or {}
        if kind == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                state["prompt"] = content
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        name = state["pending"].pop(block.get("tool_use_id"), None)
                        if name:
                            ledger.tools.append(Tooling(
                                moment(record.get("timestamp")) or 0.0,
                                short(record.get("sessionId") or "?"), name,
                                len(json.dumps(block.get("content") or "")),
                            ))
                    elif block.get("type") == "text" and (block.get("text") or "").strip():
                        state["prompt"] = block["text"]
            continue
        if kind != "assistant":
            continue
        usage = message.get("usage")
        model = message.get("model")
        at = moment(record.get("timestamp"))
        tag = f"cc|{message.get('id')}|{record.get('requestId')}"
        call = ledger.calls.get(tag)
        # Tool blocks are registered whatever else this record carries: a
        # content-block record without usage still names tools whose results
        # must be matched later.
        if call is None and usage and model and model != "<synthetic>" and at is not None:
            creation = usage.get("cache_creation") or {}
            call = Call(
                at=at, source="cc", session=short(record.get("sessionId") or "?"),
                project=Path(record.get("cwd") or "?").name, model=model,
                counts=usage_counts(usage),
                thinking=(usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0,
                cache_5m=creation.get("ephemeral_5m_input_tokens") or 0,
                cache_1h=creation.get("ephemeral_1h_input_tokens") or 0,
                prompt=state["prompt"],
            )
            ledger.calls[tag] = call
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                state["pending"][block.get("id")] = block.get("name")
                if call is not None and block.get("name") in FANOUT_TOOLS:
                    call.fanout = True


def read_codex(lines: list[str], ledger: Ledger, path: Path) -> None:
    """Codex token_count events, one per API request.

    Each event reports both a running total and that request's own usage, and
    the two reconcile exactly, so reading incrementally is also correct.
    """
    state = ledger.carry.setdefault(
        path, {"session": short(path.stem[-36:]), "project": "?", "model": "?",
               "prompt": "", "pending": {}}
    )
    for record in decode_all(lines):
        payload = record.get("payload") or {}
        at = moment(record.get("timestamp")) or 0.0
        match record.get("type"):
            case "session_meta":
                state["session"] = short(payload.get("session_id") or state["session"])
                state["project"] = Path(payload.get("cwd") or state["project"]).name
            case "turn_context":
                state["model"] = payload.get("model") or state["model"]
                state["project"] = Path(payload.get("cwd") or state["project"]).name
            case "response_item":
                if payload.get("type") in ("custom_tool_call", "function_call"):
                    state["pending"][payload.get("call_id")] = payload.get("name") or "?"
                elif payload.get("type") in ("custom_tool_call_output", "function_call_output"):
                    name = state["pending"].pop(payload.get("call_id"), None)
                    if name:
                        ledger.tools.append(Tooling(
                            at, state["session"], name,
                            len(json.dumps(payload.get("output") or "")),
                        ))
            case "event_msg":
                match payload.get("type"):
                    case "user_message":
                        state["prompt"] = payload.get("message") or state["prompt"]
                    case "token_count":
                        stamp = record.get("timestamp") or ""
                        if (gauges := payload.get("rate_limits")) and stamp > ledger.limits_at:
                            ledger.limits, ledger.limits_at = gauges, stamp
                        usage = (payload.get("info") or {}).get("last_token_usage")
                        if not usage:
                            continue
                        tag = f"cx|{state['session']}|{stamp}|{usage.get('total_tokens')}"
                        if tag in ledger.calls:
                            continue
                        cached = usage.get("cached_input_tokens") or 0
                        ledger.calls[tag] = Call(
                            at=at, source="cx", session=state["session"],
                            project=state["project"], model=state["model"],
                            counts=[max((usage.get("input_tokens") or 0) - cached, 0),
                                    usage.get("cache_write_input_tokens") or 0, cached,
                                    usage.get("output_tokens") or 0],
                            thinking=usage.get("reasoning_output_tokens") or 0,
                            prompt=state["prompt"],
                        )


def transcripts(horizon: float) -> Iterator[tuple[Path, str]]:
    """Transcript files touched since the horizon, with their source tag."""
    trees = ((CLAUDE_ROOT, "cc", "*/*.jsonl"), (CODEX_ROOT, "cx", "*/*/*/rollout-*.jsonl"))
    for root, source, pattern in trees:
        for path in root.glob(pattern):
            try:
                if path.stat().st_mtime >= horizon:
                    yield path, source
            except OSError:
                continue


def new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete lines appended since the offset, and the offset to resume from."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:  # rewritten or rotated
        offset = 0
    if size == offset:
        return [], offset
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            blob = handle.read()
    except OSError:
        return [], offset
    end = blob.rfind(b"\n")
    if end < 0:  # a partial line; wait for the rest
        return [], offset
    return blob[:end].decode("utf-8", "replace").splitlines(), offset + end + 1


def records(path: Path) -> Iterator[dict]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return
    yield from decode_all(text.splitlines())


def decode_all(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            yield record


# ------------------------------------------------------------------ utilities


def usage_counts(usage: dict) -> list[int]:
    return [
        usage.get("input_tokens") or 0,
        usage.get("cache_creation_input_tokens") or 0,
        usage.get("cache_read_input_tokens") or 0,
        usage.get("output_tokens") or 0,
    ]


def weighted(counts: Iterable[int]) -> float:
    """Tokens in input-token equivalents, so cache reads stop dominating."""
    return sum(w * c for w, c in zip(WEIGHTS, counts, strict=True))


def sparkline(series: list[float], ceiling: float, floor: float = 0.0) -> str:
    """Render a series as block characters, scaled between floor and ceiling."""
    if ceiling <= floor:
        return "·" * len(series)
    span_ = ceiling - floor
    return "".join(
        SPARK[min(len(SPARK) - 1, max(0, int((v - floor) / span_ * len(SPARK))))] if v else "·"
        for v in series
    )


def resample(series: list[float], width: int) -> list[float]:
    """Squeeze a series to at most width points by averaging each bucket."""
    if width <= 0 or len(series) <= width:
        return series
    step = len(series) / width
    return [
        sum(series[int(i * step):max(int((i + 1) * step), int(i * step) + 1)])
        / max(int((i + 1) * step) - int(i * step), 1)
        for i in range(width)
    ]


def quantity(value: float) -> str:
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:.0f}"


def span(seconds: float) -> str:
    minutes = int(max(seconds, 0) // 60)
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{mins:02d}m"
    return f"{mins}m" if minutes else f"{int(seconds)}s"


def bar(percent: float, width: int = 8) -> str:
    filled = min(width, max(0, round(percent / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def tint(percent: float) -> str:
    return "red" if percent >= 85 else "yellow" if percent >= 60 else "green"


def clock(epoch: float | None) -> str:
    return f"{datetime.fromtimestamp(epoch, UTC).astimezone():%H:%M}" if epoch else "?"


def label(model: str) -> str:
    """Model name minus the vendor prefix and build date nobody reads."""
    for prefix in ("claude-", "gpt-"):
        model = model.removeprefix(prefix)
    return re.sub(r"-\d{8}$", "", model)


def trim(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def short(identifier: str) -> str:
    return identifier[:8]


def moment(text: str | None) -> float | None:
    """An ISO-8601 timestamp as unix seconds."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def fresh(path: Path, ttl: float = 86400) -> bool:
    try:
        return time.time() - path.stat().st_mtime < ttl
    except OSError:
        return False


if __name__ == "__main__":
    main()
