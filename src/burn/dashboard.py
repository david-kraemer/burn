"""The live screen, composed from a snapshot and a view.

Nothing here mutates either argument. The same pair always renders the same
frame, so a regression is a string comparison.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from burn import views
from burn.analysis import current_block, dollars, lanes, projected
from burn.format import clock, label, quantity, span, sparkline, tint, trim
from burn.keys import BINDINGS, HELP_TEXT
from burn.model import BLOCK, CLAUDE, CODEX, Row, Snapshot
from burn.state import FILTER, HELP, SORTS, TOOLS, View, cursor, rows, viewport
from burn.widgets import meter

LANE_WIDTH = 48
METER_WIDTH = 24
ZOOM_LINES = 13

# Masthead, meters, burn lanes, the table header, the scroll indicator, the key
# bar, and the blank lines between them. Everything else is table.
CHROME = 13

AGENT_STYLE = {CLAUDE: "cyan", CODEX: "magenta"}
FRESH = 20  # a session first seen this recently is highlighted
IDLE = 120  # a session silent this long is dimmed


def dashboard(snapshot: Snapshot, view: View, height: int = 24) -> Group:
    """Meters, burn history, the session table, and a key bar."""
    table = rows(view, snapshot)
    if view.mode == HELP:
        return Group(masthead(snapshot, view, table), Text(""), helpscreen(), keybar(view))

    at = cursor(view, table)
    capacity = max(3, height - CHROME - (ZOOM_LINES if view.zoomed else 0))
    shown, offset = viewport(table, at, capacity)

    body: list[RenderableType] = [
        masthead(snapshot, view, table),
        Text(""),
        meters(snapshot, view),
        Text(""),
        burn_lanes(snapshot, view),
        Text(""),
        sessions(shown, view, snapshot.at, at, offset),
        scrollbar(len(table), len(shown), offset),
    ]
    if view.zoomed:
        body += [Text(""), zoom(snapshot, view, table, at)]
    body += [Text(""), keybar(view)]
    return Group(*body)


def masthead(snapshot: Snapshot, view: View, table: list[Row]) -> Text:
    """The one-line summary: uptime, how many sessions, and what is filtered."""
    active = sum(1 for row in table if snapshot.at - row.last <= IDLE)
    line = Text.assemble(
        ("burn", "bold"),
        f"  {datetime.now(UTC).astimezone():%H:%M:%S}  ",
        ("up ", "dim"),
        span(snapshot.at - view.started),
        "   ",
        (f"{len(table)} sessions", "bold"),
        f", {active} active   ",
        ("window ", "dim"),
        span(view.seconds),
    )
    if view.source:
        line.append(f"   agent {view.source}", style="yellow")
    if view.needle:
        line.append(f"   /{view.needle}", style="yellow")
    if view.paused:
        line.append("   PAUSED", style="black on yellow")
    return line


def meters(snapshot: Snapshot, view: View) -> Group:
    """htop-style bracket meters, one per quota window."""
    return Group(claude_meter(snapshot, view), *codex_meters(snapshot))


def claude_meter(snapshot: Snapshot, view: View) -> Text:
    """The five-hour block: what it has cost, and where it is heading."""
    made = [c for c in snapshot.calls if c.source == CLAUDE]
    start = current_block(made, snapshot.at)
    if start is None:
        return Text.assemble(("Claude 5h ", "cyan"), ("  idle", "dim"))

    inside = [c for c in made if c.at >= start]
    spent = sum(c.weight for c in inside)
    money = sum(dollars(c) for c in inside)
    elapsed = snapshot.at - start
    left = max(BLOCK - elapsed, 0.0)
    # With no quota recorded on disk there is no denominator, so the meter
    # tracks the block's clock unless a limit was supplied.
    if view.limit:
        fraction, tail = spent / view.limit, f"{quantity(spent)}/{quantity(view.limit)}"
    else:
        fraction, tail = elapsed / BLOCK, quantity(spent)
    return Text.assemble(
        ("Claude 5h ", "cyan"),
        meter(fraction, METER_WIDTH),
        f" {tail}",
        (f" →{quantity(projected(spent, elapsed, left))}", "dim"),
        (f"  ${money:.2f}→${projected(money, elapsed, left):.2f}" if money else ""),
        (f"  {span(left)} left", "dim"),
    )


def codex_meters(snapshot: Snapshot) -> list[Text]:
    """One line per quota window Codex has reported, shortest first."""
    if not snapshot.gauges:
        return [Text.assemble(("Codex     ", "magenta"), ("  no rate-limit event seen", "dim"))]
    return [
        Text.assemble(
            (f"Codex {span(gauge.window_minutes * 60):>4} ", "magenta"),
            meter(gauge.used_percent / 100, METER_WIDTH, tint(gauge.used_percent)),
            f" {gauge.used_percent:5.1f}%",
            (f"  resets {clock(gauge.resets_at)}", "dim"),
        )
        for gauge in snapshot.gauges
    ]


def burn_lanes(snapshot: Snapshot, view: View) -> RenderableType:
    """Weighted burn per bucket across the window, one lane per agent."""
    windowed = snapshot.since(view.seconds).from_agent(view.source)
    series = lanes(windowed.calls, snapshot.at, view.seconds, LANE_WIDTH)
    step = view.seconds / LANE_WIDTH
    drawn = [
        Text.assemble(
            (f"burn {source} ", "dim"),
            (sparkline(values, max(values)), AGENT_STYLE.get(source, "")),
            (f"  peak {quantity(max(values))}/{span(step)}", "dim"),
        )
        for source, values in sorted(series.items())
        if any(values)
    ]
    return Group(*drawn) if drawn else Text("burn      (idle)", style="dim")


def sessions(shown: list[Row], view: View, now: float, at: int, offset: int) -> Table:
    """The process table, with a cursor and a marked sort column."""
    table = Table(box=None, pad_edge=False, collapse_padding=True, header_style="bold")
    table.add_column(" ", width=1, no_wrap=True)
    for name, width in (("SRC", 3), ("SESSION", 9), ("PROJECT", 10), ("MODEL", 9)):
        table.add_column(heading(name, view), width=width, no_wrap=True)
    for name, width in (("CTX", 6), ("HIT", 5), ("THINK", 6), ("WEIGHT", 7), ("RATE", 6), ("$", 5)):
        table.add_column(heading(name, view), width=width, justify="right", no_wrap=True)

    for index, row in enumerate(shown, start=offset):
        here = index == at
        style = "on grey23" if here else ("dim" if now - row.last > IDLE else "")
        table.add_row(
            "▸" if here else " ",
            Text(row.source, style=AGENT_STYLE.get(row.source, "")),
            Text(
                row.session + ("⑂" if row.fanout else ""),
                style="green" if now - row.first < FRESH else "",
            ),
            trim(row.project, 10),
            trim(label(row.model), 9),
            quantity(row.ctx),
            Text(f"{row.cache:.0f}%", style=tint(100 - row.cache)),
            quantity(row.think),
            Text(quantity(row.weight), style="bold"),
            Text(quantity(row.rate) if row.rate else "·", style="green" if row.rate else "dim"),
            f"{row.cost:.2f}" if row.cost else "·",
            style=style,
        )
    if not shown:
        table.add_row(" ", *["·"] * 10)
    return table


def scrollbar(total: int, drawn: int, offset: int) -> Text:
    """Say what is off-screen, the way a pager does."""
    above, below = offset, total - offset - drawn
    parts = [f"▲ {above} above"] * bool(above) + [f"▼ {below} below"] * bool(below)
    return Text("  " + "   ".join(parts), style="dim") if parts else Text("")


def zoom(snapshot: Snapshot, view: View, table: list[Row], at: int) -> RenderableType:
    """The detail pane under the table, for whichever session is selected."""
    if not table:
        return Text("  no session selected", style="dim")
    chosen = table[at].session
    narrowed = snapshot.since(view.seconds)
    picked = Snapshot(
        at=narrowed.at,
        calls=tuple(c for c in narrowed.calls if c.session == chosen),
        tools=tuple(t for t in narrowed.tools if t.session == chosen),
    )
    if not picked.calls:
        return Text(f"  {chosen}: no calls in this window", style="dim")
    return views.turns(picked, 6) if view.panel != TOOLS else views.tools(picked, 6)


def helpscreen() -> RenderableType:
    """Every binding, since a top-alike is worthless if you must read the source."""
    layout = Table.grid(padding=(0, 3))
    layout.add_column(style="bold cyan", justify="right")
    layout.add_column()
    for key, what in HELP_TEXT:
        layout.add_row(key, what)
    return Group(
        Text("  Keys", style="bold"),
        Text(""),
        layout,
        Text(""),
        Text(
            "  Columns: CTX is the current context size, HIT the share of input\n"
            "  served from cache, THINK reasoning tokens, WEIGHT tokens in\n"
            "  input-token equivalents, RATE the weighted burn over five minutes.\n"
            "  ⑂ marks a session that fanned out to subagents.",
            style="dim",
        ),
    )


def keybar(view: View) -> Text:
    """htop's function-key strip, in the letters this program actually reads."""
    if view.mode == FILTER:
        return Text.assemble(
            ("filter", "black on yellow"),
            " ",
            view.draft,
            ("▏", "bold"),
            ("   ↵ accept   esc cancel", "dim"),
        )
    bar = Text()
    for key, name in BINDINGS:
        bar.append(key, style="dim")
        bar.append(name, style="black on cyan")
        bar.append(" ")
    return bar


def heading(name: str, view: View) -> Text:
    """Mark the column the table is sorted on, the way top does."""
    for field, column in SORTS:
        if column == name and view.sort == field:
            return Text(name + ("▼" if view.reverse else "▲"), style="black on cyan")
    return Text(name)
