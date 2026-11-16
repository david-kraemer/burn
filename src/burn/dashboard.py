"""Render the live screen from a snapshot and view state; neither is mutated."""

from __future__ import annotations

import datetime

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from .analysis import call_dollars, calls_block, calls_lanes, spend_projected
from .format import (
    epoch_deadline,
    model_label,
    number_quantity,
    percent_tint,
    seconds_span,
    series_sparkline,
    text_trim,
)
from .keys import BINDINGS, HELP_TEXT
from .model import BLOCK, CLAUDE, CODEX, Quota, Row, Snapshot, Spend
from .state import FILTER, HELP, SORTS, TOOLS, View, table_viewport, view_cursor, view_rows
from .views import report_tools, report_turns
from .widgets import widget_meter

LANE_WIDTH = 48
METER_WIDTH = 24
ZOOM_LINES = 13

# Masthead, burn lanes, table, scroll indicator, and key bar. Count quota
# meters separately. The number of windows can change.
CHROME = 10

AGENT_STYLE = {CLAUDE: "cyan", CODEX: "magenta"}
FRESH = 20  # a session first seen this recently is highlighted
IDLE = 120  # a session silent this long is dimmed

# Placeholder for an unavailable value. Keep the width of the replaced value.
UNREAD = "      ·"


def snapshot_dashboard(snapshot: Snapshot, view: View, height: int = 24) -> Group:
    """Render meters, burn history, the session table, and key bar."""
    table = view_rows(view, snapshot)
    if view.mode == HELP:
        return Group(_masthead(snapshot, view, table), Text(""), _helpscreen(), _keybar(view))

    at = view_cursor(view, table)
    drawn = _meters(snapshot, view)
    capacity = max(3, height - CHROME - len(drawn) - (ZOOM_LINES if view.zoomed else 0))
    shown, offset = table_viewport(table, at, capacity)

    body: list[RenderableType] = [
        _masthead(snapshot, view, table),
        Text(""),
        Group(*drawn),
        Text(""),
        _burn_lanes(snapshot, view),
        Text(""),
        _sessions(shown, view, snapshot.at, at, offset),
        _scrollbar(len(table), len(shown), offset),
    ]
    if view.zoomed:
        body += [Text(""), _zoom(snapshot, view, table, at)]
    body += [Text(""), _keybar(view)]
    return Group(*body)


# -------------------------------------------------------------------- internals


def _masthead(snapshot: Snapshot, view: View, table: list[Row]) -> Text:
    """Render the one-line summary."""
    active = sum(1 for row in table if snapshot.at - row.last <= IDLE)
    line = Text.assemble(
        ("burn", "bold"),
        f"  {datetime.datetime.now(datetime.UTC).astimezone():%H:%M:%S}  ",
        ("up ", "dim"),
        seconds_span(snapshot.at - view.started),
        "   ",
        (f"{len(table)} sessions", "bold"),
        f", {active} active   ",
        ("window ", "dim"),
        seconds_span(view.seconds),
    )
    if view.source:
        line.append(f"   agent {view.source}", style="yellow")
    if view.needle:
        line.append(f"   /{view.needle}", style="yellow")
    if view.paused:
        line.append("   PAUSED", style="black on yellow")
    return line


def _meters(snapshot: Snapshot, view: View) -> list[Text]:
    """Render one bracket meter per quota window."""
    return [*_claude_meters(snapshot, view), *_codex_meters(snapshot)]


def _claude_meters(snapshot: Snapshot, view: View) -> list[Text]:
    """Render Claude quota windows and extra-usage credits."""
    reading = snapshot.reading
    if reading is None:
        return [_local_meter(snapshot, view)]
    drawn = [
        _quota_meter(window, snapshot.at, reading.pending) for window in reading.windows
    ]
    if reading.spend is not None:
        drawn.append(_credit_meter(reading.spend, reading.pending))
    return drawn or [_local_meter(snapshot, view)]


def _quota_meter(window: Quota, now: float, unread: bool = False) -> Text:
    """Render one Anthropic quota window.

    If a window is pending, keep its row and width. Show no value until the
    request completes. A reset on a later day carries its date.
    """
    if unread:
        return Text.assemble(
            (f"Claude {window.name:<5.5}", "cyan"),
            widget_meter(0.0, METER_WIDTH, "dim"),
            (UNREAD, "dim"),
            (f"  {window.scope}", "dim") if window.scope else "",
        )
    return Text.assemble(
        (f"Claude {window.name:<5.5}", "cyan"),
        widget_meter(window.used_percent / 100, METER_WIDTH, percent_tint(window.used_percent)),
        f" {window.used_percent:5.1f}%",
        (f"  {window.scope}", "dim") if window.scope else "",
        (f"  resets {epoch_deadline(window.resets_at, now)}", "dim"),
        ("  governing", "yellow") if window.active else "",
    )


def _credit_meter(credits: Spend, unread: bool = False) -> Text:
    """Render extra-usage credits used against their cap."""
    if unread:
        return Text.assemble(
            ("Claude cred ", "cyan"), widget_meter(0.0, METER_WIDTH, "dim"), (UNREAD, "dim")
        )
    used = credits.used_percent
    return Text.assemble(
        ("Claude cred ", "cyan"),
        widget_meter(used / 100, METER_WIDTH, percent_tint(used)),
        f" {used:5.1f}%",
        (f"  ${credits.used:,.2f} of ${credits.cap:,.2f}", "dim"),
    )


def _local_meter(snapshot: Snapshot, view: View) -> Text:
    """Render a five-hour estimate from local transcripts.

    This meter measures weighted tokens. It does not measure account quota. If
    ``--limit`` is not set, show elapsed time. Reported quota windows replace it.
    """
    made = [c for c in snapshot.calls if c.source == CLAUDE]
    start = calls_block(made, snapshot.at)
    if start is None:
        return Text.assemble(("Claude 5h ", "cyan"), ("  idle", "dim"))

    inside = [c for c in made if c.at >= start]
    spent = sum(c.weight for c in inside)
    money = sum(call_dollars(c) for c in inside)
    elapsed = snapshot.at - start
    left = max(BLOCK - elapsed, 0.0)
    if view.limit:
        fraction, style = spent / view.limit, "cyan"
        tail = f"{number_quantity(spent)}/{number_quantity(view.limit)}"
    else:
        # No allowance: show elapsed time. Dim the meter to mark it as an estimate.
        fraction, style = elapsed / BLOCK, "dim"
        tail = f"{number_quantity(spent)} spent, no --limit set"
    return Text.assemble(
        ("Claude 5h ", "cyan"),
        widget_meter(fraction, METER_WIDTH, style),
        f" {tail}",
        (f" →{number_quantity(spend_projected(spent, elapsed, left))}", "dim"),
        (f"  ${money:.2f}→${spend_projected(money, elapsed, left):.2f}" if money else ""),
        (f"  {seconds_span(left)} left", "dim"),
    )


def _codex_meters(snapshot: Snapshot) -> list[Text]:
    """Render Codex quota windows from shortest to longest."""
    if not snapshot.gauges:
        return [Text.assemble(("Codex     ", "magenta"), ("  no rate-limit event seen", "dim"))]
    return [
        Text.assemble(
            (f"Codex {seconds_span(gauge.window_minutes * 60):>4} ", "magenta"),
            widget_meter(gauge.used_percent / 100, METER_WIDTH, percent_tint(gauge.used_percent)),
            f" {gauge.used_percent:5.1f}%",
            (f"  resets {epoch_deadline(gauge.resets_at, snapshot.at)}", "dim"),
        )
        for gauge in snapshot.gauges
    ]


def _burn_lanes(snapshot: Snapshot, view: View) -> RenderableType:
    """Render weighted burn by agent and time bucket."""
    windowed = snapshot.since(view.seconds).from_agent(view.source)
    series = calls_lanes(windowed.calls, snapshot.at, view.seconds, LANE_WIDTH)
    step = view.seconds / LANE_WIDTH
    drawn = [
        Text.assemble(
            (f"burn {source} ", "dim"),
            (series_sparkline(values, max(values)), AGENT_STYLE.get(source, "")),
            (f"  peak {number_quantity(max(values))}/{seconds_span(step)}", "dim"),
        )
        for source, values in sorted(series.items())
        if any(values)
    ]
    return Group(*drawn) if drawn else Text("burn      (idle)", style="dim")


def _sessions(shown: list[Row], view: View, now: float, at: int, offset: int) -> Table:
    """Render the session table with selection and sort indicators."""
    table = Table(box=None, pad_edge=False, collapse_padding=True, header_style="bold")
    table.add_column(" ", width=1, no_wrap=True)
    for name, width in (("SRC", 3), ("SESSION", 13), ("PROJECT", 10), ("MODEL", 9)):
        table.add_column(_heading(name, view), width=width, no_wrap=True)
    for name, width in (("CTX", 6), ("HIT", 5), ("THINK", 6), ("WEIGHT", 7), ("RATE", 6), ("$", 5)):
        table.add_column(_heading(name, view), width=width, justify="right", no_wrap=True)

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
            text_trim(row.project, 10),
            text_trim(model_label(row.model), 9),
            number_quantity(row.ctx),
            Text(f"{row.cache:.0f}%", style=percent_tint(100 - row.cache)),
            number_quantity(row.think),
            Text(number_quantity(row.weight), style="bold"),
            Text(
                number_quantity(row.rate) if row.rate else "·",
                style="green" if row.rate else "dim",
            ),
            f"{row.cost:.2f}" if row.cost else "·",
            style=style,
        )
    if not shown:
        table.add_row(" ", *["·"] * 10)
    return table


def _scrollbar(total: int, drawn: int, offset: int) -> Text:
    """Report rows that are off-screen."""
    above, below = offset, total - offset - drawn
    parts = []
    if above:
        parts.append(f"▲ {above} above")
    if below:
        parts.append(f"▼ {below} below")
    if not parts:
        return Text("")
    return Text("  " + "   ".join(parts), style="dim")


def _zoom(snapshot: Snapshot, view: View, table: list[Row], at: int) -> RenderableType:
    """Render the detail pane for the selected session."""
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
    detail = report_turns(picked, 6) if view.panel != TOOLS else report_tools(picked, 6)
    # Keyed by session, not Row.key: attribution needs the whole call
    # sequence, spanning every project row the session tabulated into.
    projects = {c.project for c in picked.calls}
    if len(projects) > 1:
        note = Text(f"  spans projects: {', '.join(sorted(projects))}", style="yellow")
        return Group(note, Text(""), detail)
    return detail


def _helpscreen() -> RenderableType:
    """Render the key reference."""
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
            "  Columns: CTX is the current context size. HIT is the share of\n"
            "  input served from cache. THINK is reasoning tokens. WEIGHT is\n"
            "  tokens in input-token equivalents. RATE is the weighted burn\n"
            "  per minute, averaged over the last five minutes. ⑂ marks a\n"
            "  session that fanned out to subagents.",
            style="dim",
        ),
    )


def _keybar(view: View) -> Text:
    """Render the key bar."""
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


def _heading(name: str, view: View) -> Text:
    """Mark the active sort column."""
    for field, column in SORTS:
        if column == name and view.sort == field:
            return Text(name + ("▼" if view.reverse else "▲"), style="black on cyan")
    return Text(name)
