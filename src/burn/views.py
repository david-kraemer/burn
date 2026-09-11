"""Render the drill-down views: each is a pure function of a snapshot."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from rich.console import Group, RenderableType
from rich.text import Text

from . import analysis
from .format import label, quantity, resample, span, sparkline, tint, trim
from .model import CACHE_TTL, Call, Snapshot, Usage, compactions
from .widgets import bar, columns, grid


def tools(snapshot: Snapshot, top: int = 15) -> RenderableType:
    """Show context growth by tool."""
    blamed = analysis.attribution(snapshot.calls, snapshot.tools)
    if not blamed:
        return Text("No attributable context growth in this window.", style="dim")

    table = columns(
        [("TOOL", 20)],
        [("CALLS", 5), ("ADDED", 7), ("PER CALL", 8), ("CARRIED", 8), ("COST", 7)],
    )
    table.add_column("", width=10, no_wrap=True)

    grand = sum(b.carried for b in blamed)
    for blame in blamed[:top]:
        share = blame.carried / max(grand, 1)
        table.add_row(
            trim(blame.name, 20),
            f"{blame.calls:,}",
            quantity(blame.added),
            quantity(blame.added / max(blame.calls, 1)),
            Text(quantity(blame.carried), style="bold"),
            f"${blame.cost:.2f}" if blame.cost else "·",
            Text(bar(100 * share, 10), style=tint(100 * share * 2)),
        )
    added = sum(b.added for b in blamed)
    return Group(
        Text.assemble(
            ("Context burn by tool", "bold"),
            f"   {quantity(added)} added → {quantity(grand)} weighted with re-reads",
        ),
        Text(""),
        table,
        Text(""),
        Text(
            "ADDED is context added by the tool. CARRIED is the weighted cost of "
            "caching it once plus every later re-read.",
            style="dim",
        ),
    )


def turns(snapshot: Snapshot, top: int = 15) -> RenderableType:
    """Show the most expensive prompts."""
    grouped: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for call in snapshot.calls:
        if call.prompt:
            grouped[(call.session, call.prompt)].append(call)
    if not grouped:
        return Text("No prompts captured in this window.", style="dim")

    table = columns(
        [("WHEN", 5), ("SESSION", 12)], [("CALLS", 5), ("WEIGHT", 7), ("COST", 6)]
    )
    table.add_column("PROMPT", width=38, no_wrap=True)

    # "Most expensive" is dollars, not tokens: a heavily-used cheap model
    # can outweigh a pricier one in tokens while costing less.
    ranked = sorted(grouped.items(), key=lambda kv: -sum(analysis.dollars(c) for c in kv[1]))
    for (session, prompt), items in ranked[:top]:
        cost = sum(analysis.dollars(c) for c in items)
        table.add_row(
            f"{datetime.fromtimestamp(items[0].at, UTC).astimezone():%H:%M}",
            session,
            str(len(items)),
            Text(quantity(sum(c.weight for c in items)), style="bold"),
            f"${cost:.2f}" if cost else "·",
            trim(" ".join(prompt.split()), 38),
        )
    return Group(Text("Most expensive turns", style="bold"), Text(""), table)


def session(snapshot: Snapshot, target: str | None) -> RenderableType:
    """Show context, cache use, and tools for one session."""
    if not target:
        return Text("Usage: burn session <session-id-prefix>", style="dim")
    items = [c for c in snapshot.calls if c.session.startswith(target)]
    if not items:
        return Text(f"No calls for session {target!r} in this window.", style="dim")

    total = sum((c.usage for c in items), Usage())
    hit = 100 * total.cache_read / max(total.cache_read + total.cache_write, 1)
    money = sum(analysis.dollars(c) for c in items)
    peak = max(c.prefix for c in items)
    models: dict[str, float] = defaultdict(float)
    for call in items:
        models[label(call.model)] += call.weight

    facts = [
        ("session", f"{items[0].session}  ·  {items[0].project}  ·  {items[0].source}"),
        ("span", f"{span(items[-1].at - items[0].at)} over {len(items)} calls"),
        ("weighted", f"{quantity(total.weight)}  (~${money:.2f})"),
        ("context", f"{quantity(items[-1].prefix)} now, peak {quantity(peak)}"),
        ("cache", f"{hit:.0f}% hits, {quantity(total.cache_write)} re-created"),
        ("thinking", quantity(sum(c.thinking for c in items))),
        ("models", ", ".join(f"{m} {quantity(w)}" for m, w in
                             sorted(models.items(), key=lambda kv: -kv[1]))),
    ]
    if collapses := compactions(items):
        facts.append(("compactions", f"{collapses} (context discarded and rebuilt)"))
    if any(c.fanout for c in items):
        facts.append(
            ("fan-out", Text("yes — subagent usage is billed but unrecorded", "yellow"))
        )

    curve = resample([float(c.prefix) for c in items], 56)
    low, high = min(curve), max(curve)
    return Group(
        Text("Session detail", style="bold"),
        Text(""),
        grid(facts),
        Text(""),
        Text.assemble(
            "context  ",
            (sparkline(curve, high, low * 0.98), "cyan"),
            f"  {quantity(low)} → {quantity(high)}",
        ),
        Text(""),
        tools(Snapshot(snapshot.at, tuple(items), snapshot.tools), 10),
    )


def cost(snapshot: Snapshot, window: int) -> RenderableType:
    """Show estimated spend by model and project."""
    table = columns([("MODEL", 26)], [("$/WEIGHTED Mtok", 16), ("SAMPLES", 8)])
    for model, (rate, samples) in sorted(analysis.rates().items()):
        table.add_row(label(model), f"{rate:.2f}", str(samples))

    spent: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for call in snapshot.calls:
        entry = spent[(call.source, call.project)]
        entry[0] += call.weight
        entry[1] += analysis.dollars(call)

    spend = columns([("PROJECT", 24)], [("WEIGHT", 8), ("COST", 8)])
    for (source, project), (weight, money) in sorted(spent.items(), key=lambda kv: -kv[1][1])[:15]:
        spend.add_row(
            f"{source}  {trim(project, 20)}", quantity(weight), f"${money:.2f}" if money else "·"
        )

    return Group(
        Text("Effective rates", style="bold"),
        Text(
            "Rates use Claude cost-state records and observed weighted tokens.\n"
            "Only sessions with matching totals are used. Codex records no cost;\n"
            "its rows show tokens only.",
            style="dim",
        ),
        Text(""),
        table,
        Text(""),
        Text(f"Spend over the last {span(window * 60)}", style="bold"),
        Text(""),
        spend,
    )


def waste(snapshot: Snapshot) -> RenderableType:
    """Show cache tokens recreated after idle periods."""
    summary = analysis.waste(snapshot.calls, CACHE_TTL)
    if not summary.total:
        return Text("No cache re-creation in this window.", style="dim")

    facts = grid([
        ("re-created after idle",
         f"{quantity(summary.idle_tokens)} over {summary.idle_events} resumes"),
        ("re-created in flow", f"{quantity(summary.churn_tokens)} (new turns, unavoidable)"),
        ("idle premium", f"{quantity(summary.premium)} weighted tokens, "
                         f"~${summary.premium * analysis.blended_rate():.2f}"),
        ("", Text(f"{100 * summary.idle_tokens / summary.total:.0f}% of all cache creation "
                  f"followed a gap longer than {span(CACHE_TTL)}", style="dim")),
    ])

    tiers = columns([("SESSION", 12)], [("5m TIER", 9), ("1h TIER", 9), ("IDLE RESUMES", 13)])
    grouped = analysis.sessions(snapshot.calls)
    ranked = sorted(
        ((name, items) for name, items in grouped.items()
         if any(c.usage.cache_write for c in items)),
        key=lambda kv: -sum(c.usage.cache_write for c in kv[1]),
    )
    for name, items in ranked[:10]:
        tiers.add_row(
            name,
            quantity(sum(c.cache_5m for c in items)),
            quantity(sum(c.cache_1h for c in items)),
            str(analysis.resumes(items, CACHE_TTL)),
        )

    return Group(
        Text("Cache re-creation", style="bold"),
        Text(""),
        facts,
        Text(""),
        tiers,
        Text(""),
        Text(
            "The 5m tier is recreated after each long idle period. The 1h tier "
            "costs more to write but lasts longer.",
            style="dim",
        ),
    )


def audit(threshold: float = 15.0) -> RenderableType:
    """Compare observed tokens with Claude's billed totals."""
    found = analysis.audit()
    if not found:
        return Text("No closed sessions with cost-state records yet.", style="dim")

    table = columns([("SESSION", 12)], [("OBSERVED", 9), ("BILLED", 9), ("UNSEEN", 7)])
    table.add_column("LIKELY CAUSE", width=24, no_wrap=True)
    # unseen > 0: burn saw fewer tokens than billed. < 0: burn overcounts.
    for entry in sorted(found, key=lambda r: -abs(r.unseen)):
        if abs(entry.unseen) < threshold:
            continue
        cause = (
            ("fan-out (Task/Workflow)" if entry.fanout else "background calls")
            if entry.unseen > 0
            else "burn overcounts vs. billed"
        )
        table.add_row(
            entry.session,
            quantity(entry.observed),
            quantity(entry.billed),
            Text(f"{entry.unseen:.0f}%", style=tint(abs(entry.unseen))),
            cause,
        )

    gaps = sorted(entry.unseen for entry in found)
    worst = max(gaps, key=abs)
    return Group(
        table,
        Text(""),
        Text(
            f"{len(gaps)} closed sessions · median unseen {gaps[len(gaps) // 2]:.1f}% "
            f"· worst {worst:.0f}%",
            style="bold",
        ),
    )
