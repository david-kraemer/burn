"""Derive attribution, quota, rate, and table data from a snapshot."""

from __future__ import annotations

import json
import os
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path

from .format import short
from .ingest import CLAUDE_ROOT, claude_usage, records
from .model import (
    BLOCK,
    CACHE_READ_WEIGHT,
    CACHE_WRITE_WEIGHT,
    FANOUT_TOOLS,
    Call,
    Row,
    Snapshot,
    Tooling,
    Usage,
    threads,
)

RATES_CACHE = Path.home() / ".cache" / "burn" / "rates.json"
RATES_TTL = 86400

# A recreated cache token costs 1.25 units. A cache read costs 0.1.
IDLE_PREMIUM = CACHE_WRITE_WEIGHT - CACHE_READ_WEIGHT


# ------------------------------------------------------------------- grouping


def sessions(calls: Iterable[Call]) -> dict[str, list[Call]]:
    """Calls grouped by session, each group in time order."""
    grouped: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        grouped[call.session].append(call)
    for group in grouped.values():
        group.sort(key=lambda c: c.at)
    return dict(grouped)


def tabulate(snapshot: Snapshot, needle: str = "") -> list[Row]:
    """Return one table row per session."""
    groups: dict[tuple[str, str, str], list[Call]] = defaultdict(list)
    for call in snapshot.calls:
        groups[(call.source, call.session, call.project)].append(call)

    rows = []
    for (source, session, project), items in groups.items():
        total = sum((c.usage for c in items), Usage())
        weight = sum(c.weight for c in items)
        rows.append(
            Row(
                source=source,
                session=session,
                project=project,
                model=items[-1].model,
                ctx=items[-1].prefix,
                cache=100 * total.cache_read / max(total.cache_read + total.cache_write, 1),
                think=sum(c.thinking for c in items),
                weight=weight,
                rate=sum(c.weight for c in items if c.at >= snapshot.at - 300) / 5,
                cost=sum(dollars(c) for c in items),
                calls=len(items),
                first=items[0].at,
                last=items[-1].at,
                fanout=any(c.fanout for c in items),
            )
        )
    if needle:
        lowered = needle.lower()
        rows = [
            row
            for row in rows
            if lowered in f"{row.session}{row.project}{row.model}{row.source}".lower()
        ]
    return rows


def lanes(calls: Iterable[Call], now: float, seconds: float, width: int) -> dict[str, list[float]]:
    """Return weighted burn by agent and time bucket."""
    step = seconds / width
    series: dict[str, list[float]] = {}
    for call in calls:
        slot = int((call.at - (now - seconds)) / step)
        if 0 <= slot < width:
            series.setdefault(call.source, [0.0] * width)[slot] += call.weight
    return series


# ---------------------------------------------------------------- attribution


@dataclass(frozen=True, slots=True)
class Blame:
    """Context added by one tool and its later read cost."""

    name: str
    added: float
    calls: int
    carried: float
    cost: float


PROMPT = "(prompt / system)"


def attribution(calls: Iterable[Call], tooling: Iterable[Tooling]) -> list[Blame]:
    """Assign context growth and later read cost to its tools.

    Growth is the increase between calls, less the assistant's output. Split
    it among intervening tools by result size. Charge each tool for later
    reads of its cached result.
    """
    events = sessions_of(tooling)
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    for session, items in sessions(calls).items():
        series = events.get(session, [])
        stamps = [event.at for event in series]
        for strand in threads(items):
            for index, (previous, call) in enumerate(pairwise(strand)):
                # The result is cached once and read by later calls in this
                # thread. A compaction starts a new thread.
                carry = CACHE_WRITE_WEIGHT + CACHE_READ_WEIGHT * max(len(strand) - index - 2, 0)
                # Growth is new context, less the model's output.
                growth = call.prefix - previous.prefix - previous.usage.output
                if growth <= 0:
                    continue
                lo = bisect_right(stamps, previous.at)
                hi = bisect_right(stamps, call.at)
                between = series[lo:hi]
                rate = per_token_cost(call.model)
                for name, portion in shares(between, growth):
                    row = totals[name]
                    row[0] += portion
                    row[1] += 1
                    row[2] += portion * carry
                    row[3] += portion * carry * rate

    blamed = [Blame(name, a, int(n), c, d) for name, (a, n, c, d) in totals.items()]
    return sorted(blamed, key=lambda b: -b.carried)


def shares(between: Sequence[Tooling], growth: float) -> Iterator[tuple[str, float]]:
    """Split one growth step among intervening tools by result size."""
    if not between:
        yield PROMPT, growth
        return
    total = sum(max(event.size, 1) for event in between)
    for event in between:
        yield event.name, growth * max(event.size, 1) / total


def sessions_of(tooling: Iterable[Tooling]) -> dict[str, list[Tooling]]:
    """Group tool results by session and sort them by time."""
    grouped: dict[str, list[Tooling]] = defaultdict(list)
    for event in tooling:
        grouped[event.session].append(event)
    for group in grouped.values():
        group.sort(key=lambda e: e.at)
    return dict(grouped)


# ----------------------------------------------------------------- cache waste


@dataclass(frozen=True, slots=True)
class Waste:
    """Cache recreation split into idle and normal churn."""

    idle_tokens: int
    idle_events: int
    churn_tokens: int

    @property
    def total(self) -> int:
        return self.idle_tokens + self.churn_tokens

    @property
    def premium(self) -> float:
        """Weighted tokens caused by an expired cache."""
        return self.idle_tokens * IDLE_PREMIUM


def waste(calls: Iterable[Call], ttl: float) -> Waste:
    """Measure cache creation after gaps longer than the cache lifetime."""
    idle_tokens = idle_events = churn = 0
    for items in sessions(calls).values():
        for previous, call in pairwise(items):
            if not call.usage.cache_write:
                continue
            if call.at - previous.at > ttl:
                idle_tokens += call.usage.cache_write
                idle_events += 1
            else:
                churn += call.usage.cache_write
    return Waste(idle_tokens, idle_events, churn)


def resumes(calls: Sequence[Call], ttl: float) -> int:
    """Count cache creations after idle gaps in one session."""
    return sum(1 for a, b in pairwise(calls) if b.usage.cache_write and b.at - a.at > ttl)


# ------------------------------------------------------------------ the block


def current_block(calls: Iterable[Call], now: float) -> float | None:
    """Return the open five-hour quota block, if one exists."""
    start = last = None
    for stamp in sorted(c.at for c in calls):
        if start is None or stamp - start >= BLOCK or stamp - last >= BLOCK:
            start = stamp - stamp % 3600
        last = stamp
    if start is None or now - start >= BLOCK:
        return None
    return start


def projected(spent: float, elapsed: float, remaining: float) -> float:
    """Project spend at the end of the quota block."""
    return spent + spent / max(elapsed, 60) * remaining


# ------------------------------------------------------------------- billing


@cache
def rates() -> Mapping[str, tuple[float, int]]:
    """Calculate dollars per weighted megatoken for each model.

    Use closed Claude sessions whose observed and billed totals match.
    Exclude sessions with unrecorded subagent traffic.
    """
    if (cached := cached_rates()) is not None:
        return cached
    pools: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for path in CLAUDE_ROOT.glob("*/*.jsonl"):
        observed, billed, fanout = reconcile(path)
        if fanout or billed <= 0 or observed <= 0 or abs(billed - observed) / billed > 0.10:
            continue
        for model, weight, cost in billed_models(path):
            pool = pools[model]
            pool[0] += cost
            pool[1] += weight
            pool[2] += 1
    solved = {m: (1e6 * p[0] / p[1], int(p[2])) for m, p in pools.items() if p[1] > 0}
    save_rates(solved)
    return solved


def per_token_cost(model: str) -> float:
    rate, _ = rates().get(model, (0.0, 0))
    return rate / 1e6


def dollars(call: Call) -> float:
    return call.weight * per_token_cost(call.model)


def blended_rate() -> float:
    """Return one average rate for totals spanning models."""
    solved = [rate for rate, _ in rates().values()]
    return sum(solved) / len(solved) / 1e6 if solved else 0.0


def cached_rates() -> Mapping[str, tuple[float, int]] | None:
    """Load a fresh, valid cached rate calculation."""
    try:
        if time.time() - RATES_CACHE.stat().st_mtime >= RATES_TTL:
            return None
        stored = json.loads(RATES_CACHE.read_text())
        return {k: (float(v[0]), int(v[1])) for k, v in stored.items()}
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return None  # Recalculate missing or invalid data.


def save_rates(solved: Mapping[str, tuple[float, int]]) -> None:
    """Save rates through a temporary file."""
    try:
        RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        scratch = RATES_CACHE.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps(solved))
        scratch.replace(RATES_CACHE)
    except OSError:
        pass  # The cache is optional.


# ---------------------------------------------------------------- audit


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """One closed session compared with Claude's billed total."""

    session: str
    observed: float
    billed: float
    fanout: bool

    @property
    def unseen(self) -> float:
        """Percent of billed tokens absent from the transcript."""
        return 100 * (self.billed - self.observed) / self.billed


def audit() -> list[Reconciliation]:
    """Return closed Claude sessions with billed totals."""
    found = []
    for path in sorted(CLAUDE_ROOT.glob("*/*.jsonl")):
        observed, billed, fanout = reconcile(path)
        if billed > 0:
            found.append(Reconciliation(short(path.stem), observed, billed, fanout))
    return found


def reconcile(path: Path) -> tuple[float, float, bool]:
    """Return observed tokens, billed tokens, and fan-out status."""
    observed = billed = 0.0
    fanout = False
    seen: set[str] = set()
    for record in records(path):
        if record.get("type") == "cost-state":
            billed = sum(w for _, w, _ in cost_state(record))
            continue
        if record.get("type") != "assistant":
            continue
        message = record.get("message") or {}
        usage = message.get("usage")
        if not usage or message.get("model") in (None, "<synthetic>"):
            continue
        fanout = fanout or any(
            isinstance(b, dict) and b.get("name") in FANOUT_TOOLS
            for b in message.get("content") or []
        )
        tag = f"{message.get('id')}|{record.get('requestId')}"
        if tag in seen:
            continue
        seen.add(tag)
        observed += claude_usage(usage).weight
    return observed, billed, fanout


def billed_models(path: Path) -> Iterator[tuple[str, float, float]]:
    """Return per-model tokens and dollars from a cost-state record."""
    for record in records(path):
        if record.get("type") == "cost-state":
            for model, weight, cost in cost_state(record):
                if weight > 0 and cost > 0:
                    yield model, weight, cost


def cost_state(record: dict) -> Iterator[tuple[str, float, float]]:
    for model, usage in (record.get("modelUsage") or {}).items():
        billed = Usage(
            input=usage.get("inputTokens") or 0,
            cache_write=usage.get("cacheCreationInputTokens") or 0,
            cache_read=usage.get("cacheReadInputTokens") or 0,
            output=usage.get("outputTokens") or 0,
        )
        yield model, billed.weight, usage.get("costUSD") or 0.0
