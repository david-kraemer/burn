"""Derive attribution, quota, rate, and table data from a snapshot."""

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import json
import os
import pathlib
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence

from .format import id_short
from .ingest import CLAUDE_ROOT, claude_usage, transcript_records
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
    calls_threads,
)

RATES_CACHE = pathlib.Path.home() / ".cache" / "burn" / "rates.json"
RATES_TTL = 86400

# A recreated cache token costs 1.25 units. A cache read costs 0.1.
IDLE_PREMIUM = CACHE_WRITE_WEIGHT - CACHE_READ_WEIGHT


# ------------------------------------------------------------------- grouping


def calls_sessions(calls: Iterable[Call]) -> dict[str, list[Call]]:
    """Calls grouped by session, each group in time order."""
    grouped: dict[str, list[Call]] = collections.defaultdict(list)
    for call in calls:
        grouped[call.session].append(call)
    for group in grouped.values():
        group.sort(key=lambda c: c.at)
    return dict(grouped)


def snapshot_table(snapshot: Snapshot, needle: str = "") -> list[Row]:
    """Return one table row per session."""
    groups: dict[tuple[str, str, str], list[Call]] = collections.defaultdict(list)
    for call in snapshot.calls:
        groups[(call.source, call.session, call.project)].append(call)

    rows = []
    for (source, session, project), items in groups.items():
        total = sum((c.usage for c in items), Usage())
        weight = sum(c.weight for c in items)
        # A subagent's last request says nothing about the session's own context.
        main = [c for c in items if not c.sidechain] or items
        rows.append(
            Row(
                source=source,
                session=session,
                project=project,
                model=main[-1].model,
                ctx=main[-1].prefix,
                cache=100 * total.cache_read / max(total.cache_read + total.cache_write, 1),
                think=sum(c.thinking for c in items),
                weight=weight,
                rate=sum(c.weight for c in items if c.at >= snapshot.at - 300) / 5,
                cost=sum(call_dollars(c) for c in items),
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


def calls_lanes(
    calls: Iterable[Call], now: float, seconds: float, width: int
) -> dict[str, list[float]]:
    """Return weighted burn by agent and time bucket."""
    step = seconds / width
    series: dict[str, list[float]] = {}
    for call in calls:
        slot = int((call.at - (now - seconds)) / step)
        if 0 <= slot < width:
            series.setdefault(call.source, [0.0] * width)[slot] += call.weight
    return series


# ---------------------------------------------------------------- attribution


@dataclasses.dataclass(frozen=True, slots=True)
class Blame:
    """Context added by one tool and its later read cost."""

    name: str
    added: float
    calls: int
    carried: float
    cost: float


PROMPT = "(prompt / system)"


@dataclasses.dataclass(frozen=True, slots=True)
class Growth:
    """One thread's growth step, pending its tool assignment."""

    start: float
    end: float
    growth: float
    carry: float
    model: str


def calls_attribution(calls: Iterable[Call], tooling: Iterable[Tooling]) -> list[Blame]:
    """Assign context growth (calls minus output) and its later read cost to intervening tools."""
    events = _sessions_of(tooling)
    totals: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    for session, items in calls_sessions(calls).items():
        steps = list(_growths(items))
        for step, between in zip(steps, _assign(steps, events.get(session, [])), strict=True):
            rate = _per_token_cost(step.model)
            for name, portion in _shares(between, step.growth):
                row = totals[name]
                row[0] += portion
                row[1] += 1
                row[2] += portion * step.carry
                row[3] += portion * step.carry * rate

    blamed = [Blame(name, a, int(n), c, d) for name, (a, n, c, d) in totals.items()]
    return sorted(blamed, key=lambda b: -b.carried)


def _growths(items: Sequence[Call]) -> Iterator[Growth]:
    """Every positive growth step across a session's threads.

    Threads can overlap in wall time (a compaction, or Codex's side threads
    under one session id), so each step is its own interval: a tool result
    is matched to the thread it actually ran in, not every overlapping one.
    """
    for strand in calls_threads(items):
        for index, (previous, call) in enumerate(itertools.pairwise(strand)):
            # Cached once, read by each later call in this thread.
            carry = CACHE_WRITE_WEIGHT + CACHE_READ_WEIGHT * max(len(strand) - index - 2, 0)
            growth = call.prefix - previous.prefix - previous.usage.output
            if growth > 0:
                yield Growth(previous.at, call.at, growth, carry, call.model)


def _assign(steps: Sequence[Growth], series: Sequence[Tooling]) -> list[list[Tooling]]:
    """Credit each tool result to the narrowest containing step, not every overlapping one."""
    between: list[list[Tooling]] = [[] for _ in steps]
    for event in series:
        best = None
        for index, step in enumerate(steps):
            if step.start < event.at <= step.end and (
                best is None or step.end - step.start < steps[best].end - steps[best].start
            ):
                best = index
        if best is not None:
            between[best].append(event)
    return between


def _shares(between: Sequence[Tooling], growth: float) -> Iterator[tuple[str, float]]:
    """Split one growth step among intervening tools by result size."""
    if not between:
        yield PROMPT, growth
        return
    total = sum(max(event.size, 1) for event in between)
    for event in between:
        yield event.name, growth * max(event.size, 1) / total


def _sessions_of(tooling: Iterable[Tooling]) -> dict[str, list[Tooling]]:
    """Group tool results by session and sort them by time."""
    grouped: dict[str, list[Tooling]] = collections.defaultdict(list)
    for event in tooling:
        grouped[event.session].append(event)
    for group in grouped.values():
        group.sort(key=lambda e: e.at)
    return dict(grouped)


# ----------------------------------------------------------------- cache waste


@dataclasses.dataclass(frozen=True, slots=True)
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
        return self.idle_tokens * IDLE_PREMIUM


def calls_waste(calls: Iterable[Call], ttl: float) -> Waste:
    """Measure cache creation after gaps longer than the cache lifetime."""
    idle_tokens = idle_events = churn = 0
    for items in calls_sessions(calls).values():
        for previous, call in itertools.pairwise(items):
            if not call.usage.cache_write:
                continue
            if call.at - previous.at > ttl:
                idle_tokens += call.usage.cache_write
                idle_events += 1
            else:
                churn += call.usage.cache_write
    return Waste(idle_tokens, idle_events, churn)


def calls_resumes(calls: Sequence[Call], ttl: float) -> int:
    """Count cache creations after idle gaps in one session."""
    return sum(1 for a, b in itertools.pairwise(calls) if b.usage.cache_write and b.at - a.at > ttl)


# ------------------------------------------------------------------ the block


def calls_block(calls: Iterable[Call], now: float) -> float | None:
    """Return the open five-hour quota block, if one exists."""
    start = last = None
    for stamp in sorted(c.at for c in calls):
        if start is None or stamp - start >= BLOCK or stamp - last >= BLOCK:
            start = stamp - stamp % 3600
        last = stamp
    if start is None or now - start >= BLOCK:
        return None
    return start



def spend_projected(spent: float, elapsed: float, remaining: float) -> float:
    """Project spend at the end of the quota block."""
    return spent + spent / max(elapsed, 60) * remaining


# ------------------------------------------------------------------- billing


def rates_solved() -> Mapping[str, tuple[float, int]]:
    """Dollars per weighted megatoken per model, from closed sessions with no subagent traffic."""
    return _rates_within(int(time.time() // RATES_TTL))


@functools.cache
def _rates_within(epoch: int) -> Mapping[str, tuple[float, int]]:
    """Solve rates, memoised per TTL-wide epoch so a live dashboard still sees cache refreshes."""
    if (cached := _cached_rates()) is not None:
        return cached
    pools: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for path in CLAUDE_ROOT.glob("*/*.jsonl"):
        observed, billed, fanout = _reconcile(path)
        if fanout or billed <= 0 or observed <= 0 or abs(billed - observed) / billed > 0.10:
            continue
        for model, weight, cost in _billed_models(path):
            pool = pools[model]
            pool[0] += cost
            pool[1] += weight
            pool[2] += 1
    solved = {m: (1e6 * p[0] / p[1], int(p[2])) for m, p in pools.items() if p[1] > 0}
    _save_rates(solved)
    return solved


def _per_token_cost(model: str) -> float:
    rate, _ = rates_solved().get(model, (0.0, 0))
    return rate / 1e6


def call_dollars(call: Call) -> float:
    return call.weight * _per_token_cost(call.model)


def rates_blended() -> float:
    """One rate spanning models, weighted by sample count, not a flat per-model average."""
    solved = list(rates_solved().values())
    samples = sum(n for _, n in solved)
    if not samples:
        return 0.0
    return sum(rate * n for rate, n in solved) / samples / 1e6


def _cached_rates() -> Mapping[str, tuple[float, int]] | None:
    """Load a fresh, valid cached rate calculation."""
    try:
        if time.time() - RATES_CACHE.stat().st_mtime >= RATES_TTL:
            return None
        stored = json.loads(RATES_CACHE.read_text())
        return {k: (float(v[0]), int(v[1])) for k, v in stored.items()}
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return None  # Recalculate missing or invalid data.


def _save_rates(solved: Mapping[str, tuple[float, int]]) -> None:
    """Save rates through a temporary file."""
    try:
        RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        scratch = RATES_CACHE.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps(solved))
        scratch.replace(RATES_CACHE)
    except OSError:
        pass  # The cache is optional.


# ---------------------------------------------------------------- audit


@dataclasses.dataclass(frozen=True, slots=True)
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


def transcripts_audit() -> list[Reconciliation]:
    """Return closed Claude sessions with billed totals."""
    found = []
    for path in sorted(CLAUDE_ROOT.glob("*/*.jsonl")):
        observed, billed, fanout = _reconcile(path)
        if billed > 0:
            found.append(Reconciliation(id_short(path.stem), observed, billed, fanout))
    return found


def _reconcile(path: pathlib.Path) -> tuple[float, float, bool]:
    """Return observed tokens, billed tokens, and fan-out status.

    The billed total covers the session's subagents, whose requests live in
    their own files under ``<session>/subagents/``; observed counts them too.
    """
    billed = 0.0
    fanout = False
    observed: dict[str, float] = {}
    files = [path, *_subagents(path)]
    for record in itertools.chain.from_iterable(map(transcript_records, files)):
        if record.get("type") == "cost-state":
            billed = sum(w for _, w, _ in _cost_state(record))
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
        if message.get("id") is None and record.get("requestId") is None:
            continue  # Nothing to tell this call from any other.
        # A call spans several records; the last one carries its final usage.
        observed[f"{message.get('id')}|{record.get('requestId')}"] = claude_usage(usage).weight
    return sum(observed.values()), billed, fanout


def _subagents(path: pathlib.Path) -> list[pathlib.Path]:
    """The subagent transcripts that belong to one main transcript."""
    return sorted(path.with_suffix("").glob("subagents/**/*.jsonl"))


def _billed_models(path: pathlib.Path) -> Iterator[tuple[str, float, float]]:
    """Per-model tokens/dollars from the final cost-state record; checkpoints replace, not add."""
    final = None
    for record in transcript_records(path):
        if record.get("type") == "cost-state":
            final = record
    if final is None:
        return
    for model, weight, cost in _cost_state(final):
        if weight > 0 and cost > 0:
            yield model, weight, cost


def _cost_state(record: dict) -> Iterator[tuple[str, float, float]]:
    for model, usage in (record.get("modelUsage") or {}).items():
        billed = Usage(
            input=usage.get("inputTokens") or 0,
            cache_write=usage.get("cacheCreationInputTokens") or 0,
            cache_read=usage.get("cacheReadInputTokens") or 0,
            output=usage.get("outputTokens") or 0,
        )
        yield model, billed.weight, usage.get("costUSD") or 0.0
