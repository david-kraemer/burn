"""Reading the two transcript trees.

Both agents append JSONL to disk as they work, so nothing needs instrumenting.
Together those trees run to hundreds of megabytes, which rules out re-parsing
them for every frame: :class:`Tailer` therefore keeps byte offsets and a little
per-file parse continuation, and that is the whole of the program's mutable
state. What it hands back -- a :class:`~burn.model.Snapshot` -- is immutable,
and every view downstream is a pure function of one.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from burn.format import moment, short
from burn.model import (
    CLAUDE,
    CODEX,
    FANOUT_TOOLS,
    Call,
    Gauge,
    Snapshot,
    Tooling,
    Usage,
)

CLAUDE_ROOT = Path.home() / ".claude" / "projects"
CODEX_ROOT = Path.home() / ".codex" / "sessions"

TREES = (
    (CLAUDE_ROOT, CLAUDE, "*/*.jsonl"),
    (CODEX_ROOT, CODEX, "*/*/*/rollout-*.jsonl"),
)

# Sessions stay in the snapshot this long past the visible window, so widening
# it with + does not blank the table until the next file actually changes.
GRACE = 3600


@dataclass(frozen=True, slots=True)
class Fragment:
    """What one batch of appended lines contributed.

    Calls arrive keyed by a request tag because a single API call is written as
    several records; folding on the tag merges them without double-counting.
    """

    calls: tuple[tuple[str, Call], ...] = ()
    fanout: tuple[str, ...] = ()
    tools: tuple[Tooling, ...] = ()
    gauges: tuple[Gauge, ...] = ()
    gauges_at: str = ""


@dataclass(slots=True)
class Carry:
    """Parse state that survives between two reads of the same file.

    A tool result can land in a batch of lines long after the ``tool_use`` that
    named it, and a Codex call inherits the model and cwd from a header record
    read minutes earlier.
    """

    session: str = "?"
    project: str = "?"
    model: str = "?"
    prompt: str = ""
    pending: dict[str, str] = field(default_factory=dict)


class Tailer:
    """Incremental reader over both transcript trees."""

    def __init__(self) -> None:
        self._offsets: dict[Path, int] = {}
        self._carry: dict[Path, Carry] = {}
        self._calls: dict[str, Call] = {}
        self._tools: list[Tooling] = []
        self._gauges: tuple[Gauge, ...] = ()
        self._gauges_at = ""

    async def sample(self, window: int) -> Snapshot:
        """Read whatever is new in both trees and return the current snapshot.

        A first sample over a week of history costs the better part of a second,
        nearly all of it parsing, and widening the window from the keyboard is
        exactly what provokes one. So the sweep, the reads and the parse all run
        off the event loop, and only the fold -- microseconds of dictionary
        work -- happens back on it. The screen keeps answering keys throughout.
        """
        now = time.time()
        horizon = now - (max(window, 300) + 60) * 60
        for path, offset, fragment in await asyncio.to_thread(self._read, horizon):
            self._offsets[path] = offset
            self._absorb(fragment)
        self._forget(now - max(window, 300) * 60 - GRACE)
        return Snapshot(
            at=now,
            calls=tuple(sorted(self._calls.values(), key=lambda c: c.at)),
            tools=tuple(sorted(self._tools, key=lambda t: t.at)),
            gauges=self._gauges,
        )

    def _read(self, horizon: float) -> list[tuple[Path, int, Fragment]]:
        """The blocking half of a sample, run in a worker thread.

        It touches the offset and carry maps, which is safe only because the
        loop keeps at most one sample in flight; :meth:`sample` is the only
        caller and enforces that by construction.
        """
        found = []
        for path, source in transcripts(horizon):
            lines, offset = tail(path, self._offsets.get(path, 0))
            if lines:
                found.append((path, offset, parse(lines, source, self._carry_for(path, source))))
        return found

    def _carry_for(self, path: Path, source: str) -> Carry:
        if path not in self._carry:
            seed = Carry(session=short(path.stem[-36:])) if source == CODEX else Carry()
            self._carry[path] = seed
        return self._carry[path]

    def _absorb(self, fragment: Fragment) -> None:
        for tag, call in fragment.calls:
            self._calls.setdefault(tag, call)
        for tag in fragment.fanout:
            if (call := self._calls.get(tag)) and not call.fanout:
                self._calls[tag] = replace(call, fanout=True)
        self._tools.extend(fragment.tools)
        if fragment.gauges and fragment.gauges_at > self._gauges_at:
            self._gauges, self._gauges_at = fragment.gauges, fragment.gauges_at

    def _forget(self, cutoff: float) -> None:
        self._calls = {tag: c for tag, c in self._calls.items() if c.at >= cutoff}
        self._tools = [t for t in self._tools if t.at >= cutoff]


# -------------------------------------------------------------------- parsing


def parse(lines: Iterable[str], source: str, carry: Carry) -> Fragment:
    """One batch of appended lines, in whichever format its tree uses."""
    reader = read_claude if source == CLAUDE else read_codex
    return reader(decode(lines), carry)


def read_claude(records: Iterable[dict], carry: Carry) -> Fragment:
    """Claude Code assistant records, merged across their content blocks.

    One API call is written as several records, one per content block, each
    repeating the same usage. Keying on (message id, request id) merges them
    without counting the tokens more than once -- and without losing the
    ``tool_use`` blocks that live in the later records, which is why tools are
    registered from any record whether or not it carries usage of its own.
    """
    calls: list[tuple[str, Call]] = []
    fanout: list[str] = []
    tools: list[Tooling] = []
    known: set[str] = set()

    for record in records:
        kind = record.get("type")
        message = record.get("message") or {}
        if kind == "user":
            tools.extend(claude_results(record, message, carry))
            continue
        if kind != "assistant":
            continue

        tag = f"cc|{message.get('id')}|{record.get('requestId')}"
        usage = message.get("usage")
        model = message.get("model")
        at = moment(record.get("timestamp"))
        if tag not in known and usage and model and model != "<synthetic>" and at is not None:
            known.add(tag)
            creation = usage.get("cache_creation") or {}
            details = usage.get("output_tokens_details") or {}
            calls.append((
                tag,
                Call(
                    at=at,
                    source=CLAUDE,
                    session=short(record.get("sessionId") or "?"),
                    project=Path(record.get("cwd") or "?").name,
                    model=model,
                    usage=claude_usage(usage),
                    thinking=details.get("thinking_tokens") or 0,
                    cache_5m=creation.get("ephemeral_5m_input_tokens") or 0,
                    cache_1h=creation.get("ephemeral_1h_input_tokens") or 0,
                    prompt=carry.prompt,
                ),
            ))
        for block in blocks(message):
            if block.get("type") == "tool_use":
                carry.pending[block.get("id")] = block.get("name")
                if block.get("name") in FANOUT_TOOLS:
                    fanout.append(tag)
    return Fragment(calls=tuple(calls), fanout=tuple(fanout), tools=tuple(tools))


def claude_results(record: dict, message: dict, carry: Carry) -> Iterator[Tooling]:
    """Tool results and user text from one ``user`` record."""
    content = message.get("content")
    if isinstance(content, str):
        if content.strip():
            carry.prompt = content
        return
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            name = carry.pending.pop(block.get("tool_use_id"), None)
            if name:
                yield Tooling(
                    at=moment(record.get("timestamp")) or 0.0,
                    session=short(record.get("sessionId") or "?"),
                    name=name,
                    size=len(json.dumps(block.get("content") or "")),
                )
        elif block.get("type") == "text" and (block.get("text") or "").strip():
            carry.prompt = block["text"]


def read_codex(records: Iterable[dict], carry: Carry) -> Fragment:
    """Codex ``token_count`` events, one per API request.

    Each event reports both a running total and that request's own usage, and
    the two reconcile exactly, so reading incrementally is also correct.
    """
    calls: list[tuple[str, Call]] = []
    tools: list[Tooling] = []
    gauges: tuple[Gauge, ...] = ()
    gauges_at = ""

    for record in records:
        payload = record.get("payload") or {}
        at = moment(record.get("timestamp")) or 0.0
        match record.get("type"):
            case "session_meta":
                carry.session = short(payload.get("session_id") or carry.session)
                carry.project = Path(payload.get("cwd") or carry.project).name
            case "turn_context":
                carry.model = payload.get("model") or carry.model
                carry.project = Path(payload.get("cwd") or carry.project).name
            case "response_item":
                tools.extend(codex_results(payload, at, carry))
            case "event_msg":
                if payload.get("type") == "user_message":
                    carry.prompt = payload.get("message") or carry.prompt
                elif payload.get("type") == "token_count":
                    stamp = record.get("timestamp") or ""
                    if (reported := payload.get("rate_limits")) and stamp > gauges_at:
                        gauges, gauges_at = codex_gauges(reported), stamp
                    if entry := codex_call(payload, at, stamp, carry):
                        calls.append(entry)
    return Fragment(
        calls=tuple(calls), tools=tuple(tools), gauges=gauges, gauges_at=gauges_at
    )


def codex_results(payload: dict, at: float, carry: Carry) -> Iterator[Tooling]:
    """Tool calls and their outputs from one ``response_item`` record."""
    kind = payload.get("type")
    if kind in ("custom_tool_call", "function_call"):
        carry.pending[payload.get("call_id")] = payload.get("name") or "?"
    elif kind in ("custom_tool_call_output", "function_call_output"):
        name = carry.pending.pop(payload.get("call_id"), None)
        if name:
            yield Tooling(at, carry.session, name, len(json.dumps(payload.get("output") or "")))


def codex_call(payload: dict, at: float, stamp: str, carry: Carry) -> tuple[str, Call] | None:
    """The request a ``token_count`` event describes, if it describes one."""
    usage = (payload.get("info") or {}).get("last_token_usage")
    if not usage:
        return None
    cached = usage.get("cached_input_tokens") or 0
    return (
        f"cx|{carry.session}|{stamp}|{usage.get('total_tokens')}",
        Call(
            at=at,
            source=CODEX,
            session=carry.session,
            project=carry.project,
            model=carry.model,
            usage=Usage(
                input=max((usage.get("input_tokens") or 0) - cached, 0),
                cache_write=usage.get("cache_write_input_tokens") or 0,
                cache_read=cached,
                output=usage.get("output_tokens") or 0,
            ),
            thinking=usage.get("reasoning_output_tokens") or 0,
            prompt=carry.prompt,
        ),
    )


def codex_gauges(reported: dict) -> tuple[Gauge, ...]:
    """Codex's quota windows, ordered shortest first and keyed by length."""
    found = [
        Gauge(
            used_percent=gauge.get("used_percent") or 0.0,
            window_minutes=gauge.get("window_minutes") or 0,
            resets_at=gauge.get("resets_at"),
        )
        for gauge in (reported.get(name) for name in ("primary", "secondary"))
        if gauge
    ]
    return tuple(sorted(found, key=lambda g: g.window_minutes))


def claude_usage(usage: dict) -> Usage:
    return Usage(
        input=usage.get("input_tokens") or 0,
        cache_write=usage.get("cache_creation_input_tokens") or 0,
        cache_read=usage.get("cache_read_input_tokens") or 0,
        output=usage.get("output_tokens") or 0,
    )


def blocks(message: dict) -> Iterator[dict]:
    for block in message.get("content") or []:
        if isinstance(block, dict):
            yield block


# --------------------------------------------------------------------- files


def transcripts(horizon: float) -> list[tuple[Path, str]]:
    """Transcript files touched since the horizon, with their source tag."""
    found = []
    for root, source, pattern in TREES:
        for path in root.glob(pattern):
            try:
                if path.stat().st_mtime >= horizon:
                    found.append((path, source))
            except OSError:
                continue
    return found


def tail(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete lines appended since the offset, and where to resume."""
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
    """Every record in a whole file, for the audits that must read all of it."""
    try:
        with path.open(errors="replace") as handle:
            yield from decode(handle)
    except OSError:
        return


def decode(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            yield record
