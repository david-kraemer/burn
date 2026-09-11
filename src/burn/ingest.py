"""Read and parse the Claude Code and Codex transcript trees."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from .format import moment, short
from .model import (
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

# Keep sessions beyond the visible window so a wider view can reuse them.
GRACE = 3600


@dataclass(frozen=True, slots=True)
class Fragment:
    """Data parsed from one batch of appended lines."""

    calls: tuple[tuple[str, Call], ...] = ()
    fanout: tuple[str, ...] = ()
    tools: tuple[Tooling, ...] = ()
    gauges: tuple[Gauge, ...] = ()
    gauges_at: str = ""


@dataclass(slots=True)
class Carry:
    """Parse state retained between reads of one file."""

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
        """Read new records and return the current snapshot.

        File scans and parsing run in a worker thread so the live view can
        continue to accept input.
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
        """Read files in a worker thread."""
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
    """Parse one batch of appended lines."""
    reader = read_claude if source == CLAUDE else read_codex
    return reader(decode(lines), carry)


def read_claude(records: Iterable[dict], carry: Carry) -> Fragment:
    """Parse Claude assistant records and merge content blocks."""
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
    """Parse tool results and user text from one ``user`` record."""
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
    """Parse Codex ``token_count`` events."""
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
    """Parse tool calls and outputs from one ``response_item`` record."""
    kind = payload.get("type")
    if kind in ("custom_tool_call", "function_call"):
        carry.pending[payload.get("call_id")] = payload.get("name") or "?"
    elif kind in ("custom_tool_call_output", "function_call_output"):
        name = carry.pending.pop(payload.get("call_id"), None)
        if name:
            yield Tooling(at, carry.session, name, len(json.dumps(payload.get("output") or "")))


def codex_call(payload: dict, at: float, stamp: str, carry: Carry) -> tuple[str, Call] | None:
    """Parse the request described by a ``token_count`` event."""
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
    """Parse and sort Codex quota windows by length."""
    found = []
    for name in ("primary", "secondary"):
        gauge = reported.get(name)
        if gauge:
            found.append(
                Gauge(
                    used_percent=gauge.get("used_percent") or 0.0,
                    window_minutes=gauge.get("window_minutes") or 0,
                    resets_at=gauge.get("resets_at"),
                )
            )
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
    """Return transcript files changed since ``horizon``."""
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
    """Read complete lines after ``offset`` and return the new offset."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:  # The file was rewritten or rotated.
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
    if end < 0:  # Wait for the rest of a partial line.
        return [], offset
    return blob[:end].decode("utf-8", "replace").splitlines(), offset + end + 1


def records(path: Path) -> Iterator[dict]:
    """Yield every record in a file."""
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
