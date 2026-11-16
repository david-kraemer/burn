"""Read and parse the Claude Code and Codex transcript trees."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import time
from collections.abc import Iterable, Iterator

from . import quota
from .format import id_short, iso_epoch
from .model import (
    CLAUDE,
    CODEX,
    FANOUT_TOOLS,
    Call,
    Gauge,
    Reading,
    Snapshot,
    Tooling,
    Usage,
)

CLAUDE_ROOT = pathlib.Path.home() / ".claude" / "projects"
CODEX_ROOT = pathlib.Path.home() / ".codex" / "sessions"

# Claude Code writes each subagent's conversation to its own file under the
# parent session's directory (``<session>/subagents/[workflows/<id>/]agent-*``).
# Those records carry the parent's sessionId, so they join the parent's row.
# Skipping them misses roughly as much usage as the main transcripts hold.
SUBAGENTS = "*/*/subagents/**/*.jsonl"

TREES = (
    (CLAUDE_ROOT, CLAUDE, "*/*.jsonl"),
    (CLAUDE_ROOT, CLAUDE, SUBAGENTS),
    (CODEX_ROOT, CODEX, "*/*/*/rollout-*.jsonl"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class Fragment:
    """Data parsed from one batch of appended lines."""

    calls: tuple[tuple[str, Call], ...] = ()
    fanout: tuple[str, ...] = ()
    tools: tuple[Tooling, ...] = ()
    gauges: tuple[Gauge, ...] = ()
    gauges_at: str = ""


@dataclasses.dataclass(slots=True)
class Carry:
    """Parse state retained between reads of one file."""

    session: str = "?"
    project: str = "?"
    model: str = "?"
    prompt: str = ""
    pending: dict[str, str] = dataclasses.field(default_factory=dict)
    # Codex's running total at its last token_count event. Codex repeats the
    # event with an unchanged total; only a change is a new request.
    total: dict | None = None


class Tailer:
    """Incremental reader over both transcript trees."""

    def __init__(self, remote: bool = True) -> None:
        self._offsets: dict[pathlib.Path, int] = {}
        self._carry: dict[pathlib.Path, Carry] = {}
        self._calls: dict[str, Call] = {}
        self._tools: list[Tooling] = []
        self._gauges: tuple[Gauge, ...] = ()
        self._gauges_at = ""
        self._remote = remote
        self._reading: Reading | None = None
        self._polling: asyncio.Task[Reading | None] | None = None
        self._polled = 0.0

    async def sample(self, window: int, settle: float = 0.0) -> Snapshot:
        """Read new records and return a snapshot.

        Run parsing in a worker thread. ``settle`` sets the maximum wait for
        quota data. If the wait expires, leave the request for the next frame.
        """
        now = time.time()
        horizon = now - (max(window, 300) + 60) * 60
        self._poll(now)
        for path, offset, fragment in await asyncio.to_thread(self._read, horizon):
            self._offsets[path] = offset
            self._absorb(fragment)
        if settle and self._polling is not None:
            await asyncio.wait({self._polling}, timeout=settle)
            self._collect()
        # Retain double the window: a widen ("+") must still find data whose
        # byte offset has already advanced past it.
        self._forget(now - max(window, 300) * 60 * 2)
        return Snapshot(
            at=now,
            calls=tuple(sorted(self._calls.values(), key=lambda c: c.at)),
            tools=tuple(sorted(self._tools, key=lambda t: t.at)),
            gauges=self._gauges,
            reading=self._reading,
        )

    def close(self) -> None:
        """Cancel an active quota poll."""
        if self._polling is not None:
            self._polling.cancel()
            self._polling = None

    def _poll(self, now: float) -> None:
        """Start or collect a quota poll without waiting for it."""
        self._collect()
        # Quota data comes from the network: poll it at the cache interval, less
        # often than transcripts, and off the frame's critical path.
        if not self._remote or self._polling is not None or now - self._polled < quota.QUOTA_TTL:
            return
        if self._reading is None:
            # Set the meter height from the cached window shape before the
            # first request. Add the values when the request completes.
            self._reading = quota.quota_shape()
        self._polled = now
        self._polling = asyncio.create_task(asyncio.to_thread(quota.quota_reading))

    def _collect(self) -> None:
        """Store a completed quota poll. Keep the last good reading on failure."""
        if self._polling is None or not self._polling.done():
            return
        task, self._polling = self._polling, None
        if task.cancelled() or task.exception() is not None:
            return
        # Keep the previous reading after a request failure. A failed request
        # does not prove that quota data is unknown.
        self._reading = task.result() or self._reading

    def _read(self, horizon: float) -> list[tuple[pathlib.Path, int, Fragment]]:
        found = []
        for path, source in _transcripts(horizon):
            stored = self._offsets.get(path, 0)
            if _rewritten(path, stored):
                # _tail() restarts from byte zero; drop the stale Carry so it
                # doesn't bleed into content read from the start.
                self._carry.pop(path, None)
            lines, offset = _tail(path, stored)
            if lines:
                found.append((path, offset, _parse(lines, source, self._carry_for(path, source))))
        return found

    def _carry_for(self, path: pathlib.Path, source: str) -> Carry:
        if path not in self._carry:
            seed = Carry(session=id_short(path.stem[-36:])) if source == CODEX else Carry()
            self._carry[path] = seed
        return self._carry[path]

    def _absorb(self, fragment: Fragment) -> None:
        for tag, call in fragment.calls:
            known = self._calls.get(tag)
            self._calls[tag] = call if known is None else _merged(known, call)
        for tag in fragment.fanout:
            if (call := self._calls.get(tag)) and not call.fanout:
                self._calls[tag] = dataclasses.replace(call, fanout=True)
        self._tools.extend(fragment.tools)
        if fragment.gauges and fragment.gauges_at > self._gauges_at:
            self._gauges, self._gauges_at = fragment.gauges, fragment.gauges_at

    def _forget(self, cutoff: float) -> None:
        self._calls = {tag: c for tag, c in self._calls.items() if c.at >= cutoff}
        self._tools = [t for t in self._tools if t.at >= cutoff]


# -------------------------------------------------------------------- parsing


def _parse(lines: Iterable[str], source: str, carry: Carry) -> Fragment:
    """Parse one batch of appended lines."""
    reader = _read_claude if source == CLAUDE else _read_codex
    return reader(_decode(lines), carry)


def _read_claude(records: Iterable[dict], carry: Carry) -> Fragment:
    """Parse Claude assistant records and merge content blocks."""
    calls: dict[str, Call] = {}
    fanout: list[str] = []
    tools: list[Tooling] = []

    for record in records:
        kind = record.get("type")
        message = record.get("message") or {}
        if kind == "user":
            tools.extend(_claude_results(record, message, carry))
            continue
        if kind != "assistant":
            continue

        tag = f"cc|{message.get('id')}|{record.get('requestId')}"
        usage = message.get("usage")
        model = message.get("model")
        at = iso_epoch(record.get("timestamp"))
        if usage and model and model != "<synthetic>" and at is not None:
            creation = usage.get("cache_creation") or {}
            details = usage.get("output_tokens_details") or {}
            call = Call(
                at=at,
                source=CLAUDE,
                session=id_short(record.get("sessionId") or "?"),
                project=pathlib.Path(record.get("cwd") or "?").name,
                model=model,
                usage=claude_usage(usage),
                thinking=details.get("thinking_tokens") or 0,
                cache_5m=creation.get("ephemeral_5m_input_tokens") or 0,
                cache_1h=creation.get("ephemeral_1h_input_tokens") or 0,
                prompt=carry.prompt,
                sidechain=bool(record.get("isSidechain")),
            )
            calls[tag] = _merged(calls[tag], call) if tag in calls else call
        for block in _blocks(message):
            if block.get("type") == "tool_use":
                carry.pending[block.get("id")] = block.get("name")
                if block.get("name") in FANOUT_TOOLS:
                    fanout.append(tag)
    return Fragment(calls=tuple(calls.items()), fanout=tuple(fanout), tools=tuple(tools))


def _merged(known: Call, later: Call) -> Call:
    """One call written as several records: the later record's usage wins.

    Claude Code writes a record per content block, each repeating the usage.
    In subagent transcripts the early records carry a partial output count;
    only the last one is final. Keep the first record's time and prompt.
    """
    return dataclasses.replace(
        later, at=known.at, prompt=known.prompt, fanout=known.fanout or later.fanout
    )


def _claude_results(record: dict, message: dict, carry: Carry) -> Iterator[Tooling]:
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
                    at=iso_epoch(record.get("timestamp")) or 0.0,
                    session=id_short(record.get("sessionId") or "?"),
                    name=name,
                    size=len(json.dumps(block.get("content") or "")),
                )
        elif block.get("type") == "text" and (block.get("text") or "").strip():
            carry.prompt = block["text"]


def _read_codex(records: Iterable[dict], carry: Carry) -> Fragment:
    """Parse Codex ``token_count`` events."""
    calls: list[tuple[str, Call]] = []
    tools: list[Tooling] = []
    gauges: tuple[Gauge, ...] = ()
    gauges_at = ""

    for record in records:
        payload = record.get("payload") or {}
        at = iso_epoch(record.get("timestamp")) or 0.0
        match record.get("type"):
            case "session_meta":
                # Ignore session_id here: a sub-agent's rollout file reports
                # its *parent's* thread id, which would corrupt attribution.
                # _carry_for already seeded carry.session from the filename.
                carry.project = pathlib.Path(payload.get("cwd") or carry.project).name
            case "turn_context":
                carry.model = payload.get("model") or carry.model
                carry.project = pathlib.Path(payload.get("cwd") or carry.project).name
            case "response_item":
                tools.extend(_codex_results(payload, at, carry))
            case "event_msg":
                if payload.get("type") == "user_message":
                    carry.prompt = payload.get("message") or carry.prompt
                elif payload.get("type") == "token_count":
                    stamp = record.get("timestamp") or ""
                    if (reported := payload.get("rate_limits")) and stamp > gauges_at:
                        gauges, gauges_at = _codex_gauges(reported), stamp
                    if entry := _codex_call(payload, at, stamp, carry):
                        calls.append(entry)
    return Fragment(
        calls=tuple(calls), tools=tuple(tools), gauges=gauges, gauges_at=gauges_at
    )


def _codex_results(payload: dict, at: float, carry: Carry) -> Iterator[Tooling]:
    """Parse tool calls and outputs from one ``response_item`` record."""
    kind = payload.get("type")
    if kind in ("custom_tool_call", "function_call"):
        carry.pending[payload.get("call_id")] = payload.get("name") or "?"
    elif kind in ("custom_tool_call_output", "function_call_output"):
        name = carry.pending.pop(payload.get("call_id"), None)
        if name:
            yield Tooling(at, carry.session, name, len(json.dumps(payload.get("output") or "")))


def _codex_call(payload: dict, at: float, stamp: str, carry: Carry) -> tuple[str, Call] | None:
    """Parse the request described by a ``token_count`` event."""
    info = payload.get("info") or {}
    usage = info.get("last_token_usage")
    if not usage:
        return None
    if (total := info.get("total_token_usage")) is not None:
        if total == carry.total:
            return None  # A repeated event: the running total did not move.
        carry.total = total
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


def _codex_gauges(reported: dict) -> tuple[Gauge, ...]:
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


def _blocks(message: dict) -> Iterator[dict]:
    for block in message.get("content") or []:
        if isinstance(block, dict):
            yield block


# --------------------------------------------------------------------- files


def _transcripts(horizon: float) -> list[tuple[pathlib.Path, str]]:
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


def _rewritten(path: pathlib.Path, offset: int) -> bool:
    """Whether ``path`` has shrunk below ``offset`` (rewritten or rotated)."""
    try:
        return path.stat().st_size < offset
    except OSError:
        return False


def _tail(path: pathlib.Path, offset: int) -> tuple[list[str], int]:
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
    # Split on "\n" only: str.splitlines() also breaks on other separators
    # legal inside a JSON string, which would corrupt a record containing one.
    return blob[:end].decode("utf-8", "replace").split("\n"), offset + end + 1


def transcript_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield every record in a file."""
    try:
        with path.open(errors="replace") as handle:
            yield from _decode(handle)
    except OSError:
        return


def _decode(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            yield record
