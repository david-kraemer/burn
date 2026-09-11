"""Immutable records used by the views."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import pairwise

__all__ = ["Call", "Gauge", "Row", "Snapshot", "Tooling", "Usage"]

# Weights are relative to one input token.
# Raw sums overstate cost because cache reads are cheaper.
CACHE_WRITE_WEIGHT = 1.25
CACHE_READ_WEIGHT = 0.1
OUTPUT_WEIGHT = 5.0

BLOCK = 300 * 60  # Claude Code's rolling quota block, in seconds
CACHE_TTL = 5 * 60  # short cache lifetime, in seconds

CLAUDE = "cc"
CODEX = "cx"

FANOUT_TOOLS = frozenset({"Task", "Agent", "Workflow"})


@dataclass(frozen=True, slots=True)
class Usage:
    """One request's four billed token classes."""

    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

    @property
    def prefix(self) -> int:
        """Tokens sent as input, including cached tokens."""
        return self.input + self.cache_write + self.cache_read

    @property
    def weight(self) -> float:
        """Usage converted to input-token equivalents."""
        return (
            self.input
            + CACHE_WRITE_WEIGHT * self.cache_write
            + CACHE_READ_WEIGHT * self.cache_read
            + OUTPUT_WEIGHT * self.output
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input + other.input,
            self.cache_write + other.cache_write,
            self.cache_read + other.cache_read,
            self.output + other.output,
        )


@dataclass(frozen=True, slots=True)
class Call:
    """One API request from either agent."""

    at: float
    source: str
    session: str
    project: str
    model: str
    usage: Usage = Usage()
    thinking: int = 0
    cache_5m: int = 0
    cache_1h: int = 0
    prompt: str = ""
    fanout: bool = False

    @property
    def prefix(self) -> int:
        return self.usage.prefix

    @property
    def weight(self) -> float:
        return self.usage.weight


@dataclass(frozen=True, slots=True)
class Tooling:
    """A tool result returned to a conversation."""

    at: float
    session: str
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class Gauge:
    """A quota window reported by Codex."""

    used_percent: float
    window_minutes: int
    resets_at: float | None = None


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Data read from disk at one point in time."""

    at: float = field(default_factory=time.time)
    calls: tuple[Call, ...] = ()
    tools: tuple[Tooling, ...] = ()
    gauges: tuple[Gauge, ...] = ()

    def since(self, seconds: float) -> Snapshot:
        """Return data from the last ``seconds``."""
        cutoff = self.at - seconds
        return replace(
            self,
            calls=tuple(c for c in self.calls if c.at >= cutoff),
            tools=tuple(t for t in self.tools if t.at >= cutoff),
        )

    def from_agent(self, source: str | None) -> Snapshot:
        """Return data for ``source``, or all data when it is None."""
        if source is None:
            return self
        sessions = {c.session for c in self.calls if c.source == source}
        return replace(
            self,
            calls=tuple(c for c in self.calls if c.source == source),
            tools=tuple(t for t in self.tools if t.session in sessions),
        )


@dataclass(frozen=True, slots=True)
class Row:
    """One session summarized for the table."""

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
    first: float
    last: float
    fanout: bool

    @property
    def key(self) -> str:
        """Identify this row across renders.

        The session id alone is not unique: a session that moved between
        project directories (a ``cd`` mid-conversation) tabulates into one
        row per project, all sharing the same session id. Selection must
        key on the full grouping, or the cursor can lock onto a row it can
        never distinguish from its neighbour.
        """
        return f"{self.source}|{self.session}|{self.project}"


def compactions(calls: Iterable[Call]) -> int:
    """Count context compactions."""
    return sum(1 for a, b in pairwise(calls) if b.prefix < a.prefix * 0.7)


def threads(calls: Iterable[Call]) -> list[list[Call]]:
    """Split interleaved calls into separately growing conversations.

    Assign each call to the open thread with the closest lower prefix. Start a
    new thread when no open thread can contain the call.
    """
    open_: list[list[Call]] = []
    for call in calls:
        fits = [t for t in open_ if t[-1].prefix <= call.prefix]
        if fits:
            max(fits, key=lambda t: t[-1].prefix).append(call)
        else:
            open_.append([call])
    return open_
