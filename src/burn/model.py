"""The immutable record every view is built from.

One normalised call type spans two very different transcript formats, so
nothing downstream of ingestion needs to know which agent produced a record.
Everything here is frozen: a :class:`Snapshot` is a value, views are functions
of that value, and the only mutable object in the program is the tailer that
produces snapshots.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import pairwise

# Relative to one input token, at Anthropic's published ratios. A raw token sum
# is ~95% cache reads and badly overstates what a session actually consumes.
CACHE_WRITE_WEIGHT = 1.25
CACHE_READ_WEIGHT = 0.1
OUTPUT_WEIGHT = 5.0

BLOCK = 300 * 60  # Claude Code's rolling quota block, in seconds
CACHE_TTL = 5 * 60  # the short ephemeral cache tier expires this fast

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
        """Everything the model was fed: the whole conversation so far."""
        return self.input + self.cache_write + self.cache_read

    @property
    def weight(self) -> float:
        """Tokens in input-token equivalents, so cache reads stop dominating."""
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
    """One API request, whichever agent made it."""

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
    """A tool result landing back in the conversation."""

    at: float
    session: str
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class Gauge:
    """One of Codex's quota windows, as last reported.

    Codex states each window's length explicitly and does not promise that
    "primary" means five hours -- for six weeks of one history the only gauge
    present was the seven-day one, under that same key. The label therefore
    comes from ``window_minutes``, never from the key.
    """

    used_percent: float
    window_minutes: int
    resets_at: float | None = None


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything harvested from disk, as of one instant."""

    at: float = field(default_factory=time.time)
    calls: tuple[Call, ...] = ()
    tools: tuple[Tooling, ...] = ()
    gauges: tuple[Gauge, ...] = ()

    def since(self, seconds: float) -> Snapshot:
        """The same snapshot narrowed to the last ``seconds`` of history."""
        cutoff = self.at - seconds
        return replace(
            self,
            calls=tuple(c for c in self.calls if c.at >= cutoff),
            tools=tuple(t for t in self.tools if t.at >= cutoff),
        )

    def from_agent(self, source: str | None) -> Snapshot:
        """The same snapshot restricted to one agent, or unchanged if None."""
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
    first: float
    last: float
    fanout: bool


def compactions(calls: Iterable[Call]) -> int:
    """How many times the context was discarded and rebuilt from a summary."""
    return sum(1 for a, b in pairwise(calls) if b.prefix < a.prefix * 0.7)


def threads(calls: Iterable[Call]) -> list[list[Call]]:
    """Split one session's calls into separately growing conversations.

    A session id does not always mean a single linear conversation. Codex runs
    side threads under the same id, so its calls arrive interleaved -- a real
    session bounces between a 162k prefix and an 88k one. Read as one
    conversation, every switch back up looks like 74k of fresh context, and the
    session accumulates 3.6M of "growth" against a context that never exceeded
    168k. A conversation only ever grows, so each call belongs to the open
    thread whose last prefix sits closest below it; a call that undercuts every
    open thread has been compacted, and starts a new one.
    """
    open_: list[list[Call]] = []
    for call in calls:
        fits = [t for t in open_ if t[-1].prefix <= call.prefix]
        if fits:
            max(fits, key=lambda t: t[-1].prefix).append(call)
        else:
            open_.append([call])
    return open_
