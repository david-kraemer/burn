"""Burn's data for other programs: one snapshot document, or a stream of them.

The document is plain data (dicts, lists, numbers, strings), the same shape
``burn --json`` prints. A long-running consumer should iterate
:func:`burn_documents`: it keeps one :class:`~burn.ingest.Tailer`, so each
document reads only the bytes appended since the last one, and quota is
requested no more often than the shared cache allows.

    import asyncio
    import burn

    print(asyncio.run(burn.burn_latest(window=60)))

    async def watch():
        async for doc in burn.burn_documents(window=60, interval=5):
            ...

Fields may be added; existing ones keep their names and meaning.
``BURN_SCHEMA`` changes only when a field is removed or redefined.

Quota freshness. ``quota_at`` is the epoch at which the ``quota`` and
``spend`` values were fetched, or null if none ever were; ``at - quota_at``
is their age. Values are refreshed at most every two minutes, and less
often while requests fail (the delay doubles up to fifteen minutes).
``quota_pending`` is true once the values are older than fifteen minutes
(``QUOTA_STALE``): they are then still present, but only so the windows'
names and order stay known; treat their percentages and reset times as
unknown, and use ``quota_at`` to say how old they are. An empty ``quota``
means no reading has ever been cached.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator

from .ingest import Tailer
from .model import Snapshot
from .quota import QUOTA_TIMEOUT
from .state import View, view_rows

__all__ = ["BURN_SCHEMA", "burn_document", "burn_documents", "burn_latest"]

BURN_SCHEMA = 1


async def burn_latest(
    window: int = 300, *, source: str | None = None, sort: str = "weight", remote: bool = True
) -> dict:
    """One document. With ``remote``, wait up to the request deadline for quota data.

    ``source`` keeps one agent's sessions ("cc" or "cx"); ``sort`` names the
    session field to order by, largest first.
    """
    tailer = Tailer(remote=remote)
    try:
        snapshot = await tailer.sample(window, settle=QUOTA_TIMEOUT if remote else 0.0)
    finally:
        tailer.close()
    return burn_document(snapshot, View(window=window, source=source, sort=sort))


async def burn_documents(
    window: int = 300,
    *,
    interval: float = 2.0,
    source: str | None = None,
    sort: str = "weight",
    remote: bool = True,
) -> AsyncIterator[dict]:
    """A document every ``interval`` seconds, read incrementally, until the consumer stops.

    The first document waits for quota data as :func:`burn_latest` does. Later ones
    never wait: a quota request runs in the background and its answer appears
    in the first document sampled after it lands.
    """
    tailer = Tailer(remote=remote)
    view = View(window=window, interval=interval, source=source, sort=sort)
    settle = QUOTA_TIMEOUT if remote else 0.0
    try:
        while True:
            due = time.monotonic() + interval
            snapshot = await tailer.sample(window, settle=settle)
            settle = 0.0
            yield burn_document(snapshot, view)
            await asyncio.sleep(max(0.0, due - time.monotonic()))
    finally:
        tailer.close()


def burn_document(snapshot: Snapshot, view: View) -> dict:
    """The machine-readable form of a snapshot: what the live view shows, as data."""
    reading = snapshot.reading
    return {
        "schema": BURN_SCHEMA,
        "at": snapshot.at,
        "window_minutes": view.window,
        "source": view.source,
        "quota": [dataclasses.asdict(window) for window in reading.windows] if reading else [],
        "quota_at": reading.at if reading else None,
        # Pending values are older than QUOTA_STALE and kept only for shape;
        # quota_at says how old. See the module docstring.
        "quota_pending": reading.pending if reading else False,
        "spend": dataclasses.asdict(reading.spend) if reading and reading.spend else None,
        "codex": [dataclasses.asdict(gauge) for gauge in snapshot.gauges],
        "sessions": [dataclasses.asdict(row) for row in view_rows(view, snapshot)],
    }
