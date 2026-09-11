"""The interactive loop.

Sampling reads a few hundred files and takes long enough to be felt, so it runs
as a task rather than inline: keystrokes are served by the event loop while a
sample is still in flight, and the screen never freezes mid-harvest. The view
has exactly one owner -- this loop -- and every keypress replaces it outright,
so there is no state to synchronise between the two.
"""

from __future__ import annotations

import asyncio
import os
import sys
import termios
import time
import tty
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from rich.console import Console
from rich.live import Live

from burn import keys as bindings
from burn.dashboard import dashboard
from burn.ingest import Tailer
from burn.model import Snapshot
from burn.state import View, rows

CSI = "\x1b["


async def monitor(console: Console, view: View) -> None:
    """Redraw on a timer, and at once on a keypress."""
    tailer = Tailer()
    async with keyboard() as pressed:
        # Take the first sample before switching to the alternate screen, so the
        # opening frame has data in it rather than flashing an empty table.
        # Keys struck meanwhile are already being queued.
        snapshot = await tailer.sample(view.window)
        with Live(console=console, screen=True, auto_refresh=False) as live:
            sampling: asyncio.Task[Snapshot] | None = None
            waiting: asyncio.Task[str] | None = None
            due = time.monotonic() + view.interval
            try:
                while True:
                    live.update(dashboard(snapshot, view, console.size.height), refresh=True)
                    now = time.monotonic()
                    if sampling is None and not view.paused and now >= due:
                        sampling = asyncio.create_task(tailer.sample(view.window))
                    if waiting is None:
                        waiting = asyncio.create_task(pressed.get())

                    # Only idle on a clock when nothing else can wake us: with a
                    # sample in flight, its completion is the next event.
                    pending = {waiting} if sampling is None else {waiting, sampling}
                    idle = view.paused or sampling is not None
                    done, _ = await asyncio.wait(
                        pending,
                        timeout=None if idle else max(0.0, due - now),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if sampling in done:
                        snapshot = sampling.result()
                        sampling = None
                        due = time.monotonic() + view.interval
                    if waiting in done:
                        key = waiting.result()
                        waiting = None
                        updated = bindings.apply(key, view, rows(view, snapshot))
                        if updated is None:
                            return
                        if updated.window != view.window:
                            due = 0.0  # a wider window needs older files read now
                        view = updated
            finally:
                for task in (sampling, waiting):
                    if task is not None:
                        task.cancel()


@asynccontextmanager
async def keyboard() -> AsyncIterator[asyncio.Queue[str]]:
    """Keypresses as a queue, or an empty one where there is no terminal."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    if not sys.stdin.isatty():
        yield queue
        return

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    loop = asyncio.get_running_loop()

    def readable() -> None:
        try:
            data = os.read(fd, 1024)
        except OSError:
            return
        for key in split(data.decode("utf-8", "replace")):
            queue.put_nowait(key)

    try:
        tty.setcbreak(fd)
        loop.add_reader(fd, readable)
        yield queue
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def split(data: str) -> list[str]:
    """One burst of input as individual keys, arrow escapes kept whole."""
    found, at = [], 0
    while at < len(data):
        if data[at : at + 2] == CSI:
            found.append(data[at : at + 3])
            at += 3
        else:
            found.append(data[at])
            at += 1
    return found
