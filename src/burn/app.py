"""Run the interactive loop.

Sampling reads a few hundred files. That takes long enough to feel, so it
runs as a task, not inline. Keystrokes are served by the event loop while
a sample is still in flight, and the screen never freezes mid-read. This
loop is the view's only owner. Every keypress replaces the view outright,
so there is no state to synchronise.
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

from . import keys as bindings
from .dashboard import dashboard
from .ingest import Tailer
from .model import Snapshot
from .state import View, rows

CSI = "\x1b["


async def monitor(console: Console, view: View) -> None:
    """Redraw on a timer or after a keypress."""
    tailer = Tailer()
    async with keyboard() as pressed:
        # Sample before entering the alternate screen. This prevents an empty
        # first frame. Keys pressed during sampling remain queued.
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

                    # Wait on the clock only when no other event can wake us.
                    # A running sample wakes us when it completes.
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
                            due = 0.0  # read older files for a wider window
                        view = updated
            finally:
                for task in (sampling, waiting):
                    if task is not None:
                        task.cancel()


@asynccontextmanager
async def keyboard() -> AsyncIterator[asyncio.Queue[str]]:
    """Return queued keypresses, or none when stdin is not a terminal."""
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
    """Read one burst of input and keep arrow escapes intact."""
    found, at = [], 0
    while at < len(data):
        if data[at : at + 2] == CSI:
            found.append(data[at : at + 3])
            at += 3
        else:
            found.append(data[at])
            at += 1
    return found
