"""The interactive loop and its terminal plumbing."""

import asyncio

import pytest
from rich.console import Console

from burn.app import monitor, split
from burn.state import View


def test_a_burst_of_input_splits_into_individual_keys():
    assert split("jk") == ["j", "k"]


def test_an_arrow_escape_stays_whole():
    assert split("\x1b[A") == ["\x1b[A"]
    assert split("j\x1b[Bk") == ["j", "\x1b[B", "k"]


def test_a_bare_escape_is_its_own_key():
    assert split("\x1b") == ["\x1b"]


class Keys:
    """A queue of scripted keypresses, standing in for the terminal."""

    def __init__(self, keys):
        self.queue = asyncio.Queue()
        for key in keys:
            self.queue.put_nowait(key)

    async def __aenter__(self):
        return self.queue

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def scripted(monkeypatch):
    """Drive the loop from a script of keys, over an empty transcript tree."""
    def drive(keys, sampler=None):
        from burn import app
        from burn.model import Snapshot

        monkeypatch.setattr(app, "keyboard", lambda: Keys(keys))
        seen = []

        async def sample(self, window):
            seen.append(window)
            return sampler(window) if sampler else Snapshot(at=1000.0)

        monkeypatch.setattr(app.Tailer, "sample", sample)
        console = Console(width=100, height=30, force_terminal=False, no_color=True)
        asyncio.run(asyncio.wait_for(monitor(console, View(interval=0.01)), timeout=5))
        return seen

    return drive


def test_the_loop_samples_then_quits_on_q(scripted):
    assert scripted(["q"]) == [300]


def test_widening_the_window_forces_a_fresh_sample(scripted):
    # A wider window needs files the last sample never opened, so + must not
    # wait out the refresh interval before reading them.
    assert scripted(["+", "q"])[-1] == 600


def test_the_loop_survives_a_key_arriving_before_the_first_sample(scripted):
    assert scripted(["j", "k", "\r", "t", "s", "/", "x", "\r", "h", "?", "q"])


def test_a_sample_in_flight_does_not_block_the_quit_key(monkeypatch):
    """Only the opening sample blocks. After that, a key always wins.

    The first sample runs before the app enters the alternate screen, so the
    opening frame has data in it. Every later sample overlaps the keyboard.
    """
    from burn import app
    from burn.model import Snapshot

    monkeypatch.setattr(app, "keyboard", lambda: Keys(["q"]))
    taken = []

    async def sample(self, window):
        taken.append(window)
        if len(taken) > 1:
            await asyncio.sleep(30)
        return Snapshot(at=1000.0)

    monkeypatch.setattr(app.Tailer, "sample", sample)
    console = Console(width=100, height=30, force_terminal=False, no_color=True)
    asyncio.run(asyncio.wait_for(monitor(console, View(interval=0.0)), timeout=3))


def test_a_real_sample_leaves_the_event_loop_free(monkeypatch, tmp_path):
    """The sweep, the reads and the parse must all run off the loop.

    A first sample over a week of history costs the better part of a second,
    nearly all of it parsing, so doing any of it inline would freeze the screen
    exactly when the user has just widened the window.
    """
    import time

    from burn import ingest

    def slow_sweep(horizon):
        time.sleep(0.4)  # stands in for the sweep, reads and parse
        return []

    monkeypatch.setattr(ingest, "transcripts", slow_sweep)

    async def race():
        ticks = 0
        sampling = asyncio.create_task(ingest.Tailer().sample(300))
        while not sampling.done():
            await asyncio.sleep(0.01)
            ticks += 1
        return ticks

    assert asyncio.run(race()) > 10
