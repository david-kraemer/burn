"""The programmatic interface: the snapshot document, once and as a stream."""

import asyncio
import json
import time

import pytest

import burn
from burn import Gauge, Quota, Reading, Snapshot, Spend, Usage, View, ingest

LEGACY_KEYS = {"at", "window_minutes", "quota", "spend", "sessions"}
QUOTA_KEYS = {"name", "group", "used_percent", "resets_at", "scope", "active"}
ROW_KEYS = {"session", "project", "weight", "rate", "cost", "ctx"}


@pytest.fixture
def reported(snapshot, call):
    reading = Reading(
        at=990.0,
        windows=(Quota("week", "weekly", 8.0, resets_at=1789797599.5, active=True),),
        spend=Spend(used=1.5, cap=10.0),
    )
    return snapshot(
        at=1000.0,
        calls=[call(at=950.0, session="aaa", usage=Usage(10, 20, 300, 40))],
        gauges=[Gauge(12.0, 10080, 1789797000)],
        reading=reading,
    )


def test_the_document_keeps_every_field_rexops_reads(reported):
    doc = burn.burn_document(reported, View(window=60))
    assert doc.keys() >= LEGACY_KEYS
    assert doc["quota"][0].keys() >= QUOTA_KEYS
    assert doc["sessions"][0].keys() >= ROW_KEYS
    assert doc["window_minutes"] == 60


def test_the_document_survives_a_json_round_trip(reported):
    doc = burn.burn_document(reported, View())
    assert json.loads(json.dumps(doc)) == doc


def test_reset_times_stay_raw_epochs(reported):
    doc = burn.burn_document(reported, View())
    assert doc["quota"][0]["resets_at"] == 1789797599.5
    assert doc["codex"] == [{"used_percent": 12.0, "window_minutes": 10080,
                             "resets_at": 1789797000}]


def test_the_document_says_how_old_its_quota_is(reported):
    import dataclasses

    doc = burn.burn_document(reported, View())
    assert (doc["schema"], doc["quota_at"], doc["quota_pending"]) == (1, 990.0, False)
    pending = dataclasses.replace(reported.reading, pending=True)
    stale = dataclasses.replace(reported, reading=pending)
    assert burn.burn_document(stale, View())["quota_pending"] is True
    assert burn.burn_document(Snapshot(at=1.0), View())["quota_at"] is None


# ------------------------------------------------------------------ sampling


def record(message_id, session="11111111-2222-3333"):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "type": "assistant", "timestamp": stamp, "sessionId": session, "cwd": "/x/proj",
        "requestId": "r", "message": {"id": message_id, "model": "claude-opus-5",
                                      "usage": {"input_tokens": 10}},
    }


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """An empty Claude tree and a log of every (file, offset) the Tailer reads from."""
    root = tmp_path / "claude"
    monkeypatch.setattr(ingest, "TREES", ((root, ingest.CLAUDE, "*/*.jsonl"),))
    reads = []
    tail = ingest._tail
    monkeypatch.setattr(
        ingest,
        "_tail",
        lambda path, offset: reads.append((path.name, offset)) or tail(path, offset),
    )
    path = root / "-proj" / "s.jsonl"
    path.parent.mkdir(parents=True)
    return path, reads


def append(path, *records):
    with path.open("a") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)


def test_latest_returns_one_document(tree):
    path, _ = tree
    append(path, record("m1"))
    doc = asyncio.run(burn.burn_latest(60, remote=False))
    assert [row["session"] for row in doc["sessions"]] == ["11111111-222"]


def test_documents_read_only_what_was_appended(tree):
    path, reads = tree
    append(path, record("m1"))

    async def run():
        found = []
        stream = burn.burn_documents(60, interval=0.01, remote=False)
        found.append(await anext(stream))
        append(path, record("m2", session="99999999-8888"))
        found.append(await anext(stream))
        await stream.aclose()
        return found

    first, second = asyncio.run(run())
    assert len(first["sessions"]) == 1
    assert len(second["sessions"]) == 2
    offsets = [offset for name, offset in reads if name == "s.jsonl"]
    assert offsets[0] == 0 and offsets[-1] > 0, "the second read must resume, not re-parse"


def test_documents_close_their_tailer_when_the_consumer_stops(tree, monkeypatch):
    closed = []
    monkeypatch.setattr(ingest.Tailer, "close", lambda self: closed.append(self))

    async def run():
        stream = burn.burn_documents(60, interval=0.01, remote=False)
        await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert len(closed) == 1


def test_documents_wait_for_quota_only_on_the_first_sample(monkeypatch):
    waits = []

    async def sample(self, window, settle=0.0):
        waits.append(settle)
        return Snapshot(at=1.0)

    monkeypatch.setattr(ingest.Tailer, "sample", sample)

    async def run():
        stream = burn.burn_documents(60, interval=0.0)
        for _ in range(3):
            await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert waits[0] > 0 and waits[1:] == [0.0, 0.0]


def test_the_public_surface_is_the_top_level_namespace():
    assert all(hasattr(burn, name) for name in burn.__all__)
    assert all(name.startswith("burn_") for name in burn.__all__ if name.islower())
