import json

import pytest

from burn.model import Call, Snapshot, Tooling, Usage


@pytest.fixture
def call():
    """A Call factory. Every field defaults except the one under test."""

    def make(at=0.0, session="s1", *, source="cc", project="p", model="claude-opus-5",
             usage=None, **kwargs):
        return Call(at=at, source=source, session=session, project=project, model=model,
                    usage=usage or Usage(), **kwargs)

    return make


@pytest.fixture
def snapshot(call):
    def make(calls=(), tools=(), at=1000.0, gauges=()):
        return Snapshot(at=at, calls=tuple(calls), tools=tuple(tools), gauges=tuple(gauges))

    return make


@pytest.fixture
def tool():
    def make(at, name, session="s1", size=100):
        return Tooling(at=at, session=session, name=name, size=size)

    return make


@pytest.fixture
def jsonl():
    """Render records as the transcript lines a parser reads."""
    return lambda *records: [json.dumps(r) for r in records]


@pytest.fixture(autouse=True)
def free_rates(monkeypatch, tmp_path):
    """Keep every test off the real billing cache and the real transcript tree."""
    from burn import analysis

    analysis._rates_within.cache_clear()
    monkeypatch.setattr(analysis, "RATES_CACHE", tmp_path / "rates.json")
    monkeypatch.setattr(analysis, "CLAUDE_ROOT", tmp_path / "empty")
    yield
    analysis._rates_within.cache_clear()
