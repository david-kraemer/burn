"""Parsing both transcript formats, and tailing files as they grow."""

import asyncio
import json
import threading
import time

import pytest

from burn.ingest import Carry, Tailer, _read_claude, _read_codex, _tail
from burn.model import Source


def claude_assistant(**overrides):
    record = {
        "type": "assistant",
        "timestamp": "2026-09-11T12:00:00Z",
        "sessionId": "session-one-two-three",
        "cwd": "/home/d/projects/burn",
        "requestId": "r1",
        "message": {
            "id": "m1",
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 40,
            },
            "content": [],
        },
    }
    record["message"].update(overrides.pop("message", {}))
    record.update(overrides)
    return record


def test_claude_call_carries_usage_project_and_session():
    fragment = _read_claude([claude_assistant()], Carry())
    (_, call), = fragment.calls
    assert call.usage.input == 10
    assert call.usage.cache_read == 30
    assert call.session == "session-one-"
    assert call.project == "burn"


def test_claude_merges_the_records_of_one_call():
    # One API call is written as several records, each repeating the usage.
    lines = [claude_assistant(), claude_assistant(message={"content": [{"type": "text"}]})]
    fragment = _read_claude(lines, Carry())
    assert len(fragment.calls) == 1


def test_claude_registers_tool_use_from_a_record_with_no_usage_of_its_own():
    # The later content-block records carry the tool_use blocks but no usage.
    # Dropping them loses every tool result that follows.
    carry = Carry()
    _read_claude(
        [
            claude_assistant(),
            claude_assistant(
                message={"usage": None, "content": [{"type": "tool_use", "id": "t1",
                                                     "name": "Bash"}]}
            ),
        ],
        carry,
    )
    result = {
        "type": "user",
        "timestamp": "2026-09-11T12:00:02Z",
        "sessionId": "session-one",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                 "content": "x" * 400}]},
    }
    fragment = _read_claude([result], carry)
    assert [t.name for t in fragment.tools] == ["Bash"]
    assert fragment.tools[0].size > 400


def test_claude_flags_fan_out():
    fragment = _read_claude(
        [claude_assistant(message={"content": [{"type": "tool_use", "id": "t", "name": "Task"}]})],
        Carry(),
    )
    assert fragment.fanout == ("cc|m1|r1",)


def test_claude_ignores_synthetic_and_unusable_records():
    assert _read_claude([claude_assistant(message={"model": "<synthetic>"})], Carry()).calls == ()
    assert _read_claude([claude_assistant(timestamp="not a date")], Carry()).calls == ()


def test_claude_attaches_the_latest_user_prompt():
    carry = Carry()
    _read_claude(
        [{"type": "user", "timestamp": "2026-09-11T11:59:00Z",
          "message": {"content": "why is this slow"}}],
        carry,
    )
    (_, call), = _read_claude([claude_assistant()], carry).calls
    assert call.prompt == "why is this slow"


def codex(kind, payload, at="2026-09-11T12:00:00Z"):
    return {"type": kind, "timestamp": at, "payload": payload}


def test_codex_call_inherits_model_and_project_from_earlier_headers():
    carry = Carry(session="s")
    _read_codex(
        [
            codex("session_meta", {"session_id": "abcdefghij", "cwd": "/home/d/spade"}),
            codex("turn_context", {"model": "gpt-5.6-luna"}),
        ],
        carry,
    )
    fragment = _read_codex(
        [codex("event_msg", {"type": "token_count", "info": {"last_token_usage": {
            "input_tokens": 100, "cached_input_tokens": 90,
            "cache_write_input_tokens": 5, "output_tokens": 7, "total_tokens": 107}}})],
        carry,
    )
    (_, call), = fragment.calls
    assert (call.model, call.project, call.source) == ("gpt-5.6-luna", "spade", Source.CODEX)
    # Codex reports cached tokens inside the input count. Burn splits them out.
    assert (call.usage.input, call.usage.cache_read) == (10, 90)


def test_codex_gauges_are_labelled_by_window_length_not_by_key():
    # For six weeks of one history the only gauge present was the seven-day
    # one, reported under the key "primary".
    fragment = _read_codex(
        [codex("event_msg", {"type": "token_count", "rate_limits": {
            "primary": {"used_percent": 12.0, "window_minutes": 10080}}})],
        Carry(),
    )
    assert [(g.window_minutes, g.used_percent) for g in fragment.gauges] == [(10080, 12.0)]


def test_codex_gauges_come_back_shortest_window_first():
    fragment = _read_codex(
        [codex("event_msg", {"type": "token_count", "rate_limits": {
            "primary": {"used_percent": 1.0, "window_minutes": 10080},
            "secondary": {"used_percent": 2.0, "window_minutes": 300}}})],
        Carry(),
    )
    assert [g.window_minutes for g in fragment.gauges] == [300, 10080]


def test_codex_pairs_tool_calls_with_their_output():
    carry = Carry(session="s")
    fragment = _read_codex(
        [
            codex("response_item", {"type": "function_call", "call_id": "c1", "name": "exec"}),
            codex("response_item", {"type": "function_call_output", "call_id": "c1",
                                    "output": "hello"}),
        ],
        carry,
    )
    assert [t.name for t in fragment.tools] == ["exec"]


def test_malformed_lines_are_skipped_not_fatal():
    from burn.ingest import _parse

    assert _parse(["{not json", "", json.dumps([1, 2])], "cc", Carry()).calls == ()


def test_tail_returns_only_whole_lines_and_resumes_where_it_stopped(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("one\ntwo\npar")
    lines, offset = _tail(path, 0)
    assert lines == ["one", "two"]

    path.write_text("one\ntwo\npartial\n")
    lines, offset = _tail(path, offset)
    assert lines == ["partial"]

    assert _tail(path, offset) == ([], offset)


def test_tail_restarts_when_a_file_shrinks(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("aaaa\nbbbb\n")
    _, offset = _tail(path, 0)
    path.write_text("cc\n")
    assert _tail(path, offset)[0] == ["cc"]


def test_tail_of_a_missing_file_is_silent(tmp_path):
    assert _tail(tmp_path / "gone.jsonl", 0) == ([], 0)


# ------------------------------------------------------------ quota polling
#
# Each test runs one Tailer with asyncio.run. Poll order affects the contract.
# Each test therefore uses an event loop.


def fake_reading(percent):
    from burn.model import Quota, Reading

    return Reading(at=0.0, windows=(Quota(name="5h", group="session", used_percent=percent),))


@pytest.fixture
def polls(monkeypatch):
    """Disable transcript reads and return scripted quota responses."""
    from burn import ingest

    monkeypatch.setattr(ingest, "_transcripts", lambda horizon: [])
    taken, answers = [], []

    def poll():
        taken.append(len(taken))
        return answers[len(taken) - 1]

    monkeypatch.setattr(ingest, "quota_reading", poll)
    return taken, answers


def sampled(tailer, times=1):
    """Sample and wait for quota data."""

    async def run():
        for _ in range(times):
            snapshot = await tailer.sample(300, settle=5.0)
        return snapshot

    return asyncio.run(run())


def test_first_sample_reports_quota(polls):
    taken, answers = polls
    answers.append(fake_reading(1.0))
    tailer = Tailer()
    assert sampled(tailer).reading.windows[0].used_percent == 1.0
    assert taken == [0]


def test_quota_is_not_polled_for_each_frame(polls):
    taken, answers = polls
    answers.extend(fake_reading(float(n)) for n in range(5))
    sampled(Tailer(), times=5)
    assert len(taken) == 1, "refreshes must not set the request interval"


def test_failed_poll_keeps_last_reading(polls, monkeypatch):
    from burn import ingest

    taken, answers = polls
    answers.extend([fake_reading(7.0), None])
    monkeypatch.setattr(ingest, "QUOTA_TTL", 0.0)
    tailer = Tailer()
    assert sampled(tailer, times=2).reading.windows[0].used_percent == 7.0
    assert len(taken) == 2


def test_poll_error_does_not_stop_sampling(polls, monkeypatch):
    from burn import ingest

    def explode():
        raise RuntimeError("network error")

    monkeypatch.setattr(ingest, "quota_reading", explode)
    assert sampled(Tailer()).reading is None


def test_no_remote_skips_quota_poll(polls):
    taken, answers = polls
    answers.append(fake_reading(1.0))
    assert sampled(Tailer(remote=False)).reading is None
    assert taken == []


def test_quota_wait_ends_at_deadline(polls, monkeypatch):
    """A stalled request must not block the terminal."""
    from burn import ingest

    release = threading.Event()
    monkeypatch.setattr(ingest, "quota_reading", lambda: release.wait(30))

    async def run():
        tailer = Tailer()
        started = time.monotonic()
        # Measure inside the loop. asyncio.run() also waits for the worker
        # thread. That wait is outside the sample deadline.
        snapshot = await tailer.sample(300, settle=0.05)
        waited = time.monotonic() - started
        release.set()
        return waited, snapshot

    waited, snapshot = asyncio.run(run())
    assert waited < 5
    assert snapshot.reading is None


def test_late_poll_updates_later_frame(polls, monkeypatch):
    from burn import ingest

    taken, answers = polls
    answers.append(fake_reading(3.0))
    release = threading.Event()
    reading = ingest.quota_reading
    monkeypatch.setattr(ingest, "quota_reading", lambda: release.wait(5) and reading())

    async def run():
        tailer = Tailer()
        missed = await tailer.sample(300, settle=0.05)
        release.set()
        return missed, await tailer.sample(300, settle=5.0)

    missed, arrived = asyncio.run(run())
    assert missed.reading is None
    assert arrived.reading.windows[0].used_percent == 3.0
    assert len(taken) == 1, "a missed deadline must not start a second request"


# ------------------------------------------------------- streamed usage, subagents


def test_claude_keeps_the_final_usage_of_a_call_written_as_several_records():
    # Subagent transcripts write the first content block with a partial output
    # count; only the call's last record carries the final usage.
    partial = claude_assistant(message={"usage": {"input_tokens": 2, "output_tokens": 1}})
    final = claude_assistant(message={"usage": {"input_tokens": 2, "output_tokens": 247}})
    (_, call), = _read_claude([partial, final], Carry()).calls
    assert call.usage.output == 247


def test_claude_marks_subagent_requests_as_sidechain():
    (_, call), = _read_claude([claude_assistant(isSidechain=True)], Carry()).calls
    assert call.sidechain
    (_, call), = _read_claude([claude_assistant()], Carry()).calls
    assert not call.sidechain


@pytest.fixture
def trees(monkeypatch, tmp_path):
    """Point the Tailer at an empty Claude tree laid out as Claude Code writes it."""
    from burn import ingest

    root = tmp_path / "claude"
    monkeypatch.setattr(
        ingest, "TREES", tuple((root, s, p) for _, s, p in ingest.TREES if s == Source.CLAUDE)
    )
    return root


def appended(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)


def recent(**overrides):
    """A Claude assistant record stamped now, so it falls inside any window."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return claude_assistant(timestamp=stamp, **overrides)


def test_usage_completed_in_a_later_read_replaces_the_partial_one(trees):
    path = trees / "-proj" / "session-one-two-three.jsonl"
    tailer = Tailer(remote=False)
    appended(path, recent(message={"usage": {"output_tokens": 1}}))
    asyncio.run(tailer.sample(60))
    appended(path, recent(message={"usage": {"output_tokens": 247}}))
    (call,) = asyncio.run(tailer.sample(60)).calls
    assert call.usage.output == 247


def test_subagent_transcripts_are_read_into_the_parent_session(trees):
    session = trees / "-proj" / "session-one-two-three"
    appended(session.with_suffix(".jsonl"), recent())
    appended(session / "subagents" / "agent-a1.jsonl",
             recent(requestId="r2", isSidechain=True, message={"id": "m2"}))
    appended(session / "subagents" / "workflows" / "wf_1" / "agent-a2.jsonl",
             recent(requestId="r3", isSidechain=True, message={"id": "m3"}))
    calls = asyncio.run(Tailer(remote=False).sample(60)).calls
    assert len(calls) == 3
    assert {c.session for c in calls} == {"session-one-"}
    assert sum(c.sidechain for c in calls) == 2


def token_count(at, last, total):
    return codex("event_msg", {"type": "token_count", "info": {
        "last_token_usage": last, "total_token_usage": total}}, at=at)


def test_codex_does_not_count_a_repeated_token_count_twice():
    # Codex re-emits token_count with an unchanged running total; the repeat
    # is not a new request.
    one = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    two = {"input_tokens": 300, "output_tokens": 30, "total_tokens": 330}
    events = [
        token_count("2026-11-03T20:51:49Z", one, one),
        token_count("2026-11-03T20:51:51Z", one, one),
        token_count("2026-11-03T20:52:30Z", two, {"input_tokens": 400, "output_tokens": 40,
                                                   "total_tokens": 440}),
    ]
    assert len(_read_codex(events, Carry(session="s")).calls) == 2


def test_codex_repeat_split_across_reads_is_still_one_request():
    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    carry = Carry(session="s")
    assert len(_read_codex([token_count("2026-11-03T20:51:49Z", usage, usage)], carry).calls) == 1
    assert _read_codex([token_count("2026-11-03T20:51:51Z", usage, usage)], carry).calls == ()
