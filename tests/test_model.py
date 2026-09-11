"""Weights, snapshot narrowing, and the thread-splitting heuristic."""

from burn.model import Snapshot, Usage, compactions, threads


def test_weight_prices_each_token_class():
    assert Usage(input=100).weight == 100
    assert Usage(cache_write=100).weight == 125
    assert Usage(cache_read=100).weight == 10
    assert Usage(output=100).weight == 500


def test_prefix_is_everything_fed_in_and_excludes_output():
    usage = Usage(input=1, cache_write=2, cache_read=3, output=4)
    assert usage.prefix == 6


def test_usage_adds_classwise():
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)


def test_since_narrows_calls_and_tools(snapshot, call, tool):
    full = snapshot(
        calls=[call(at=100), call(at=900)], tools=[tool(100, "Read"), tool(900, "Bash")]
    )
    recent = full.since(200)
    assert [c.at for c in recent.calls] == [900]
    assert [t.name for t in recent.tools] == ["Bash"]


def test_from_agent_keeps_the_tools_of_the_sessions_it_keeps(snapshot, call, tool):
    full = snapshot(
        calls=[call(session="a", source="cc"), call(session="b", source="cx")],
        tools=[tool(0, "Read", session="a"), tool(0, "exec", session="b")],
    )
    assert [t.name for t in full.from_agent("cc").tools] == ["Read"]
    assert full.from_agent(None) is full


def test_threads_separate_interleaved_conversations(call):
    # Codex runs side threads under one session id: read as one conversation,
    # every switch back up looks like a huge injection of fresh context.
    calls = [call(usage=Usage(cache_read=n)) for n in (100, 160, 110, 170)]
    strands = threads(calls)
    assert [[c.prefix for c in strand] for strand in strands] == [[100, 160, 170], [110]]


def test_threads_start_a_new_strand_after_a_compaction(call):
    calls = [call(usage=Usage(cache_read=n)) for n in (100, 200, 20, 40)]
    assert [[c.prefix for c in s] for s in threads(calls)] == [[100, 200], [20, 40]]


def test_compactions_counts_collapses(call):
    calls = [call(usage=Usage(cache_read=n)) for n in (100, 200, 20, 40)]
    assert compactions(calls) == 1


def test_empty_snapshot_is_usable():
    assert Snapshot().calls == ()
