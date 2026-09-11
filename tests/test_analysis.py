"""Attribution, quota blocks, cache waste, and the billing cache."""

import json

from burn import analysis
from burn.analysis import (
    PROMPT,
    attribution,
    cached_rates,
    current_block,
    lanes,
    resumes,
    save_rates,
    tabulate,
    waste,
)
from burn.model import BLOCK, Snapshot, Usage


def growing(call, prefixes, session="s1", start=0.0, step=10.0, output=0):
    """A session whose context grows through the given prefixes."""
    return [
        call(at=start + i * step, session=session, usage=Usage(cache_read=p, output=output))
        for i, p in enumerate(prefixes)
    ]


# ----------------------------------------------------------------- attribution


def test_growth_with_no_tool_between_is_charged_to_the_prompt(call):
    blamed = attribution(growing(call, [100, 300]), [])
    assert [(b.name, b.added) for b in blamed] == [(PROMPT, 200)]


def test_growth_is_split_in_proportion_to_result_size(call, tool):
    calls = growing(call, [0, 300])
    tools = [tool(5, "Read", size=200), tool(6, "Grep", size=100)]
    added = {b.name: b.added for b in attribution(calls, tools)}
    assert added == {"Read": 200.0, "Grep": 100.0}


def test_the_assistants_own_output_is_not_blamed_on_a_tool(call, tool):
    # Growth is what entered the conversation, less what the model itself wrote.
    calls = growing(call, [0, 300], output=100)
    added = {b.name: b.added for b in attribution(calls, [tool(5, "Read")])}
    assert added == {"Read": 200.0}


def test_tools_tied_exactly_on_the_step_boundary_all_get_credit(call, tool):
    # Three results landing on the same timestamp as the call that read them
    # must each be credited, not collapsed onto whichever sorts first.
    calls = growing(call, [0, 100, 200], step=10)
    tools = [tool(10.0, name) for name in ("Read", "Grep", "Glob")]
    added = {b.name: b.added for b in attribution(calls, tools)}
    assert {"Read", "Grep", "Glob"} <= added.keys()
    assert sum(added[n] for n in ("Read", "Grep", "Glob")) == 100
    # The second step ran no tools at all, so its growth is the prompt's.
    assert added[PROMPT] == 100


def test_carried_exceeds_added_because_the_prefix_is_re_read(call, tool):
    calls = growing(call, [0, 100, 100, 100, 100])
    (blame,) = attribution(calls, [tool(5, "Read")])
    assert blame.added == 100
    # Written once at 1.25, then re-read by each of the three later calls at 0.1.
    assert blame.carried == 100 * (1.25 + 0.1 * 3)


def test_attribution_does_not_bill_a_compaction_as_fresh_growth(call, tool):
    # The jump back up to 400 is a separate thread, not 300 tokens of new input.
    calls = growing(call, [100, 400, 110, 420])
    added = sum(b.added for b in attribution(calls, [tool(5, "Read")]))
    assert added == 320  # 300 in one strand, 20 in the other


def test_attribution_is_empty_without_growth(call):
    assert attribution(growing(call, [100, 100]), []) == []


# ------------------------------------------------------------------- the block


def test_no_block_when_the_last_one_has_closed(call):
    assert current_block([call(at=0.0)], now=BLOCK + 1) is None


def test_a_block_starts_on_the_hour_of_its_first_request(call):
    start = current_block([call(at=7200 + 130)], now=7200 + 200)
    assert start == 7200


def test_five_idle_hours_open_a_new_block(call):
    calls = [call(at=3600.0), call(at=3600.0 + 2 * BLOCK + 5)]
    assert current_block(calls, now=3600.0 + 2 * BLOCK + 10) == 3600.0 + 2 * BLOCK


def test_no_calls_means_no_block():
    assert current_block([], now=100.0) is None


# -------------------------------------------------------------------- the table


def test_a_row_summarises_one_session(call):
    calls = [
        call(at=0.0, usage=Usage(cache_write=100, cache_read=900), thinking=5),
        call(at=10.0, usage=Usage(cache_write=0, cache_read=1000), thinking=7),
    ]
    (row,) = tabulate(Snapshot(at=10.0, calls=tuple(calls)))
    assert row.calls == 2
    assert row.think == 12
    assert row.ctx == 1000  # the latest prefix, not the sum
    assert round(row.cache) == 95


def test_the_filter_matches_session_project_model_and_agent(snapshot, call):
    calls = [call(session="aaa", project="burn"), call(session="bbb", project="spade")]
    assert len(tabulate(snapshot(calls=calls), "spade")) == 1
    assert len(tabulate(snapshot(calls=calls), "OPUS")) == 2
    assert tabulate(snapshot(calls=calls), "nothing") == []


def test_lanes_bucket_each_agent_separately(call):
    calls = [
        call(at=95.0, source="cc", usage=Usage(input=10)),
        call(at=99.0, source="cx", usage=Usage(input=20)),
    ]
    series = lanes(calls, now=100.0, seconds=10.0, width=10)
    assert series["cc"][5] == 10
    assert series["cx"][9] == 20


# ------------------------------------------------------------------- cache waste


def test_only_a_gap_longer_than_the_cache_life_counts_as_idle(call):
    calls = [
        call(at=0.0, usage=Usage(cache_read=10)),
        call(at=60.0, usage=Usage(cache_write=100)),   # in flow
        call(at=1000.0, usage=Usage(cache_write=500)),  # after a long gap
    ]
    summary = waste(calls, ttl=300)
    assert (summary.idle_tokens, summary.idle_events, summary.churn_tokens) == (500, 1, 100)
    assert summary.premium == 500 * (1.25 - 0.1)
    assert resumes(calls, ttl=300) == 1


def test_no_cache_creation_means_no_waste(call):
    assert waste([call(at=0.0), call(at=9999.0)], ttl=300).total == 0


# ----------------------------------------------------------------------- rates


def test_a_corrupt_rates_cache_is_ignored_not_fatal():
    analysis.RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    analysis.RATES_CACHE.write_text("{not json at all")
    assert cached_rates() is None


def test_a_well_formed_but_wrongly_shaped_cache_is_ignored():
    analysis.RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    analysis.RATES_CACHE.write_text(json.dumps({"model": "not a pair"}))
    assert cached_rates() is None


def test_rates_round_trip_through_the_cache():
    save_rates({"claude-opus-5": (5.35, 61)})
    assert cached_rates() == {"claude-opus-5": (5.35, 61)}


def test_an_unwritable_cache_directory_is_survivable(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "RATES_CACHE", tmp_path / "nope" / "x" / "rates.json")
    (tmp_path / "nope").write_text("i am a file, not a directory")
    save_rates({"m": (1.0, 1)})  # must not raise
    assert cached_rates() is None


def test_rates_are_solved_once_and_memoised(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis, "cached_rates", lambda: calls.append(1) or {"m": (2.0, 1)})
    assert analysis.rates() == {"m": (2.0, 1)}
    assert analysis.rates() == {"m": (2.0, 1)}
    assert len(calls) == 1
