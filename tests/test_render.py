"""Rendering: every view must produce a frame from any snapshot, empty included."""

import pytest
from rich.console import Console

from burn import views
from burn.dashboard import _scrollbar, snapshot_dashboard
from burn.model import Gauge, Quota, Reading, Snapshot, Spend, Usage
from burn.state import HELP, TURNS, View, table_viewport


def plain(renderable, width=120, height=40):
    console = Console(width=width, height=height, force_terminal=False, no_color=True)
    with console.capture() as captured:
        console.print(renderable)
    return captured.get()


@pytest.fixture
def busy(snapshot, call, tool):
    calls = [
        call(at=900.0, session="aaa", project="burn", usage=Usage(100, 200, 3000, 40),
             thinking=7, cache_1h=200, prompt="why is this slow"),
        call(at=950.0, session="aaa", project="burn", usage=Usage(0, 0, 3400, 60),
             prompt="why is this slow"),
        call(at=960.0, session="bbb", project="spade", source="cx",
             usage=Usage(50, 0, 900, 10)),
    ]
    return snapshot(
        calls=calls,
        tools=[tool(920.0, "Bash", session="aaa", size=800)],
        gauges=[Gauge(used_percent=17.0, window_minutes=300, resets_at=1789145166)],
    )


@pytest.mark.parametrize("render", [
    lambda s: views.report_tools(s),
    lambda s: views.report_turns(s),
    lambda s: views.report_session(s, "aaa"),
    lambda s: views.report_session(s, None),
    lambda s: views.report_session(s, "nosuch"),
    lambda s: views.report_cost(s, 300),
    lambda s: views.report_waste(s),
    lambda s: snapshot_dashboard(s, View()),
    lambda s: snapshot_dashboard(s, View(mode=HELP)),
    lambda s: snapshot_dashboard(s, View(zoomed=True, selected="cc|aaa|burn")),
    lambda s: snapshot_dashboard(s, View(zoomed=True, panel=TURNS, selected="cc|aaa|burn")),
    lambda s: snapshot_dashboard(s, View(source="cx")),
    lambda s: snapshot_dashboard(s, View(needle="nothing matches")),
    lambda s: snapshot_dashboard(s, View(limit=4e7)),
    lambda s: snapshot_dashboard(s, View(paused=True)),
])
def test_every_view_renders_from_a_full_snapshot(render, busy):
    assert plain(render(busy)).strip()


@pytest.mark.parametrize("render", [
    lambda s: views.report_tools(s),
    lambda s: views.report_turns(s),
    lambda s: views.report_session(s, "aaa"),
    lambda s: views.report_waste(s),
    lambda s: views.report_cost(s, 300),
    lambda s: snapshot_dashboard(s, View()),
])
def test_every_view_renders_from_an_empty_snapshot(render):
    assert plain(render(Snapshot(at=1000.0))) is not None


def test_the_dashboard_names_the_sessions_and_their_projects(busy):
    frame = plain(snapshot_dashboard(busy, View()))
    assert "aaa" in frame and "burn" in frame and "2 sessions" in frame


def test_the_filter_narrows_the_table_and_is_shown_in_the_masthead(busy):
    frame = plain(snapshot_dashboard(busy, View(needle="spade")))
    assert "/spade" in frame and "1 sessions" in frame and "aaa" not in frame


def test_the_agent_filter_drops_the_other_agents_sessions(busy):
    assert "bbb" not in plain(snapshot_dashboard(busy, View(source="cc")))


def test_a_codex_gauge_is_labelled_by_its_window_length(busy):
    assert "Codex 5h00m" in plain(snapshot_dashboard(busy, View()))


REPORTED = Reading(
    at=1000.0,
    windows=(
        Quota(name="5h", group="session", used_percent=2.0, resets_at=1789512599),
        Quota(name="week", group="weekly", used_percent=8.0, resets_at=1789797599, active=True),
        Quota(name="week", group="weekly", used_percent=6.0, resets_at=1789797599, scope="Fable"),
    ),
    spend=Spend(used=261.32, cap=750.0),
)


def test_reported_window_shows_its_percentage(busy, snapshot):
    frame = plain(snapshot_dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    assert "Claude 5h" in frame and "2.0%" in frame and "8.0%" in frame


def test_scoped_window_names_the_model(busy, snapshot):
    drawn = snapshot_dashboard(snapshot(calls=busy.calls, reading=REPORTED), View())
    assert "Fable" in plain(drawn)


def test_governing_window_is_marked(busy, snapshot):
    drawn = snapshot_dashboard(snapshot(calls=busy.calls, reading=REPORTED), View())
    assert "governing" in plain(drawn)


def test_credits_show_usage_and_cap(busy, snapshot):
    frame = plain(snapshot_dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    assert "$261.32 of $750.00" in frame


def test_reported_quota_replaces_local_estimate(busy, snapshot):
    """Do not show a local estimate with reported quota data."""
    reported = snapshot(calls=busy.calls, reading=REPORTED)
    assert "no --limit set" not in plain(snapshot_dashboard(reported, View()))
    assert "no --limit set" in plain(snapshot_dashboard(snapshot(calls=busy.calls), View()))


def test_empty_reading_uses_local_estimate(busy, snapshot):
    blank = snapshot(calls=busy.calls, reading=Reading(at=1000.0))
    assert "no --limit set" in plain(snapshot_dashboard(blank, View()))


def test_the_sorted_column_is_marked_in_the_header(busy):
    assert "WEIGHT▼" in plain(snapshot_dashboard(busy, View()))
    assert "CTX▲" in plain(snapshot_dashboard(busy, View(sort="ctx", reverse=False)))


def test_a_fan_out_session_is_marked(snapshot, call):
    marked = snapshot(calls=[call(at=1000.0, session="aaa", fanout=True)])
    assert "aaa⑂" in plain(snapshot_dashboard(marked, View()))


def test_the_zoom_pane_reports_a_session_with_nothing_in_the_window(busy):
    frame = plain(snapshot_dashboard(busy, View(zoomed=True, selected="cc|aaa|burn", window=0)))
    assert "no session selected" in frame or "no calls in this window" in frame


@pytest.mark.parametrize("reading", [None, REPORTED])
def test_dashboard_fits_available_height(reading, busy, snapshot, call):
    """The layout accounts for the number of reported windows."""
    crowded = snapshot(
        calls=[call(at=1000.0, session=f"s{n:03d}", usage=Usage(input=n)) for n in range(60)],
        gauges=busy.gauges,
        reading=reading,
    )
    assert len(plain(snapshot_dashboard(crowded, View()), height=24).splitlines()) <= 24


def test_the_scrollbar_says_what_is_off_screen():
    assert _scrollbar(total=50, drawn=10, offset=20).plain == "  ▲ 20 above   ▼ 20 below"
    assert _scrollbar(total=10, drawn=10, offset=0).plain == ""


def test_the_viewport_keeps_the_cursor_inside_itself():
    rows = list(range(100))
    shown, offset = table_viewport(rows, at=50, capacity=10)
    assert offset <= 50 < offset + len(shown)
    assert table_viewport(rows, at=0, capacity=10)[1] == 0
    assert table_viewport(rows, at=99, capacity=10)[1] == 90
    assert table_viewport(rows[:5], at=0, capacity=10) == (rows[:5], 0)


def test_pending_window_shows_no_value(busy, snapshot):
    import dataclasses

    waiting = snapshot(calls=busy.calls, reading=dataclasses.replace(REPORTED, pending=True))
    frame = plain(snapshot_dashboard(waiting, View()))
    assert "Claude 5h" in frame, "draw the row before its value is known"
    assert "2.0%" not in frame and "$261.32" not in frame


def brackets(frame):
    """Return each meter bar's line and column positions."""
    return [
        (n, line.index("["), line.index("]"))
        for n, line in enumerate(frame.splitlines())
        if "[" in line and "]" in line
    ]


def test_values_do_not_change_meter_positions(busy, snapshot):
    """Meter rows and columns stay fixed while values load."""
    import dataclasses

    pending = snapshot(calls=busy.calls, reading=dataclasses.replace(REPORTED, pending=True))
    waiting = plain(snapshot_dashboard(pending, View()))
    arrived = plain(snapshot_dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    assert len(waiting.splitlines()) == len(arrived.splitlines())
    assert brackets(waiting) == brackets(arrived)


def test_pending_quota_does_not_use_local_estimate(busy, snapshot):
    """The pending quota keeps the meter layout fixed."""
    import dataclasses

    waiting = snapshot(calls=busy.calls, reading=dataclasses.replace(REPORTED, pending=True))
    assert "no --limit set" not in plain(snapshot_dashboard(waiting, View()))


# ---------------------------------------------------------------- reset times


def local(*fields):
    """Epoch of a wall-clock time in the local zone."""
    import datetime

    return datetime.datetime(*fields).timestamp()


def test_a_reset_today_still_carries_its_date():
    from burn.format import epoch_deadline

    assert epoch_deadline(local(2026, 11, 16, 15, 30)).startswith("Mon Nov 16  3:30 PM ")


def test_reset_times_have_one_width():
    from burn.format import epoch_deadline

    times = [local(2026, 11, 2, 1, 5), local(2026, 11, 21, 12, 59), local(2026, 11, 16, 9, 0)]
    assert len({len(epoch_deadline(t)) for t in [*times, None]}) == 1


def test_countdowns_have_one_width():
    from burn.format import COUNTDOWN_WIDTH, epoch_countdown

    now = local(2026, 11, 16, 9, 0)
    shown = [epoch_countdown(now + s, now) for s in (30, 45 * 60, 3 * 3600, 4 * 86400 + 12 * 3600)]
    assert shown[-1] == "in  4d12h"
    assert {len(text) for text in shown} == {COUNTDOWN_WIDTH}
    assert epoch_countdown(now - 180, now) == "    3m ago"


def test_a_weekly_quota_meter_says_which_day_it_resets(busy, snapshot):
    now = local(2026, 11, 16, 9, 0)
    reading = Reading(at=now, windows=(
        Quota(name="5h", group="session", used_percent=2.0, resets_at=now + 3600),
        Quota(name="week", group="weekly", used_percent=8.0, resets_at=now + 3 * 86400),
    ))
    frame = plain(snapshot_dashboard(snapshot(at=now, calls=busy.calls, reading=reading), View()))
    (five,) = [line for line in frame.splitlines() if "Claude 5h" in line]
    (week,) = [line for line in frame.splitlines() if "Claude week" in line]
    assert "Mon Nov 16 10:00 AM" in five and "in  1h00m" in five
    assert "Thu Nov 19  9:00 AM" in week and "in  3d00h" in week
    assert five.index("resets") == week.index("resets")


def test_a_codex_gauge_resetting_on_another_day_carries_its_date(snapshot):
    now = local(2026, 11, 16, 9, 0)
    gauge = Gauge(used_percent=5.0, window_minutes=10080, resets_at=now + 2 * 86400)
    assert "Wed Nov 18" in plain(snapshot_dashboard(snapshot(at=now, gauges=[gauge]), View()))


def test_scope_and_governing_trail_the_reset_columns(busy, snapshot):
    frame = plain(snapshot_dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    rows = [line for line in frame.splitlines() if line.startswith("Claude ") and "resets" in line]
    assert len({line.index("resets") for line in rows}) == 1
    assert len({line.index(" in ") for line in rows}) == 1
    assert all(line.index("Fable") > line.index(" in ") for line in rows if "Fable" in line)
