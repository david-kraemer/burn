"""Rendering: every view must produce a frame from any snapshot, empty included."""

import pytest
from rich.console import Console

from burn import views
from burn.dashboard import dashboard, scrollbar
from burn.model import Gauge, Quota, Reading, Snapshot, Spend, Usage
from burn.state import HELP, TURNS, View, viewport


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
    lambda s: views.tools(s),
    lambda s: views.turns(s),
    lambda s: views.session(s, "aaa"),
    lambda s: views.session(s, None),
    lambda s: views.session(s, "nosuch"),
    lambda s: views.cost(s, 300),
    lambda s: views.waste(s),
    lambda s: dashboard(s, View()),
    lambda s: dashboard(s, View(mode=HELP)),
    lambda s: dashboard(s, View(zoomed=True, selected="cc|aaa|burn")),
    lambda s: dashboard(s, View(zoomed=True, panel=TURNS, selected="cc|aaa|burn")),
    lambda s: dashboard(s, View(source="cx")),
    lambda s: dashboard(s, View(needle="nothing matches")),
    lambda s: dashboard(s, View(limit=4e7)),
    lambda s: dashboard(s, View(paused=True)),
])
def test_every_view_renders_from_a_full_snapshot(render, busy):
    assert plain(render(busy)).strip()


@pytest.mark.parametrize("render", [
    lambda s: views.tools(s),
    lambda s: views.turns(s),
    lambda s: views.session(s, "aaa"),
    lambda s: views.waste(s),
    lambda s: views.cost(s, 300),
    lambda s: dashboard(s, View()),
])
def test_every_view_renders_from_an_empty_snapshot(render):
    assert plain(render(Snapshot(at=1000.0))) is not None


def test_the_dashboard_names_the_sessions_and_their_projects(busy):
    frame = plain(dashboard(busy, View()))
    assert "aaa" in frame and "burn" in frame and "2 sessions" in frame


def test_the_filter_narrows_the_table_and_is_shown_in_the_masthead(busy):
    frame = plain(dashboard(busy, View(needle="spade")))
    assert "/spade" in frame and "1 sessions" in frame and "aaa" not in frame


def test_the_agent_filter_drops_the_other_agents_sessions(busy):
    assert "bbb" not in plain(dashboard(busy, View(source="cc")))


def test_a_codex_gauge_is_labelled_by_its_window_length(busy):
    assert "Codex 5h00m" in plain(dashboard(busy, View()))


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
    frame = plain(dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    assert "Claude 5h" in frame and "2.0%" in frame and "8.0%" in frame


def test_scoped_window_names_the_model(busy, snapshot):
    assert "Fable" in plain(dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))


def test_governing_window_is_marked(busy, snapshot):
    assert "governing" in plain(dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))


def test_credits_show_usage_and_cap(busy, snapshot):
    frame = plain(dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    assert "$261.32 of $750.00" in frame


def test_reported_quota_replaces_local_estimate(busy, snapshot):
    """Do not show a local estimate with reported quota data."""
    reported = snapshot(calls=busy.calls, reading=REPORTED)
    assert "no --limit set" not in plain(dashboard(reported, View()))
    assert "no --limit set" in plain(dashboard(snapshot(calls=busy.calls), View()))


def test_empty_reading_uses_local_estimate(busy, snapshot):
    blank = snapshot(calls=busy.calls, reading=Reading(at=1000.0))
    assert "no --limit set" in plain(dashboard(blank, View()))


def test_the_sorted_column_is_marked_in_the_header(busy):
    assert "WEIGHT▼" in plain(dashboard(busy, View()))
    assert "CTX▲" in plain(dashboard(busy, View(sort="ctx", reverse=False)))


def test_a_fan_out_session_is_marked(snapshot, call):
    marked = snapshot(calls=[call(at=1000.0, session="aaa", fanout=True)])
    assert "aaa⑂" in plain(dashboard(marked, View()))


def test_the_zoom_pane_reports_a_session_with_nothing_in_the_window(busy):
    frame = plain(dashboard(busy, View(zoomed=True, selected="cc|aaa|burn", window=0)))
    assert "no session selected" in frame or "no calls in this window" in frame


@pytest.mark.parametrize("reading", [None, REPORTED])
def test_dashboard_fits_available_height(reading, busy, snapshot, call):
    """The layout accounts for the number of reported windows."""
    crowded = snapshot(
        calls=[call(at=1000.0, session=f"s{n:03d}", usage=Usage(input=n)) for n in range(60)],
        gauges=busy.gauges,
        reading=reading,
    )
    assert len(plain(dashboard(crowded, View()), height=24).splitlines()) <= 24


def test_the_scrollbar_says_what_is_off_screen():
    assert scrollbar(total=50, drawn=10, offset=20).plain == "  ▲ 20 above   ▼ 20 below"
    assert scrollbar(total=10, drawn=10, offset=0).plain == ""


def test_the_viewport_keeps_the_cursor_inside_itself():
    rows = list(range(100))
    shown, offset = viewport(rows, at=50, capacity=10)
    assert offset <= 50 < offset + len(shown)
    assert viewport(rows, at=0, capacity=10)[1] == 0
    assert viewport(rows, at=99, capacity=10)[1] == 90
    assert viewport(rows[:5], at=0, capacity=10) == (rows[:5], 0)


def test_pending_window_shows_no_value(busy, snapshot):
    from dataclasses import replace

    waiting = snapshot(calls=busy.calls, reading=replace(REPORTED, pending=True))
    frame = plain(dashboard(waiting, View()))
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
    from dataclasses import replace

    waiting = plain(dashboard(snapshot(calls=busy.calls, reading=replace(REPORTED, pending=True)),
                              View()))
    arrived = plain(dashboard(snapshot(calls=busy.calls, reading=REPORTED), View()))
    assert len(waiting.splitlines()) == len(arrived.splitlines())
    assert brackets(waiting) == brackets(arrived)


def test_pending_quota_does_not_use_local_estimate(busy, snapshot):
    """The pending quota keeps the meter layout fixed."""
    from dataclasses import replace

    waiting = snapshot(calls=busy.calls, reading=replace(REPORTED, pending=True))
    assert "no --limit set" not in plain(dashboard(waiting, View()))
