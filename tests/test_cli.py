"""Argument parsing and view dispatch."""

import pytest
from rich.console import Console

from burn.cli import VIEWS, main, parser, render
from burn.model import Gauge, Snapshot, Usage
from burn.state import View


def plain(renderable):
    console = Console(width=110, height=40, force_terminal=False, no_color=True)
    with console.capture() as captured:
        console.print(renderable)
    return captured.get()


@pytest.fixture
def busy(snapshot, call, tool):
    return snapshot(
        at=1000.0,
        calls=[
            call(at=950.0, session="aaa", project="burn", usage=Usage(10, 20, 300, 40),
                 prompt="ask"),
            call(at=960.0, session="bbb", project="spade", source="cx",
                 usage=Usage(5, 0, 90, 1)),
        ],
        tools=[tool(955.0, "Bash", session="aaa")],
        gauges=[Gauge(12.0, 10080, None)],
    )


def test_the_default_view_is_live():
    assert parser().parse_args([]).view == "live"


def test_scientific_notation_is_accepted_for_the_limit():
    assert parser().parse_args(["--limit", "40e6"]).limit == 4e7


def test_an_unknown_view_is_rejected():
    with pytest.raises(SystemExit):
        parser().parse_args(["nonsense"])


@pytest.mark.parametrize("name", [v for v in VIEWS if v != "verify"])
def test_every_view_dispatches_and_renders(name, busy, monkeypatch):
    args = parser().parse_args([name, "aaa"] if name == "session" else [name])
    assert plain(render(args, View(), busy, 40)) is not None


def test_verify_reads_the_transcript_tree_not_the_snapshot(monkeypatch):
    from burn import analysis, views

    monkeypatch.setattr(analysis, "audit", lambda: [])
    monkeypatch.setattr(views.analysis, "audit", lambda: [])
    args = parser().parse_args(["verify"])
    assert "No closed sessions" in plain(render(args, View(), Snapshot(), 40))


def test_the_source_flag_narrows_the_one_shot_views(busy):
    args = parser().parse_args(["turns", "--source", "cx"])
    assert "aaa" not in plain(render(args, View(source="cx"), busy, 40))


def test_once_prints_a_frame_and_returns(monkeypatch, capsys):
    from burn import cli

    async def sample(self, window, settle=0.0):
        return Snapshot(at=1000.0)

    monkeypatch.setattr(cli.Tailer, "sample", sample)
    main(["--once"])
    assert "burn" in capsys.readouterr().out


def test_only_live_view_requests_quota(monkeypatch):
    from burn import cli

    asked = []

    async def sample(self, window, settle=0.0):
        asked.append((self._remote, settle > 0))
        return Snapshot(at=1000.0)

    monkeypatch.setattr(cli.Tailer, "sample", sample)
    main(["--once"])
    main(["--once", "--no-remote"])
    main(["waste"])
    assert asked == [(True, True), (False, False), (False, False)]
