"""Argument parsing and view dispatch."""

import pytest
from rich.console import Console

from burn.cli import VIEWS, _parser, _render, main
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
    assert _parser().parse_args([]).view == "live"


def test_scientific_notation_is_accepted_for_the_limit():
    assert _parser().parse_args(["--limit", "40e6"]).limit == 4e7


def test_an_unknown_view_is_rejected():
    with pytest.raises(SystemExit):
        _parser().parse_args(["nonsense"])


@pytest.mark.parametrize("name", [v for v in VIEWS if v != "verify"])
def test_every_view_dispatches_and_renders(name, busy, monkeypatch):
    args = _parser().parse_args([name, "aaa"] if name == "session" else [name])
    assert plain(_render(args, View(), busy, 40)) is not None


def test_verify_reads_the_transcript_tree_not_the_snapshot(monkeypatch):
    from burn import analysis, views

    monkeypatch.setattr(analysis, "transcripts_audit", lambda: [])
    monkeypatch.setattr(views, "transcripts_audit", lambda: [])
    args = _parser().parse_args(["verify"])
    assert "No closed sessions" in plain(_render(args, View(), Snapshot(), 40))


def test_the_source_flag_narrows_the_one_shot_views(busy):
    args = _parser().parse_args(["turns", "--source", "cx"])
    assert "aaa" not in plain(_render(args, View(source="cx"), busy, 40))


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


def test_json_document_lists_one_row_per_session(busy):
    from burn.api import burn_document

    doc = burn_document(busy, View(window=60))
    assert {row["session"] for row in doc["sessions"]} == {"aaa", "bbb"}
    assert doc["quota"] == [] and doc["spend"] is None


# --------------------------------------------------------- programmatic output


def test_follow_without_json_is_rejected():
    with pytest.raises(SystemExit):
        main(["--follow"])


def test_json_with_a_detail_view_is_rejected():
    with pytest.raises(SystemExit):
        main(["tools", "--json"])


def test_json_keeps_the_sort_flag(monkeypatch, capsys, busy):
    import json

    from burn import cli

    async def sample(self, window, settle=0.0):
        return busy

    monkeypatch.setattr(cli.Tailer, "sample", sample)
    main(["--json", "--no-remote", "--sort", "session", "--window", "60"])
    doc = json.loads(capsys.readouterr().out)
    assert [row["session"] for row in doc["sessions"]] == ["bbb", "aaa"]
    assert doc["window_minutes"] == 60


def test_follow_prints_one_compact_document_per_line(monkeypatch, capsys, busy):
    import json

    from burn import burn_document, cli

    async def documents(window, **kwargs):
        for _ in range(3):
            yield burn_document(busy, View(window=window))

    monkeypatch.setattr(cli, "burn_documents", documents)
    main(["--json", "--follow", "--no-remote"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and all(" " not in line for line in lines)
    assert all(json.loads(line)["sessions"] for line in lines)


def follower(tmp_path):
    """A real ``burn --json --follow`` process over an empty home directory."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "HOME": str(tmp_path)}
    return subprocess.Popen(
        [sys.executable, "-m", "burn", "--json", "--follow", "--no-remote", "--interval", "0.05"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )


def test_follow_exits_quietly_when_the_reader_goes_away(tmp_path):
    process = follower(tmp_path)
    assert process.stdout.readline().startswith(b"{")
    process.stdout.close()  # `| head -1`
    assert process.wait(10) == 0
    assert process.stderr.read() == b""


def test_follow_exits_quietly_on_interrupt(tmp_path):
    import signal

    process = follower(tmp_path)
    assert process.stdout.readline().startswith(b"{")
    process.send_signal(signal.SIGINT)
    assert process.wait(10) == 0
    assert b"Traceback" not in process.stderr.read()


# ---------------------------------------------------------------- plain text


@pytest.mark.parametrize("name", ["tools", "turns", "session", "cost", "waste"])
def test_piped_views_print_plain_ascii(name, monkeypatch, capsys, busy):
    from burn import cli

    async def sample(self, window, settle=0.0):
        return busy

    monkeypatch.setattr(cli.Tailer, "sample", sample)
    main([name, "aaa"] if name == "session" else [name])
    out = capsys.readouterr().out
    assert out.strip() and out.isascii() and "\x1b" not in out
    assert all(line == line.rstrip() for line in out.splitlines())


def test_plain_rendering_maps_glyphs_one_for_one():
    from rich.text import Text

    from burn.cli import _plain as flat

    assert flat(Text("██░░ ▁▂▇ · ▸ → …")) == "##.. _.* - > > ~\n"


def test_a_piped_live_view_prints_one_frame_instead_of_taking_the_screen(monkeypatch, capsys):
    from burn import cli

    async def sample(self, window, settle=0.0):
        return Snapshot(at=1000.0)

    monkeypatch.setattr(cli.Tailer, "sample", sample)
    monkeypatch.setattr(cli, "console_monitor", lambda *a, **k: pytest.fail("took the screen"))
    main(["--no-remote"])
    assert capsys.readouterr().out.startswith("burn ")
