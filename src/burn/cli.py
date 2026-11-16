"""Parse arguments and render one view.

    burn                 live session monitor
    burn tools           context growth by tool
    burn turns           most expensive prompts
    burn session <id>    session detail
    burn cost            estimated spend
    burn waste           cache re-creation
    burn verify          compare observed and billed Claude tokens
    burn --json          one snapshot as JSON: quota windows and per-session rows
    burn --json --follow one compact JSON document per line, every --interval

One-shot views print plain ASCII text, without colour, when stdout is not a
terminal, or with --plain.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
from collections.abc import Callable

from rich.console import Console, RenderableType

from .api import burn_documents, burn_latest
from .app import console_monitor
from .dashboard import snapshot_dashboard
from .ingest import Tailer
from .model import Snapshot
from .quota import QUOTA_TIMEOUT
from .state import SORTS, View
from .views import (
    report_audit,
    report_cost,
    report_session,
    report_tools,
    report_turns,
    report_waste,
)

LIVE = "live"
VIEWS = (LIVE, "tools", "turns", "session", "cost", "waste", "verify")

# Plain output: no screen to fit, so draw every row, and lines as wide as needed.
PLAIN_WIDTH = 200
PLAIN_HEIGHT = 10_000

# One ASCII character per glyph, so rendered columns keep their alignment.
ASCII = str.maketrans({
    "▁": "_", "▂": ".", "▃": ":", "▄": "~", "▅": "=", "▆": "+", "▇": "*", "█": "#",
    "░": ".", "·": "-", "▸": ">", "⑂": "*", "▲": "^", "▼": "v", "▏": "|",
    "→": ">", "↑": "^", "↓": "v", "↵": "<", "…": "~", "—": "-",
})


def main(argv: list[str] | None = None) -> None:
    spec = _parser()
    args = spec.parse_args(argv)
    if args.follow and not args.json:
        spec.error("--follow needs --json")
    if args.json and args.view != LIVE:
        spec.error(f"--json prints the live view's data; it does not apply to {args.view!r}")
    view = View(
        window=args.window,
        interval=args.interval,
        limit=args.limit,
        sort=args.sort,
        source=args.source,
    )
    try:
        if args.json:
            asyncio.run(_follow(args) if args.follow else _dump(args))
            return
        # The live screen needs a terminal; piped, it prints one frame.
        if args.view == LIVE and not args.once and not _plain_wanted(args):
            asyncio.run(console_monitor(Console(), view, remote=not args.no_remote))
            return
        asyncio.run(_report(args, view))
    except KeyboardInterrupt:
        # cbreak mode leaves ISIG enabled, so Ctrl-C arrives as SIGINT, not
        # as keys.QUIT's "\x03" byte. Quit exactly as "q" would.
        pass
    except BrokenPipeError:
        # The reader went away (`burn --json --follow | head -1`). Point stdout
        # at /dev/null so the interpreter's final flush cannot raise again.
        with contextlib.suppress(OSError, ValueError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------- internals


def _positive(kind: type, text: str, noun: str) -> int | float:
    """Parse a flag value, rejecting zero or negative numbers."""
    value = kind(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive {noun}")
    return value


def _positive_minutes(text: str) -> int:
    """Parse ``--window``: a non-positive value silently renders an empty, future-cutoff view."""
    return _positive(int, text, "number of minutes")


def _positive_seconds(text: str) -> float:
    """Parse ``--interval``: a non-positive value spins the live loop as fast as it can."""
    return _positive(float, text, "number of seconds")


def _positive_count(text: str) -> int:
    """Parse ``--top``: negative is valid Python slicing but not the intended "show top N"."""
    return _positive(int, text, "count")


def _non_negative_limit(text: str) -> float:
    """Parse ``--limit``: 0.0 means "unset" (falsy); negative is truthy and must be rejected."""
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def _parser() -> argparse.ArgumentParser:
    spec = argparse.ArgumentParser(
        prog="burn",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    spec.add_argument("view", nargs="?", default=LIVE, choices=VIEWS)
    spec.add_argument("target", nargs="?", help="session ID prefix")
    spec.add_argument(
        "--window", type=_positive_minutes, default=300, help="history in minutes (default: 300)"
    )
    spec.add_argument(
        "--interval", type=_positive_seconds, default=2.0, help="refresh interval in seconds"
    )
    spec.add_argument(
        "--limit",
        type=_non_negative_limit,
        default=0.0,
        help="five-hour Claude allowance in weighted tokens (accepts 40e6)",
    )
    spec.add_argument("--sort", default="weight", choices=[field for field, _ in SORTS])
    spec.add_argument("--once", action="store_true", help="print one live-view frame and exit")
    spec.add_argument(
        "--no-remote",
        action="store_true",
        help="read local transcripts; skip the quota request",
    )
    spec.add_argument("--source", choices=["cc", "cx"], help="show one agent only")
    spec.add_argument(
        "--json", action="store_true", help="print one snapshot as JSON for other programs"
    )
    spec.add_argument(
        "--follow",
        action="store_true",
        help="with --json: print a document per line every --interval seconds until stopped",
    )
    spec.add_argument(
        "--plain",
        action="store_true",
        help="print ASCII without colour (the default when stdout is not a terminal)",
    )
    spec.add_argument(
        "--top", type=_positive_count, default=15, help="rows in detail views (default: 15)"
    )
    return spec


def _plain_wanted(args: argparse.Namespace) -> bool:
    """Whether to print plain text: asked for, or stdout is a pipe or file."""
    return args.plain or not sys.stdout.isatty()


async def _report(args: argparse.Namespace, view: View) -> None:
    """Render one snapshot and exit."""
    # The live view shows quota meters. It is the only view that requests quota
    # data.
    remote = args.view == LIVE and not args.no_remote
    # A one-shot view has no later frame. If remote quota is enabled, wait for
    # the request.
    settle = QUOTA_TIMEOUT if remote else 0.0
    snapshot = await Tailer(remote=remote).sample(args.window, settle=settle)
    if _plain_wanted(args):
        sys.stdout.write(_plain(_render(args, view, snapshot, PLAIN_HEIGHT)))
        return
    console = Console()
    console.print(_render(args, view, snapshot, console.size.height))


def _plain(renderable: RenderableType) -> str:
    """Render as ASCII text with no escape codes and no trailing blanks."""
    console = Console(
        file=io.StringIO(), width=PLAIN_WIDTH, color_system=None, force_terminal=False,
        highlight=False, emoji=False,
    )
    with console.capture() as captured:
        console.print(renderable)
    lines = captured.get().translate(ASCII).splitlines()
    return "".join(line.rstrip() + "\n" for line in lines)


async def _dump(args: argparse.Namespace) -> None:
    """Print quota windows and per-session rows as one JSON document."""
    doc = await burn_latest(
        args.window, source=args.source, sort=args.sort, remote=not args.no_remote
    )
    print(json.dumps(doc))


async def _follow(args: argparse.Namespace) -> None:
    """Print one compact JSON document per line, flushed, until stopped or the pipe closes."""
    async for doc in burn_documents(
        args.window,
        interval=args.interval,
        source=args.source,
        sort=args.sort,
        remote=not args.no_remote,
    ):
        sys.stdout.write(json.dumps(doc, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def _render(
    args: argparse.Namespace, view: View, snapshot: Snapshot, height: int
) -> RenderableType:
    """Return the requested view."""
    if args.view == LIVE:
        return snapshot_dashboard(snapshot, view, height)
    if args.view == "verify":
        return report_audit()
    narrowed = snapshot.since(view.seconds).from_agent(view.source)
    chosen: dict[str, Callable[[], RenderableType]] = {
        "tools": lambda: report_tools(narrowed, args.top),
        "turns": lambda: report_turns(narrowed, args.top),
        "session": lambda: report_session(narrowed, args.target),
        "cost": lambda: report_cost(narrowed, args.window),
        "waste": lambda: report_waste(narrowed),
    }
    return chosen[args.view]()
