"""Parse arguments and render one view.

    burn                 live session monitor
    burn tools           context growth by tool
    burn turns           most expensive prompts
    burn session <id>    session detail
    burn cost            estimated spend
    burn waste           cache re-creation
    burn verify          compare observed and billed Claude tokens
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable

from rich.console import Console, RenderableType

from . import quota, views
from .app import monitor
from .dashboard import dashboard
from .ingest import Tailer
from .model import Snapshot
from .state import SORTS, View

LIVE = "live"
VIEWS = (LIVE, "tools", "turns", "session", "cost", "waste", "verify")


def positive(kind: type, text: str, noun: str) -> int | float:
    """Parse a flag value, rejecting zero or negative numbers."""
    value = kind(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive {noun}")
    return value


def positive_minutes(text: str) -> int:
    """Parse ``--window``: a non-positive value silently renders an empty, future-cutoff view."""
    return positive(int, text, "number of minutes")


def positive_seconds(text: str) -> float:
    """Parse ``--interval``: a non-positive value spins the live loop as fast as it can."""
    return positive(float, text, "number of seconds")


def positive_count(text: str) -> int:
    """Parse ``--top``: negative is valid Python slicing but not the intended "show top N"."""
    return positive(int, text, "count")


def non_negative_limit(text: str) -> float:
    """Parse ``--limit``: 0.0 means "unset" (falsy); negative is truthy and must be rejected."""
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def parser() -> argparse.ArgumentParser:
    spec = argparse.ArgumentParser(
        prog="burn",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    spec.add_argument("view", nargs="?", default=LIVE, choices=VIEWS)
    spec.add_argument("target", nargs="?", help="session ID prefix")
    spec.add_argument(
        "--window", type=positive_minutes, default=300, help="history in minutes (default: 300)"
    )
    spec.add_argument(
        "--interval", type=positive_seconds, default=2.0, help="refresh interval in seconds"
    )
    spec.add_argument(
        "--limit",
        type=non_negative_limit,
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
        "--top", type=positive_count, default=15, help="rows in detail views (default: 15)"
    )
    return spec


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    view = View(
        window=args.window,
        interval=args.interval,
        limit=args.limit,
        sort=args.sort,
        source=args.source,
    )
    try:
        if args.view == LIVE and not args.once:
            asyncio.run(monitor(Console(), view, remote=not args.no_remote))
            return
        asyncio.run(report(args, view))
    except KeyboardInterrupt:
        # cbreak mode leaves ISIG enabled, so Ctrl-C arrives as SIGINT, not
        # as keys.QUIT's "\x03" byte. Quit exactly as "q" would.
        pass


async def report(args: argparse.Namespace, view: View) -> None:
    """Render one snapshot and exit."""
    console = Console()
    # The live view shows quota meters. It is the only view that requests quota
    # data.
    remote = args.view == LIVE and not args.no_remote
    # A one-shot view has no later frame. If remote quota is enabled, wait for
    # the request.
    settle = quota.TIMEOUT if remote else 0.0
    snapshot = await Tailer(remote=remote).sample(args.window, settle=settle)
    console.print(render(args, view, snapshot, console.size.height))


def render(
    args: argparse.Namespace, view: View, snapshot: Snapshot, height: int
) -> RenderableType:
    """Return the requested view."""
    if args.view == LIVE:
        return dashboard(snapshot, view, height)
    if args.view == "verify":
        return views.audit()
    narrowed = snapshot.since(view.seconds).from_agent(view.source)
    chosen: dict[str, Callable[[], RenderableType]] = {
        "tools": lambda: views.tools(narrowed, args.top),
        "turns": lambda: views.turns(narrowed, args.top),
        "session": lambda: views.session(narrowed, args.target),
        "cost": lambda: views.cost(narrowed, args.window),
        "waste": lambda: views.waste(narrowed),
    }
    return chosen[args.view]()


if __name__ == "__main__":
    main()
