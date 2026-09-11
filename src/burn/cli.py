"""Command line: parse arguments, take one snapshot, render one view.

    burn                 live per-session monitor
    burn tools           which tools are inflating context
    burn turns           the most expensive prompts
    burn session <id>    one session in detail
    burn cost            spend, at rates calibrated from your own billing
    burn waste           cache re-creation you are paying for
    burn verify          audit against Claude's own totals
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable

from rich.console import Console, RenderableType

from burn import views
from burn.app import monitor
from burn.dashboard import dashboard
from burn.ingest import Tailer
from burn.model import Snapshot
from burn.state import SORTS, View

LIVE = "live"
VIEWS = (LIVE, "tools", "turns", "session", "cost", "waste", "verify")


def parser() -> argparse.ArgumentParser:
    spec = argparse.ArgumentParser(
        prog="burn",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    spec.add_argument("view", nargs="?", default=LIVE, choices=VIEWS)
    spec.add_argument("target", nargs="?", help="session id prefix, for the session view")
    spec.add_argument("--window", type=int, default=300, help="minutes of history (default 300)")
    spec.add_argument("--interval", type=float, default=2.0, help="live refresh seconds")
    spec.add_argument(
        "--limit",
        type=float,
        default=0.0,
        help="your 5h weighted-token allowance, to turn the Claude meter into a "
        "real gauge (accepts 40e6)",
    )
    spec.add_argument("--sort", default="weight", choices=[field for field, _ in SORTS])
    spec.add_argument("--once", action="store_true", help="one frame of the live view, then exit")
    spec.add_argument("--source", choices=["cc", "cx"], help="restrict to one agent")
    spec.add_argument("--top", type=int, default=15, help="rows in the drill-down views")
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
    if args.view == LIVE and not args.once:
        asyncio.run(monitor(Console(), view))
        return
    asyncio.run(report(args, view))


async def report(args: argparse.Namespace, view: View) -> None:
    """One snapshot, one frame, no loop."""
    console = Console()
    snapshot = await Tailer().sample(args.window)
    console.print(render(args, view, snapshot, console.size.height))


def render(
    args: argparse.Namespace, view: View, snapshot: Snapshot, height: int
) -> RenderableType:
    """The renderable for whichever view was asked for."""
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
