"""Scalar formatting: numbers, durations and names, as strings."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from itertools import pairwise

SPARK = "▁▂▃▄▅▆▇█"


def quantity(value: float) -> str:
    """Format a token count with three significant figures."""
    tiers = ((1e9, "B"), (1e6, "M"), (1e3, "k"))
    for index, (limit, suffix) in enumerate(tiers):
        if abs(value) >= limit:
            scaled = round(value / limit, 1)
            # 999,999 can round to "1000.0" in its own tier; re-scale up.
            if abs(scaled) >= 1000 and index > 0:
                limit, suffix = tiers[index - 1]
                scaled = value / limit
            return f"{scaled:.1f}{suffix}"
    return f"{value:.0f}"


def span(seconds: float) -> str:
    """Format a duration as days, hours, minutes, or seconds."""
    seconds = max(seconds, 0)
    minutes = int(seconds // 60)
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{mins:02d}m"
    return f"{mins}m" if minutes else f"{int(seconds)}s"


def clock(epoch: float | None) -> str:
    """Format a Unix timestamp in local time."""
    return f"{datetime.fromtimestamp(epoch, UTC).astimezone():%H:%M}" if epoch is not None else "?"


def moment(text: str | None) -> float | None:
    """Parse an ISO 8601 timestamp, or return None."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def label(model: str) -> str:
    """Remove the vendor prefix and build date from a model name."""
    for prefix in ("claude-", "gpt-"):
        model = model.removeprefix(prefix)
    return re.sub(r"-\d{8}$", "", model)


def trim(text: str, width: int) -> str:
    """Clip text to ``width`` and add an ellipsis when needed."""
    if len(text) <= width:
        return text
    if width <= 0:
        return ""
    return text[: width - 1] + "…"


def short(identifier: str) -> str:
    """Shorten a session ID; it's a grouping key, so keep 12 chars for collision resistance."""
    return identifier[:12]


def tint(percent: float) -> str:
    """Return green, yellow, or red for a percentage."""
    return "red" if percent >= 85 else "yellow" if percent >= 60 else "green"


def sparkline(series: list[float], ceiling: float, floor: float = 0.0) -> str:
    """Render a scaled series as block characters."""
    if ceiling <= floor:
        return "·" * len(series)
    reach = ceiling - floor
    return "".join(
        SPARK[min(len(SPARK) - 1, max(0, int((v - floor) / reach * len(SPARK))))] if v else "·"
        for v in series
    )


def resample(series: list[float], width: int) -> list[float]:
    """Reduce a series to at most ``width`` points by averaging buckets."""
    if width <= 0 or len(series) <= width:
        return series
    step = len(series) / width
    # Last edge is len(series), not round(width * step), to avoid dropping
    # the series' final point to float drift.
    edges = [round(i * step) for i in range(width)] + [len(series)]
    return [sum(series[lo : max(hi, lo + 1)]) / max(hi - lo, 1) for lo, hi in pairwise(edges)]
