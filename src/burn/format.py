"""Scalar formatting: numbers, durations and names, as strings."""

from __future__ import annotations

import re
from datetime import UTC, datetime

SPARK = "▁▂▃▄▅▆▇█"


def quantity(value: float) -> str:
    """A token count at three significant figures: 1.2M, 940k, 37."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:.0f}"


def span(seconds: float) -> str:
    """A duration as 2d04h, 4h17m, 9m or 30s."""
    minutes = int(max(seconds, 0) // 60)
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{mins:02d}m"
    return f"{mins}m" if minutes else f"{int(seconds)}s"


def clock(epoch: float | None) -> str:
    """A unix timestamp as local wall-clock time."""
    return f"{datetime.fromtimestamp(epoch, UTC).astimezone():%H:%M}" if epoch else "?"


def moment(text: str | None) -> float | None:
    """An ISO-8601 timestamp as unix seconds, or None if unparseable."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def label(model: str) -> str:
    """Model name minus the vendor prefix and build date nobody reads."""
    for prefix in ("claude-", "gpt-"):
        model = model.removeprefix(prefix)
    return re.sub(r"-\d{8}$", "", model)


def trim(text: str, width: int) -> str:
    """Text clipped to width, with an ellipsis where it was cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


def short(identifier: str) -> str:
    """A session id at the length that still distinguishes them."""
    return identifier[:8]


def tint(percent: float) -> str:
    """Green below 60, yellow to 85, red above: the usual traffic light."""
    return "red" if percent >= 85 else "yellow" if percent >= 60 else "green"


def sparkline(series: list[float], ceiling: float, floor: float = 0.0) -> str:
    """A series as block characters, scaled between floor and ceiling."""
    if ceiling <= floor:
        return "·" * len(series)
    reach = ceiling - floor
    return "".join(
        SPARK[min(len(SPARK) - 1, max(0, int((v - floor) / reach * len(SPARK))))] if v else "·"
        for v in series
    )


def resample(series: list[float], width: int) -> list[float]:
    """A series squeezed to at most ``width`` points by averaging each bucket."""
    if width <= 0 or len(series) <= width:
        return series
    step = len(series) / width
    buckets = ((int(i * step), int((i + 1) * step)) for i in range(width))
    return [sum(series[lo : max(hi, lo + 1)]) / max(hi - lo, 1) for lo, hi in buckets]

