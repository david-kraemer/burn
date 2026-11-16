"""Scalar formatting: numbers, durations and names, as strings."""

from __future__ import annotations

import datetime
import itertools
import re
import time

__all__ = []

SPARK = "▁▂▃▄▅▆▇█"

# Widest seconds_span output: "10d03h". Wider spans push the row out by one.
SPAN_WIDTH = 6
COUNTDOWN_WIDTH = len("in ") + SPAN_WIDTH

# Heat bands, each a percentage floor and its color, cool to hot. Yellow
# starts at 60 and red at 85, the old warning thresholds; the extra bands
# split the calm range and the red zone.
HEAT = (
    (0, "#5e60ce"),  # indigo
    (15, "#4895ef"),  # blue
    (30, "#4cc9f0"),  # cyan
    (45, "#52b788"),  # green
    (60, "#ffd60a"),  # yellow
    (72, "#ff9f1c"),  # orange
    (85, "#e63946"),  # red
    (95, "bold #ff006e"),  # hot pink
)


def number_quantity(value: float) -> str:
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


def seconds_span(seconds: float) -> str:
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


def epoch_deadline(epoch: float | None) -> str:
    """Format a reset time in full, local, at a fixed width: ``Sat Nov 21 01:00 EST``.

    Always dated, even today: a weekly reset is days away, and rows that
    switch between bare and dated forms do not line up. The day is
    space-padded and the clock is 24-hour, zero-padded, so stacked rows align.
    """
    if epoch is None:
        # Same width as a real reading; the zone name sets it, so ask the zone.
        return "?".ljust(len(epoch_deadline(time.time())))
    local = datetime.datetime.fromtimestamp(epoch, datetime.UTC).astimezone()
    return f"{epoch_stamp(epoch)} {local:%Z}"


def epoch_stamp(epoch: float) -> str:
    """Format a past or future time, local, dated, at a fixed width: ``Sat Nov 21 01:00``.

    The zone-free form of :func:`epoch_deadline`, for columns of local times.
    """
    local = datetime.datetime.fromtimestamp(epoch, datetime.UTC).astimezone()
    return f"{local:%a %b} {local.day:>2} {local:%H:%M}"


def epoch_countdown(epoch: float | None, now: float) -> str:
    """Format the time until ``epoch`` at a fixed width: ``in  4d12h`` or ``  3m ago``."""
    if epoch is None:
        return "?".ljust(COUNTDOWN_WIDTH)
    if epoch >= now:
        return f"in {seconds_span(epoch - now):>{SPAN_WIDTH}}"
    return f"{seconds_span(now - epoch):>{SPAN_WIDTH}} ago"


def iso_epoch(text: str | None) -> float | None:
    """Parse an ISO 8601 timestamp, or return None."""
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def model_label(model: str) -> str:
    """Remove the vendor prefix and build date from a model name."""
    for prefix in ("claude-", "gpt-"):
        model = model.removeprefix(prefix)
    return re.sub(r"-\d{8}$", "", model)


def text_trim(text: str, width: int) -> str:
    """Clip text to ``width`` and add an ellipsis when needed."""
    if len(text) <= width:
        return text
    if width <= 0:
        return ""
    return text[: width - 1] + "…"


def id_short(identifier: str) -> str:
    """Shorten a session ID; it's a grouping key, so keep 12 chars for collision resistance."""
    return identifier[:12]


def percent_tint(percent: float) -> str:
    """Return the heat band color for a percentage: cool at 0, hot at 100."""
    tint = HEAT[0][1]
    for floor, color in HEAT:
        if percent >= floor:
            tint = color
    return tint


def series_sparkline(series: list[float], ceiling: float, floor: float = 0.0) -> str:
    """Render a scaled series as block characters."""
    if ceiling <= floor:
        return "·" * len(series)
    reach = ceiling - floor
    return "".join(
        SPARK[min(len(SPARK) - 1, max(0, int((v - floor) / reach * len(SPARK))))] if v else "·"
        for v in series
    )


def series_resample(series: list[float], width: int) -> list[float]:
    """Reduce a series to at most ``width`` points by averaging buckets."""
    if width <= 0 or len(series) <= width:
        return series
    step = len(series) / width
    # Last edge is len(series), not round(width * step), to avoid dropping
    # the series' final point to float drift.
    edges = [round(i * step) for i in range(width)] + [len(series)]
    return [
        sum(series[lo : max(hi, lo + 1)]) / max(hi - lo, 1) for lo, hi in itertools.pairwise(edges)
    ]
