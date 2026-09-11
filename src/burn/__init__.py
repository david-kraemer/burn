"""Token burn inspection for Claude Code and Codex.

Both agents append JSONL transcripts to disk as they work, so nothing needs to
be instrumented. :mod:`burn.ingest` tails those files into an immutable
:class:`~burn.model.Snapshot`; everything else -- analysis, the drill-down
views, the live dashboard and the key handling -- is a pure function of one.
"""

from burn.model import Call, Gauge, Row, Snapshot, Tooling, Usage

__all__ = ["Call", "Gauge", "Row", "Snapshot", "Tooling", "Usage"]
__version__ = "0.2.0"
