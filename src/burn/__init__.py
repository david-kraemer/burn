"""Monitor token use in Claude Code and Codex.

The public surface lives here. Functions carry the ``burn_`` prefix; records
keep their class names. Submodules are implementation detail.

    import asyncio
    import burn

    doc = asyncio.run(burn.burn_latest(window=60))
"""

from .api import BURN_SCHEMA, burn_document, burn_documents, burn_latest
from .ingest import Tailer
from .model import Call, Gauge, Quota, Reading, Row, Snapshot, Spend, Tooling, Usage
from .state import View

__all__ = [
    "BURN_SCHEMA",
    "Call",
    "Gauge",
    "Quota",
    "Reading",
    "Row",
    "Snapshot",
    "Spend",
    "Tailer",
    "Tooling",
    "Usage",
    "View",
    "burn_document",
    "burn_documents",
    "burn_latest",
]
