"""Read Claude quota windows from Anthropic.

Claude Code does not write quota records to transcript files. Its ``/usage``
view gets quota data from an account endpoint. This module uses the same
endpoint and OAuth token. Claude Code stores the token in the login keychain.
Read the token for each request. Do not refresh or replace the token.

The endpoint reports percentages. It does not report a token allowance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import ssl
import subprocess
import time
import urllib.error
import urllib.request

from .format import iso_epoch
from .model import Quota, Reading, Spend

__all__ = []

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
KEYCHAIN_SERVICE = "Claude Code-credentials"
OAUTH_BETA = "oauth-2025-04-20"

QUOTA_TIMEOUT = 4.0  # seconds; the live dashboard must not wait for the network

# Each ``burn --once`` starts a new process. An in-process interval would not
# limit requests across processes. The cache shares the interval and provides
# the window shape for the first frame.
CACHE = pathlib.Path.home() / ".cache" / "burn" / "quota.json"

# Request interval. The cache carries it across processes. The lease stops
# instances on the same interval from requesting together.
QUOTA_TTL = 120.0
# Data older than this interval keeps its window shape. It does not keep its values.
QUOTA_STALE = 900.0
# Maximum delay after a rate-limit response.
MAX_BACKOFF = 900.0

# Window names in shortest-first order. Put unknown names last.
ORDER = ("session", "weekly", "monthly")

NAMES = {"session": "5h", "weekly_all": "week", "weekly_scoped": "week"}


def quota_reading() -> Reading | None:
    """Return cached or reported quota data.

    Return the last window shape if a request fails. Mark its values as
    pending if the data is stale.
    """
    stored, blocked = _cached()
    now = time.time()
    # Draw current data and blocked data as they stand. TTL is shorter than
    # STALE. Current data is never marked pending.
    if not _due(stored, blocked, now) or not _leased():
        return _usable(stored)
    try:
        answer = _reported()
    finally:
        _release()
    return _recorded(answer, stored, now)


def quota_shape() -> Reading | None:
    """Return the cached window shape without a network request."""
    return _usable(_cached()[0])


@dataclasses.dataclass(frozen=True, slots=True)
class Response:
    """Result from one quota request."""

    payload: dict | None = None
    retry_after: float = 0.0


# -------------------------------------------------------------------- internals


def _due(stored: Reading | None, blocked: float, now: float) -> bool:
    """Return true if the data expired and the rate-limit delay ended."""
    return now >= blocked and (stored is None or now - stored.at >= QUOTA_TTL)


def _usable(stored: Reading | None) -> Reading | None:
    """Return current values, or the window shape if values are stale."""
    if stored is None:
        return None
    if time.time() - stored.at < QUOTA_STALE:
        return stored
    return dataclasses.replace(stored, pending=True)


def _recorded(answer: Response, stored: Reading | None, now: float) -> Reading | None:
    """Save one response and return the data to draw."""
    if answer.retry_after:
        _save(stored, retry_at=now + answer.retry_after)
        return _usable(stored)
    if answer.payload is None:
        return _usable(stored)
    taken = Reading(at=now, windows=_windows(answer.payload), spend=_spend(answer.payload))
    _save(taken)
    return taken


def _reported() -> Response:
    """Request quota data once. Return an empty response on failure.

    The keychain read and the request share one deadline. Quitting cannot
    cancel this work. The terminal waits for TIMEOUT, not for both timeouts.
    """
    deadline = time.monotonic() + QUOTA_TIMEOUT
    if (bearer := _token()) is None:
        return Response()
    request = urllib.request.Request(
        USAGE_URL,
        headers={"Authorization": f"Bearer {bearer}", "anthropic-beta": OAUTH_BETA},
    )
    if (left := deadline - time.monotonic()) <= 0.0:
        return Response()  # The keychain used the whole deadline.
    try:
        with urllib.request.urlopen(request, timeout=left, context=_trust()) as response:
            # The endpoint must report an object. Any other JSON value fails.
            return Response(payload=_answered(json.load(response)))
    except urllib.error.HTTPError as refused:
        # Several one-shot runs can exceed the endpoint limit. Respect the
        # requested wait time.
        return Response(retry_after=_patience(refused))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return Response()


def _answered(payload: object) -> dict | None:
    """Return the reported object. Return None for any other JSON value."""
    return payload if isinstance(payload, dict) else None


def _patience(refused: urllib.error.HTTPError) -> float:
    """Return the requested wait time within the allowed range."""
    if refused.code not in (429, 503):
        return 0.0
    try:
        asked = float((refused.headers or {}).get("Retry-After") or 0.0)
    except (TypeError, ValueError):
        asked = 0.0
    return min(max(asked, QUOTA_TTL), MAX_BACKOFF)


def _token() -> str | None:
    """Return the current OAuth token. Return None if it is unavailable or expired."""
    try:
        found = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=QUOTA_TIMEOUT,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        credentials = json.loads(found.stdout)["claudeAiOauth"]
    except (ValueError, KeyError, TypeError):
        return None
    # Claude Code replaces expired tokens in the keychain. Do not write tokens
    # from burn.
    if credentials.get("expiresAt", 0) / 1000 <= time.time():
        return None
    return credentials.get("accessToken")


def _trust() -> ssl.SSLContext | None:
    """Return a TLS context that verifies certificates.

    Use certifi if it is available. Otherwise, use the system trust store.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _windows(payload: dict) -> tuple[Quota, ...]:
    """Convert reported limits to windows. Sort them by duration."""
    found = [_quota(entry) for entry in payload.get("limits", ()) if isinstance(entry, dict)]
    rank = {name: index for index, name in enumerate(ORDER)}
    return tuple(sorted(found, key=lambda q: (rank.get(q.group, len(ORDER)), q.name)))


def _quota(entry: dict) -> Quota:
    """Convert one reported limit to a quota window."""
    kind = str(entry.get("kind") or "?")
    model = ((entry.get("scope") or {}).get("model") or {}).get("display_name")
    return Quota(
        name=NAMES.get(kind, kind),
        group=str(entry.get("group") or kind.split("_")[0]),
        used_percent=float(entry.get("percent") or 0.0),
        resets_at=iso_epoch(entry.get("resets_at")),
        scope=model,
        active=bool(entry.get("is_active")),
    )


def _spend(payload: dict) -> Spend | None:
    """Return extra-usage credits if they are enabled and valid."""
    block = payload.get("spend")
    if not isinstance(block, dict) or not block.get("enabled"):
        return None
    try:
        return Spend(used=_money(block["used"]), cap=_money(block["limit"]))
    except (KeyError, TypeError, ValueError):
        return None


def _money(amount: dict) -> float:
    """Convert a minor currency value to a major currency value."""
    return amount["amount_minor"] / 10 ** amount.get("exponent", 2)


def _cached() -> tuple[Reading | None, float]:
    """Return the last saved reading and the next request time."""
    try:
        stored = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return None, 0.0  # Request new data for missing or invalid data.
    if not isinstance(stored, dict):
        return None, 0.0
    blocked = float(stored.get("retry_at") or 0.0)
    if not stored.get("at"):
        return None, blocked  # A rate-limit response occurred before a reading.
    try:
        return Reading(
            at=float(stored["at"]),
            windows=tuple(Quota(**window) for window in stored["windows"]),
            spend=Spend(**stored["spend"]) if stored.get("spend") else None,
        ), blocked
    except (ValueError, TypeError, KeyError):
        return None, blocked


def _leased() -> bool:
    """Claim the right to make the next request.

    Instances started together can read the cache at the same moment. They
    can then all request data at every interval. Only one process can create
    the claim. A claim lasts longer than a request. A claim from a dead
    process expires.
    """
    lease = CACHE.with_suffix(".lease")
    try:
        lease.parent.mkdir(parents=True, exist_ok=True)
        if time.time() - lease.stat().st_mtime < QUOTA_TIMEOUT * 2:
            return False  # Another instance is part-way through a request.
        lease.unlink(missing_ok=True)  # A process died while holding the claim.
    except OSError:
        pass  # No claim exists, or the claim cannot be read.
    try:
        os.close(os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False  # Another instance created the claim first.
    except OSError:
        return True  # The claim is optional. Do not block the request.
    return True


def _release() -> None:
    """Remove the request claim."""
    with contextlib.suppress(OSError):
        CACHE.with_suffix(".lease").unlink(missing_ok=True)


def _save(taken: Reading | None, retry_at: float = 0.0) -> None:
    """Save a reading with a temporary file."""
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        scratch = CACHE.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps(_record(taken, retry_at)))
        scratch.replace(CACHE)
    except OSError:
        pass  # The cache is optional.


def _record(taken: Reading | None, retry_at: float = 0.0) -> dict:
    """Convert a reading to JSON-compatible values."""
    if taken is None:
        return {"at": None, "windows": [], "spend": None, "retry_at": retry_at}
    return {
        "at": taken.at,
        "windows": [dataclasses.asdict(window) for window in taken.windows],
        "spend": dataclasses.asdict(taken.spend) if taken.spend is not None else None,
        "retry_at": retry_at,
    }
