"""Parse Claude quota data. Do not estimate missing values."""

import contextlib
import io
import json
import subprocess
import sys
import time
import urllib.error
from dataclasses import replace

import pytest

from burn import quota
from burn.model import Quota, Reading, Spend

# The offline fixture replaces the module attribute. Keep the real function so
# the request path can be tested.
REPORTED = quota.reported


def opened(payload):
    """Return a urlopen-like context manager over a JSON body."""
    return contextlib.closing(io.BytesIO(json.dumps(payload).encode()))

PAYLOAD = {
    "limits": [
        {"kind": "weekly_all", "group": "weekly", "percent": 8, "severity": "normal",
         "resets_at": "2026-09-19T05:59:59+00:00", "scope": None, "is_active": True},
        {"kind": "session", "group": "session", "percent": 2, "severity": "normal",
         "resets_at": "2026-09-15T22:49:59+00:00", "scope": None, "is_active": False},
        {"kind": "weekly_scoped", "group": "weekly", "percent": 6, "severity": "normal",
         "resets_at": "2026-09-19T05:59:59+00:00",
         "scope": {"model": {"id": None, "display_name": "Fable"}}, "is_active": False},
    ],
    "spend": {
        "used": {"amount_minor": 26132, "currency": "USD", "exponent": 2},
        "limit": {"amount_minor": 75000, "currency": "USD", "exponent": 2},
        "enabled": True,
    },
}


def test_session_window_is_first():
    assert [w.name for w in quota.windows(PAYLOAD)] == ["5h", "week", "week"]


def test_scoped_window_has_model_name():
    scoped = [w for w in quota.windows(PAYLOAD) if w.scope]
    assert [(w.scope, w.used_percent) for w in scoped] == [("Fable", 6.0)]


def test_active_window_governs():
    assert [w.used_percent for w in quota.windows(PAYLOAD) if w.active] == [8.0]


def test_reset_time_is_parsed_as_epoch():
    session = quota.windows(PAYLOAD)[0]
    assert session.resets_at == pytest.approx(1789512599.0)


def test_unknown_window_keeps_name_and_is_last():
    payload = {"limits": [{"kind": "lunar", "group": "lunar", "percent": 3}, *PAYLOAD["limits"]]}
    assert [w.name for w in quota.windows(payload)] == ["5h", "week", "week", "lunar"]


def test_malformed_limits_yield_no_windows():
    assert quota.windows({}) == ()
    assert quota.windows({"limits": ["nonsense", None]}) == ()


def test_credits_use_major_currency_units():
    assert quota.spend(PAYLOAD) == Spend(used=261.32, cap=750.0)


def test_disabled_or_missing_credits_return_none():
    assert quota.spend({}) is None
    assert quota.spend({"spend": {**PAYLOAD["spend"], "enabled": False}}) is None
    assert quota.spend({"spend": {"enabled": True}}) is None


def test_zero_cap_gives_zero_percent():
    assert Spend(used=5.0, cap=0.0).used_percent == 0.0


# --------------------------------------------------------------------- token


def credentials(**fields):
    return json.dumps({"claudeAiOauth": {"accessToken": "sk-live", **fields}})


def keychain(monkeypatch, stdout):
    monkeypatch.setattr(
        quota.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr=""),
    )


def test_live_token_is_read_from_keychain(monkeypatch):
    keychain(monkeypatch, credentials(expiresAt=(time.time() + 3600) * 1000))
    assert quota.token() == "sk-live"


def test_expired_token_is_not_refreshed(monkeypatch):
    """Claude Code replaces expired keychain tokens. Do not write a token."""
    keychain(monkeypatch, credentials(expiresAt=(time.time() - 1) * 1000))
    assert quota.token() is None


def test_missing_or_invalid_keychain_item_returns_none(monkeypatch):
    keychain(monkeypatch, "not json")
    assert quota.token() is None
    keychain(monkeypatch, json.dumps({"other": {}}))
    assert quota.token() is None

    def absent(*args, **kwargs):
        raise subprocess.CalledProcessError(44, "security")

    monkeypatch.setattr(quota.subprocess, "run", absent)
    assert quota.token() is None


def test_missing_token_skips_request(monkeypatch):
    monkeypatch.setattr(quota, "token", lambda: None)
    monkeypatch.setattr(quota.urllib.request, "urlopen", lambda *a, **k: pytest.fail("polled"))
    assert REPORTED().payload is None


# ------------------------------------------------------------------ deadline


def test_keychain_delay_shortens_the_request(monkeypatch):
    """The keychain read and the request share one deadline.

    Quitting cannot cancel this work. Their sum would delay the terminal.
    """
    monkeypatch.setattr(quota, "TIMEOUT", 1.0)
    monkeypatch.setattr(quota, "token", lambda: time.sleep(0.4) or "sk-live")
    allowed = []
    monkeypatch.setattr(
        quota.urllib.request, "urlopen",
        lambda *a, timeout, **k: allowed.append(timeout) or opened(PAYLOAD),
    )
    assert REPORTED().payload == PAYLOAD
    assert allowed[0] < 0.7, "the request must not restart the deadline"


def test_spent_deadline_skips_the_request(monkeypatch):
    monkeypatch.setattr(quota, "TIMEOUT", 0.05)
    monkeypatch.setattr(quota, "token", lambda: time.sleep(0.1) or "sk-live")
    monkeypatch.setattr(
        quota.urllib.request, "urlopen", lambda *a, **k: pytest.fail("requested past deadline")
    )
    assert REPORTED() == quota.Response()


# --------------------------------------------------------- malformed answers


def test_only_objects_are_payloads():
    assert quota.answered({"limits": []}) == {"limits": []}
    assert quota.answered(["nonsense"]) is None
    assert quota.answered("nonsense") is None
    assert quota.answered(None) is None


def test_non_object_answer_leaves_no_reading(monkeypatch):
    """The parser reads an object. Do not pass a list to it."""
    monkeypatch.setattr(quota, "token", lambda: "sk-live")
    monkeypatch.setattr(quota.urllib.request, "urlopen", lambda *a, **k: opened(["nonsense"]))
    monkeypatch.setattr(quota, "reported", REPORTED)
    assert quota.reading() is None


def test_unreachable_endpoint_returns_no_quota(monkeypatch):
    monkeypatch.setattr(quota, "reported", lambda: quota.Response())
    assert quota.reading() is None


def test_reading_contains_windows_credits_and_timestamp(monkeypatch):
    monkeypatch.setattr(quota, "reported", lambda: quota.Response(PAYLOAD))
    taken = quota.reading()
    assert isinstance(taken, Reading)
    assert taken.spend == Spend(261.32, 750.0)
    assert all(isinstance(w, Quota) for w in taken.windows)
    assert taken.at == pytest.approx(time.time(), abs=5)


# --------------------------------------------------------------------- cache


@pytest.fixture
def polled(monkeypatch):
    """Count requests and return the sample payload."""
    made = []

    def reported():
        made.append(len(made))
        return quota.Response(PAYLOAD)

    monkeypatch.setattr(quota, "reported", reported)
    return made


def test_reading_survives_cache_round_trip(polled):
    taken = quota.reading()
    assert quota.cached() == (taken, 0.0)


def test_fresh_cache_skips_request(polled):
    quota.reading()
    quota.reading()
    assert len(polled) == 1, "a second process must not request in this interval"


def test_stale_cache_is_refreshed(polled, monkeypatch):
    quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    quota.reading()
    assert len(polled) == 2


def test_unreachable_endpoint_keeps_current_reading(polled, monkeypatch):
    """A failed request does not remove the current reading."""
    taken = quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    monkeypatch.setattr(quota, "reported", lambda: quota.Response())
    assert quota.reading() == taken


def test_stale_reading_keeps_windows_but_not_values(
    polled, monkeypatch
):
    """Keep the layout. Do not show stale percentages as current values."""
    taken = quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    monkeypatch.setattr(quota, "STALE", 0.0)
    monkeypatch.setattr(quota, "reported", lambda: quota.Response())
    unread = quota.reading()
    assert unread.pending
    assert unread.windows == taken.windows
    assert replace(unread, pending=False) == taken


def test_corrupt_cache_returns_no_reading(polled):
    quota.reading()
    quota.CACHE.write_text("{ not json")
    assert quota.cached() == (None, 0.0)
    quota.CACHE.write_text(json.dumps({"at": 1.0, "windows": [{"bogus": 1}]}))
    assert quota.cached() == (None, 0.0)


def test_unwritable_cache_does_not_break_reading(polled, monkeypatch, tmp_path):
    """The cache is optional. A failed write must not remove the reading."""
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not the cache directory")
    monkeypatch.setattr(quota, "CACHE", blocked / "quota.json")
    assert quota.reading() is not None
    assert quota.cached() == (None, 0.0)


def test_missing_cache_and_endpoint_return_none(monkeypatch):
    monkeypatch.setattr(quota, "reported", lambda: quota.Response())
    assert quota.reading() is None


# ---------------------------------------------------------------- rate limits


def refusal(code, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(quota.USAGE_URL, code, "refused", headers, None)


def test_rate_limit_wait_uses_retry_after():
    assert quota.patience(refusal(429, "161")) == 161.0


def test_rate_limit_without_valid_delay_uses_ttl():
    assert quota.patience(refusal(429)) == quota.TTL
    assert quota.patience(refusal(429, "soon")) == quota.TTL


def test_rate_limit_delay_is_capped():
    assert quota.patience(refusal(429, "999999")) == quota.MAX_BACKOFF


def test_other_http_errors_do_not_delay_request():
    assert quota.patience(refusal(401)) == 0.0
    assert quota.patience(refusal(500)) == 0.0


def test_rate_limit_delays_next_process(polled, monkeypatch):
    """The cache applies the delay to later one-shot processes."""
    taken = quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    monkeypatch.setattr(quota, "reported", lambda: quota.Response(retry_after=300.0))
    assert quota.reading() == taken

    asked = []
    monkeypatch.setattr(quota, "reported", lambda: asked.append(1) or quota.Response(PAYLOAD))
    assert quota.reading() == taken
    assert asked == [], "burn must apply the requested delay to later processes"


def test_rate_limit_wait_expires(polled, monkeypatch):
    quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    monkeypatch.setattr(quota, "reported", lambda: quota.Response(retry_after=-1.0))
    quota.reading()
    monkeypatch.setattr(quota, "reported", lambda: quota.Response(PAYLOAD))
    assert quota.reading().windows, "an expired delay must allow a request"


def test_rate_limit_without_cache_delays_next_run(monkeypatch):
    monkeypatch.setattr(quota, "reported", lambda: quota.Response(retry_after=300.0))
    assert quota.reading() is None
    monkeypatch.setattr(quota, "reported", lambda: pytest.fail("request made during delay"))
    assert quota.reading() is None


# --------------------------------------------------------------------- claim


def test_claim_blocks_a_concurrent_request(polled, monkeypatch):
    """Instances started together can reach the request. The claim allows one request."""
    taken = quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    assert quota.leased(), "the claim must be free after a completed request"
    monkeypatch.setattr(quota, "reported", lambda: pytest.fail("requested under a claim"))
    assert quota.reading() == taken
    quota.release()


def test_expired_claim_is_reclaimed(polled, monkeypatch):
    """A dead process must not block later instances."""
    quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)
    assert quota.leased()
    monkeypatch.setattr(quota, "TIMEOUT", 0.0)  # Every claim is older than a request.
    assert quota.reading().windows


def test_failed_request_drops_claim(polled, monkeypatch):
    monkeypatch.setattr(quota, "reported", lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        quota.reading()
    assert quota.leased(), "a raising request must still drop the claim"


WORKER = '''
"""One burn process. Request quota data when all sibling processes are ready."""
import os, pathlib, sys, time
from burn import quota

cache, gate, wanted = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
quota.CACHE = cache

def reported():
    (gate / f"request.{os.getpid()}").write_text("")
    time.sleep(0.3)  # A real request takes time. The claim must outlast it.
    return quota.Response({"limits": [{"kind": "session", "group": "session", "percent": 5}]})

quota.reported = reported
(gate / f"ready.{os.getpid()}").write_text("")
while len(list(gate.glob("ready.*"))) < wanted:
    time.sleep(0.005)  # Make the processes cross the cache check together.
quota.reading()
'''


def test_claim_limits_simultaneous_processes(tmp_path):
    """Separate processes can make separate calls. The claim limits them to one request."""
    gate = tmp_path / "gate"
    gate.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    argv = [sys.executable, str(worker), str(tmp_path / "quota.json"), str(gate), "6"]
    running = [subprocess.Popen(argv) for _ in range(6)]
    assert [instance.wait(30) for instance in running] == [0] * 6
    made = list(gate.glob("request.*"))
    assert len(made) == 1, f"{len(made)} of 6 processes requested at once"
    assert not (tmp_path / "quota.lease").exists(), "the claim must not outlive the request"


def test_unusable_cache_does_not_block_request(polled, monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not the cache directory")
    monkeypatch.setattr(quota, "CACHE", blocked / "quota.json")
    assert quota.leased() and quota.leased()


# --------------------------------------------------------------------- shape


def test_cached_shape_is_available_before_request(polled, monkeypatch):
    """The first frame can draw rows without waiting for the network."""
    taken = quota.reading()
    monkeypatch.setattr(quota, "reported", lambda: pytest.fail("shape requested from network"))
    monkeypatch.setattr(quota, "STALE", 0.0)
    drawn = quota.shape()
    assert drawn.pending
    assert [w.name for w in drawn.windows] == [w.name for w in taken.windows]


def test_current_shape_carries_values(polled, monkeypatch):
    quota.reading()
    monkeypatch.setattr(quota, "TTL", 0.0)  # Refresh due, but values are still current.
    monkeypatch.setattr(quota, "reported", lambda: pytest.fail("shape requested from network"))
    assert not quota.shape().pending


def test_no_shape_exists_before_first_reading():
    assert quota.shape() is None
