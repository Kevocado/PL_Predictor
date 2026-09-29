"""Guard refusals are terminal: a blocked fetch must fail fast, not retry.

The network guard (`tests/conftest.py::_block_network`) exists so the offline
suite fails fast and legibly. Every `_fetch_with_retry` wrapper in
`pl_predictor/data/` that catches broad `Exception` risks treating the guard's
refusal as a transient network error: 3 attempts with 2s+4s sleeps (~6s per
blocked fetch) and a final `Failed after N attempts` error that misattributes
a policy refusal to an upstream outage.

These tests pin the contract, per retry wrapper:

1. a guard refusal is attempted exactly once, with zero sleeps, and the
   guard's own error (why + remedy) propagates unwrapped;
2. a genuine transient error still retries with the documented backoff;
3. the refusal type is a `RuntimeError` subclass, so every existing
   degrade-on-failure path (`except RuntimeError` in `load_xg_data`,
   `_load_season_*`, `fetch_current_season_partial`, ...) keeps working.

Attempt counts are observed, not inferred: each driver counts real calls to
the fetch function and records real `time.sleep` calls per module. A test
that cannot tell "attempted once" from "attempted 3x" is decoration.
"""
from __future__ import annotations

import importlib

import pytest
import requests

from pl_predictor.data.network_blocked import NetworkBlockedError

EXPECTED_SLEEPS = [2.0, 4.0]  # backoff * (attempt + 1), attempts=3, backoff=2.0


def _record_sleeps(monkeypatch, module_path):
    """Record every sleep the wrapper under test asks for. Returns the list."""
    mod = importlib.import_module(module_path)
    sleeps: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _blocked_fetch():
    """Inject the refusal even when cache warming disables the socket guard.

    The real socket guard is covered separately in test_network_guard.py.
    Retry behavior must be testable without making an outbound connection.
    """
    raise NetworkBlockedError(
        "outbound network blocked in tests; warm the cache with PL_ALLOW_NETWORK=1"
    )


def _flaky_fetch(state, sentinel):
    """A genuinely transient fetch: fails twice, then succeeds."""

    def _fetch(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return sentinel

    return _fetch


def _drive_fn_wrapper(monkeypatch, module_path, fetch):
    """Drive `clubelo`/`pulselive`/`understat_shots`-shaped `_fetch_with_retry(fn)`."""
    mod = importlib.import_module(module_path)
    sleeps = _record_sleeps(monkeypatch, module_path)
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return fetch(*args, **kwargs)

    return mod, sleeps, calls, counting


def _drive_scraper_wrapper(monkeypatch, module_path, attr, fetch):
    """Drive `understat`/`football_data`-shaped `_fetch_with_retry(season)`.

    These wrappers construct their penaltyblog scraper internally, so the
    observed fetch is injected by replacing the scraper class with a fake
    whose method delegates to `fetch` (and counts constructions too).
    """
    import penaltyblog as pb

    mod = importlib.import_module(module_path)
    sleeps = _record_sleeps(monkeypatch, module_path)
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return fetch(*args, **kwargs)

    class _FakeScraper:
        def __init__(self, *args, **kwargs):
            pass

        def get_fixtures(self):
            return counting()

    monkeypatch.setattr(pb.scrapers, attr, _FakeScraper)
    return mod, sleeps, calls


FN_WRAPPERS = [
    "pl_predictor.data.clubelo",
    "pl_predictor.data.pulselive",
    "pl_predictor.data.understat_shots",
]
SCRAPER_WRAPPERS = [
    ("pl_predictor.data.understat", "Understat"),
    ("pl_predictor.data.football_data", "FootballData"),
]
ALL_WRAPPER_PATHS = FN_WRAPPERS + [path for path, _ in SCRAPER_WRAPPERS]


def _terminal_case(module_path, monkeypatch):
    if module_path in FN_WRAPPERS:
        mod, sleeps, calls, counting = _drive_fn_wrapper(monkeypatch, module_path, _blocked_fetch)
        with pytest.raises(NetworkBlockedError) as excinfo:
            mod._fetch_with_retry(counting)
    else:
        attr = dict(SCRAPER_WRAPPERS)[module_path]
        mod, sleeps, calls = _drive_scraper_wrapper(monkeypatch, module_path, attr, _blocked_fetch)
        with pytest.raises(NetworkBlockedError) as excinfo:
            mod._fetch_with_retry("2023")
    return sleeps, calls, excinfo


@pytest.mark.parametrize("module_path", ALL_WRAPPER_PATHS)
def test_guard_refusal_is_attempted_exactly_once(module_path, monkeypatch):
    """The guard's refusal is a known state, not a transient failure.

    Would fail pre-fix as attempts == 3 with sleeps == [2.0, 4.0]: the
    `except Exception` retry loop treats the refusal like a flaky upstream.
    """
    sleeps, calls, _excinfo = _terminal_case(module_path, monkeypatch)
    assert calls["n"] == 1, (
        f"{module_path} retried a guard refusal {calls['n']}x -- "
        "a blocked network never resolves by waiting"
    )
    assert sleeps == [], (
        f"{module_path} slept {sleeps} on a guard refusal -- "
        "the guard's whole job is to fail fast"
    )


@pytest.mark.parametrize("module_path", ALL_WRAPPER_PATHS)
def test_guard_refusal_propagates_unwrapped_and_legible(module_path, monkeypatch):
    """The test sees the guard's why + remedy, not a `Failed after N` flake.

    Would fail pre-fix with `RuntimeError: Failed after 3 attempts`: the
    wrapper's final raise misattributes a policy refusal to an upstream
    outage -- exactly the defect class this repo keeps recording.
    """
    _, _, excinfo = _terminal_case(module_path, monkeypatch)
    message = str(excinfo.value)
    assert "outbound network blocked" in message, f"guard reason lost: {message!r}"
    assert "PL_ALLOW_NETWORK=1" in message, f"remedy lost: {message!r}"
    assert "Failed after" not in message, f"refusal misattributed as flake: {message!r}"
    assert type(excinfo.value).__name__ == "NetworkBlockedError", (
        f"refusal type is {type(excinfo.value).__name__}, not the dedicated "
        "NetworkBlockedError -- a bare RuntimeError cannot be told apart "
        "from a genuine fetch failure without message-sniffing"
    )


def _retry_case(module_path, monkeypatch):
    sentinel = object()
    state = {"calls": 0}
    fetch = _flaky_fetch(state, sentinel)
    if module_path in FN_WRAPPERS:
        mod, sleeps, _calls, counting = _drive_fn_wrapper(monkeypatch, module_path, fetch)
        result = mod._fetch_with_retry(counting)
    else:
        attr = dict(SCRAPER_WRAPPERS)[module_path]
        mod, sleeps, _calls = _drive_scraper_wrapper(monkeypatch, module_path, attr, fetch)
        result = mod._fetch_with_retry("2023")
    return result, sentinel, state, sleeps


@pytest.mark.parametrize("module_path", ALL_WRAPPER_PATHS)
def test_genuine_transient_errors_still_retry(module_path, monkeypatch):
    """Making refusals terminal must not disarm the retry loop for real flakes."""
    result, sentinel, state, sleeps = _retry_case(module_path, monkeypatch)
    assert result is sentinel
    assert state["calls"] == 3, f"expected 3 attempts, saw {state['calls']}"
    assert sleeps == EXPECTED_SLEEPS, f"expected backoff {EXPECTED_SLEEPS}, saw {sleeps}"


def test_network_blocked_error_is_a_runtime_error():
    """Degrade-on-failure paths catch `RuntimeError`: the refusal must reach them."""
    from pl_predictor.data.network_blocked import NetworkBlockedError

    assert issubclass(NetworkBlockedError, RuntimeError), (
        "NetworkBlockedError must stay a RuntimeError subclass or the "
        "`except RuntimeError` skip-and-continue paths silently stop catching it"
    )
