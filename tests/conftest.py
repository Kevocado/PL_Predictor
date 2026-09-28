"""Shared test isolation.

`fpl_api` memoises its fixtures at module level (`_fixtures_memory_cache`, a
60-second TTL) and backs off failures in `_failed_fetches`. Any test that
drives a real request through the app warms that memo, and a later test which
stubs `fpl_api.requests.get` then never sees its own stub — it reads real FPL
data and fails for a reason that has nothing to do with the code under test.

`tests/test_fpl_outcomes.py::test_fetch_fixtures_uses_the_last_successful_cache_when_fpl_is_unavailable`
is the casualty: it sorts after the /facts tests, which are the first to make a
real request through `pl_predictor.api.main`.

Snapshot and restore both module-level dicts around every test, so no test can
inherit another test's warm cache or backoff window.

## The network guard

The suite used to have no bound on network I/O, and a fresh clone has an empty
`data/cache/` (it is gitignored, deliberately -- see `.gitignore`). So
`pytest tests/` on a clean checkout went to the network for every match it
needed, and the understat shot loader fetches **one file per match** with a
0.3s delay between uncached ones. That is ~4,600 requests for the full
historical window.

Measured on a genuinely cold worktree of `origin/main` (2026-09-28), for
`tests/test_features.py` alone -- five tests:

    before this guard   707.02s   2 failed
    after this guard     12.05s   5 passed

and the full suite went from not finishing in >1000s to **709.64s,
415 passed, 27 skipped**.

The guard raises on an outbound connect instead of letting it proceed, so a
test that needs data it does not have fails immediately and says which command
warms it. That is the whole point: a suite that fetches is a suite whose
runtime depends on a third party's uptime, and a suite that hangs is not a
check -- a developer who cannot run it locally stops running it, and a green
build becomes the only signal.

A test that genuinely needs the network opts in with `@pytest.mark.network`.
None do today; the marker exists so that adding one is a deliberate act rather
than an accident, and so the opt-out is greppable.
"""
from __future__ import annotations

import os
import socket

import pytest

from pl_predictor.data import fpl_api
from pl_predictor.data.network_blocked import NetworkBlockedError

WARM_CACHE_HINT = """\
This test reached the network, and the suite blocks it so that runtime does not
depend on a third party's uptime. The data it wanted is cached under data/cache/
(gitignored), so it warms by running the code once with the guard disabled:

    PL_ALLOW_NETWORK=1 PYTHONPATH=src .venv/bin/python -m pytest <this file> -q

That is slow the first time -- the understat shot loader fetches one file per
match -- and fast once cached. If this test is supposed to work offline, it
should not be reading real data at all: patch the loader with monkeypatch
instead, which is what most of this suite already does. If it genuinely needs
the network, mark it `@pytest.mark.network` and it will be allowed through.
"""


# Captured at import, before any test runs, so the teardown can be checked
# against the genuine stdlib originals rather than against whatever happens to
# be installed. A test asserting "the socket is restored" while its own guard is
# active would otherwise compare a patch to a patch.
PRISTINE_SOCKET_CONNECT = socket.socket.connect
PRISTINE_CREATE_CONNECTION = socket.create_connection


@pytest.fixture(autouse=True)
def _block_network(request):
    """Fail fast on an outbound connection unless the test opts in.

    Autouse and per-test rather than one session-wide patch, so the socket is
    genuinely restored between tests and a leaked `monkeypatch` elsewhere cannot
    leave the suite permanently deaf.
    """
    if request.node.get_closest_marker("network") or os.environ.get("PL_ALLOW_NETWORK") == "1":
        yield
        return
    original = socket.socket.connect
    original_create = socket.create_connection

    def blocked(self, address, *args, **kwargs):
        raise NetworkBlockedError(f"outbound network blocked in tests: {address!r}\n{WARM_CACHE_HINT}")

    def blocked_create(address, *args, **kwargs):
        raise NetworkBlockedError(f"outbound network blocked in tests: {address!r}\n{WARM_CACHE_HINT}")

    socket.socket.connect = blocked
    socket.create_connection = blocked_create
    try:
        yield
    finally:
        socket.socket.connect = original
        socket.create_connection = original_create


@pytest.fixture(autouse=True)
def _isolate_fpl_module_caches():
    """Keep fpl_api's module-level memo and backoff window out of every test."""
    saved_cache = fpl_api._fixtures_memory_cache
    saved_failed = dict(fpl_api._failed_fetches)
    try:
        yield
    finally:
        fpl_api._fixtures_memory_cache = saved_cache
        fpl_api._failed_fetches.clear()
        fpl_api._failed_fetches.update(saved_failed)
