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
"""
from __future__ import annotations

import pytest

from pl_predictor.data import fpl_api


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
