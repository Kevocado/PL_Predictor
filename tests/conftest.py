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
from datetime import datetime, timezone

import pytest

from pl_predictor.data import fpl_api
from pl_predictor.data.network_blocked import NetworkBlockedError

# --- the shared /facts builders ------------------------------------------
#
# These used to live in `tests/test_facts.py`, and
# `tests/test_facts_record_reads_pre_kickoff.py` reached across and imported
# them with `from tests.test_facts import EVENT_ID, _snapshot`.
#
# That import only resolves when the running Python happens to have the repo
# ROOT on `sys.path`, so that `tests` resolves as a package -- but there is no
# `tests/__init__.py`, no `pythonpath` setting and no `rootdir` config, so it
# does NOT resolve the way CI runs it. CI sets only `PYTHONPATH=src` and runs
# `pytest tests/ -q -m "not network"`, which gave
# `ModuleNotFoundError: No module named 'tests'` at collection time.
#
# A test importing another test *as a package* is the root cause: it depends on
# a developer's cwd rather than on anything the repo declares. So the builders
# live here, in the one module pytest always loads for `tests/`, and both test
# modules import them from `conftest`. No `sys.path`, no package marker, no cwd.

#: Frozen "now" for the /facts bundle, so a test cannot pass by accident on a
#: fixture that happens to sit on the other side of a kick-off.
FACTS_NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)

#: The one fixture id every /facts test builds around.
FACTS_EVENT_ID = "100"


def facts_detail(**over):
    """A fixture-detail block, the shape `facts._detail()` is given."""
    detail = {
        "event_id": FACTS_EVENT_ID,
        "team_home": "Sunderland",
        "team_away": "Chelsea",
        "commence_time": "2026-11-08T14:00:00Z",
        "home_win": {"prob": 0.36, "implied": None, "edge": None},
        "draw": {"prob": 0.26, "implied": None, "edge": None},
        "away_win": {"prob": 0.38, "implied": None, "edge": None},
        "over_2_5": {"prob": 0.55, "implied": None, "edge": None},
        "under_2_5": {"prob": 0.45, "implied": None, "edge": None},
        "predicted_total_goals": 2.9,
        "btts_yes_prob": 0.57,
        "top_scoreline": "1-1",
        "has_live_odds": False,
        "value_bet_flags": [],
        "home_recent_form": ["W", "D", "L"],
        "away_recent_form": ["L", "W", "W"],
        "data_confidence": "established",
    }
    detail.update(over)
    return detail


def facts_card(**over):
    """A fixture card as the public snapshot carries it."""
    card = {
        "event_id": FACTS_EVENT_ID,
        "team_home": "Sunderland",
        "team_away": "Chelsea",
        "commence_time": "2026-11-08T14:00:00Z",
        "finished": False,
        "actual_goals_home": None,
        "actual_goals_away": None,
        "predicted_home_win": 0.36,
        "predicted_draw": 0.26,
        "predicted_away_win": 0.38,
        "backfilled": False,
        "has_live_odds": False,
        "value_bet_flags": [],
    }
    card.update(over)
    return card


def facts_player(suffix, name, prob, team):
    """One entry in a fixture's player block."""
    return {
        "player_id": 1000 + int(suffix),
        "name": name,
        "position": "MID",
        "anytime_goal_prob": prob,
        "status": "a",
        "team": team,
    }


def facts_snapshot(detail=None, cards=None, players=None, gameweek=9):
    """A whole public snapshot for the one fixture above."""
    card_list = cards if cards is not None else [facts_card()]
    player_block = players if players is not None else {
        "home_players": [
            facts_player("1", "Wilson", 0.62, "Sunderland"),
            facts_player("2", "Jones", 0.31, "Sunderland"),
        ],
        "away_players": [facts_player("3", "Blue", 0.44, "Chelsea")],
    }
    return {
        "current_gameweek": gameweek,
        "fixtures_by_gameweek": {str(gameweek): {"gameweek": gameweek, "fixtures": card_list}},
        "fixture_detail_by_event_id": {FACTS_EVENT_ID: detail if detail is not None else facts_detail()},
        "fixture_players_by_event_id": {FACTS_EVENT_ID: player_block},
    }

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


@pytest.fixture
def no_other_competitions(monkeypatch):
    """Use the PL-only baseline in tests unrelated to the live cup calendar.

    Its six-hour cache can expire between CI cache warming and the offline
    gate. Calendar loaders and cross-competition features have their own
    tests; opt in here only where those live fixtures are incidental.
    """
    from pl_predictor.data import other_competitions

    monkeypatch.setattr(
        other_competitions, "get_team_fixture_calendar",
        lambda: other_competitions._EMPTY.copy(),
    )
