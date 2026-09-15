"""fixtures.py — upcoming EPL fixtures.

Primary source: `sportsbook_api.fetch_epl_fixtures` (RapidAPI's Sportsbook
API's own event list — no odds needed, just fixtures). Falls back to the
official FPL API's `/fixtures/` + `/bootstrap-static/` endpoints (team names
only, no market odds) when `SPORTSBOOK_API_KEY` isn't configured or that
provider is unavailable, so the app still shows upcoming matches either way.

`odds_api.py` (The Odds API, this project's original odds source) is no
longer used here — see its module docstring: it exhausts its monthly credit
quota too often for day-to-day use and is kept only in case it's ever worth
reverting to or supplementing with, same as `odds/value_bets.py`'s pricing
already stopped depending on it.
"""

from __future__ import annotations

import pandas as pd
import requests

from . import fpl_api
from .sportsbook_api import SportsbookAPIKeyMissing, fetch_epl_fixtures
from .team_names import to_canonical


def _fixtures_from_fpl_api() -> pd.DataFrame:
    # Reuse fpl_api's cached/offline-safe feed rather than bypassing it with
    # raw requests. Otherwise a temporary FPL DNS/API outage makes every
    # future fixture disappear even though we already have a valid cache.
    bootstrap = fpl_api.fetch_bootstrap()
    fixtures = fpl_api.fetch_fixtures()

    team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}

    # `not f["finished"]` is already the correct, per-fixture test for
    # "still to be played" — a prior version of this also required
    # `event >= bootstrap["is_next"] gameweek`, which silently dropped the
    # still-unplayed matches of the *current, in-progress* gameweek (FPL's
    # `is_next` flag points at the gameweek *after* the current one while
    # the current one is still underway, e.g. it's GW3 while GW2 has only
    # 5 of its 10 matches played). Confirmed live: that extra filter was
    # the reason GW2's remaining fixtures disappeared from the Fixtures
    # tab entirely instead of showing alongside its completed matches.
    upcoming = [f for f in fixtures if not f["finished"] and f.get("kickoff_time")]

    rows = [
        {
            "event_id": f["id"],
            "gameweek": f.get("event"),
            "commence_time": f["kickoff_time"],
            "team_home": to_canonical(team_names[f["team_h"]], source="fpl"),
            "team_away": to_canonical(team_names[f["team_a"]], source="fpl"),
            "has_odds": False,
        }
        for f in upcoming
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
    return df


def _future_only(df: pd.DataFrame) -> pd.DataFrame:
    """Defense-in-depth: never rely solely on the upstream API to have
    already dropped a fixture that's kicked off — the Sportsbook API pulls
    pre-match lines at kickoff (usually), and FPL's `finished` flag can lag
    a live match by a while, but neither is guaranteed instantaneous.
    `commence_time` mixes tz-aware (Sportsbook API, UTC) and naive (FPL)
    values — same normalization idiom as `tracking/store.py::_naive`."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    naive_commence = df["commence_time"].apply(lambda ts: ts.tz_localize(None) if ts.tzinfo is not None else ts)
    return df[naive_commence >= now].reset_index(drop=True)


def get_upcoming_fixtures(force_refresh: bool = False) -> pd.DataFrame:
    """Returns columns: event_id, commence_time, team_home, team_away,
    has_odds. `has_odds` tells callers whether `sportsbook_api.fetch_epl_odds()`
    will have market data for this fixture."""
    try:
        df = fetch_epl_fixtures(force_refresh=force_refresh)
    except SportsbookAPIKeyMissing:
        df = _fixtures_from_fpl_api()
    except requests.RequestException as exc:
        # Same fallback as a missing key: a rate limit (150 requests/day,
        # see sportsbook_api.py's module docstring), or a transient outage
        # shouldn't take the Fixtures tab (or the public snapshot build)
        # down entirely -- fall back to fixtures-with-no-odds, same as
        # before a key was ever configured.
        print(f"[fixtures] Sportsbook API unavailable, falling back to FPL fixtures (no market odds): {exc}")
        df = _fixtures_from_fpl_api()
    return _future_only(df)


def get_all_remaining_fixtures() -> pd.DataFrame:
    """Every unplayed fixture for the rest of the season, regardless of
    whether live odds exist for it yet. The Sportsbook API only lists
    matches close enough to kickoff to have a posted line, which is fine
    for the Fixtures tab but wrong for anything projecting the *whole*
    remaining season (e.g. the projected table) — this always uses the FPL
    API's full fixture list instead."""
    return _future_only(_fixtures_from_fpl_api())


def list_current_teams() -> list[str]:
    """This season's teams, derived from the full remaining fixture list —
    always accurate to whichever teams are actually in the league right
    now (promotions/relegations included), no hardcoded roster to keep in
    sync each August."""
    df = get_all_remaining_fixtures()
    if df.empty:
        return []
    return sorted(set(df["team_home"]) | set(df["team_away"]))
