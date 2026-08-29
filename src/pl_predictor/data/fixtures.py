"""fixtures.py — upcoming EPL fixtures.

Primary source: the event list embedded in The Odds API's response (one call
gives fixtures *and* odds together). Falls back to the official FPL API's
`/fixtures/` + `/bootstrap-static/` endpoints (team names only, no market
odds) when `ODDS_API_KEY` isn't configured, so the app still shows upcoming
matches before the user sets up a key.
"""

from __future__ import annotations

import pandas as pd

from .odds_api import OddsAPIKeyMissing, fetch_epl_odds_raw
from . import fpl_api
from .team_names import to_canonical


def _fixtures_from_odds_api(gameweek_key: str, force_refresh: bool) -> pd.DataFrame:
    events = fetch_epl_odds_raw(gameweek_key=gameweek_key, force_refresh=force_refresh)
    rows = [
        {
            "event_id": e["id"],
            "commence_time": e["commence_time"],
            "team_home": to_canonical(e["home_team"], source="odds_api"),
            "team_away": to_canonical(e["away_team"], source="odds_api"),
            "has_odds": bool(e.get("bookmakers")),
        }
        for e in events
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
    return df


def _fixtures_from_fpl_api() -> pd.DataFrame:
    # Reuse fpl_api's cached/offline-safe feed rather than bypassing it with
    # raw requests. Otherwise a temporary FPL DNS/API outage makes every
    # future fixture disappear even though we already have a valid cache.
    bootstrap = fpl_api.fetch_bootstrap()
    fixtures = fpl_api.fetch_fixtures()

    team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    next_event = next((e["id"] for e in bootstrap["events"] if e.get("is_next")), None)

    upcoming = [f for f in fixtures if not f["finished"] and f.get("kickoff_time")]
    if next_event is not None:
        upcoming = [f for f in upcoming if f["event"] is None or f["event"] >= next_event]

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
    already dropped a fixture that's kicked off — The Odds API pulls
    pre-match lines at kickoff (usually), and FPL's `finished` flag can lag
    a live match by a while, but neither is guaranteed instantaneous.
    `commence_time` mixes tz-aware (Odds API, UTC) and naive (FPL) values —
    same normalization idiom as `tracking/store.py::_naive`."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    naive_commence = df["commence_time"].apply(lambda ts: ts.tz_localize(None) if ts.tzinfo is not None else ts)
    return df[naive_commence >= now].reset_index(drop=True)


def get_upcoming_fixtures(gameweek_key: str = "current", force_refresh: bool = False) -> pd.DataFrame:
    """Returns columns: event_id, commence_time, team_home, team_away,
    has_odds. `has_odds` tells callers whether `odds_api.fetch_epl_odds()`
    will have market data for this fixture."""
    try:
        df = _fixtures_from_odds_api(gameweek_key=gameweek_key, force_refresh=force_refresh)
    except OddsAPIKeyMissing:
        df = _fixtures_from_fpl_api()
    return _future_only(df)


def get_all_remaining_fixtures() -> pd.DataFrame:
    """Every unplayed fixture for the rest of the season, regardless of
    whether live odds exist for it yet. The Odds API only lists matches
    close enough to kickoff to have a posted line, which is fine for the
    Fixtures tab but wrong for anything projecting the *whole* remaining
    season (e.g. the projected table) — this always uses the FPL API's full
    fixture list instead."""
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
