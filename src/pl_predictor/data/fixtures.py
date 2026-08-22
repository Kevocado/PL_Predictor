"""fixtures.py — upcoming EPL fixtures.

Primary source: the event list embedded in The Odds API's response (one call
gives fixtures *and* odds together). Falls back to the official FPL API's
`/fixtures/` + `/bootstrap-static/` endpoints (team names only, no market
odds) when `ODDS_API_KEY` isn't configured, so the app still shows upcoming
matches before the user sets up a key.
"""

from __future__ import annotations

import requests
import pandas as pd

from ..config import FPL_API_BASE_URL
from .odds_api import OddsAPIKeyMissing, fetch_epl_odds_raw
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
    bootstrap = requests.get(f"{FPL_API_BASE_URL}/bootstrap-static/", timeout=30).json()
    fixtures = requests.get(f"{FPL_API_BASE_URL}/fixtures/", timeout=30).json()

    team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    next_event = next((e["id"] for e in bootstrap["events"] if e.get("is_next")), None)

    upcoming = [f for f in fixtures if not f["finished"] and f.get("kickoff_time")]
    if next_event is not None:
        upcoming = [f for f in upcoming if f["event"] is None or f["event"] >= next_event]

    rows = [
        {
            "event_id": f["id"],
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


def get_upcoming_fixtures(gameweek_key: str = "current", force_refresh: bool = False) -> pd.DataFrame:
    """Returns columns: event_id, commence_time, team_home, team_away,
    has_odds. `has_odds` tells callers whether `odds_api.fetch_epl_odds()`
    will have market data for this fixture."""
    try:
        return _fixtures_from_odds_api(gameweek_key=gameweek_key, force_refresh=force_refresh)
    except OddsAPIKeyMissing:
        return _fixtures_from_fpl_api()
