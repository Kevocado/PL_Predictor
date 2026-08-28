"""other_competitions.py — fixture-congestion data: matches a Premier League
club plays outside the league itself (Champions League, Europa League,
Conference League, FA Cup, EFL Cup), so `features/rest_days.py` and
`features/fixture_congestion.py` can see a team's *true* recent match load
instead of just its PL fixtures.

Two sources, split by what's actually free:
- Champions League: football-data.org (competition id 2001), the same
  key/host `data/football_data_org.py` already uses for the Premier League
  itself, via that module's `fetch_matches(competition_id=..., season=...)`.
  Confirmed live: the free tier only reaches back a couple of completed
  seasons (a `season` older than that returns 403) — treated the same as
  every other partial-coverage source in this project (best-effort, never a
  hard failure).
- Europa League / Conference League / FA Cup / EFL Cup: none of these are on
  football-data.org's free tier (confirmed against its published coverage
  page — free tier is exactly PL/CL/top-5-European-leagues/Eredivisie/
  Primeira Liga/Brasileirão/World Cup/Euros). ESPN's site API
  (`data/espn.py` already uses this host for confirmed lineups) is the only
  remaining free, keyless option. Confirmed live — see
  `config.py::ESPN_SOCCER_BASE_URL`'s comment; every fetch here still
  degrades to an empty DataFrame on any failure rather than raising, so an
  unreachable or unexpectedly-shaped ESPN response never blocks training or
  live serving (current season only — no historical-season parameter was
  found for these four).

Output shape (`get_team_fixture_calendar`): long format, one row per
(team, date, competition) — `team` is the canonical PL short name
(`team_names.CANONICAL_TEAMS`); rows for the opponent side are dropped
entirely, since a congestion feature only needs "did our PL team play on
this date," not the opponent's identity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import ESPN_CUP_COMPETITIONS, ESPN_SOCCER_BASE_URL, OTHER_COMPETITIONS_CACHE_DIR
from . import football_data, football_data_org
from .team_names import CANONICAL_TEAMS, to_canonical

CHAMPIONS_LEAGUE_COMPETITION_ID = 2001
# Current season + up to 2 back; the free tier's actual depth limit is
# confirmed live but not documented by football-data.org, so this just
# probes and quietly stops once a season 403s (see `_fetch_cl_season`).
CHAMPIONS_LEAGUE_SEASON_LOOKBACK = 2

_EMPTY = pd.DataFrame(columns=["team", "date", "competition"])


def _cl_cache_path(season_start_year: int) -> Path:
    return OTHER_COMPETITIONS_CACHE_DIR / "champions_league" / f"{season_start_year}.csv"


def _to_team_rows(matches_df: pd.DataFrame, competition: str) -> pd.DataFrame:
    """`football_data_org.fetch_matches`'s wide (team_home/team_away) shape
    -> long (team, date) rows, English clubs only."""
    if matches_df.empty:
        return _EMPTY.copy()
    rows = []
    for _, m in matches_df.iterrows():
        for team in (m["team_home"], m["team_away"]):
            if team in CANONICAL_TEAMS:
                rows.append({"team": team, "date": m["commence_time"], "competition": competition})
    return pd.DataFrame(rows, columns=["team", "date", "competition"])


def _fetch_cl_season(season_start_year: int, force_refresh: bool) -> pd.DataFrame:
    cache_path = _cl_cache_path(season_start_year)
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, parse_dates=["date"])

    matches = football_data_org.fetch_matches(
        competition_id=CHAMPIONS_LEAGUE_COMPETITION_ID, season=season_start_year
    )
    df = _to_team_rows(matches, "Champions League")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_champions_league_matches() -> pd.DataFrame:
    """Every Champions League fixture (played or scheduled) involving a
    current-or-recent Premier League club, across whatever seasons the free
    tier actually grants access to. The current season is always
    force-refreshed (fixtures/results change weekly); past seasons are
    cached forever, same split as `football_data.py`'s own
    fetch_season/fetch_current_season_partial."""
    frames = []
    for offset in range(CHAMPIONS_LEAGUE_SEASON_LOOKBACK + 1):
        year = football_data.CURRENT_SEASON_START_YEAR - offset
        try:
            df = _fetch_cl_season(year, force_refresh=(offset == 0))
        except (football_data_org.FootballDataOrgKeyMissing, requests.RequestException):
            # A missing key means no CL data at all; an HTTP error for an
            # older season is the free tier's depth limit, not a real
            # failure — either way, degrade rather than block training.
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return _EMPTY.copy()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def _fetch_espn_cup(label: str, slug: str) -> pd.DataFrame:
    """One ESPN cup competition's schedule for the current season (played +
    upcoming) — these congestion features only need to be accurate for
    recent/live rest-day accounting, not the full multi-season training
    window. Wrapped so ANY failure (network, timeout, unexpected shape)
    returns empty rather than raising — see module docstring."""
    year = football_data.CURRENT_SEASON_START_YEAR
    try:
        resp = requests.get(
            f"{ESPN_SOCCER_BASE_URL}/{slug}/scoreboard",
            # `limit` matters: ESPN's default cap is 100 events, and without
            # it a wide date range returns qualifying-round matches from
            # dozens of countries first (chronological order) rather than
            # ever reaching the Premier League clubs' own fixtures later in
            # the season — confirmed live per-competition.
            params={"dates": f"{year}0701-{year + 1}0701", "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except (requests.RequestException, ValueError, KeyError):
        return _EMPTY.copy()

    rows = []
    for event in events:
        date = event.get("date")
        if not date:
            continue
        for competition in event.get("competitions", []):
            for competitor in competition.get("competitors", []):
                raw_name = competitor.get("team", {}).get("displayName", "")
                name = to_canonical(raw_name, source="espn")
                if name in CANONICAL_TEAMS:
                    rows.append({"team": name, "date": date, "competition": label})
    return pd.DataFrame(rows, columns=["team", "date", "competition"])


def fetch_cup_matches() -> pd.DataFrame:
    """Europa League, Conference League, FA Cup, EFL Cup fixtures for
    current Premier League clubs, from ESPN (see module docstring for why
    football-data.org can't supply these). Not cached to disk the way the
    Champions League fetch is — draws/replays change these more often, and
    ESPN's own date-range query already returns a full season in one call
    per competition."""
    frames = [_fetch_espn_cup(label, slug) for label, slug in ESPN_CUP_COMPETITIONS.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _EMPTY.copy()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def get_team_fixture_calendar() -> pd.DataFrame:
    """Every non-PL fixture (Champions League, Europa League, Conference
    League, FA Cup, EFL Cup) for a current-or-recent Premier League club,
    long format: one row per (team, date, competition). This is the single
    entry point `features/rest_days.py` and `features/fixture_congestion.py`
    consume — an empty result here (e.g. ESPN unreachable, or no key
    configured for football-data.org) degrades those features to their
    PL-only baseline rather than failing."""
    frames = [fetch_champions_league_matches(), fetch_cup_matches()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _EMPTY.copy()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["team", "date", "competition"]).reset_index(drop=True)
