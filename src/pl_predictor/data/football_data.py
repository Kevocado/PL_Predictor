"""football_data.py — cached historical match data (results, corners, cards,
closing odds) via penaltyblog's football-data.co.uk scraper.

Mirrors the cache-or-fetch pattern from FPL_Optimizer/historical_data.py:
check the cache file first, otherwise fetch (with retry), write to cache,
return.
"""

from __future__ import annotations

import time

import pandas as pd
import penaltyblog as pb

from ..config import COMPETITION, FOOTBALL_DATA_CACHE_DIR

CURRENT_SEASON_START_YEAR = 2026  # 2026-27 season; bump each August


def season_str(start_year: int) -> str:
    """e.g. 2023 -> '2023-2024', the format penaltyblog's FootballData expects."""
    return f"{start_year}-{start_year + 1}"


def default_completed_seasons(n: int = 8) -> list[str]:
    """The `n` most recently *completed* seasons, oldest first. The current
    season (in progress) is deliberately excluded here — use
    `fetch_current_season_partial()` for in-progress results."""
    latest_completed = CURRENT_SEASON_START_YEAR - 1
    start_years = range(latest_completed - n + 1, latest_completed + 1)
    return [season_str(y) for y in start_years]


def _fetch_with_retry(season: str, attempts: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    last_err = None
    for attempt in range(attempts):
        try:
            scraper = pb.scrapers.FootballData(COMPETITION, season)
            return scraper.get_fixtures()
        except Exception as exc:  # noqa: BLE001 - retry on any transient fetch failure
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {season} after {attempts} attempts") from last_err


def fetch_season(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """Cache-or-fetch a single season's match data."""
    cache_path = FOOTBALL_DATA_CACHE_DIR / f"{season}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        df = _fetch_with_retry(season)
        df = df.reset_index(drop=True)
        df.to_csv(cache_path, index=False)

    df["season"] = season
    return df


def load_training_data(seasons: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Concatenate several seasons of historical match data, sorted by date."""
    seasons = seasons or default_completed_seasons()
    frames = [fetch_season(s, force_refresh=force_refresh) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_current_season_partial(force_refresh: bool = True) -> pd.DataFrame | None:
    """Matches played so far in the current (in-progress) season. Refreshes
    by default since this data changes weekly; returns None if the season
    hasn't started / football-data.co.uk has no file for it yet."""
    season = season_str(CURRENT_SEASON_START_YEAR)
    try:
        return fetch_season(season, force_refresh=force_refresh)
    except RuntimeError:
        return None
