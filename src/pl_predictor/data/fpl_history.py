"""fpl_history.py — cached historical per-player-per-gameweek data (minutes,
goals, assists) for the player-prediction feature.

Ported from FPL_Optimizer/historical_data.py — same source (vaastav's
Fantasy-Premier-League GitHub archive), same cache-or-fetch pattern as this
project's own data/football_data.py. Season format here is FPL's own
"2023-24" (2-digit year), distinct from football_data.py's "2023-2024".
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from ..config import FPL_HISTORY_BASE_URL, FPL_HISTORY_CACHE_DIR

CURRENT_SEASON_START_YEAR = 2026  # keep in sync with data/football_data.py


def season_str(start_year: int) -> str:
    """e.g. 2023 -> '2023-24', FPL's own season format."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def default_completed_seasons(n: int = 4) -> list[str]:
    latest_completed = CURRENT_SEASON_START_YEAR - 1
    return [season_str(y) for y in range(latest_completed - n + 1, latest_completed + 1)]


def _fetch_csv_with_retry(url: str, attempts: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    last_err = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
        except requests.exceptions.RequestException as exc:
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_err


def fetch_season_gw_data(season: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """One row per player per gameweek for `season` (FPL format, e.g.
    '2023-24'). Returns None if the archive has no data for that season yet
    (e.g. the season just started)."""
    cache_path = FPL_HISTORY_CACHE_DIR / f"{season}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
    else:
        url = f"{FPL_HISTORY_BASE_URL}/{season}/gws/merged_gw.csv"
        try:
            df = _fetch_csv_with_retry(url)
        except RuntimeError:
            return None
        df.to_csv(cache_path, index=False)

    df["season"] = season
    return df


def load_player_gw_history(seasons: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Concatenate several seasons of player-gameweek data, sorted by
    kickoff time. Columns include (per vaastav's archive): element, name,
    team, GW, minutes, goals_scored, assists, was_home, kickoff_time."""
    seasons = seasons or default_completed_seasons()
    frames = []
    for season in seasons:
        df = fetch_season_gw_data(season, force_refresh=force_refresh)
        if df is None or df.empty:
            print(f"  ! Skipping {season}: no FPL history data available")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No FPL player-gameweek history could be loaded")

    df = pd.concat(frames, ignore_index=True)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    return df.sort_values("kickoff_time").reset_index(drop=True)
