"""understat.py — cached team-level expected-goals (xG) per match.

Actual goals are noisy (a team can dominate on chances and still lose 0-1);
xG is a smoother, lower-variance read on underlying performance. Research
consensus is that xG doesn't clearly beat a goals-fit Dixon-Coles model
outright, so this is added as an *extra* rolling feature (see
`features/xg_form.py`), not a replacement for the goals-based scoreline
model.

One call per season via penaltyblog's Understat scraper — cheap, same
cache-or-fetch pattern as `data/football_data.py`.
"""

from __future__ import annotations

import time

import pandas as pd
import penaltyblog as pb

from ..config import UNDERSTAT_CACHE_DIR
from .team_names import to_canonical

COMPETITION = "ENG Premier League"
CURRENT_SEASON_START_YEAR = 2026  # keep in sync with data/football_data.py


def default_completed_seasons(n: int = 8) -> list[str]:
    """Understat's own season format is just the start year, e.g. '2023'
    for the 2023-24 season."""
    latest_completed = CURRENT_SEASON_START_YEAR - 1
    return [str(y) for y in range(latest_completed - n + 1, latest_completed + 1)]


def _fetch_with_retry(season: str, attempts: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    last_err = None
    for attempt in range(attempts):
        try:
            return pb.scrapers.Understat(COMPETITION, season).get_fixtures()
        except Exception as exc:  # noqa: BLE001 - retry on any transient fetch failure
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch Understat {season} after {attempts} attempts") from last_err


def fetch_season(season: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = UNDERSTAT_CACHE_DIR / f"{season}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        df = _fetch_with_retry(season)
        df = df.reset_index(drop=True)
        df.to_csv(cache_path, index=False)

    df["team_home"] = df["team_home"].apply(lambda n: to_canonical(n, source="understat"))
    df["team_away"] = df["team_away"].apply(lambda n: to_canonical(n, source="understat"))
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "team_home", "team_away", "xg_home", "xg_away", "goals_home", "goals_away"]]


def load_xg_data(seasons: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    seasons = seasons or default_completed_seasons()
    frames = []
    for season in seasons:
        try:
            frames.append(fetch_season(season, force_refresh=force_refresh))
        except RuntimeError as exc:
            print(f"  ! Skipping Understat {season}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["date", "team_home", "team_away", "xg_home", "xg_away", "goals_home", "goals_away"])

    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
