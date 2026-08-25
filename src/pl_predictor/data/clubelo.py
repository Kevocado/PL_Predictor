"""clubelo.py — cross-league team-strength ratings from clubelo.com.

Used only as a promoted-team cold-start prior (see
`features/promoted_team_prior.py`) — the project's own Elo/Pi ratings have
no history at all for a club with zero matches in the loaded football-data
window, so they fall back to a flat league average. ClubElo covers England's
second tier and most of Europe, which is exactly the gap: a club promoted
into the Premier League already has a real, dated ClubElo rating from its
prior division.

STATUS NOTE (see `config.py::CLUBELO_BASE_URL`): the live API has not been
reachable from this project's development environment — TCP connects, the
server never responds. This module is written and unit-tested against a
mocked HTTP response (`tests/test_clubelo.py`) but has never run against a
real fetch. Confirm the response shape (`Rank,Club,Country,Level,Elo,From,To`
CSV, per api.clubelo.com's documented format) and the licence/terms for
reuse before relying on it for anything beyond a manual spot-check — see
docs/AI_CONTINUITY.md's EXP-2026-05 entry.

Endpoint: `GET {CLUBELO_BASE_URL}/{YYYY-MM-DD}` returns a CSV snapshot of
every tracked club's rating *as of* that date. A past date's snapshot is
historical fact once fetched, so it caches forever, same as pulselive.py's
per-match stats.
"""

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from ..config import CLUBELO_BASE_URL, CLUBELO_CACHE_DIR
from .team_names import to_canonical


def _fetch_with_retry(fn, attempts: int = 3, backoff: float = 2.0):
    last_err = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry on any transient fetch failure
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"clubelo fetch failed after {attempts} attempts") from last_err


def _cache_path(date: str) -> Path:
    return CLUBELO_CACHE_DIR / f"{date}.csv"


def fetch_ratings_asof(date: str, force_refresh: bool = False) -> pd.DataFrame:
    """Every tracked club's ClubElo rating as of `date` (`"YYYY-MM-DD"`),
    with team names mapped to this project's canonical convention. Columns:
    `team`, `elo`, `level` (1 = top flight in that club's country, 2 =
    second tier, ...). A past date's snapshot never changes once fetched,
    so this caches forever per date, mirroring `pulselive.fetch_match_stats`.

    Only rows in `Country == "ENG"` and `Level` in `{1, 2}` are kept — this
    project only ever needs a promoted team's rating from the Premier
    League or Championship, not the full global club list ClubElo tracks.
    """
    cache_path = _cache_path(date)
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    def _get():
        resp = requests.get(f"{CLUBELO_BASE_URL}/{date}", timeout=30)
        resp.raise_for_status()
        return resp.text

    raw = _fetch_with_retry(_get)
    full = pd.read_csv(StringIO(raw))
    england = full[(full["Country"] == "ENG") & (full["Level"].isin([1, 2]))].copy()
    england["team"] = england["Club"].apply(lambda name: to_canonical(name, source="clubelo"))
    result = england[["team", "Elo", "Level"]].rename(columns={"Elo": "elo", "Level": "level"})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    return result


def team_rating_asof(team: str, date: str, force_refresh: bool = False) -> float | None:
    """One canonical team's ClubElo rating as of `date`, or None if that
    team isn't in ClubElo's ENG level-1/2 snapshot for that date (e.g. a
    misspelled/unmapped name — see `data/team_names.py::_CLUBELO_ALIASES` —
    or a club outside England's top two tiers)."""
    ratings = fetch_ratings_asof(date, force_refresh=force_refresh)
    match = ratings[ratings["team"] == team]
    if match.empty:
        return None
    return float(match.iloc[0]["elo"])
