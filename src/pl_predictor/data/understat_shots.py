"""understat_shots.py — per-match shot-level data (situation: OpenPlay,
FromCorner, SetPiece, DirectFreekick, Penalty) from Understat, for the
shot-situation-split feature (`features/shot_situation.py`).

Unlike `data/understat.py::fetch_season` (one request per *season*, already
used for match-level xG totals), getting a situation breakdown needs shot-
level detail — one *additional* request per match via penaltyblog's
`Understat.get_shots(understat_id)`, confirmed directly against the live
API (not exposed by `get_fixtures()`'s match-level summary at all). For the
current 8-season window that's on the order of 2,500-3,000 new requests,
one-time, then cached forever — same cache-or-fetch idiom as everywhere
else in this package, just keyed per match instead of per season (hence a
dedicated `UNDERSTAT_SHOTS_CACHE_DIR` rather than reusing
`UNDERSTAT_CACHE_DIR`).

`understat.py::fetch_season`'s own cache drops `understat_id` (it only
keeps match-level summary columns), so this module fetches+caches the raw
season fixture list itself (with `understat_id` intact) rather than
depending on that trimmed shape.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..config import UNDERSTAT_SHOTS_CACHE_DIR
from .team_names import to_canonical
from .understat import COMPETITION

# Every Understat shot situation except "OpenPlay" is a dead-ball
# restart — corners, direct/indirect free kicks, penalties — confirmed
# directly against live data (the full category set is exactly these five).
SET_PIECE_SITUATIONS = {"FromCorner", "SetPiece", "DirectFreekick", "Penalty"}


def _fetch_with_retry(fn, *args, attempts: int = 3, backoff: float = 2.0):
    last_err = None
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - retry on any transient fetch failure
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed after {attempts} attempts") from last_err


def _fetch_season_fixtures_with_id(season: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"_fixtures_{season}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, parse_dates=["date"])
    scraper = pb.scrapers.Understat(COMPETITION, season)
    df = _fetch_with_retry(scraper.get_fixtures)
    df = df.reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_match_shots(scraper: pb.scrapers.Understat, understat_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """Cache-or-fetch one match's shots. `scraper` is a single `Understat`
    instance reused across a whole season's matches (construction is cheap
    — no network call — but reusing one instance avoids rebuilding it
    per-match for no reason). A genuinely shot-less match (e.g. Understat
    has no log for it) writes a columnless CSV that `read_csv` can't parse
    back — treated as a cache miss rather than crashing the whole run, and
    re-marked with an explicit empty-but-readable file so it isn't retried
    forever."""
    cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"{understat_id}.csv"
    if cache_path.exists() and not force_refresh:
        try:
            return pd.read_csv(cache_path)
        except pd.errors.EmptyDataError:
            pass
    df = _fetch_with_retry(scraper.get_shots, understat_id)
    df = df.reset_index()  # get_shots sets a synthetic string 'id' as the index
    if df.empty and df.columns.empty:
        df = pd.DataFrame(columns=["situation", "h_a", "x_g", "team_home", "team_away"])
    df.to_csv(cache_path, index=False)
    return df


def _aggregate_match(shots: pd.DataFrame) -> tuple[dict, dict]:
    """Returns (home_row, away_row): {set_piece_xg_share, total_xg} per
    side. `x_g` comes back as a string from Understat's API (Arrow string
    dtype) — must be cast before summing. `share` is None (not 0) for a
    side with zero shots that match, same "genuinely unknown, not zero"
    convention as the rest of this codebase's NaN-safe features."""
    shots = shots.copy()
    shots["x_g"] = shots["x_g"].astype(float)
    shots["is_set_piece"] = shots["situation"].isin(SET_PIECE_SITUATIONS)

    def _side(h_a: str) -> dict:
        side_shots = shots[shots["h_a"] == h_a]
        total_xg = float(side_shots["x_g"].sum())
        set_piece_xg = float(side_shots.loc[side_shots["is_set_piece"], "x_g"].sum())
        share = set_piece_xg / total_xg if total_xg > 0 else None
        return {"total_xg": total_xg, "set_piece_xg_share": share}

    return _side("h"), _side("a")


_SHOT_SITUATION_COLS = ["date", "team_home", "team_away", "home_set_piece_xg_share", "away_set_piece_xg_share"]


def _load_season_shot_situation(season: str, force_refresh: bool, request_delay: float) -> pd.DataFrame:
    """One row per match for a single season — cached as ONE file per
    season (`_aggregate_{season}.csv`), not re-derived from ~380 individual
    per-match shot CSVs on every call. Reading 380 small CSVs with
    `pd.read_csv` per call is genuinely slow (each call has real per-call
    overhead, same lesson as the XGBoost per-row-prediction fix earlier this
    session) — this made `/api/fixtures` time out in production once a
    caller (`FixtureFeatureContext`) started constructing this on every
    request. The per-match files (`{understat_id}.csv`) stay as the
    fetch-level cache; this is a second, coarser cache layer on top."""
    agg_cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"_aggregate_{season}.csv"
    if agg_cache_path.exists() and not force_refresh:
        return pd.read_csv(agg_cache_path, parse_dates=["date"])

    fixtures = _fetch_season_fixtures_with_id(season, force_refresh=force_refresh)
    scraper = pb.scrapers.Understat(COMPETITION, season)
    rows = []
    for _, fx in fixtures.iterrows():
        understat_id = str(fx["understat_id"])
        cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"{understat_id}.csv"
        was_cached = cache_path.exists()
        try:
            shots = fetch_match_shots(scraper, understat_id, force_refresh=force_refresh)
        except RuntimeError as exc:
            print(f"  ! Skipping shots for match {understat_id}: {exc}")
            continue
        if not was_cached and request_delay:
            time.sleep(request_delay)

        if shots.empty:
            continue
        home_row, away_row = _aggregate_match(shots)
        rows.append(
            {
                "date": fx["date"],
                "team_home": to_canonical(str(fx["team_home"]), source="understat"),
                "team_away": to_canonical(str(fx["team_away"]), source="understat"),
                "home_set_piece_xg_share": home_row["set_piece_xg_share"],
                "away_set_piece_xg_share": away_row["set_piece_xg_share"],
            }
        )

    df = pd.DataFrame(rows, columns=_SHOT_SITUATION_COLS)
    df.to_csv(agg_cache_path, index=False)
    return df


_DOMINANCE_COLS = [
    "date",
    "team_home",
    "team_away",
    "home_total_xg",
    "away_total_xg",
    "home_non_penalty_xg",
    "away_non_penalty_xg",
    "home_shots",
    "away_shots",
    "home_xg_per_shot",
    "away_xg_per_shot",
    "home_open_play_xg_share",
    "away_open_play_xg_share",
    "home_set_piece_xg_share",
    "away_set_piece_xg_share",
    "home_avg_shot_distance",
    "away_avg_shot_distance",
]


def _aggregate_match_dominance(shots: pd.DataFrame) -> tuple[dict, dict]:
    """Richer per-side match aggregate than `_aggregate_match`: total and
    non-penalty xG, shot count, xG per shot, open-play/set-piece xG share,
    and average shot distance-to-goal. Distance (not a "quality" score) is
    deliberate: Understat's `x`/`y` are already normalized pitch
    coordinates (`x=1` is the opponent's goal line, `y=0.5` is the pitch's
    vertical center), so a *lower* average distance means shots taken from
    better positions on average — no separate scale to define or justify.
    Player names and free-text action labels (`player`, `player_assisted`,
    `last_action`, `shot_type`) are deliberately excluded, per the numeric-
    only, no-unstable-free-text-features rule for this feature set's v1."""
    shots = shots.copy()
    shots["x_g"] = shots["x_g"].astype(float)
    shots["x"] = shots["x"].astype(float)
    shots["y"] = shots["y"].astype(float)
    shots["is_set_piece"] = shots["situation"].isin(SET_PIECE_SITUATIONS)
    shots["is_penalty"] = shots["situation"] == "Penalty"
    shots["is_open_play"] = shots["situation"] == "OpenPlay"
    shots["distance_to_goal"] = np.sqrt((1 - shots["x"]) ** 2 + (shots["y"] - 0.5) ** 2)

    def _side(h_a: str) -> dict:
        side_shots = shots[shots["h_a"] == h_a]
        n_shots = len(side_shots)
        total_xg = float(side_shots["x_g"].sum())
        non_penalty_xg = float(side_shots.loc[~side_shots["is_penalty"], "x_g"].sum())
        set_piece_xg = float(side_shots.loc[side_shots["is_set_piece"], "x_g"].sum())
        open_play_xg = float(side_shots.loc[side_shots["is_open_play"], "x_g"].sum())
        return {
            "total_xg": total_xg,
            "non_penalty_xg": non_penalty_xg,
            "shots": n_shots,
            "xg_per_shot": total_xg / n_shots if n_shots > 0 else None,
            "open_play_xg_share": open_play_xg / total_xg if total_xg > 0 else None,
            "set_piece_xg_share": set_piece_xg / total_xg if total_xg > 0 else None,
            "avg_shot_distance": float(side_shots["distance_to_goal"].mean()) if n_shots > 0 else None,
        }

    return _side("h"), _side("a")


def _load_season_match_dominance(season: str, force_refresh: bool, request_delay: float) -> pd.DataFrame:
    """Same one-row-per-match, cache-per-season shape as
    `_load_season_shot_situation`, but built via `_aggregate_match_dominance`
    and cached to a separately-versioned file (`_aggregate_v2_{season}.csv`)
    so the older `set_piece_xg_share`-only cache is never silently
    reinterpreted as this richer shape. Reuses the exact same per-match
    fetch-or-cache raw files (`{understat_id}.csv`) `_load_season_shot_
    situation` already populates — for any season already fetched, this
    makes zero new network calls."""
    agg_cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"_aggregate_v2_{season}.csv"
    if agg_cache_path.exists() and not force_refresh:
        return pd.read_csv(agg_cache_path, parse_dates=["date"])

    fixtures = _fetch_season_fixtures_with_id(season, force_refresh=force_refresh)
    scraper = pb.scrapers.Understat(COMPETITION, season)
    rows = []
    for _, fx in fixtures.iterrows():
        understat_id = str(fx["understat_id"])
        cache_path = UNDERSTAT_SHOTS_CACHE_DIR / f"{understat_id}.csv"
        was_cached = cache_path.exists()
        try:
            shots = fetch_match_shots(scraper, understat_id, force_refresh=force_refresh)
        except RuntimeError as exc:
            print(f"  ! Skipping dominance aggregate for match {understat_id}: {exc}")
            continue
        if not was_cached and request_delay:
            time.sleep(request_delay)

        if shots.empty:
            continue
        home_row, away_row = _aggregate_match_dominance(shots)
        row = {
            "date": fx["date"],
            "team_home": to_canonical(str(fx["team_home"]), source="understat"),
            "team_away": to_canonical(str(fx["team_away"]), source="understat"),
        }
        row.update({f"home_{key}": value for key, value in home_row.items()})
        row.update({f"away_{key}": value for key, value in away_row.items()})
        rows.append(row)

    df = pd.DataFrame(rows, columns=_DOMINANCE_COLS)
    df.to_csv(agg_cache_path, index=False)
    return df


def load_match_dominance_data(
    seasons: list[str] | None = None, force_refresh: bool = False, request_delay: float = 0.3
) -> pd.DataFrame:
    """One row per match with the richer dominance aggregate (see
    `_DOMINANCE_COLS`) — the match-dominance research candidate feature
    set. Same caching/refresh caveats as `load_shot_situation_data`."""
    from . import understat as understat_mod

    seasons = seasons or understat_mod.default_completed_seasons()
    frames = []
    for season in seasons:
        try:
            frames.append(_load_season_match_dominance(season, force_refresh, request_delay))
        except RuntimeError as exc:
            print(f"  ! Skipping match-dominance {season}: {exc}")

    if not frames:
        return pd.DataFrame(columns=_DOMINANCE_COLS)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_shot_situation_data(
    seasons: list[str] | None = None, force_refresh: bool = False, request_delay: float = 0.3
) -> pd.DataFrame:
    """One row per match: date, team_home, team_away,
    home_set_piece_xg_share, away_set_piece_xg_share. `request_delay` only
    applies to genuinely new (uncached) requests — a rerun against an
    already-cached season costs nothing extra and doesn't sleep at all.

    NOTE: like `understat.load_xg_data`, this does not automatically pick up
    new matches in an in-progress season unless `force_refresh=True` — the
    per-season aggregate cache above is the reason: an in-progress season's
    aggregate would otherwise never be considered stale. Not currently a
    problem in practice since this feature isn't wired into any model's
    `feature_cols` (see `features/build.py`'s docstring note)."""
    from . import understat as understat_mod

    seasons = seasons or understat_mod.default_completed_seasons()
    frames = []
    for season in seasons:
        try:
            frames.append(_load_season_shot_situation(season, force_refresh, request_delay))
        except RuntimeError as exc:
            print(f"  ! Skipping shot-situation {season}: {exc}")

    if not frames:
        return pd.DataFrame(columns=_SHOT_SITUATION_COLS)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
