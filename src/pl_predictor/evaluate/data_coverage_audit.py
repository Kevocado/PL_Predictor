"""data_coverage_audit.py — pre-flight report on what's actually cached.

Run before trusting any expanded-window training/evaluation run (e.g. the
12-season match-dominance research this backs): fixture counts per season,
missing-field rates, cache-file age, canonical-name join hit rate, and
season availability per source. Purely reads the existing on-disk caches —
makes no network calls itself, so it's always safe to run.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import (
    FOOTBALL_DATA_CACHE_DIR,
    FPL_HISTORY_CACHE_DIR,
    UNDERSTAT_CACHE_DIR,
    UNDERSTAT_SHOTS_CACHE_DIR,
)
from ..data import football_data, fpl_history, understat
from ..data.team_names import CANONICAL_TEAMS


def _cache_age_days(path: Path) -> float | None:
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 86400


def audit_football_data(seasons: list[str] | None = None) -> pd.DataFrame:
    seasons = seasons or football_data.default_completed_seasons(n=12)
    rows = []
    for season in seasons:
        cache_path = FOOTBALL_DATA_CACHE_DIR / f"{season}.csv"
        n_fixtures = None
        missing_field_rate = None
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            n_fixtures = len(df)
            key_cols = [c for c in ("goals_home", "goals_away", "hs", "as", "hc", "ac", "hy", "ay") if c in df.columns]
            missing_field_rate = float(df[key_cols].isna().mean().mean()) if key_cols else None
        rows.append(
            {
                "source": "football_data",
                "season": season,
                "cached": cache_path.exists(),
                "n_fixtures": n_fixtures,
                "missing_field_rate": missing_field_rate,
                "cache_age_days": _cache_age_days(cache_path),
            }
        )
    return pd.DataFrame(rows)


def audit_understat_xg(seasons: list[str] | None = None) -> pd.DataFrame:
    """Note: the raw cache file (`{season}.csv`) intentionally stores each
    team's *unmapped* Understat name — `understat.py::fetch_season` applies
    `to_canonical` at read time on every call (cache hit or not), precisely
    so the alias table can improve without invalidating the cache. This
    audit therefore calls `fetch_season` itself (the actual production code
    path) rather than reading the cache file directly — reading the raw
    file would report a false join-rate problem that doesn't exist in
    anything the model actually consumes."""
    seasons = seasons or understat.default_completed_seasons(n=12)
    rows = []
    for season in seasons:
        cache_path = UNDERSTAT_CACHE_DIR / f"{season}.csv"
        n_fixtures = None
        canonical_join_rate = None
        if cache_path.exists():
            df = understat.fetch_season(season)
            n_fixtures = len(df)
            teams = set(df["team_home"].unique()) | set(df["team_away"].unique())
            canonical_join_rate = len(teams & CANONICAL_TEAMS) / len(teams) if teams else None
        rows.append(
            {
                "source": "understat_xg",
                "season": season,
                "cached": cache_path.exists(),
                "n_fixtures": n_fixtures,
                "canonical_join_rate": canonical_join_rate,
                "cache_age_days": _cache_age_days(cache_path),
            }
        )
    return pd.DataFrame(rows)


def audit_understat_shots(seasons: list[str] | None = None) -> pd.DataFrame:
    """Per season: whether the fixture-id list is cached, how many of its
    matches have a per-match shot file cached, and which aggregate layer(s)
    exist. `n_matches_shot_cached < n_matches_expected` means a fetch is
    still needed for that season before the richer dominance features can
    be trusted for it."""
    seasons = seasons or understat.default_completed_seasons(n=12)
    rows = []
    for season in seasons:
        fixtures_path = UNDERSTAT_SHOTS_CACHE_DIR / f"_fixtures_{season}.csv"
        n_matches_expected = None
        n_matches_shot_cached = None
        if fixtures_path.exists():
            fixtures = pd.read_csv(fixtures_path)
            n_matches_expected = len(fixtures)
            n_matches_shot_cached = sum(
                (UNDERSTAT_SHOTS_CACHE_DIR / f"{understat_id}.csv").exists()
                for understat_id in fixtures["understat_id"]
            )
        rows.append(
            {
                "source": "understat_shots",
                "season": season,
                "fixtures_cached": fixtures_path.exists(),
                "n_matches_expected": n_matches_expected,
                "n_matches_shot_cached": n_matches_shot_cached,
                "aggregate_v1_cached": (UNDERSTAT_SHOTS_CACHE_DIR / f"_aggregate_{season}.csv").exists(),
                "aggregate_v2_cached": (UNDERSTAT_SHOTS_CACHE_DIR / f"_aggregate_v2_{season}.csv").exists(),
            }
        )
    return pd.DataFrame(rows)


def audit_fpl_history(seasons: list[str] | None = None) -> pd.DataFrame:
    seasons = seasons or fpl_history.default_completed_seasons(n=8)
    rows = []
    for season in seasons:
        cache_path = FPL_HISTORY_CACHE_DIR / f"{season}.csv"
        rows.append(
            {
                "source": "fpl_history",
                "season": season,
                "cached": cache_path.exists(),
                "cache_age_days": _cache_age_days(cache_path),
            }
        )
    return pd.DataFrame(rows)


def run_full_audit(match_seasons: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """One call for the pre-flight report every expanded-window run should
    check first. `match_seasons` is football-data.co.uk's own season format
    (e.g. `"2014-2015"`) and governs the football_data/understat audits,
    translated to each source's own season-string convention internally
    (Understat: `"2014"`). `fpl_history` uses its own default range
    regardless — its coverage is intentionally shallower than the match
    data (see docs/AI_CONTINUITY.md's vaastav-archive caveat), so forcing
    it onto the same 12-season window would just report it as missing for
    seasons it was never expected to cover."""
    match_seasons = match_seasons or football_data.default_completed_seasons(n=12)
    understat_seasons = [season.split("-")[0] for season in match_seasons]
    return {
        "football_data": audit_football_data(match_seasons),
        "understat_xg": audit_understat_xg(understat_seasons),
        "understat_shots": audit_understat_shots(understat_seasons),
        "fpl_history": audit_fpl_history(),
    }


def print_audit_summary(audit: dict[str, pd.DataFrame]) -> None:
    for name, df in audit.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    print_audit_summary(run_full_audit())
