"""squad_change.py — a leading indicator of off-season squad-strength
change: what fraction of a team's total playing-time minutes last season
came from players still registered to that team this season.

Motivation (see docs/AI_CONTINUITY.md EXP-2026-18): the model's only
channel for team strength is Elo/Pi ratings carried over from last season
plus this-season match evidence — there is no signal for a real squad
change (transfers in/out) until it's been proven by results, which takes
many games. This is a candidate leading indicator, available before a
ball is kicked, that's orthogonal to both of those.

No-lookahead note, worth reading before changing this file: "who's on the
squad this season" cannot be known from match-*minutes* data before any
matches are played — but FPL's gameweek-1 squad list for a season is
fixed at that season's price-release/deadline, well before kickoff, and
includes every registered squad member regardless of whether they ever
play. Confirmed directly against a real season's data: a GW1 slice has
dozens of 0-minute rows per club, far more than a matchday squad — this
is genuinely the full registered squad, not who started or was even
named to a bench. Using "has a GW1 row at all" (not "played minutes in
GW1") as the current-season squad signal is what makes this safe to
compute for every fixture of a season, including GW1 itself, without
waiting for any match to happen.
"""

from __future__ import annotations

import pandas as pd

from ..data import football_data, fpl_history, team_names


def _team_rows(history: pd.DataFrame, team: str, season: str) -> pd.DataFrame:
    season_rows = history[(history["season"] == season) & history["team"].notna()]
    canonical = season_rows["team"].map(lambda t: team_names.to_canonical(t, source="fpl"))
    return season_rows[canonical == team]


def squad_continuity(team: str, start_year: int, history: pd.DataFrame) -> float | None:
    """Fraction of `team`'s total minutes in the season starting
    `start_year - 1` that were played by players still registered to
    `team` for the season starting `start_year` (see module docstring for
    why "registered", not "played", is the safe signal to use here).

    Joins players across the season boundary by `name`, not FPL's numeric
    `element` id — confirmed directly against real data that `element` is
    **not stable across seasons** (e.g. Bruno Fernandes is element `333`
    in the 2022-23 archive and `373` in 2023-24, despite being the same
    player at the same club both seasons). Grouping by `element` across a
    season boundary would silently merge different players' minutes under
    a reused id. `name` isn't watertight either (a mid-season formatting
    change, or two players sharing a name, would misfire) but is far more
    reliable than a per-season-reassigned integer for this specific
    cross-season join; this module's own no-lookahead test pins the
    expected behavior against this real Bruno Fernandes case.

    Returns `None` when either season has no data for `team` at all —
    e.g. freshly promoted into `start_year`, or relegated out of
    `start_year - 1`. This feature is about an established team's squad
    changing, not about promotion, which `features/cold_start.py`
    already handles."""
    prior_season = fpl_history.season_str(start_year - 1)
    current_season = fpl_history.season_str(start_year)

    prior_rows = _team_rows(history, team, prior_season)
    if prior_rows.empty:
        return None
    prior_minutes = prior_rows.groupby("name")["minutes"].sum()
    total_prior_minutes = prior_minutes.sum()
    if total_prior_minutes == 0:
        return None

    current_rows = _team_rows(history, team, current_season)
    if current_rows.empty:
        return None
    gw1 = current_rows[current_rows["GW"] == current_rows["GW"].min()]
    registered_this_season = set(gw1["name"])

    retained_minutes = prior_minutes[prior_minutes.index.isin(registered_this_season)].sum()
    return float(retained_minutes / total_prior_minutes)


def team_season_continuity_table(seasons: list[str], history: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, team) with a `squad_continuity` value —
    `seasons` in football-data's own "YYYY-YYYY" format (matches
    `features.build.build_training_frame`'s own `season` column), so a
    caller can merge this onto a fixture frame by `team_home`/`team_away`
    separately (see `evaluate/squad_change_prior.py`, which does exactly
    that for `evaluate.walk_forward.prepare_folds`'s `extra_feature_frame`
    mechanism — this module only computes the signal, it doesn't shape it
    onto fixtures itself).

    Loads its own FPL history covering one extra season back (needed for
    the earliest requested season's own "prior season" lookup) unless
    `history` is already loaded."""
    start_years = sorted({int(s.split("-")[0]) for s in seasons})
    fpl_seasons = [fpl_history.season_str(y) for y in range(start_years[0] - 1, start_years[-1] + 1)]
    if history is None:
        history = fpl_history.load_player_gw_history(seasons=fpl_seasons)

    canonical_teams = sorted(
        {team_names.to_canonical(t, source="fpl") for t in history["team"].dropna().unique()}
    )

    rows = []
    for start_year in start_years:
        fd_season = football_data.season_str(start_year)
        for team in canonical_teams:
            continuity = squad_continuity(team, start_year, history)
            if continuity is not None:
                rows.append({"season": fd_season, "team": team, "squad_continuity": continuity})
    return pd.DataFrame(rows, columns=["season", "team", "squad_continuity"])
