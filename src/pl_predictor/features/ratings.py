"""ratings.py — team-strength ratings as pre-match features.

Elo and Pi ratings are stateful and updated match-by-match, so matches must
be replayed in chronological order. The critical discipline (easy to get
wrong, and the single easiest way to leak a match's own result into its own
pre-match feature): read each team's rating *before* calling
`update_ratings` for that match, then update after.

Massey/Colley are whole-history batch solves, so they're computed as-of a
cutoff date rather than replayed incrementally.
"""

from __future__ import annotations

import pandas as pd
import penaltyblog as pb

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}  # penaltyblog.ratings.Elo convention


def replay_elo(matches_df: pd.DataFrame, k: float = 20.0, home_field_advantage: float = 100.0) -> pd.DataFrame:
    """Returns a frame aligned to `matches_df`'s index with columns
    `elo_home`, `elo_away` — each team's Elo rating *before* that match."""
    df = matches_df.sort_values("date")
    elo = pb.ratings.Elo(k=k, home_field_advantage=home_field_advantage)

    home_ratings, away_ratings = [], []
    for _, row in df.iterrows():
        home, away = row["team_home"], row["team_away"]
        home_ratings.append(elo.get_team_rating(home))
        away_ratings.append(elo.get_team_rating(away))
        elo.update_ratings(home, away, _RESULT_CODE[row["ftr"]])

    out = pd.DataFrame({"elo_home": home_ratings, "elo_away": away_ratings}, index=df.index)
    return out.reindex(matches_df.index)


def replay_pi_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Returns `pi_home`, `pi_away` — each team's Pi rating before that
    match, analogous to `replay_elo`."""
    df = matches_df.sort_values("date")
    pi = pb.ratings.PiRatingSystem()

    home_ratings, away_ratings = [], []
    for _, row in df.iterrows():
        home, away = row["team_home"], row["team_away"]
        home_ratings.append(pi.get_team_rating(home))
        away_ratings.append(pi.get_team_rating(away))
        goal_diff = int(row["goals_home"] - row["goals_away"])
        pi.update_ratings(home, away, goal_diff, date=row["date"])

    out = pd.DataFrame({"pi_home": home_ratings, "pi_away": away_ratings}, index=df.index)
    return out.reindex(matches_df.index)


def fit_elo(matches_df: pd.DataFrame, k: float = 20.0, home_field_advantage: float = 100.0) -> pb.ratings.Elo:
    """Fit Elo over the full history and return the model itself (current
    ratings as of the last match) — used for scoring upcoming fixtures, as
    opposed to `replay_elo` which produces historical training features."""
    df = matches_df.sort_values("date")
    elo = pb.ratings.Elo(k=k, home_field_advantage=home_field_advantage)
    for _, row in df.iterrows():
        elo.update_ratings(row["team_home"], row["team_away"], _RESULT_CODE[row["ftr"]])
    return elo


def fit_pi_ratings(matches_df: pd.DataFrame) -> pb.ratings.PiRatingSystem:
    """Analogous to `fit_elo`: current Pi ratings as of the last match, for
    scoring upcoming fixtures."""
    df = matches_df.sort_values("date")
    pi = pb.ratings.PiRatingSystem()
    for _, row in df.iterrows():
        goal_diff = int(row["goals_home"] - row["goals_away"])
        pi.update_ratings(row["team_home"], row["team_away"], goal_diff, date=row["date"])
    return pi


def _gameweek_lookup(fd_org_matches: pd.DataFrame | None) -> pd.DataFrame | None:
    """`matches_df` (football-data.co.uk, replayed above) has no gameweek
    column — only `fd_org_matches` (football-data.org) does. Both describe
    the same real matches from different providers, so join on team pair +
    nearest date rather than assuming identical date formatting between
    the two (same idea `_fixture_actual_stats` in routes.py uses for a
    different pair of mismatched sources)."""
    if fd_org_matches is None or fd_org_matches.empty:
        return None
    lookup = fd_org_matches[["team_home", "team_away", "commence_time", "matchday"]].dropna(subset=["matchday"]).copy()
    if lookup.empty:
        return None
    lookup = lookup.rename(columns={"commence_time": "date"})
    lookup["date"] = pd.to_datetime(lookup["date"])
    if lookup["date"].dt.tz is not None:
        lookup["date"] = lookup["date"].dt.tz_localize(None)
    return lookup


def _nearest_gameweek(lookup: pd.DataFrame | None, home: str, away: str, date, max_days: int = 3) -> int | None:
    if lookup is None:
        return None
    candidates = lookup[(lookup["team_home"] == home) & (lookup["team_away"] == away)]
    if candidates.empty:
        return None
    date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date)
    deltas = (candidates["date"] - date).abs()
    nearest_idx = deltas.idxmin()
    if deltas.loc[nearest_idx] > pd.Timedelta(days=max_days):
        return None
    return int(candidates.loc[nearest_idx, "matchday"])


def team_rating_timeseries(
    matches_df: pd.DataFrame, fd_org_matches: pd.DataFrame | None = None, k: float = 20.0, home_field_advantage: float = 100.0
) -> pd.DataFrame:
    """Long-format *post*-match Elo/Pi rating for both sides of every match
    — for charting a team's rating trend over time. `replay_elo`/
    `replay_pi_ratings` return *pre*-match values instead, since those feed
    model training and must never see the match's own outcome; here the
    match's outcome is exactly what we want reflected.

    `fd_org_matches`, when given, attaches a `gameweek` to each row (see
    `_gameweek_lookup`/`_nearest_gameweek`) — used to group the chart by
    gameweek visually without changing its date-based x-axis."""
    df = matches_df.sort_values("date")
    elo = pb.ratings.Elo(k=k, home_field_advantage=home_field_advantage)
    pi = pb.ratings.PiRatingSystem()
    gameweek_lookup = _gameweek_lookup(fd_org_matches)

    rows = []
    for _, row in df.iterrows():
        home, away = row["team_home"], row["team_away"]
        elo.update_ratings(home, away, _RESULT_CODE[row["ftr"]])
        pi.update_ratings(home, away, int(row["goals_home"] - row["goals_away"]), date=row["date"])
        gameweek = _nearest_gameweek(gameweek_lookup, home, away, row["date"])

        for team in (home, away):
            rows.append(
                {"date": row["date"], "team": team, "elo": elo.get_team_rating(team), "pi": pi.get_team_rating(team), "gameweek": gameweek}
            )

    return pd.DataFrame(rows)


def massey_colley_snapshot(matches_df: pd.DataFrame, as_of_date) -> dict:
    """Massey and Colley ratings computed on matches strictly before
    `as_of_date` — a point-in-time snapshot, not a per-match replay."""
    cutoff_df = matches_df[matches_df["date"] < as_of_date]
    if cutoff_df.empty:
        return {"massey": {}, "colley": {}}

    massey = pb.ratings.Massey(
        cutoff_df["goals_home"], cutoff_df["goals_away"], cutoff_df["team_home"], cutoff_df["team_away"]
    )
    colley = pb.ratings.Colley(
        cutoff_df["goals_home"], cutoff_df["goals_away"], cutoff_df["team_home"], cutoff_df["team_away"]
    )
    return {"massey": massey.get_ratings(), "colley": colley.get_ratings()}
