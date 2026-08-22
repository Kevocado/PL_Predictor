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
