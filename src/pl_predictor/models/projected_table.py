"""projected_table.py — current standings plus a projected final table.

The projection is deterministic, not simulated: `FootballProbabilityGrid`
already exposes `expected_points_home`/`expected_points_away` for a fixture
(3*P(win) + 1*P(draw), the standard "expected points" definition used
throughout football analytics), so projecting the rest of the season is
just summing those over each team's remaining fixtures — no Monte Carlo
sampling needed for a points table.
"""

from __future__ import annotations

import pandas as pd

from ..features.rolling_form import to_team_perspective
from . import scoreline


def compute_standings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Actual current-season standings from played matches so far. Columns:
    team, played, points, goals_for, goals_against, goal_diff."""
    long_df = to_team_perspective(matches_df)
    standings = (
        long_df.groupby("team")
        .agg(played=("points", "size"), points=("points", "sum"), goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"))
        .reset_index()
    )
    standings["goal_diff"] = standings["goals_for"] - standings["goals_against"]
    return standings.sort_values(["points", "goal_diff"], ascending=False).reset_index(drop=True)


def project_table(model, upcoming_fixtures_df: pd.DataFrame, standings: pd.DataFrame) -> list[dict]:
    """Adds each team's summed expected points/goal-diff from their
    remaining fixtures onto their current actual totals, ranked by
    projected final points."""
    projected_points = {row["team"]: 0.0 for _, row in standings.iterrows()}
    projected_gd = {row["team"]: 0.0 for _, row in standings.iterrows()}

    for _, fixture in upcoming_fixtures_df.iterrows():
        home, away = fixture["team_home"], fixture["team_away"]
        pred = scoreline.predict_fixture(model, home, away)

        for team in (home, away):
            projected_points.setdefault(team, 0.0)
            projected_gd.setdefault(team, 0.0)

        projected_points[home] += pred["home_win"] * 3 + pred["draw"] * 1
        projected_points[away] += pred["away_win"] * 3 + pred["draw"] * 1
        goal_diff_expectation = pred["home_goal_expectation"] - pred["away_goal_expectation"]
        projected_gd[home] += goal_diff_expectation
        projected_gd[away] -= goal_diff_expectation

    standings_by_team = standings.set_index("team")
    rows = []
    for team in projected_points:
        current = standings_by_team.loc[team] if team in standings_by_team.index else None
        current_points = float(current["points"]) if current is not None else 0.0
        current_gd = float(current["goal_diff"]) if current is not None else 0.0
        current_played = int(current["played"]) if current is not None else 0

        rows.append(
            {
                "team": team,
                "played": current_played,
                "current_points": current_points,
                "projected_points": round(current_points + projected_points[team], 1),
                "projected_goal_diff": round(current_gd + projected_gd[team], 1),
            }
        )

    rows.sort(key=lambda r: (-r["projected_points"], -r["projected_goal_diff"]))
    for i, row in enumerate(rows, start=1):
        row["projected_position"] = i
    return rows
