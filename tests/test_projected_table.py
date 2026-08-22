"""Sanity checks for the deterministic expected-points table projection."""

import pandas as pd
import penaltyblog as pb
import pytest

from pl_predictor.models.projected_table import compute_standings, project_table


@pytest.fixture(scope="module")
def fitted_model():
    # A tiny synthetic league where "Strong" always beats "Weak" at a
    # realistic scoreline, fit on enough matches for the model to learn it
    # decisively.
    rows = []
    for _ in range(10):
        rows.append({"team_home": "Strong", "team_away": "Weak", "goals_home": 3, "goals_away": 0})
        rows.append({"team_home": "Weak", "team_away": "Strong", "goals_home": 0, "goals_away": 3})
    df = pd.DataFrame(rows)
    model = pb.models.DixonColesGoalModel(
        df["goals_home"].to_numpy().astype(float),
        df["goals_away"].to_numpy().astype(float),
        df["team_home"].to_numpy().astype(str),
        df["team_away"].to_numpy().astype(str),
    )
    model.fit()
    return model


def test_compute_standings_empty_is_safe():
    empty = pd.DataFrame(columns=["team_home", "team_away", "goals_home", "goals_away", "date", "ftr", "season"])
    standings = compute_standings(empty)
    assert standings.empty


def test_dominant_team_projects_top(fitted_model):
    empty_standings = pd.DataFrame(columns=["team", "played", "points", "goals_for", "goals_against", "goal_diff"])
    remaining_fixtures = pd.DataFrame(
        [
            {"team_home": "Strong", "team_away": "Weak"},
            {"team_home": "Weak", "team_away": "Strong"},
        ]
    )
    table = project_table(fitted_model, remaining_fixtures, empty_standings)
    by_team = {row["team"]: row for row in table}

    assert by_team["Strong"]["projected_position"] == 1
    assert by_team["Strong"]["projected_points"] > by_team["Weak"]["projected_points"]
    assert by_team["Strong"]["projected_goal_diff"] > 0
    assert by_team["Weak"]["projected_goal_diff"] < 0
