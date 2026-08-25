import pandas as pd
import pytest

from pl_predictor.evaluate import covariate_poisson_research as cpr


def _synthetic_matches():
    dates = pd.date_range("2023-08-01", periods=6, freq="7D")
    rows = [
        {"date": dates[0], "team_home": "A", "team_away": "B", "goals_home": 2, "goals_away": 1, "elo_home": 1600, "elo_away": 1400, "pi_home": 0.3, "pi_away": -0.1},
        {"date": dates[1], "team_home": "B", "team_away": "C", "goals_home": 0, "goals_away": 0, "elo_home": 1420, "elo_away": 1500, "pi_home": -0.05, "pi_away": 0.2},
        {"date": dates[2], "team_home": "C", "team_away": "A", "goals_home": 1, "goals_away": 3, "elo_home": 1490, "elo_away": 1620, "pi_home": 0.15, "pi_away": 0.35},
        {"date": dates[3], "team_home": "A", "team_away": "C", "goals_home": 2, "goals_away": 0, "elo_home": 1630, "elo_away": 1480, "pi_home": 0.4, "pi_away": 0.1},
        {"date": dates[4], "team_home": "B", "team_away": "A", "goals_home": 1, "goals_away": 2, "elo_home": 1430, "elo_away": 1640, "pi_home": -0.02, "pi_away": 0.42},
        {"date": dates[5], "team_home": "C", "team_away": "B", "goals_home": 1, "goals_away": 1, "elo_home": 1495, "elo_away": 1440, "pi_home": 0.18, "pi_away": 0.0},
    ]
    return pd.DataFrame(rows)


def test_team_perspective_produces_two_rows_per_match_with_signed_covariates():
    matches = _synthetic_matches()
    long_df = cpr._team_perspective(matches)

    assert len(long_df) == 2 * len(matches)
    first_home_row = long_df.iloc[0]
    assert first_home_row["attack_team"] == "A"
    assert first_home_row["defence_team"] == "B"
    assert first_home_row["is_home"] == 1.0
    assert first_home_row["elo_diff_signed"] == pytest.approx(1600 - 1400)

    # The mirrored away-perspective row for the same match has the sign flipped.
    away_row = long_df.iloc[len(matches)]
    assert away_row["attack_team"] == "B"
    assert away_row["defence_team"] == "A"
    assert away_row["elo_diff_signed"] == pytest.approx(1400 - 1600)


def test_fit_and_predict_roundtrip_produces_valid_grids():
    matches = _synthetic_matches()
    train_df, val_df = matches.iloc[:4], matches.iloc[4:]

    model, columns, team_categories = cpr.fit(train_df, covariate_cols=["elo_diff_signed", "pi_diff_signed"])
    grids = cpr.predict_grids_batch(
        model, columns, team_categories, val_df, covariate_cols=["elo_diff_signed", "pi_diff_signed"]
    )

    assert len(grids) == len(val_df)
    for grid in grids:
        probs = grid.home_win + grid.draw + grid.away_win
        assert probs == pytest.approx(1.0, abs=1e-6)


def test_predict_handles_a_team_unseen_at_fit_time_without_erroring():
    matches = _synthetic_matches()
    train_df = matches[matches["team_home"].isin(["A", "B"]) & matches["team_away"].isin(["A", "B"])]
    unseen_team_fixture = pd.DataFrame(
        [{"date": pd.Timestamp("2023-10-01"), "team_home": "A", "team_away": "NewTeam", "goals_home": 1, "goals_away": 1, "elo_home": 1600, "elo_away": 1500, "pi_home": 0.3, "pi_away": 0.0}]
    )

    model, columns, team_categories = cpr.fit(train_df, covariate_cols=["elo_diff_signed", "pi_diff_signed"])
    grids = cpr.predict_grids_batch(
        model, columns, team_categories, unseen_team_fixture, covariate_cols=["elo_diff_signed", "pi_diff_signed"]
    )

    assert len(grids) == 1
    assert grids[0].home_win + grids[0].draw + grids[0].away_win == pytest.approx(1.0, abs=1e-6)
