import pandas as pd
import pytest

from pl_predictor.models import covariate_poisson


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


class _FakeRatingSystem:
    def __init__(self, ratings: dict, default: float):
        self.ratings = ratings
        self.default = default

    def get_team_rating(self, team):
        return self.ratings.get(team, self.default)


class _FakeContext:
    def __init__(self, elo: dict, pi: dict):
        self.elo = _FakeRatingSystem(elo, default=1500.0)
        self.pi = _FakeRatingSystem(pi, default=0.0)


def test_fit_returns_model_with_expected_teams():
    matches = _synthetic_matches()
    model = covariate_poisson.fit(matches)
    assert model.teams == ["A", "B", "C"]
    assert model.context is None


def test_predict_from_row_returns_a_valid_probability_grid():
    matches = _synthetic_matches()
    model = covariate_poisson.fit(matches.iloc[:4])

    grid = model.predict_from_row(matches.iloc[4])
    assert grid.home_win + grid.draw + grid.away_win == pytest.approx(1.0, abs=1e-6)


def test_predict_requires_context_for_live_fixtures():
    matches = _synthetic_matches()
    model = covariate_poisson.fit(matches)
    with pytest.raises(RuntimeError, match="context must be set"):
        model.predict("A", "B")


def test_predict_uses_context_elo_pi_and_returns_valid_grid():
    matches = _synthetic_matches()
    model = covariate_poisson.fit(matches)
    model.context = _FakeContext(elo={"A": 1700, "B": 1450, "C": 1500}, pi={"A": 0.5, "B": 0.0, "C": 0.1})

    grid = model.predict("A", "B")
    assert grid.home_win + grid.draw + grid.away_win == pytest.approx(1.0, abs=1e-6)
    # Given A is rated much stronger than B on both Elo and Pi, A should be
    # the clear favorite -- a sanity check the covariates are actually wired
    # in the right direction, not just present.
    assert grid.home_win > grid.away_win


def test_predict_grids_batch_matches_predict_from_row_per_fixture():
    matches = _synthetic_matches()
    train_df, val_df = matches.iloc[:4], matches.iloc[4:]
    model = covariate_poisson.fit(train_df)

    batch_grids = covariate_poisson.predict_grids_batch(model, val_df)
    for i, (_, row) in enumerate(val_df.iterrows()):
        single_grid = model.predict_from_row(row)
        assert batch_grids[i].home_win == pytest.approx(single_grid.home_win, abs=1e-6)
        assert batch_grids[i].away_win == pytest.approx(single_grid.away_win, abs=1e-6)


def test_predict_from_row_handles_a_team_unseen_at_fit_time():
    matches = _synthetic_matches()
    train_df = matches[matches["team_home"].isin(["A", "B"]) & matches["team_away"].isin(["A", "B"])]
    model = covariate_poisson.fit(train_df)

    unseen_row = pd.Series(
        {"team_home": "A", "team_away": "NewTeam", "elo_home": 1600, "elo_away": 1500, "pi_home": 0.3, "pi_away": 0.0}
    )
    grid = model.predict_from_row(unseen_row)
    assert grid.home_win + grid.draw + grid.away_win == pytest.approx(1.0, abs=1e-6)
