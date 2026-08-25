import numpy as np
import pandas as pd
import penaltyblog as pb
import pytest

from pl_predictor.models import scoreline
from pl_predictor.models.scoreline import bootstrap_ci, multiclass_top_label_ece


def test_multiclass_top_label_ece_is_zero_for_perfectly_calibrated_predictions():
    # Model always predicts the true class with certainty -> perfect calibration.
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert multiclass_top_label_ece(probs, outcomes) == pytest.approx(0.0)


def test_multiclass_top_label_ece_penalizes_overconfident_wrong_predictions():
    # Confident (0.9) but always wrong -> high calibration error.
    probs = np.array([[0.9, 0.05, 0.05]] * 4)
    outcomes = np.array([1, 2, 1, 2])
    ece = multiclass_top_label_ece(probs, outcomes)
    assert ece > 0.5


def test_bootstrap_ci_brackets_the_true_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.2, scale=0.05, size=200)
    low, high = bootstrap_ci(values, n_resamples=500)
    assert low < values.mean() < high


def test_bootstrap_ci_handles_empty_input():
    low, high = bootstrap_ci(np.array([]))
    assert np.isnan(low) and np.isnan(high)


class _FakeModel:
    """Minimal stand-in satisfying scoreline.predict_fixture's contract:
    no .context (not feature-driven), a fixed .teams list, and a
    deterministic .predict so overrides are easy to verify precisely."""

    def __init__(self, teams, lam_home, lam_away):
        self.teams = teams
        self.lam_home = lam_home
        self.lam_away = lam_away

    def predict(self, home, away, max_goals=10):
        return pb.models.create_dixon_coles_grid(self.lam_home, self.lam_away, rho=0.0, max_goals=max_goals)


def test_predict_fixture_market_override_replaces_only_that_markets_fields():
    primary = _FakeModel(["A", "B"], lam_home=1.5, lam_away=1.0)
    override = _FakeModel(["A", "B"], lam_home=2.5, lam_away=0.2)

    plain = scoreline.predict_fixture(primary, "A", "B")
    overridden = scoreline.predict_fixture(primary, "A", "B", market_overrides={"over_2_5": override})

    # The overridden market's fields differ from the primary-only result...
    assert overridden["over_2_5"] != plain["over_2_5"]
    assert overridden["under_2_5"] != plain["under_2_5"]
    # ...and match what the override model alone would have produced.
    override_only = scoreline.predict_fixture(override, "A", "B")
    assert overridden["over_2_5"] == pytest.approx(override_only["over_2_5"])
    assert overridden["under_2_5"] == pytest.approx(override_only["under_2_5"])

    # Every other market stays exactly the primary model's own value.
    for field in ("home_win", "draw", "away_win", "btts_yes", "btts_no", "top_scorelines"):
        assert overridden[field] == plain[field]

    assert overridden["market_model_overrides"] == ["over_2_5"]
    assert "market_model_overrides" not in plain


def test_predict_fixtures_batch_market_override_matches_per_fixture_result():
    primary = _FakeModel(["A", "B", "C"], lam_home=1.5, lam_away=1.0)
    override = _FakeModel(["A", "B", "C"], lam_home=2.5, lam_away=0.2)
    fixtures_df = pd.DataFrame([{"team_home": "A", "team_away": "B"}, {"team_home": "B", "team_away": "C"}])

    batch_results = scoreline.predict_fixtures_batch(primary, fixtures_df, market_overrides={"btts": override})

    for i, row in fixtures_df.iterrows():
        single = scoreline.predict_fixture(primary, row["team_home"], row["team_away"], market_overrides={"btts": override})
        assert batch_results[i]["btts_yes"] == pytest.approx(single["btts_yes"])
        assert batch_results[i]["home_win"] == pytest.approx(single["home_win"])
        assert batch_results[i]["market_model_overrides"] == ["btts"]
