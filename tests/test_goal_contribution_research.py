import pandas as pd

from pl_predictor.evaluate.goal_contribution_research import (
    build_goal_contribution_frame,
    build_projected_team_player_features,
    summarise_team_unit_experiment,
)


def test_goal_contribution_frame_has_only_prior_form_features():
    frame, _ = build_goal_contribution_frame(seasons=["2023-24"])
    first_rows = frame.sort_values(["season", "element", "kickoff_time"]).groupby(["season", "element"], sort=False).head(1)
    assert (first_rows["goals_per90_last3"] == 0).all()
    assert (first_rows["assists_per90_last3"] == 0).all()
    assert (first_rows["expected_minutes_pre_match"] == 0).all()


def test_team_unit_features_are_eight_causal_expected_xi_aggregates():
    features = build_projected_team_player_features(seasons=["2023-24"])
    units = [column for column in features if column.endswith("_unit_strength")]

    assert len(units) == 8
    assert {"home_gk_unit_strength", "away_fwd_unit_strength"} <= set(units)
    first = features.sort_values("kickoff_date").iloc[0]
    assert first[units].fillna(0).sum() == 0


def test_team_unit_summary_cannot_promote_automatically():
    report = summarise_team_unit_experiment(
        pd.DataFrame(
            [
                {"model": "production_features", "rps": 0.20, "brier": 0.40, "ece": 0.04, "log_loss": 0.90, "coverage": 1.0},
                {"model": "role_unit_strength", "rps": 0.19, "brier": 0.39, "ece": 0.04, "log_loss": 0.88, "coverage": 1.0},
            ]
        )
    )

    assert report["promotion_eligible"] is False
    assert report["candidate_metrics"]["rps"] == 0.19
