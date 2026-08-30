import pandas as pd
import pytest

from pl_predictor.evaluate.goal_contribution_research import (
    _historical_role_quality,
    build_goal_contribution_frame,
    build_projected_team_player_features,
    summarise_team_unit_experiment,
)


def test_historical_role_quality_ceilings_are_equalised_across_positions():
    """The no-lookahead research feature must not grant a role a higher ceiling."""
    maxed_out = pd.DataFrame(
        [
            {
                "position": "GK", "saves_per90_last10": 2.25,
                "clean_sheets_per90_last10": 0.5, "bps_last10": 13 / 3,
                "starts_last10": 10.0,
                "minutes_ema": 90.0,
            },
            {
                "position": "DEF", "clean_sheets_per90_last10": 0.5,
                "defensive_contribution_last10": 50.0,
                "expected_goal_involvements_per90_last10": 0.3,
                "bps_last10": 40 / 9, "starts_last10": 10.0, "minutes_ema": 90.0,
            },
            {
                "position": "MID", "expected_goal_involvements_per90_last10": 0.52,
                "expected_assists_per90_last10": 0.4, "goals_per90_last10": 4 / 9,
                "starts_last10": 10.0, "minutes_ema": 90.0,
            },
            {
                "position": "FWD", "expected_goals_per90_last10": 14 / 38,
                "expected_goal_involvements_per90_last10": 0.45,
                "goals_per90_last10": 5 / 12, "starts_last10": 10.0, "minutes_ema": 90.0,
            },
        ]
    )

    result = _historical_role_quality(maxed_out)

    # Each role contributes 50% of its former own ceiling. Rescaling makes
    # that 23 points for every role, well below the global 92 cap.
    assert result.tolist() == pytest.approx([73.0, 73.0, 73.0, 73.0], abs=0.1)


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
