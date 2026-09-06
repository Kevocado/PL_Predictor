"""Failing tests for player shots / SoT market (Task 1 of 2026-09-05 plan)."""
import pytest
from pl_predictor.features import player_form
from pl_predictor.models import player_goals


def test_rate_stats_includes_shots_and_shots_on_target():
    assert "shots" in player_form.RATE_STATS
    assert "shots_on_target" in player_form.RATE_STATS


def test_predict_player_computes_shots_scale():
    # If shots_scale logic missing, this fails with KeyError or missing keys
    result = player_goals.predict_player(
        rates={"avg_minutes": 90, "shots_per90": 2.0, "shots_on_target_per90": 0.7},
        team_goal_expectation=1.8, availability=1.0,
    )
    assert "expected_shots" in result
    assert "expected_shots_on_target" in result
    assert "anytime_shot_on_target_prob" in result
    assert result["expected_shots"] > 0
    assert result["expected_shots_on_target"] > 0
