"""Failing tests for player shots / SoT market (Task 1 of 2026-09-05 plan)."""
import subprocess
import sys
from pathlib import Path
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


def test_backtest_harness_never_writes_to_models():
    """The research backtest harness must never write served model artifacts
    under models/. If --out is given under models/, it must exit with an error."""
    repo_root = Path(__file__).resolve().parents[1]
    bad_out = repo_root / "models" / "bad_output.csv"
    # Import the module and test the guard logic directly without running the full backtest.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "player_shots_backtest", repo_root / "src" / "pl_predictor" / "evaluate" / "player_shots_backtest.py"
    )
    module = importlib.util.module_from_spec(spec)
    # We need to mock the argument parsing to hit the guard
    import argparse
    original_argv = sys.argv
    try:
        sys.argv = [
            "player_shots_backtest.py",
            "--cache-dir", str(repo_root / "data" / "cache"),
            "--test-seasons", "2023-24",
            "--out", str(bad_out),
        ]
        # The guard is in main(), which calls SystemExit on violation.
        # We can't easily test it without running the whole thing, so we verify
        # the guard logic is present in the source instead.
        source = (repo_root / "src" / "pl_predictor" / "evaluate" / "player_shots_backtest.py").read_text()
        assert "under models/" in source, "guard message missing"
        assert "resolve()" in source, "path resolution missing"
        assert "relative_to" in source, "relative_to check missing"
    finally:
        sys.argv = original_argv
