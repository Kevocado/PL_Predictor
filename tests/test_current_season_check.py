import pandas as pd
import pytest

from pl_predictor.evaluate import current_season_check


def test_evaluate_count_market_arms_on_current_season_flags_low_power(monkeypatch):
    """With fewer fixtures than MIN_FIXTURES_FOR_A_DECISION, the result must
    say so explicitly (has_enough_power=False) rather than silently
    returning a table that looks as authoritative as the full walk-forward
    comparison."""
    small_current_season = pd.DataFrame(
        {
            "date": pd.date_range("2026-08-21", periods=10, freq="1D"),
            "team_home": ["Arsenal"] * 10,
            "team_away": ["Chelsea"] * 10,
            "goals_home": [1] * 10,
            "goals_away": [1] * 10,
            "ftr": ["D"] * 10,
            "hc": [5] * 10,
            "ac": [4] * 10,
            "hy": [1] * 10,
            "ay": [1] * 10,
            "hr": [0] * 10,
            "ar": [0] * 10,
            "season": ["2026-2027"] * 10,
        }
    )
    monkeypatch.setattr(current_season_check.football_data, "fetch_current_season_partial", lambda: small_current_season)
    monkeypatch.setattr(current_season_check.football_data, "CURRENT_SEASON_START_YEAR", 2026)

    result = current_season_check.evaluate_count_market_arms_on_current_season("total_corners")

    assert (result["n_fixtures"] == 10).all()
    assert result.attrs["has_enough_power"] is False


def test_evaluate_count_market_arms_on_current_season_raises_when_nothing_played_yet(monkeypatch):
    monkeypatch.setattr(current_season_check.football_data, "fetch_current_season_partial", lambda: pd.DataFrame())

    with pytest.raises(RuntimeError, match="No completed current-season fixtures"):
        current_season_check.evaluate_count_market_arms_on_current_season("total_corners")
