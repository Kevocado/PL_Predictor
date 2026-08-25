import pandas as pd

from pl_predictor.api.routes import _fixture_actual_stats


def test_completed_fixture_uses_final_team_statistics_when_available():
    matches = pd.DataFrame([{
        "team_home": "Arsenal", "team_away": "Chelsea", "date": pd.Timestamp("2026-08-22T15:00:00Z"),
        "goals_home": 2, "goals_away": 1, "hs": 14, "as": 7, "hst": 6, "ast": 3,
        "hp": 61.4, "ap": 38.6, "hc": 5, "ac": 2, "hf": 9, "af": 12,
        "hy": 1, "ay": 3, "hr": 0, "ar": 0,
    }])

    stats = _fixture_actual_stats(matches, "Arsenal", "Chelsea", pd.Timestamp("2026-08-22T15:00:00Z"))

    assert stats is not None
    assert stats.home["Goals"] == 2
    assert stats.away["Corners"] == 2
    assert stats.home["Possession %"] == 61.4
