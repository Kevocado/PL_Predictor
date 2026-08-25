import pandas as pd

from pl_predictor.evaluate import power_ranking_research


def test_future_points_per_game_uses_only_future_matches():
    future = pd.DataFrame(
        [
            {"team_home": "A", "team_away": "B", "goals_home": 2, "goals_away": 0},
            {"team_home": "A", "team_away": "C", "goals_home": 1, "goals_away": 1},
        ]
    )

    result = power_ranking_research._future_points_per_game(future)

    assert result == {"A": 2.0, "B": 0.0, "C": 1.0}
