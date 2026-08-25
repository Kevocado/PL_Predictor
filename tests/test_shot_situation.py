import pandas as pd
import pytest

from pl_predictor.features import shot_situation


def test_attach_shot_situation_features_includes_the_most_recent_prior_match():
    """Same regression as tests/test_xg_form.py — this module shares the
    identical merge_asof pattern (and, before the fix, the identical
    one-match-stale bug)."""
    dates = pd.date_range("2023-08-01", periods=3, freq="7D")
    shot_df = pd.DataFrame(
        [
            {"date": dates[0], "team_home": "A", "team_away": "B", "home_set_piece_xg_share": 0.1, "away_set_piece_xg_share": 0.4},
            {"date": dates[1], "team_home": "B", "team_away": "A", "home_set_piece_xg_share": 0.5, "away_set_piece_xg_share": 0.3},
            {"date": dates[2], "team_home": "A", "team_away": "B", "home_set_piece_xg_share": 0.2, "away_set_piece_xg_share": 0.6},
        ]
    )
    target_date = dates[-1] + pd.Timedelta(days=7)
    matches_df = pd.DataFrame([{"date": target_date, "team_home": "A", "team_away": "B"}])

    out, _ = shot_situation.attach_shot_situation_features(matches_df, shot_df)

    # Team A's set_piece_xg_share across its 3 known matches: 0.1, 0.3, 0.2
    assert out.loc[0, "home_set_piece_xg_share_last_5"] == pytest.approx((0.1 + 0.3 + 0.2) / 3)
