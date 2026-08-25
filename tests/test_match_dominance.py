import pandas as pd
import pytest

from pl_predictor.features import match_dominance


def _dominance_df():
    dates = pd.date_range("2023-08-01", periods=3, freq="7D")
    rows = []
    for i, d in enumerate(dates):
        home_team, away_team = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append(
            {
                "date": d,
                "team_home": home_team,
                "team_away": away_team,
                "home_total_xg": 1.0 + i,
                "away_total_xg": 0.5,
                "home_non_penalty_xg": 1.0 + i,
                "away_non_penalty_xg": 0.5,
                "home_shots": 10,
                "away_shots": 6,
                "home_xg_per_shot": 0.1,
                "away_xg_per_shot": 0.1,
                "home_open_play_xg_share": 0.8,
                "away_open_play_xg_share": 0.6,
                "home_set_piece_xg_share": 0.2,
                "away_set_piece_xg_share": 0.4,
                "home_avg_shot_distance": 0.15,
                "away_avg_shot_distance": 0.2,
            }
        )
    return pd.DataFrame(rows)


def test_build_rolling_dominance_includes_each_row_own_match():
    """`build_rolling_dominance`'s raw long frame is intentionally *not*
    pre-match — a team's first-match row equals that match's own value,
    not NaN. The no-lookahead guarantee lives in `attach_dominance_
    features`'s merge_asof join (see that function's docstring), not here;
    see test_attach_dominance_features_includes_the_most_recent_prior_match
    for the actual leakage-safety check."""
    long_df, cols = match_dominance.build_rolling_dominance(_dominance_df())
    first_rows = long_df.sort_values("date").groupby("team", sort=False).head(1)
    assert first_rows["total_xg_last_5"].tolist() == pytest.approx(first_rows["total_xg"].tolist())


def test_attach_dominance_features_includes_the_most_recent_prior_match():
    dominance_df = _dominance_df()
    target_date = dominance_df["date"].iloc[-1] + pd.Timedelta(days=7)
    matches_df = pd.DataFrame([{"date": target_date, "team_home": "A", "team_away": "B"}])

    out, _ = match_dominance.attach_dominance_features(matches_df, dominance_df)

    # Team A's total_xg across its 3 matches (as home on i=0,2, away on i=1): 1.0, 0.5, 3.0
    assert out.loc[0, "home_total_xg_last_5"] == pytest.approx((1.0 + 0.5 + 3.0) / 3)


def test_attach_dominance_features_empty_input_returns_nan_columns():
    matches_df = pd.DataFrame([{"date": pd.Timestamp("2023-08-01"), "team_home": "A", "team_away": "B"}])
    out, cols = match_dominance.attach_dominance_features(matches_df, pd.DataFrame())
    assert out[cols].isna().all().all()
