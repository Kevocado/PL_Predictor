from pl_predictor.evaluate.goal_contribution_research import build_projected_team_player_features
from pl_predictor.evaluate.walk_forward import prepare_folds


def test_prepare_folds_merges_extra_feature_frame():
    """EXP-2026-03's walk-forward follow-up depends on `prepare_folds` being
    able to inject a candidate feature set (built on a different season-
    string convention than football_data's) onto every fold without
    touching the baseline. Uses two seasons FPL history actually covers so
    the merge produces real, non-zero values rather than an all-fallback
    left join."""
    aggregates = build_projected_team_player_features(seasons=["2022-23", "2023-24"])
    aggregate_cols = [
        column
        for column in aggregates.columns
        if column.startswith(("home_projected_", "away_projected_", "home_top_", "away_top_"))
    ]

    candidate_folds = prepare_folds(
        seasons=["2023-2024", "2024-2025"],
        min_train_seasons=1,
        extra_feature_frame=aggregates,
        extra_feature_cols=aggregate_cols,
    )
    baseline_folds = prepare_folds(seasons=["2023-2024", "2024-2025"], min_train_seasons=1)

    assert len(candidate_folds) == len(baseline_folds) == 1
    x_train = candidate_folds[0]["X_train"]
    assert all(column in x_train.columns for column in aggregate_cols)
    assert (x_train["home_projected_goal_rate"] != 0).mean() > 0.5

    # The baseline fold must be completely unaffected by the candidate's
    # extra columns — this is an additive, opt-in merge, not a shared state.
    assert not any("projected" in column for column in baseline_folds[0]["X_train"].columns)
