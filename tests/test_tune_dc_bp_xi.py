from pl_predictor.evaluate.tune_dc_bp_xi import evaluate_xi_grid, prepare_folds, summarize


def test_evaluate_xi_grid_produces_one_row_per_xi_per_fold():
    folds = prepare_folds(seasons=["2023-2024", "2024-2025", "2025-2026"], min_train_seasons=2)
    assert len(folds) == 1

    result = evaluate_xi_grid("dixon_coles", folds, xi_grid=[0.0018, 0.005])

    assert len(result) == 2
    assert set(result["xi"]) == {0.0018, 0.005}
    assert (result["rps"] > 0).all()


def test_summarize_averages_across_folds():
    folds = prepare_folds(seasons=["2022-2023", "2023-2024", "2024-2025", "2025-2026"], min_train_seasons=2)
    assert len(folds) == 2

    result = evaluate_xi_grid("bivariate_poisson", folds, xi_grid=[0.0018])
    summary = summarize(result)

    assert len(summary) == 1
    assert summary.iloc[0]["rps"] == result["rps"].mean()
