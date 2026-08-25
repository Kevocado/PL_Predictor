import pandas as pd

from pl_predictor.evaluate.model_selection_by_segment import _is_cold_start_involved, evaluate_models_by_segment


def test_is_cold_start_involved_flags_either_side():
    val_df = pd.DataFrame(
        {
            "confidence_home": ["current", "blended", "none", "current"],
            "confidence_away": ["current", "current", "current", "blended"],
        }
    )
    result = _is_cold_start_involved(val_df).tolist()
    assert result == [False, True, True, True]


def test_evaluate_models_by_segment_covers_all_three_models_and_segments():
    result = evaluate_models_by_segment(seasons=["2023-2024", "2024-2025", "2025-2026"], min_train_seasons=2)

    assert set(result["model"]) == {"dixon_coles", "bivariate_poisson", "ml_scoreline"}
    assert set(result["segment"]) == {"overall", "cold_start_involved", "established_only"}
    # overall must equal cold_start_involved + established_only fixture counts
    overall_n = result[(result["segment"] == "overall") & (result["model"] == "ml_scoreline")]["n_fixtures"].iloc[0]
    cold_n = result[(result["segment"] == "cold_start_involved") & (result["model"] == "ml_scoreline")]["n_fixtures"].iloc[0]
    established_n = result[(result["segment"] == "established_only") & (result["model"] == "ml_scoreline")]["n_fixtures"].iloc[0]
    assert overall_n == cold_n + established_n
