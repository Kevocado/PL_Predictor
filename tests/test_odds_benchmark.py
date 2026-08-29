import pandas as pd

from pl_predictor.evaluate.odds_benchmark import closing_odds_benchmark


def test_closing_benchmark_is_explicitly_not_deployable():
    result = closing_odds_benchmark(pd.DataFrame({
        "psch": [2.0, 3.0], "pscd": [3.0, 3.2], "psca": [4.0, 2.5], "ftr": ["H", "A"],
    }))
    assert result["available"] is True
    assert result["deployable"] is False
    assert result["n_fixtures"] == 2
