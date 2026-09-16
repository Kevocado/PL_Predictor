"""Unit tests for the pure, fast parts of research/preseason_vs_current_season.py
(chunking and metric aggregation) -- the full run_study() end-to-end is
deliberately untested here, same as evaluate/walk_forward.py itself: it
does real multi-season feature engineering + several XGBoost fits and
takes minutes, not something a unit test suite should pay for on every
run."""

import numpy as np
import pandas as pd

from pl_predictor.research.preseason_vs_current_season import _metrics_from_probs, chunk_by_gameweek


def test_chunk_by_gameweek_splits_in_chronological_order():
    df = pd.DataFrame({"date": pd.to_datetime([f"2024-08-{d:02d}" for d in range(1, 26)]), "match": range(1, 26)})
    # Shuffle to confirm chunking sorts by date first, not input order.
    shuffled = df.sample(frac=1, random_state=0)

    chunks = chunk_by_gameweek(shuffled, chunk_size=10)

    assert [len(c) for c in chunks] == [10, 10, 5]
    assert chunks[0]["match"].tolist() == list(range(1, 11))
    assert chunks[1]["match"].tolist() == list(range(11, 21))
    assert chunks[2]["match"].tolist() == list(range(21, 26))


def test_chunk_by_gameweek_empty_input():
    df = pd.DataFrame(columns=["date"])
    assert chunk_by_gameweek(df) == []


def test_metrics_from_probs_matches_perfect_prediction():
    # Three matches, model puts 100% on the actual outcome each time --
    # RPS and Brier should both come out to (near) zero.
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])

    metrics = _metrics_from_probs(probs, outcomes)

    assert metrics["n"] == 3
    assert metrics["rps"] < 1e-6
    assert metrics["brier"] < 1e-6
