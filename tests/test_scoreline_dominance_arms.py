import numpy as np
import pytest

from pl_predictor.evaluate.scoreline_dominance_arms import _bootstrap_ci, _multiclass_top_label_ece


def test_multiclass_top_label_ece_is_zero_for_perfectly_calibrated_predictions():
    # Model always predicts the true class with certainty -> perfect calibration.
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert _multiclass_top_label_ece(probs, outcomes) == pytest.approx(0.0)


def test_multiclass_top_label_ece_penalizes_overconfident_wrong_predictions():
    # Confident (0.9) but always wrong -> high calibration error.
    probs = np.array([[0.9, 0.05, 0.05]] * 4)
    outcomes = np.array([1, 2, 1, 2])
    ece = _multiclass_top_label_ece(probs, outcomes)
    assert ece > 0.5


def test_bootstrap_ci_brackets_the_true_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.2, scale=0.05, size=200)
    low, high = _bootstrap_ci(values, n_resamples=500)
    assert low < values.mean() < high


def test_bootstrap_ci_handles_empty_input():
    low, high = _bootstrap_ci(np.array([]))
    assert np.isnan(low) and np.isnan(high)
