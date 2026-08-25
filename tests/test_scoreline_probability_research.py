import numpy as np
import pytest

from pl_predictor.evaluate import scoreline_probability_research as research


def test_probability_metrics_are_finite_for_valid_1x2_predictions():
    outcomes = np.array([0, 1, 2])
    probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])

    metrics = research._metrics(outcomes, probabilities)

    assert set(metrics) == {"rps", "brier", "log_loss", "ece"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_blend_weights_are_non_negative_and_sum_to_one():
    probability_sets = [
        np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]),
        np.array([[0.4, 0.4, 0.2], [0.3, 0.3, 0.4]]),
        np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4]]),
    ]

    weights = research._blend_weights(probability_sets, np.array([0, 2]))

    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0)
