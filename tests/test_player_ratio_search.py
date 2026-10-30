"""Unit checks for the ratio-feature search. Pure-function tests only: the
full frame build reads the cached FPL and Understat data and is run by the
search script, not by pytest."""
import numpy as np
import pandas as pd

from pl_predictor.evaluate import player_ratio_search as prs


def test_lagged_rolling_sum_never_sees_its_own_row():
    frame = pd.DataFrame({
        "season": ["2024-25"] * 4 + ["2024-25"],
        "element": [1, 1, 1, 2, 2],
        "shots": [2.0, 5.0, 7.0, 1.0, 9.0],
    })
    base = prs.lagged_rolling_sums(frame, ["season", "element"], ["shots"], (3,), prefix="S")

    # The first appearance of player 1 has no prior rows at all.
    assert np.isnan(base.loc[0, "S3_shots"])
    # The second sees only the first row (2.0), never its own 5.0.
    assert base.loc[1, "S3_shots"] == 2.0

    # Changing a row's own value must not change its own feature.
    changed = frame.copy()
    changed.loc[2, "shots"] = 100.0
    after = prs.lagged_rolling_sums(changed, ["season", "element"], ["shots"], (3,), prefix="S")
    assert after.loc[2, "S3_shots"] == 7.0  # prior rows only: 2.0 + 5.0


def test_lagged_rolling_sum_does_not_cross_players():
    frame = pd.DataFrame({"season": ["2024-25"] * 3, "element": [1, 2, 2], "shots": [4.0, 3.0, 8.0]})
    out = prs.lagged_rolling_sums(frame, ["season", "element"], ["shots"], (3,), prefix="S")
    assert np.isnan(out.loc[1, "S3_shots"])  # player 2's first row has no history
    assert out.loc[2, "S3_shots"] == 3.0


def test_holm_adjust_known_example():
    adjusted = prs.holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_adjust_caps_at_one():
    assert prs.holm_adjust(np.array([0.9, 0.95]))[1] == 1.0


def test_poisson_deviance_zero_at_perfect_fit_and_zero_count_case():
    y = np.array([0.0, 3.0])
    assert np.allclose(prs.poisson_deviance(y, np.array([0.0 + 1e-12, 3.0])), [0.0, 0.0], atol=1e-6)
    # y = 0 with mean mu contributes 2*mu.
    assert np.isclose(prs.poisson_deviance(np.array([0.0]), np.array([0.5]))[0], 1.0)


def test_binary_log_loss_matches_closed_form():
    assert np.isclose(prs.binary_log_loss(np.array([1.0]), np.array([0.25]))[0], -np.log(0.25))


def test_candidate_pool_is_capped_unique_and_prefixed():
    defs = prs.candidate_definitions()
    names = [d["name"] for d in defs]
    assert len(names) == len(set(names))
    assert len(names) <= prs.POOL_CAP
    assert all(n.startswith("cand_") for n in names)
    assert not any("kp" in n or "key_pass" in n for n in names)


def test_expected_calibration_error_zero_when_calibrated_in_bin():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert prs.expected_calibration_error(y, p) == 0.0
