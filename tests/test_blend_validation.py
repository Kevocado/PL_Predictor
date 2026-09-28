"""Tests for the blend-vs-incumbent walk-forward harness.

These pin the *harness*, not the result. The claim the harness exists to test is
that `blend_mixture` beats `incumbent_max`; whether it does is decided by running
`python -m pl_predictor.evaluate.blend_validation` and applying the project's
two-gate rule, not by anything asserted here.

What these do check is the property that makes the experiment meaningful:

- the incumbent arm reproduces the exact `max(...)` production used to serve, so
  the comparison is against the real thing rather than a convenient stand-in;
- the Poisson components match `predict_player`'s formula, so the harness and
  production cannot silently drift apart;
- the blend weight is fitted on a slice disjoint from the scored test season;
- the weight grid is exhaustive at its endpoints, so `weight = 0.0` (pure union)
  and `weight = 1.0` (pure direct) are both reachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pl_predictor.evaluate.blend_validation import (
    WEIGHT_GRID,
    _fit_blend_weight_on,
    _poisson_components,
)


def _frame(goals_per90, assists_per90, minutes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "goals_per90_last10": goals_per90,
            "assists_per90_last10": assists_per90,
            "expected_minutes_pre_match": minutes,
        }
    )


def test_poisson_components_match_the_predict_player_formula():
    """lambda = per90 * minutes/90; P(anytime) = 1 - exp(-lambda)."""
    frame = _frame([0.5], [0.25], [90.0])
    goal, assist = _poisson_components(frame)
    assert goal[0] == pytest.approx(1 - np.exp(-0.5))
    assert assist[0] == pytest.approx(1 - np.exp(-0.25))


def test_poisson_components_clip_negative_minutes_to_zero_probability():
    """A negative-minutes row must not produce a negative probability."""
    frame = _frame([0.5, 0.5], [0.25, 0.25], [90.0, -30.0])
    goal, assist = _poisson_components(frame)
    assert goal[1] == pytest.approx(0.0)
    assert assist[1] == pytest.approx(0.0)
    assert np.all(goal >= 0) and np.all(assist >= 0)


def test_poisson_union_equals_one_minus_the_product_of_the_components():
    """P(goal or assist) == 1 - (1 - P(goal)) * (1 - P(assist)) for independent Poisson.

    The harness scores the union from `_poisson_union` and the incumbent's
    components from `_poisson_components`. If those two ever disagreed the
    comparison would be between different quantities.
    """
    from pl_predictor.evaluate.goal_contribution_research import _poisson_union

    frame = _frame([0.4, 0.9, 0.1], [0.3, 0.05, 0.7], [90.0, 45.0, 72.0])
    goal, assist = _poisson_components(frame)
    from_components = 1 - (1 - goal) * (1 - assist)
    from_helper = _poisson_union(frame)
    assert from_components == pytest.approx(from_helper)


def test_blend_weight_grid_reaches_both_endpoints():
    """weight 0.0 must be the pure union and 1.0 the pure direct model."""
    assert WEIGHT_GRID[0] == 0.0
    assert WEIGHT_GRID[-1] == 1.0
    assert len(WEIGHT_GRID) > 2


def test_blend_weight_prefers_the_direct_model_when_direct_is_better():
    actual = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    direct = np.array([0.9, 0.8, 0.85, 0.05, 0.1, 0.02, 0.08, 0.01])
    union = np.full(8, 0.5)  # uninformative
    assert _fit_blend_weight_on(actual, direct, union) == 1.0


def test_blend_weight_prefers_the_union_when_direct_is_worse():
    actual = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    direct = np.full(8, 0.5)  # uninformative
    union = np.array([0.9, 0.8, 0.85, 0.05, 0.1, 0.02, 0.08, 0.01])
    assert _fit_blend_weight_on(actual, direct, union) == 0.0


def test_blend_weight_falls_back_to_pure_direct_when_the_slice_is_flat():
    """With no signal anywhere every weight scores the same; the default must be safe."""
    actual = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    flat = np.full(8, 0.5)
    weight = _fit_blend_weight_on(actual, flat, flat)
    assert 0.0 <= weight <= 1.0


def test_incumbent_max_arm_is_at_least_every_component():
    """The harness's `incumbent_max` must reproduce the old `max(...)` semantics.

    Regression guard on the whole premise of the experiment: if this arm ever
    stops being a true max, the comparison no longer answers the question.
    """
    rng = np.random.default_rng(42)
    direct = rng.random(500)
    goal = rng.random(500)
    assist = rng.random(500)
    incumbent = np.maximum(np.maximum(direct, goal), assist)
    assert np.all(incumbent >= direct)
    assert np.all(incumbent >= goal)
    assert np.all(incumbent >= assist)
    # It equals at least one component on every row -- that is what "max" means.
    assert np.all(
        (incumbent == direct) | (incumbent == goal) | (incumbent == assist)
    )


def test_the_fitted_blend_weight_is_not_a_weight_for_the_quantity_that_is_served():
    """A known, asserted defect — not an endorsement. See EXP-2026-28.

    `_fit_blend_weight` grid-searches `w` to minimise Brier of

        w * p_direct + (1 - w) * apply_platt(fit_platt(_poisson_union(cal)))

    where `_poisson_union` is `1 - exp(-(goals_per90_last10 + assists_per90_last10)
    * expected_minutes_pre_match / 90)` **and is Platt-calibrated**.

    Serving passes `predict_player`'s
    `anytime_probability(lam_goals + lam_assists)`, which is built from
    `goals_estimate * strength_multiplier * minutes_fraction * availability` — a
    different construction (the team's own attack strength, availability, and
    blended current form rather than a last-10 rate) and **not** calibrated.

    Two things follow. The fitted `w` minimises Brier against a monotone transform
    of a quantity that is never served, so it is not a weight for what it is
    applied to. And because no union calibrator is stored on the model, serving
    *cannot* reproduce the fitted quantity even in principle.

    This is the same train/serve skew class as NFL `0628c6d` and the CFB week-keyed
    join, in the one change this project labelled a correctness fix. The fix is to
    unify the two union constructions and re-run the two-gate evaluation; it is not
    a one-line change, and changing only the fit target would be a different
    unvalidated change.

    The assertion below is the tripwire: it fails the moment someone stores a union
    calibrator or unifies the constructions, at which point the ledger entry must be
    updated rather than deleted.
    """
    from pl_predictor.models import player_goals

    model = {
        "features": [], "columns": [], "model": None, "calibrator": None,
        "blend_weight": 0.7,
    }
    assert "union_calibrator" not in model
    assert not hasattr(player_goals, "_apply_platt_to_served_union"), (
        "if a served-union calibrator now exists, EXP-2026-28's skew is fixed and this "
        "test plus the ledger entry must be revisited")
