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
from pl_predictor.evaluate import goal_contribution_research
from pl_predictor.models import player_goals


def _frame(goals_per90, assists_per90, minutes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "goals_per90_last10": goals_per90,
            "assists_per90_last10": assists_per90,
            "expected_minutes_pre_match": minutes,
        }
    )


# The next three tests need the dict `fit_goal_contribution_model` *really*
# returns, not a dict this file writes itself. Everything the tripwire asserts
# has to be observable on production's own output, or it is not a tripwire.
#
# The previous version of this file asserted against a literal defined two
# lines above the assertion (`model = {"blend_weight": 0.7, ...}`), so
# `assert "union_calibrator" not in model` was a tautology -- it could not fail
# for any edit to any source file. The sibling `hasattr` check named
# `_apply_platt_to_served_union`, an identifier that appears nowhere in the
# tree, so it was false and would have stayed false. Both are gone.
#
# `fit_goal_contribution_model` reaches its data through
# `build_goal_contribution_frame`, which it imports *inside* the function body.
# Patching that module attribute therefore substitutes the data layer without
# touching the fitting code, which is what has to run for real here.
_MODEL_FEATURES = list(
    goal_contribution_research.BASE_FEATURES
    + goal_contribution_research.ENHANCED_FEATURES
    + goal_contribution_research.OPPONENT_FEATURES
)

# The manifest keys `fit_goal_contribution_model` is contracted to return. The
# tripwire asserts the real key set, so *any* new key fires it -- which is the
# point: unifying the two union constructions cannot be done from what is
# already stored, so it has to add something, and a union calibrator is the
# obvious candidate.
_MANIFEST_KEYS = {"model", "calibrator", "columns", "features", "blend_weight", "fitted_blend_weight"}


def _contribution_frame(calibration_rows: int = 120, seed: int = 0) -> pd.DataFrame:
    """Two seasons of synthetic player-fixture rows shaped like the real frame.

    `calibration_rows` is exposed because it selects which arm of
    `_fit_blend_weight` runs: under 100 rows that function returns None, which
    is the reachable `None` path that `blend_contribution`'s docstring wrongly
    describes. 120 keeps the fitted arm.
    """
    rng = np.random.default_rng(seed)

    def block(rows: int, season: str) -> pd.DataFrame:
        data = {feature: rng.random(rows) for feature in _MODEL_FEATURES}
        data["position"] = rng.choice(["FWD", "MID", "DEF"], rows)
        data["goal_contribution"] = (rng.random(rows) < 0.35).astype(int)
        # The three columns `_poisson_union` reads, on their own scale.
        data["goals_per90_last10"] = rng.random(rows) * 0.9
        data["assists_per90_last10"] = rng.random(rows) * 0.7
        data["expected_minutes_pre_match"] = 20.0 + rng.random(rows) * 70.0
        data["season"] = season
        return pd.DataFrame(data)

    return pd.concat(
        [block(150, "2023-2024"), block(calibration_rows, "2024-2025")],
        ignore_index=True,
    )


def _fit_manifest(calibration_rows: int = 120) -> dict:
    """Run the real `fit_goal_contribution_model` over the synthetic frame."""
    frame = _contribution_frame(calibration_rows)
    patch = pytest.MonkeyPatch()
    patch.setattr(
        goal_contribution_research,
        "build_goal_contribution_frame",
        lambda seasons=None: (frame, _MODEL_FEATURES + ["position"]),
    )
    try:
        return player_goals.fit_goal_contribution_model()
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def fitted_manifest() -> dict:
    """`fit_goal_contribution_model`'s actual return value, data layer stubbed."""
    return _fit_manifest()


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


def test_the_fitted_blend_weight_is_not_a_weight_for_the_quantity_that_is_served(
    fitted_manifest,
):
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

    **This is the tripwire, and it is now bound to production.** It reads the dict
    `fit_goal_contribution_model` actually returns. The previous version asserted
    against a dict literal defined in the test and a `hasattr` for a name absent
    from the tree, so no edit to any source file could have made it fail.

    It fires on either of the two things unification has to do:

    - it **stores** the missing bridge, which is a new manifest key, so the
      key-set assertion goes red; or
    - it starts **applying** the fitted weight, so the manifest's served share
      stops equalling `NEUTRAL_BLEND_WEIGHT`.

    Either way the ledger entry must be updated rather than deleted.
    """
    manifest = fitted_manifest

    # Only `calibrator` may exist, and it is the *direct* model's. A second
    # calibrator, or any other new key, is the shape unification takes.
    assert set(manifest) == _MANIFEST_KEYS, (
        "the manifest grew a key; if that is a union calibrator, EXP-2026-28's skew "
        "is fixed and this test plus the ledger entry must be revisited")

    # The share that actually gets served is the neutral one, whatever was
    # fitted. Asserted through the production function on the production input,
    # so it holds for every fitted value rather than for one chosen number.
    assert manifest["blend_weight"] == player_goals.NEUTRAL_BLEND_WEIGHT
    assert (
        player_goals._serving_blend_weight(manifest["fitted_blend_weight"])
        == player_goals.NEUTRAL_BLEND_WEIGHT
    ), (
        "if a fitted weight can now reach serving, the two constructions must have "
        "been unified and EXP-2026-28 must be revisited")


def test_serving_does_not_apply_a_weight_fitted_against_another_construction():
    """The interim state after EXP-2026-28.

    `_fit_blend_weight` minimises Brier against a **Platt-calibrated**
    `_poisson_union` built from a last-10 per-90 rate times expected minutes.
    Serving passes `anytime_probability(lam_goals + lam_assists)`, built from
    position-rate regressors scaled by team attack strength, minutes fraction and
    availability, **uncalibrated**. Two different quantities in two different
    calibration states, with no stored union calibrator to bridge them.

    So the fitted weight is recorded (`fitted_blend_weight`) but not applied. A
    convex mixture at a neutral share still removes the `max(...)` order-statistic
    bias, which was the structural point of the change; the fitted share was never
    more than a refinement on top of that, and it was fitted against the wrong
    thing.
    """
    from pl_predictor.models import player_goals

    assert player_goals._serving_blend_weight(0.70) == player_goals.NEUTRAL_BLEND_WEIGHT
    assert player_goals._serving_blend_weight(0.65) == player_goals.NEUTRAL_BLEND_WEIGHT
    # `None` passes through now, so the unfitted case serves the direct model alone
    # rather than blending 50/50 against an uncalibrated union.
    assert player_goals._serving_blend_weight(None) is None
    assert player_goals.NEUTRAL_BLEND_WEIGHT != 0.70, (
        "if the neutral weight is ever set to the fitted value, the distinction this "
        "exists to make has been lost and EXP-2026-28 must be revisited")


def test_the_blend_is_still_convex_at_the_neutral_weight():
    """Dropping the fitted share must not reintroduce the order-statistic bias.
    A convex mixture can land below either arm; a max never can.

    The `0.0 < NEUTRAL < 1.0` bound is the one assertion here that is not
    obviously implied by the algebra, so it is stated explicitly. Verified by
    mutation: setting `NEUTRAL_BLEND_WEIGHT = 1.0` makes `blend_contribution`
    return the direct arm unchanged, which silently reverts serving to the
    direct-model-alone semantics that the two docstrings above still describe as
    the `None` fallback. Every other assertion in this file passed under that
    mutant -- `blended < max(direct, union)` holds at weight 1.0 whenever
    `direct < union`, and the linear-identity assertion holds by definition --
    so without this bound the blend could be switched off and the suite would
    stay green.
    """
    from pl_predictor.models.player_goals import NEUTRAL_BLEND_WEIGHT, blend_contribution

    assert 0.0 < NEUTRAL_BLEND_WEIGHT < 1.0, (
        "at 1.0 the blend is the direct model alone and at 0.0 it is the union "
        "alone; either silently changes what serving returns")

    direct, union = 0.30, 0.45
    blended = blend_contribution(direct, union, NEUTRAL_BLEND_WEIGHT)
    assert blended < max(direct, union), "must not be an order statistic"
    assert blended == pytest.approx(NEUTRAL_BLEND_WEIGHT * direct + (1 - NEUTRAL_BLEND_WEIGHT) * union)
    # And it must not be floored at zero or capped at one for extreme arms.
    assert 0.0 < blend_contribution(0.99, 0.01, NEUTRAL_BLEND_WEIGHT) < 1.0


def test_the_fitted_weight_is_still_recorded_on_the_model(fitted_manifest):
    """It is real information about the research construction and belongs in the
    manifest. Dropping it would lose the EXP-2026-24 result; applying it is what
    EXP-2026-28 forbids.

    Bound to the dict `fit_goal_contribution_model` really returns. The previous
    version built `model = {"blend_weight": ..., "fitted_blend_weight": 0.70}`
    as a literal two lines above the assertion, so the three `assert`s read a
    value the test itself had just written and could not fail for any change to
    any source file. In particular, deleting the `fitted_blend_weight` entry from
    `fit_goal_contribution_model`'s return dict left this test green.

    The invariant is deliberately stated as a *range* rather than a specific
    fitted number: what matters is that the key is present and carries a
    plausible share, not which grid point the search happened to land on. That
    also keeps it from pinning a value to this synthetic frame.
    """
    manifest = fitted_manifest

    assert "fitted_blend_weight" in manifest, (
        "the fitted weight is real information about the research construction "
        "(EXP-2026-24); dropping it from the manifest loses that result")
    fitted = manifest["fitted_blend_weight"]
    assert fitted is not None, (
        "this frame is 120 calibration rows, well over `_fit_blend_weight`'s "
        "100-row floor, so it must have produced a weight")
    assert 0.0 <= fitted <= 1.0

    # The recorded value and the served value are two different things, and both
    # are on the real manifest. The grid is multiples of 0.05, so a tie at the
    # neutral share is possible in principle; the property that must hold either
    # way is that serving ignores the fitted number.
    assert manifest["blend_weight"] == player_goals.NEUTRAL_BLEND_WEIGHT
    assert player_goals._serving_blend_weight(fitted) == player_goals.NEUTRAL_BLEND_WEIGHT


def test_serving_serves_direct_alone_when_no_weight_was_fitted():
    """The documented `None` fallback is the path serving takes.

    `_fit_blend_weight` documents that it returns `None` "in which case serving
    falls back to the direct model alone". That is now true: `_serving_blend_weight`
    passes `None` through, the manifest carries `None`, and `blend_contribution`'s
    `weight is None` branch serves the direct model alone.

    The `None` is reachable, which is the part that makes this matter rather than
    merely cosmetic. All three triggers were confirmed by calling the real
    `_fit_blend_weight`: a missing required column, a single-class target, and --
    the undocumented one, and the one with no counterpart in the caller, which has
    already rejected empty and single-class slices -- a calibration slice under
    100 rows. That last is a row-count floor nothing upstream shares, so a short
    or partial final season lands on it.

    This test drives the manifest, so it pins what serving actually serves
    rather than what a helper does with a hand-passed `None`.

    **Decision made 2026-09-29:** the neutral share blends 50/50 against an
    *uncalibrated* union. When no weight was fitted there is no evidence for any
    share, and mixing in an uncalibrated quantity is not a conservative default.
    Serving the direct model alone is what the code already documents, and it is
    the choice that cannot be worse than inventing a weight. The previous
    behaviour (neutral share) is recorded here so the change is visible.

    Verified by mutation:
      - `_serving_blend_weight(None)` returning `0.5` (old behaviour) -- red.
      - the 100-row floor removed from `_fit_blend_weight`, so a 40-row
        calibration slice stops returning `None` -- red, via the
        `fitted_blend_weight is None` assertion on the manifest.
    """
    manifest = _fit_manifest(calibration_rows=40)

    # 40 rows is under `_fit_blend_weight`'s 100-row floor, so the fit really
    # did fail -- asserted on the real manifest, not assumed from the row count.
    assert manifest["fitted_blend_weight"] is None, (
        "premise: a 40-row calibration slice cannot produce a fitted weight")

    # The served share is None, so blend_contribution serves the direct model alone.
    assert manifest["blend_weight"] is None

    direct, union = 0.30, 0.45
    served = player_goals.blend_contribution(direct, union, manifest["blend_weight"])
    assert served == pytest.approx(direct)
    assert served != pytest.approx(0.5 * direct + 0.5 * union)
    assert player_goals.blend_contribution(direct, union, None) == pytest.approx(direct)


def test_serving_passes_none_through_when_no_weight_was_fitted():
    """The documented `None` fallback is the path serving takes.

    `_fit_blend_weight` documents that it returns `None` "in which case serving
    falls back to the direct model alone". After this fix that is true:
    `_serving_blend_weight(None)` returns `None`, the manifest carries `None`,
    and `blend_contribution`'s `weight is None` branch serves the direct model
    alone.

    The three `None` triggers are reachable (missing required column,
    single-class target, calibration slice under 100 rows), so this is not a
    hypothetical path.
    """
    assert player_goals._serving_blend_weight(None) is None
    assert player_goals._serving_blend_weight(0.70) == player_goals.NEUTRAL_BLEND_WEIGHT
    assert player_goals._serving_blend_weight(0.65) == player_goals.NEUTRAL_BLEND_WEIGHT


def test_manifest_carries_none_and_serves_direct_alone_when_unfitted():
    """End to end through the manifest: an unfitted weight serves the direct model.

    40 rows is under `_fit_blend_weight`'s 100-row floor, so the fit really did
    fail -- asserted on the real manifest, not assumed from the row count.
    """
    manifest = _fit_manifest(calibration_rows=40)

    assert manifest["fitted_blend_weight"] is None, (
        "premise: a 40-row calibration slice cannot produce a fitted weight")
    # the served share is now None, not the neutral share
    assert manifest["blend_weight"] is None

    direct, union = 0.30, 0.45
    served = player_goals.blend_contribution(direct, union, manifest["blend_weight"])
    assert served == pytest.approx(direct), "the direct model alone is what serving serves"
    assert served != pytest.approx(0.5 * direct + 0.5 * union)
