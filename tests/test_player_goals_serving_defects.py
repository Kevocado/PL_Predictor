"""Regression tests for the three serving-path defects found in the
2026-09-27 player-model audit.

1. `predict_goal_contribution` dropped `was_home` on the floor. The model is
   fitted with `was_home` in BASE_FEATURES (populated from the FPL archive),
   but neither `blended_current_form` nor `current_start_features` emits that
   key, so every served row took the `or 0.0` default -- home players were
   scored with the fitted home-advantage coefficient contributing a constant
   offset instead of an effect. The served model was therefore not the model
   that was validated.

2. `rank_team_players` combined the direct G+A classifier and the Poisson
   union with `max(a, b, c)`. A max of separately-calibrated estimators is
   not a calibrated estimator of anything, and `max >= (a+b)/2` biases
   upward by an amount that grows exactly where the two models disagree most
   -- i.e. for the least reliable players. The weight is now fitted on the
   held-out calibration season and applied unconditionally.

3. `predict_player` reads a `context` object to scale shot estimates, but
   `rank_team_players` never passed one, so `shots_scale` was 1.0 on every
   live call and `expected_shots`/`expected_shots_on_target` were unscaled.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


class _FakeModel:
    """Minimal stand-in exposing the attributes predict_goal_contribution reads."""

    class _Fixed:
        def __init__(self, probability: float):
            self._probability = probability

        def predict_proba(self, matrix):
            return np.tile([1.0 - self._probability, self._probability], (len(matrix), 1))

    def __init__(self, probability: float, features: list[str], columns: list[str]):
        self.model = self._Fixed(probability)
        self.calibrator = self._Fixed(probability)
        self.features = features
        self.columns = columns


def test_was_home_is_passed_through_instead_of_defaulting_to_zero():
    """`was_home` must come from the `is_home` argument, not the `or 0.0` default."""
    from pl_predictor.models.player_goals import predict_goal_contribution

    seen: dict[str, float] = {}

    class _ModelRecorder:
        """`predict_goal_contribution` hands the classifier a one-row DataFrame."""

        def __init__(self, columns):
            self.columns = columns

        def predict_proba(self, matrix):
            row = matrix.reindex(columns=self.columns).iloc[0]
            seen["was_home"] = float(row["was_home"])
            seen["goals_per90_last10"] = float(row["goals_per90_last10"])
            return np.array([[0.4, 0.6]])

    class _CalibratorRecorder:
        """The Platt calibrator is handed a raw 1-D logit array instead."""

        def predict_proba(self, matrix):
            return np.array([[0.4, 0.6]])

    columns = ["goals_per90_last10", "was_home", "position_FWD"]
    model = {
        "model": _ModelRecorder(columns),
        "calibrator": _CalibratorRecorder(),
        "columns": columns,
        "features": ["goals_per90_last10", "was_home"],
    }

    rates = {"goals_per90_last10": 0.4}
    predict_goal_contribution(rates, {}, "FWD", model, is_home=True)
    assert seen["was_home"] == 1.0
    assert seen["goals_per90_last10"] == 0.4

    predict_goal_contribution(rates, {}, "FWD", model, is_home=False)
    assert seen["was_home"] == 0.0


def test_was_home_argument_overrides_a_conflicting_rates_value():
    """`is_home` must win over a `was_home` already present in `rates`.

    This is the opposite precedence to `predict_player`'s `dict.get` default,
    and it is deliberate: `is_home` describes the fixture being predicted,
    while any `was_home` in the feature dicts was built from the player's
    *historical* rows and describes some past match. The test supplies a
    genuinely conflicting value so the override is observable, rather than
    passing because the key happens to be absent.
    """
    from pl_predictor.models.player_goals import predict_goal_contribution

    seen: dict[str, float] = {}

    class _ModelRecorder:
        def __init__(self, columns):
            self.columns = columns

        def predict_proba(self, matrix):
            seen["was_home"] = float(matrix.reindex(columns=self.columns).iloc[0]["was_home"])
            return np.array([[0.4, 0.6]])

    class _CalibratorRecorder:
        def predict_proba(self, matrix):
            return np.array([[0.4, 0.6]])

    columns = ["was_home"]
    model = {
        "model": _ModelRecorder(columns),
        "calibrator": _CalibratorRecorder(),
        "columns": columns,
        "features": ["was_home"],
    }

    # rates claims the player was at home; the fixture being priced is away.
    predict_goal_contribution({"was_home": 1.0}, {}, "FWD", model, is_home=False)
    assert seen["was_home"] == 0.0, "is_home must override a conflicting rates value"

    # And the mirror case: rates says away, the fixture being priced is home.
    predict_goal_contribution({"was_home": 0.0}, {}, "FWD", model, is_home=True)
    assert seen["was_home"] == 1.0


def test_was_home_is_supplied_when_absent_from_both_feature_dicts():
    """Without the fix the comprehension's `or 0.0` default silently supplied 0.0."""
    from pl_predictor.models.player_goals import predict_goal_contribution

    seen: dict[str, float] = {}

    class _ModelRecorder:
        def __init__(self, columns):
            self.columns = columns

        def predict_proba(self, matrix):
            seen["was_home"] = float(matrix.reindex(columns=self.columns).iloc[0]["was_home"])
            return np.array([[0.4, 0.6]])

    class _CalibratorRecorder:
        def predict_proba(self, matrix):
            return np.array([[0.4, 0.6]])

    columns = ["was_home"]
    model = {
        "model": _ModelRecorder(columns),
        "calibrator": _CalibratorRecorder(),
        "columns": columns,
        "features": ["was_home"],
    }

    predict_goal_contribution({}, {}, "FWD", model, is_home=True)
    assert seen["was_home"] == 1.0


def test_blend_weight_is_used_instead_of_a_max_floor():
    """A fitted mixture must be able to land below either component."""
    from pl_predictor.models.player_goals import blend_contribution

    direct, union, weight = 0.30, 0.55, 0.0
    blended = blend_contribution(direct, union, weight)
    # A pure-union weight returns the union exactly -- never a max() above it.
    assert blended == pytest.approx(0.55)

    blended_full = blend_contribution(direct, union, 1.0)
    assert blended_full == pytest.approx(0.30)


def test_blend_defaults_to_the_direct_model_when_no_weight_is_fitted():
    from pl_predictor.models.player_goals import blend_contribution

    assert blend_contribution(0.30, 0.55, None) == pytest.approx(0.30)


def test_blend_is_a_convex_combination_never_exceeding_either_ceiling():
    """The old `max()` could never return below max(direct, goal, assist)."""
    from pl_predictor.models.player_goals import blend_contribution

    for weight in np.linspace(0.0, 1.0, 21):
        value = blend_contribution(0.80, 0.10, float(weight))
        assert 0.10 - 1e-9 <= value <= 0.80 + 1e-9


def test_expected_shots_respect_the_passed_context_scale():
    """`shots_scale` from a context must actually reach expected_shots."""
    from pl_predictor.models.player_goals import LEAGUE_AVERAGE_TEAM_SHOTS, predict_player

    rates = {
        "avg_minutes": 90.0,
        "goals_per90": 0.2,
        "assists_per90": 0.1,
        "shots_per90": 2.0,
        "shots_on_target_per90": 0.8,
    }
    form = pd.DataFrame(
        [[LEAGUE_AVERAGE_TEAM_SHOTS * 2.0, LEAGUE_AVERAGE_TEAM_SHOTS * 2.0]],
        index=["home"],
        columns=["home_last_10_shots_for", "away_last_10_shots_for"],
    )

    class _Ctx:
        def __init__(self, form):
            self.form = form

    unscaled = predict_player(rates, 1.35, 1.0, expected_minutes=90.0, is_home=True)
    scaled = predict_player(
        rates, 1.35, 1.0, expected_minutes=90.0, is_home=True, context=_Ctx(form)
    )

    # Doubling the team's shot volume must double expected shots.
    assert scaled["expected_shots"] == pytest.approx(unscaled["expected_shots"] * 2.0, rel=1e-6)
    assert scaled["expected_shots_on_target"] == pytest.approx(
        unscaled["expected_shots_on_target"] * 2.0, rel=1e-6
    )
    # Goal probability is independent of shot volume and must not move.
    assert scaled["anytime_goal_prob"] == pytest.approx(unscaled["anytime_goal_prob"])


def test_context_scale_reaches_exactly_three_fields_and_no_others():
    """`shots_scale` feeds lam_shots and lam_shots_on_target, so it moves three
    served fields: expected_shots, expected_shots_on_target, and
    anytime_shot_on_target_prob (via 1 - exp(-lam_sot)). The shot-on-target
    *probability* is where the 0.9781 defect was observed, so it is easy to
    forget the fix reaches it too. Everything else must be untouched.
    """
    from pl_predictor.models.player_goals import LEAGUE_AVERAGE_TEAM_SHOTS, predict_player

    rates = {
        "avg_minutes": 90.0,
        "goals_per90": 0.3,
        "assists_per90": 0.15,
        "shots_per90": 1.5,
        "shots_on_target_per90": 0.6,
        "saves_per90": 0.4,
    }
    form = pd.DataFrame(
        [[LEAGUE_AVERAGE_TEAM_SHOTS * 3.0, LEAGUE_AVERAGE_TEAM_SHOTS]],
        index=["home"],
        columns=["home_last_10_shots_for", "away_last_10_shots_for"],
    )

    class _Ctx:
        def __init__(self, form):
            self.form = form

    common = dict(expected_minutes=90.0, is_home=True)
    unscaled = predict_player(rates, 1.35, 1.0, **common)
    scaled = predict_player(rates, 1.35, 1.0, context=_Ctx(form), **common)

    # The three fields that legitimately move, monotonically in the scale.
    assert scaled["expected_shots"] > unscaled["expected_shots"]
    assert scaled["expected_shots_on_target"] > unscaled["expected_shots_on_target"]
    assert scaled["anytime_shot_on_target_prob"] > unscaled["anytime_shot_on_target_prob"]

    # Everything else is shot-volume independent and must be identical.
    for field in (
        "expected_goals",
        "expected_assists",
        "anytime_goal_prob",
        "anytime_assist_prob",
        "anytime_goal_contribution_prob",
        "expected_saves",
    ):
        assert scaled[field] == unscaled[field], f"{field} must not depend on shot volume"


def test_anytime_probability_helper_stays_a_poisson_union():
    """Guard the identity the union baseline relies on."""
    from pl_predictor.models.player_goals import anytime_probability

    lam = 0.8
    assert anytime_probability(lam) == pytest.approx(1 - math.exp(-lam))
