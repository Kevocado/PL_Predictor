"""player_goals.py — anytime goalscorer / assist probability per player.

    λ_player = player's expected-goals-per-appearance estimate
               × (this fixture's team expected goals / league-average team goals)
               × (expected minutes this match / 90, from recent appearances)
               × live availability multiplier

Ties player predictions to the already-fitted scoreline model's
fixture-specific team strength (`models.scoreline.predict_fixture`'s
`home_goal_expectation`/`away_goal_expectation`) rather than an independent
player regressor, so a player's chance rises/falls with how much the match
model expects their team to score in this specific fixture. Availability
reuses the exact live-status business rule already proven in
`FPL_Optimizer/scout.py`: zero out injured/suspended/unavailable players,
scale doubtful ones by their `chance_of_playing_next_round`.

The "expected-goals-per-appearance estimate" itself: `evaluate/
player_stat_reliability.py` tested whether FPL's ICT Index (and the wider
stat surface `features/player_form.py` computes) actually predicts a
player's future output beyond the plain rolling goals/assists rate this
formula used to rely on alone. Result — not assumed, measured on a real
held-out season: `threat` adds real incremental signal for goals (R² on a
goals~[rate, threat] regression beats goals~rate alone), `creativity` does
the same for assists; the ICT Index's `influence` sub-component does not
(near-zero/negative gain — redundant with the existing rate, not reliable
on its own). So `predict_player` now blends in `threat`/`creativity` via a
small linear regression fitted on real historical data (`fit_reliability_coefficients`),
not the raw rate alone — `influence` and the other more marginal
candidates (bps, bonus, xG/xA on their own) are left out, matching "keep
only what earns it" the same way match-model features were tested this
session. `reliability_coeffs` is optional specifically so this stays
backward compatible (omit it to fall back to the plain rate, e.g. in a unit
test with no fitted coefficients handy)."""

from __future__ import annotations

import math
import unicodedata

import numpy as np
import pandas as pd

from ..data import fpl_api, fpl_history
from ..data.team_names import to_canonical
from ..features import player_form
from . import scoreline

# Reuse the same reference constant scoreline.py already uses for an
# average-strength team, so "how much stronger/weaker is this fixture's
# team than average" has one consistent definition across the project.
LEAGUE_AVERAGE_TEAM_GOALS = scoreline.FALLBACK_GOAL_EXPECTANCY

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# (target rate key, reliability-tested stat key) -> which of player_form's
# blended_current_form output keys the reliability-adjusted estimate reads.
_RELIABILITY_SPEC = {
    "goals": ("goals_per90", "threat"),
    "assists": ("assists_per90", "creativity"),
}

LEAGUE_AVERAGE_TEAM_SHOTS = 13.1  # measured over last 3 completed seasons

LINEUP_FEATURES = [
    "starts_last3", "starts_last5", "starts_last10", "sub_rate_last3", "sub_rate_last5",
    "minutes_last3", "minutes_last5", "minutes_last10", "minutes_ema", "start_streak",
]
RATE_FEATURES = [
    "goals_per90_last3", "goals_per90_last5", "goals_per90_last10",
    "assists_per90_last3", "assists_per90_last5", "assists_per90_last10",
    "expected_goals_per90_last3", "expected_goals_per90_last5", "expected_goals_per90_last10",
    "expected_assists_per90_last3", "expected_assists_per90_last5", "expected_assists_per90_last10",
    "threat_last3", "threat_last5", "threat_last10",
    "creativity_last3", "creativity_last5", "creativity_last10", "was_home",
]
STARTER_MINUTES = 82.9
SUBSTITUTE_MINUTES = 39.9

# Imported, not mirrored. An earlier version duplicated this tuple and justified
# it with "a runtime import here would be circular" -- which is false on both
# counts: `goal_contribution_research` does not import this module, and
# `fit_goal_contribution_model` forty lines below already does a deferred import
# from it. A duplicated feature list is a list that can drift, and a wrong
# justification is worse than none because the next reader preserves it.
from ..evaluate.goal_contribution_research import OPPONENT_FEATURES as _OPPONENT_DEFENCE_FEATURES


def fit_reliability_coefficients(seasons: list[str] | None = None) -> dict:
    """Fits the two small linear regressions `evaluate/
    player_stat_reliability.py` validated (goals ~ [goals_per90_last10,
    threat_last10], assists ~ [assists_per90_last10, creativity_last10]) on
    *all* available historical seasons (not held out — this is for
    production use, unlike the reliability study's own train/test split).
    Cheap (a couple of features, tens of thousands of rows, sub-second) —
    meant to be refit on a lightweight periodic cadence (see
    `api/routes.py`'s cache for this), not persisted to disk like the match
    model's much more expensive XGBoost fits."""
    from sklearn.linear_model import LinearRegression

    seasons = seasons or fpl_history.default_completed_seasons()
    df = fpl_history.load_player_gw_history(seasons=seasons)
    played, _ = player_form.build_historical_player_form(df)

    coeffs = {}
    for target_key, (rate_stat, extra_stat) in _RELIABILITY_SPEC.items():
        rate_col, extra_col = f"{rate_stat}_last10", f"{extra_stat}_last10"
        target_col = "goals_scored" if target_key == "goals" else "assists"
        d = played.dropna(subset=[rate_col, extra_col, target_col])
        if len(d) < 100:
            continue
        model = LinearRegression().fit(d[[rate_col, extra_col]].to_numpy(), d[target_col].to_numpy())
        coeffs[target_key] = {
            "intercept": float(model.intercept_),
            "coef_rate": float(model.coef_[0]),
            "coef_extra": float(model.coef_[1]),
        }
    return coeffs


def fit_goal_contribution_model(seasons: list[str] | None = None) -> dict:
    """Fit the walk-forward-winning direct, Platt-calibrated G+A model.

    The research evaluator owns the held-out comparison; this production fit
    uses the same enhanced feature set and reserves the newest completed
    season for calibration. Goal and assist probabilities intentionally keep
    their existing specialised models.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ..evaluate.goal_contribution_research import (
        BASE_FEATURES,
        ENHANCED_FEATURES,
        OPPONENT_FEATURES,
        build_goal_contribution_frame,
    )

    frame, _ = build_goal_contribution_frame(seasons)
    all_features = [feature for feature in BASE_FEATURES + ENHANCED_FEATURES + OPPONENT_FEATURES if feature in frame] + ["position"]
    available_seasons = sorted(frame["season"].unique())
    if len(available_seasons) < 2:
        return {}
    calibration_season = available_seasons[-1]
    fit = frame[frame["season"] != calibration_season]
    calibration = frame[frame["season"] == calibration_season]
    if fit.empty or calibration.empty or fit["goal_contribution"].nunique() < 2 or calibration["goal_contribution"].nunique() < 2:
        return {}

    def matrix(rows: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
        result = pd.get_dummies(rows[all_features], columns=["position"], dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if columns is not None:
            result = result.reindex(columns=columns, fill_value=0.0)
        return result, result.columns.tolist()

    X_fit, columns = matrix(fit)
    X_calibration, _ = matrix(calibration, columns)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_fit, fit["goal_contribution"])
    calibration_prob = model.predict_proba(X_calibration)[:, 1]
    logits = np.log(np.clip(calibration_prob, 1e-6, 1 - 1e-6) / np.clip(1 - calibration_prob, 1e-6, 1))
    calibrator = LogisticRegression(max_iter=1000).fit(logits.reshape(-1, 1), calibration["goal_contribution"])
    calibrated = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]

    # Fit the direct-vs-union mixture share on the same held-out calibration
    # season, so both components are compared on data neither was fitted on.
    # The Poisson union is built from the same pre-match rate and minutes
    # columns the research harness uses, then Platt-calibrated on the same
    # slice so neither side of the mixture is privileged.
    blend_weight = _fit_blend_weight(calibration, calibrated)

    return {
        "model": model,
        "calibrator": calibrator,
        "columns": columns,
        "features": all_features,
        # NOT `blend_weight`. See `_serving_blend_weight` for why the fitted
        # number is recorded but not applied.
        "blend_weight": _serving_blend_weight(blend_weight),
        "fitted_blend_weight": blend_weight,
    }


# The share given to the direct classifier at serving time, when the two arms are
# not commensurable.
#
# `_fit_blend_weight` minimises Brier against
# `w * p_direct + (1 - w) * apply_platt(_poisson_union(calibration))`, where
# `_poisson_union` is `1 - exp(-(goals_per90_last10 + assists_per90_last10) *
# expected_minutes_pre_match / 90)` and is **Platt-calibrated**. Serving passes
# `predict_player`'s `anytime_probability(lam_goals + lam_assists)`, whose lambdas
# come from position-rate regressors scaled by the team's attack strength,
# minutes fraction and availability.
#
# Those are two different quantities, in two different calibration states, and no
# union calibrator is stored — so serving cannot reproduce the quantity the weight
# was fitted against even in principle. Applying `w` to it is not a small
# imprecision: the blend's whole point is that a convex combination of a calibrated
# and an uncalibrated probability is itself uncalibrated, which is the same
# objection that disqualified the `max(...)` this replaced.
#
# So the fitted weight is recorded as `fitted_blend_weight` -- it is real
# information about the research construction and belongs in the manifest -- and
# serving uses a neutral share until the two constructions are unified. The
# structural win is unaffected: a convex mixture with no order-statistic bias is
# better than `max(...)` at *any* weight, including this one.
NEUTRAL_BLEND_WEIGHT = 0.5


def _serving_blend_weight(fitted: float | None) -> float | None:
    """The weight serving may actually use.

    `None` when no weight was fitted, so `blend_contribution`'s `weight is None`
    branch serves the direct model alone -- which is what `_fit_blend_weight`'s
    docstring has always promised. The neutral share only when a weight was
    actually fitted.

    Kept as a function rather than a constant at the call site so that when the
    constructions are unified there is exactly one place to change, and so a test
    can assert the two are deliberately distinct today.

    It is called once, from `fit_goal_contribution_model`, with either a fitted
    float or `None`. The three `None` triggers are all reachable (a missing
    required column, a single-class target, a calibration slice under 100 rows),
    so this is not a hypothetical path.

    **Why the neutral share is not used for the unfitted case:** the neutral
    share blends 50/50 against an *uncalibrated* union. When no weight was
    fitted there is no evidence for any share, and mixing in an uncalibrated
    quantity is not a conservative default -- it is an undisclosed behaviour
    change. Serving the direct model alone is what the code already documents,
    and it is the choice that cannot be worse than inventing a weight.
    """
    if fitted is None:
        return None
    return NEUTRAL_BLEND_WEIGHT


def _fit_blend_weight(calibration: pd.DataFrame, direct_probability: np.ndarray) -> float | None:
    """Grid-search the direct-model share that minimises calibration Brier.

    Returns None when no weight can be fitted for this slice. All three triggers
    below were verified by calling this function, not inferred:

    - a required column is absent (`goals_per90_last10`, `assists_per90_last10`
      or `expected_minutes_pre_match` -- absent for early seasons);
    - the target is single-class on this slice; or
    - the slice has fewer than 100 rows.

    **All three are reachable.** The third is the one that was undocumented and
    the easiest to hit: it is a row-count floor with no counterpart in
    `fit_goal_contribution_model`, which has already returned `{}` for a
    single-class or empty calibration slice, so by the time this is called the
    100-row floor is the only guard left standing. A short final season -- the
    first weeks of a new one, a partial cache, a truncated history -- produces
    `None`.

    When no weight is fitted, `_serving_blend_weight` preserves `None`, so
    `blend_contribution` serves the direct model alone when available. A fitted
    weight instead selects the neutral share for serving; the fitted value is
    recorded separately in the model manifest.
    """
    from sklearn.metrics import brier_score_loss

    from ..evaluate.goal_contribution_research import _apply_platt, _fit_platt, _poisson_union

    required = ["goals_per90_last10", "assists_per90_last10", "expected_minutes_pre_match"]
    if any(column not in calibration.columns for column in required):
        return None
    actual = calibration["goal_contribution"].to_numpy()
    if len(np.unique(actual)) < 2 or len(actual) < 100:
        return None

    union_raw = _poisson_union(calibration)
    union = _apply_platt(_fit_platt(union_raw, actual), union_raw)

    best_weight, best_score = 1.0, float("inf")
    for weight in np.linspace(0.0, 1.0, 21):
        score = brier_score_loss(actual, weight * direct_probability + (1.0 - weight) * union)
        if score < best_score:
            best_score, best_weight = score, float(weight)
    return best_weight


def predict_goal_contribution(
    rates: dict,
    start_features: dict,
    position: str,
    contribution_model: dict | None,
    is_home: bool = False,
    opponent_defence: dict | None = None,
) -> float | None:
    """Return calibrated direct P(goal or assist), or None without a fit.

    `was_home` is in the fitted feature set (`BASE_FEATURES`) and is populated
    from the FPL archive at training time, but neither
    `features.player_form.blended_current_form` nor `current_start_features`
    emits that key at serving time. Without the explicit branch below, the
    `or 0.0` default silently scored every home player with
    `was_home = 0.0`, so the fitted home-advantage coefficient contributed a
    constant offset rather than an effect and the served model was not the
    model that had been validated. `predict_player` already special-cased the
    same key for its position-rate models; this is the same fix for the direct
    G+A model.

    `is_home` deliberately wins over any `was_home` already present in
    `rates`/`start_features`, which is the opposite precedence to
    `predict_player`'s `dict.get` default. The reason is provenance, not
    convenience: `is_home` describes *the fixture this call is predicting*,
    which is known exactly, whereas any `was_home` reachable through the
    feature dicts was built by `blended_current_form` from the player's own
    *historical* rows and therefore describes some past match. Substituting a
    past match's venue for the one being priced would be a train/serve skew of
    exactly the kind this function's docstring above is fixing. The two
    functions differ on precedence deliberately, and
    `test_was_home_argument_overrides_a_conflicting_rates_value` pins it.

    `opponent_defence` does the same job for the three `opponent_defence_last*`
    terms (EXP-2026-25), for the same reason and with the same precedence: they
    describe *the fixture being priced*, and nothing reachable through
    `blended_current_form` describes it -- that function rolls the player's own
    history, which has no notion of who is being faced. A caller that cannot
    resolve the opponent passes `None` and gets 0.0, which is the league-average
    prior the fitted model was calibrated against, not a missing feature.
    """
    if not contribution_model:
        return None
    values = {
        feature: rates.get(feature, start_features.get(feature, 0.0)) or 0.0
        for feature in contribution_model["features"]
        if feature != "position"
    }
    if "was_home" in values:
        values["was_home"] = 1.0 if is_home else 0.0
    for feature in _OPPONENT_DEFENCE_FEATURES:
        if feature in values:
            supplied = (opponent_defence or {}).get(feature)
            values[feature] = float(supplied) if supplied is not None else 0.0
    values["position"] = position
    matrix = pd.get_dummies(pd.DataFrame([values]), columns=["position"], dtype=float)
    matrix = matrix.reindex(columns=contribution_model["columns"], fill_value=0.0)
    raw_probability = float(contribution_model["model"].predict_proba(matrix)[:, 1][0])
    logit = math.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1))
    return float(contribution_model["calibrator"].predict_proba(np.array([[logit]]))[:, 1][0])


def blend_contribution(
    direct_probability: float | None,
    union_probability: float,
    weight: float | None,
) -> float:
    """Combine the direct G+A classifier with the Poisson union.

    `weight` is the share given to the direct classifier. Serving passes the
    neutral share when a weight was fitted on the held-out calibration season
    (see `fit_goal_contribution_model`). When no weight was fitted, serving
    passes `None` and uses the direct model alone when available, or the union
    if no direct probability is available.

    This replaces a previous `max(direct, anytime_goal, anytime_assist)`. A
    max of separately-calibrated estimators is not a calibrated estimator of
    anything, and because `max(a, b) >= (a + b) / 2` it biased every
    probability upward by an amount that grew with the disagreement between
    the two models -- that is, it shrank least toward the base rate exactly
    for the players both models were least sure about. A fitted convex
    mixture has no such order-statistic bias, and it is free to land below
    either component, which is information about which model to trust where.
    """
    if direct_probability is None:
        return float(union_probability)
    if weight is None:
        return float(direct_probability)
    share = min(max(float(weight), 0.0), 1.0)
    return float(share * direct_probability + (1.0 - share) * union_probability)


def fit_lineup_model(seasons: list[str] | None = None):
    """Train a calibrated, leakage-safe probability-of-start classifier."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    seasons = seasons or fpl_history.default_completed_seasons()
    history = fpl_history.load_player_gw_history(seasons=seasons)
    rows, _ = player_form.build_historical_start_features(history)
    train = rows.dropna(subset=LINEUP_FEATURES + ["started"])
    if len(train) < 500 or train["started"].nunique() < 2:
        return None
    tail_size = max(500, len(train) // 5)
    fit_rows = train.iloc[:-tail_size]
    calibration = train.iloc[-tail_size:]
    classifier = GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=30, random_state=42)
    classifier.fit(fit_rows[LINEUP_FEATURES], fit_rows["started"])
    # Gradient boosting's raw probabilities are sharp. Calibrate them on a
    # chronological tail so fixture consumers receive useful probabilities.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(classifier.predict_proba(calibration[LINEUP_FEATURES])[:, 1], calibration["started"])
    return {"classifier": classifier, "calibrator": calibrator}


def predict_lineup(start_features: dict, lineup_model: dict | None = None) -> dict:
    """Return a start probability and expected minutes for the next match."""
    fallback_start = float(start_features.get("starts_last5", 0.0) or 0.0)
    if lineup_model is None:
        probability_start = fallback_start
    else:
        vector = np.array([[float(start_features.get(feature, 0.0) or 0.0) for feature in LINEUP_FEATURES]])
        raw_probability = lineup_model["classifier"].predict_proba(vector)[:, 1][0]
        probability_start = float(lineup_model["calibrator"].predict([raw_probability])[0])
    probability_sub = float(start_features.get("sub_rate_last5", 0.0) or 0.0)
    expected_minutes = probability_start * STARTER_MINUTES + (1 - probability_start) * probability_sub * SUBSTITUTE_MINUTES
    return {
        "probability_start": min(max(probability_start, 0.0), 1.0),
        "predicted_starter": probability_start >= 0.5,
        "expected_minutes": min(max(expected_minutes, 0.0), 90.0),
    }


def fit_position_rate_models(seasons: list[str] | None = None) -> dict:
    """Fit separate Ridge rate models for defenders, midfielders and forwards."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seasons = seasons or fpl_history.default_completed_seasons()
    history = fpl_history.load_player_gw_history(seasons=seasons)
    played, _ = player_form.build_historical_player_form(history)
    models = {}
    for position in ("DEF", "MID", "FWD"):
        position_rows = played[played["position"] == position].copy()
        position_rows["was_home"] = position_rows["was_home"].astype(float)
        train = position_rows.dropna(subset=RATE_FEATURES)
        if len(train) < 250:
            continue
        for target, target_col in (("goals", "goals_scored"), ("assists", "assists")):
            target_rate = train[target_col] / train["minutes"] * 90
            model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
            model.fit(train[RATE_FEATURES], target_rate)
            models[(position, target)] = model
    return models


def anytime_probability(lam: float) -> float:
    return 1 - math.exp(-lam)


def _expected_per_appearance(rates: dict, target_key: str, reliability_coeffs: dict | None) -> float:
    """The reliability-adjusted estimate for one target (goals or assists)
    if fitted coefficients are available and the extra stat is present in
    `rates`; falls back to the plain per-90 rate otherwise (e.g. no
    coefficients passed, or a brand-new player with no `threat`/`creativity`
    history yet)."""
    rate_stat, extra_stat = _RELIABILITY_SPEC[target_key]
    plain_rate = rates.get(rate_stat, 0.0) or 0.0

    coeffs = (reliability_coeffs or {}).get(target_key)
    extra_val = rates.get(extra_stat)
    if coeffs is None or extra_val is None:
        return plain_rate

    estimate = coeffs["intercept"] + coeffs["coef_rate"] * plain_rate + coeffs["coef_extra"] * extra_val
    return max(estimate, 0.0)


def predict_player(
    rates: dict,
    team_goal_expectation: float,
    availability: float,
    reliability_coeffs: dict | None = None,
    expected_minutes: float | None = None,
    position: str | None = None,
    is_home: bool = False,
    position_rate_models: dict | None = None,
    is_penalty_taker: bool = False,
    is_set_piece_taker: bool = False,
    context: object | None = None,
    opponent_defence: dict | None = None,
) -> dict:
    """`rates` is `features.player_form.blended_current_form`'s output.
    `reliability_coeffs` (from `fit_reliability_coefficients`) is optional —
    pass it to use the reliability-adjusted goals/assists estimate;
    omit for the plain per-90 rate (e.g. in tests)."""
    strength_multiplier = team_goal_expectation / LEAGUE_AVERAGE_TEAM_GOALS
    minutes_fraction = min((expected_minutes if expected_minutes is not None else rates["avg_minutes"]) / 90, 1.0)
    scale = strength_multiplier * minutes_fraction * availability

    goals_estimate = _expected_per_appearance(rates, "goals", reliability_coeffs)
    assists_estimate = _expected_per_appearance(rates, "assists", reliability_coeffs)

    has_current_rate_features = any(rates.get(feature) is not None for feature in RATE_FEATURES if feature != "was_home")
    if position_rate_models and position and has_current_rate_features:
        vector = np.array([[float(rates.get(feature, 1.0 if feature == "was_home" and is_home else 0.0) or 0.0) for feature in RATE_FEATURES]])
        goal_model = position_rate_models.get((position, "goals"))
        assist_model = position_rate_models.get((position, "assists"))
        if goal_model is not None:
            goals_estimate = max(float(goal_model.predict(vector)[0]), 0.0)
        if assist_model is not None:
            assists_estimate = max(float(assist_model.predict(vector)[0]), 0.0)

    shots_scale = 1.0
    if context is not None:
        form = getattr(context, "form", None)
        if form is not None:
            shots_scale = (
                (form.loc["home", "home_last_10_shots_for"] if is_home else form.loc["away", "away_last_10_shots_for"])
                / LEAGUE_AVERAGE_TEAM_SHOTS
            )
    shots_scale = max(float(shots_scale or 1.0), 0.0)

    lam_goals = goals_estimate * scale
    lam_assists = assists_estimate * scale
    shots_estimate = rates.get("shots_per90", 0.0) or 0.0
    shots_on_target_estimate = rates.get("shots_on_target_per90", 0.0) or 0.0
    lam_shots = shots_estimate * scale * shots_scale
    lam_shots_on_target = shots_on_target_estimate * scale * shots_scale
    if is_penalty_taker:
        lam_goals += 0.15 * minutes_fraction * availability
    if is_set_piece_taker:
        lam_assists += 0.10 * minutes_fraction * availability

    # "saves" is one of RATE_STATS (features/player_form.py), so
    # blended_current_form already computes saves_per90 straight from FPL's
    # own history (a native column, unlike shots -- no Understat crosswalk
    # needed) -- this was simply never read out of `rates` before. Scaled
    # by minutes/availability only, not team_goal_expectation/strength_
    # multiplier: those describe the *scoring* team's attack, not the
    # opponent's, and this function isn't given the opponent's own
    # attacking strength to scale against -- a real simplification (a
    # keeper facing a stronger attack should expect more saves), flagged
    # here rather than silently assumed away.
    saves_estimate = rates.get("saves_per90", 0.0) or 0.0
    lam_saves = saves_estimate * minutes_fraction * availability

    return {
        "expected_goals": lam_goals,
        "expected_assists": lam_assists,
        "anytime_goal_prob": anytime_probability(lam_goals),
        "anytime_assist_prob": anytime_probability(lam_assists),
        "anytime_goal_contribution_prob": anytime_probability(lam_goals + lam_assists),
        "expected_shots": lam_shots,
        "expected_shots_on_target": lam_shots_on_target,
        "anytime_shot_on_target_prob": anytime_probability(lam_shots_on_target),
        "expected_saves": lam_saves,
    }


def _merge_shots_into_history(history: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One player's live current-season history (from `fetch_player_summary`,
    no shots columns) plus that same player's own Understat `{date, shots,
    shots_on_target}` rows -- joined on match date, same idea as
    `_load_history_with_shots`'s bulk training-frame merge, just scoped to
    a single player's live per-request history. Left join: an unmatched
    date (their Understat data hasn't been merged in yet, or this specific
    match has none) leaves NaN, never a fabricated number -- the same
    convention `player_form.py`'s existing per-RATE_STATS-column presence
    checks already rely on."""
    if history.empty or shots.empty or "kickoff_time" not in history.columns:
        return history
    history = history.copy()
    history["_match_date"] = pd.to_datetime(history["kickoff_time"]).dt.date
    shots = shots.copy()
    shots["date"] = pd.to_datetime(shots["date"]).dt.date
    merged = history.merge(shots, left_on="_match_date", right_on="date", how="left").drop(columns=["_match_date", "date"], errors="ignore")
    return merged


def rank_team_players(
    team: str,
    team_goal_expectation: float,
    bootstrap: dict,
    current_event: int | None,
    position_priors: dict,
    limit: int = 8,
    reliability_coeffs: dict | None = None,
    lineup_model: dict | None = None,
    position_rate_models: dict | None = None,
    contribution_model: dict | None = None,
    is_home: bool = False,
    confirmed_starters: list[str] | None = None,
    confirmed_starter_ids: set[int] | None = None,
    player_shots_by_element: dict[int, pd.DataFrame] | None = None,
    context: object | None = None,
    opponent_defence: dict | None = None,
) -> list[dict]:
    """Ranked (by anytime-goal probability) list of a team's players for one
    fixture, given that fixture's team expected goals from the scoreline
    model. `reliability_coeffs` (from `fit_reliability_coefficients`) is
    optional — pass it for the reliability-adjusted goals/assists estimate.

    `player_shots_by_element` (FPL element id -> that player's own
    Understat-sourced {date, shots, shots_on_target} rows) is the live-
    serving counterpart to `_load_history_with_shots`'s training-frame
    merge: `fetch_player_summary`'s live per-player history has no shots
    columns at all (FPL's API doesn't provide them), so without this,
    `blended_current_form` never sees a `shots`/`shots_on_target` column to
    compute a rate from -- `predict_player`'s `rates.get("shots_per90",
    0.0)` then silently returns 0.0, indistinguishable from "genuinely no
    shots" (confirmed live: every player in every fixture)."""
    teams_by_id = {t["id"]: t["name"] for t in bootstrap["teams"]}
    team_id = next(
        (tid for tid, name in teams_by_id.items() if to_canonical(name, source="fpl") == team),
        None,
    )
    if team_id is None:
        return []

    confirmed_starters = confirmed_starters or []
    confirmed_names = {_normalise_name(name) for name in confirmed_starters}
    confirmed_starter_ids = confirmed_starter_ids or set()
    confirmed_elements = [
        element for element in bootstrap["elements"]
        if element["team"] == team_id
        and (int(element["id"]) in confirmed_starter_ids or _element_matches_confirmed_name(element, confirmed_names))
    ]
    elements = confirmed_elements if len(confirmed_elements) >= 9 else [element for element in bootstrap["elements"] if element["team"] == team_id]
    lineup_confirmed = len(confirmed_elements) >= 9

    # First pass: fetch each player's history once and predict their start
    # probability off the trained lineup model. Early in a season that
    # probability can clear 0.5 for nobody on a team (too little current-
    # season signal) — in that case, fall back to whoever actually started
    # this team's most recent match (last row of their own history, which
    # FPL includes with 0 minutes for an unused sub, so it reflects the
    # team's real most recent XI) rather than leaving every player unmarked.
    player_data = []
    for el in elements:
        position = POSITION_MAP.get(el["element_type"], "Unknown")
        history, prior_season = fpl_api.fetch_player_summary(el["id"], current_event)
        if player_shots_by_element and el["id"] in player_shots_by_element:
            history = _merge_shots_into_history(history, player_shots_by_element[el["id"]])
        rates, confidence = player_form.blended_current_form(history, prior_season, position, position_priors)
        start_features = player_form.current_start_features(history, fallback_minutes=rates["avg_minutes"])
        lineup = (
            {"predicted_starter": True, "expected_minutes": 90.0}
            if lineup_confirmed
            else predict_lineup(start_features, lineup_model)
        )
        player_data.append((el, position, history, rates, confidence, start_features, lineup))

    if not lineup_confirmed and not any(lineup["predicted_starter"] for *_, lineup in player_data):
        recent_starter_ids = set()
        for el, _position, history, *_rest in player_data:
            if history.empty:
                continue
            last_row = history.sort_values("GW").iloc[-1]
            started = bool(int(last_row.get("starts", 0) or 0)) or int(last_row.get("minutes", 0) or 0) >= 60
            if started:
                recent_starter_ids.add(el["id"])
        if recent_starter_ids:
            player_data = [
                (
                    el,
                    position,
                    history,
                    rates,
                    confidence,
                    start_features,
                    {"predicted_starter": True, "expected_minutes": max(lineup["expected_minutes"], 75.0)}
                    if el["id"] in recent_starter_ids
                    else lineup,
                )
                for el, position, history, rates, confidence, start_features, lineup in player_data
            ]

    results = []
    for el, position, history, rates, confidence, start_features, lineup in player_data:
        availability = fpl_api.availability_multiplier(el["status"], el.get("chance_of_playing_next_round"))
        is_penalty_taker = _is_primary_taker(el, "penalties_order")
        is_set_piece_taker = any(
            _is_primary_taker(el, field)
            for field in ("corners_and_indirect_freekicks_order", "direct_freekicks_order")
        )
        pred = predict_player(
            rates, team_goal_expectation, availability, reliability_coeffs,
            expected_minutes=lineup["expected_minutes"], position=position, is_home=is_home,
            position_rate_models=position_rate_models, is_penalty_taker=is_penalty_taker,
            is_set_piece_taker=is_set_piece_taker, context=context,
            opponent_defence=opponent_defence,
        )
        direct_contribution = predict_goal_contribution(
            rates, start_features, position, contribution_model, is_home=is_home,
            opponent_defence=opponent_defence,
        )
        pred["anytime_goal_contribution_prob"] = blend_contribution(
            direct_contribution,
            pred["anytime_goal_contribution_prob"],
            (contribution_model or {}).get("blend_weight"),
        )

        results.append(
            {
                "player_id": el["id"],
                "name": el["web_name"],
                "position": position,
                "status": el["status"],
                "news": el.get("news", ""),
                "availability": availability,
                "confidence": confidence,
                "predicted_starter": lineup["predicted_starter"],
                "expected_minutes": lineup["expected_minutes"] * availability,
                "confirmed_starter": lineup_confirmed,
                "is_penalty_taker": is_penalty_taker,
                "is_set_piece_taker": is_set_piece_taker,
                **pred,
            }
        )

    results.sort(key=lambda r: -r["anytime_goal_prob"])
    return results[: max(limit, len(confirmed_elements))]


def _is_primary_taker(element: dict, field: str) -> bool:
    try:
        return int(element.get(field, 0)) == 1
    except (TypeError, ValueError):
        return False


def _normalise_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed.casefold() if char.isalnum())


def _element_matches_confirmed_name(element: dict, confirmed_names: set[str]) -> bool:
    first_name = str(element.get("first_name", ""))
    second_name = str(element.get("second_name", ""))
    full_name = _normalise_name(f"{first_name}{second_name}")
    web_name = _normalise_name(element.get("web_name", ""))
    first_and_surname_parts = [_normalise_name(f"{first_name}{part}") for part in second_name.split()]
    return any(
        confirmed_name in candidate
        for confirmed_name in confirmed_names
        for candidate in (full_name, web_name, *first_and_surname_parts)
        if confirmed_name
    )
