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

    from ..evaluate.goal_contribution_research import BASE_FEATURES, ENHANCED_FEATURES, build_goal_contribution_frame

    frame, _ = build_goal_contribution_frame(seasons)
    all_features = [feature for feature in BASE_FEATURES + ENHANCED_FEATURES if feature in frame] + ["position"]
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
    return {"model": model, "calibrator": calibrator, "columns": columns, "features": all_features}


def predict_goal_contribution(rates: dict, start_features: dict, position: str, contribution_model: dict | None) -> float | None:
    """Return calibrated direct P(goal or assist), or None without a fit."""
    if not contribution_model:
        return None
    values = {
        feature: rates.get(feature, start_features.get(feature, 0.0)) or 0.0
        for feature in contribution_model["features"]
        if feature != "position"
    }
    values["position"] = position
    matrix = pd.get_dummies(pd.DataFrame([values]), columns=["position"], dtype=float)
    matrix = matrix.reindex(columns=contribution_model["columns"], fill_value=0.0)
    raw_probability = float(contribution_model["model"].predict_proba(matrix)[:, 1][0])
    logit = math.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1))
    return float(contribution_model["calibrator"].predict_proba(np.array([[logit]]))[:, 1][0])


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

    lam_goals = goals_estimate * scale
    lam_assists = assists_estimate * scale
    if is_penalty_taker:
        lam_goals += 0.15 * minutes_fraction * availability
    if is_set_piece_taker:
        lam_assists += 0.10 * minutes_fraction * availability

    return {
        "expected_goals": lam_goals,
        "expected_assists": lam_assists,
        "anytime_goal_prob": anytime_probability(lam_goals),
        "anytime_assist_prob": anytime_probability(lam_assists),
        "anytime_goal_contribution_prob": anytime_probability(lam_goals + lam_assists),
    }


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
) -> list[dict]:
    """Ranked (by anytime-goal probability) list of a team's players for one
    fixture, given that fixture's team expected goals from the scoreline
    model. `reliability_coeffs` (from `fit_reliability_coefficients`) is
    optional — pass it for the reliability-adjusted goals/assists estimate."""
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

    results = []
    for el in elements:
        position = POSITION_MAP.get(el["element_type"], "Unknown")

        history, prior_season = fpl_api.fetch_player_summary(el["id"], current_event)
        rates, confidence = player_form.blended_current_form(history, prior_season, position, position_priors)
        start_features = player_form.current_start_features(history, fallback_minutes=rates["avg_minutes"])
        lineup = (
            {"predicted_starter": True, "expected_minutes": 90.0}
            if lineup_confirmed
            else predict_lineup(start_features, lineup_model)
        )
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
            is_set_piece_taker=is_set_piece_taker,
        )
        direct_contribution = predict_goal_contribution(rates, start_features, position, contribution_model)
        if direct_contribution is not None:
            pred["anytime_goal_contribution_prob"] = max(
                direct_contribution,
                pred["anytime_goal_prob"],
                pred["anytime_assist_prob"],
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
