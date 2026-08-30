"""Role-aware, evidence-weighted Player Hub ratings.

The scores are descriptive, not forecasts.  They consume only the cached FPL
bootstrap response, so opening Player Hub never fans out into player-history
requests or changes a fixture prediction.  Quality is a fixed role-aware
scale, Form is a deliberately constrained short-term lift, and Impact is the
only score affected by availability for the next gameweek.
"""

from __future__ import annotations

import pandas as pd


ROLE_PRIORS = {"GK": 50.0, "DEF": 50.0, "MID": 50.0, "FWD": 50.0}
ROLE_LABELS = {
    "GK": {"saves": "Saves and shot prevention", "clean_sheets": "Clean sheets", "bps": "Bonus-point system"},
    "DEF": {"clean_sheets": "Clean sheets", "defensive_contribution": "Defensive contribution", "expected_goal_involvements": "Expected goal involvements"},
    "MID": {"expected_goal_involvements": "Expected goal involvements", "expected_assists": "Chance creation", "threat": "Attacking threat"},
    "FWD": {"expected_goals": "Expected goals", "expected_goal_involvements": "Expected goal involvements", "goals_scored": "Goals per 90"},
}


def _number(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _availability(element: dict) -> float:
    if element.get("status", "a") in {"i", "s", "u"}:
        return 0.0
    if element.get("status") == "d":
        return _number(element.get("chance_of_playing_next_round")) / 100 if element.get("chance_of_playing_next_round") is not None else 0.5
    return 1.0


def _expected_minutes(element: dict, availability: float) -> float:
    minutes = _number(element.get("minutes"))
    starts = _number(element.get("starts"))
    appearances = max(_number(element.get("appearances")), starts, minutes / 60, 1)
    return min(90.0, (25.0 + 65.0 * min(1.0, starts / appearances)) * availability)


def _per90(element: dict, field: str) -> float:
    return _number(element.get(field)) / max(_number(element.get("minutes")), 1.0) * 90.0


def _role_components(element: dict, position: str) -> dict[str, float]:
    """Fixed-scale components.  They are deliberately not percentiles."""
    xg = _per90(element, "expected_goals")
    xa = _per90(element, "expected_assists")
    xgi = _per90(element, "expected_goal_involvements")
    goals = _per90(element, "goals_scored")
    if position == "GK":
        return {
            "saves": min(18.0, _per90(element, "saves") * 4.0),
            "clean_sheets": min(17.0, _per90(element, "clean_sheets") * 17.0),
            "bps": min(13.0, _per90(element, "bps") * 1.5),
        }
    if position == "DEF":
        return {
            "clean_sheets": min(18.0, _per90(element, "clean_sheets") * 18.0),
            "defensive_contribution": min(12.0, _per90(element, "defensive_contribution") * 0.12),
            "expected_goal_involvements": min(15.0, xgi * 25.0),
            "bps": min(8.0, _per90(element, "bps") * 0.9),
        }
    if position == "FWD":
        return {
            "expected_goals": min(28.0, xg * 38.0),
            "expected_goal_involvements": min(18.0, xgi * 20.0),
            "goals_scored": min(10.0, goals * 12.0),
        }
    return {
        "expected_goal_involvements": min(26.0, xgi * 25.0),
        "expected_assists": min(12.0, xa * 15.0),
        "threat": min(6.0, _per90(element, "threat") * 0.06),
        "creativity": min(6.0, _per90(element, "creativity") * 0.06),
        "goals_scored": min(8.0, goals * 9.0),
    }


def _quality_score(element: dict, position: str) -> tuple[float, str]:
    components = _role_components(element, position)
    strongest = max(components, key=components.get, default="expected_goal_involvements")
    raw = min(92.0, ROLE_PRIORS.get(position, 50.0) + sum(components.values()))
    minutes, starts = _number(element.get("minutes")), _number(element.get("starts"))
    confidence = min(1.0, (minutes / 900.0 + starts / 10.0) / 2.0)
    quality = ROLE_PRIORS.get(position, 50.0) * (1.0 - confidence) + raw * confidence
    label = ROLE_LABELS.get(position, ROLE_LABELS["MID"]).get(strongest, strongest.replace("_", " ").title())
    return round(min(92.0, max(0.0, quality)), 1), label


def _form_score(element: dict, position: str) -> float:
    """Return an earned 0–15 lift, gated by real opportunity evidence."""
    minutes, starts = _number(element.get("minutes")), _number(element.get("starts"))
    if minutes < 360.0 or starts < 4.0:
        return 0.0
    xgi = _per90(element, "expected_goal_involvements")
    output = _per90(element, "goals_scored") + _per90(element, "assists")
    opportunity = min(1.0, (minutes / 900.0 + starts / 10.0) / 2.0)
    if position == "GK":
        underlying = min(1.0, (_per90(element, "saves") / 4.0 + _per90(element, "clean_sheets")) / 2.0)
        actual = min(1.0, (_per90(element, "clean_sheets") + _per90(element, "bps") / 12.0) / 2.0)
    elif position == "DEF":
        underlying = min(1.0, (xgi / 0.45 + _per90(element, "clean_sheets")) / 2.0)
        actual = min(1.0, (output / 0.45 + _per90(element, "clean_sheets")) / 2.0)
    elif position == "FWD":
        underlying = min(1.0, (_per90(element, "expected_goals") / 0.75 + xgi / 0.95) / 2.0)
        actual = min(1.0, output / 0.85)
    else:
        underlying = min(1.0, xgi / 0.75)
        actual = min(1.0, output / 0.75)
    return round(min(15.0, 6.0 * underlying + 5.0 * actual + 4.0 * opportunity), 1)


def rate_bootstrap_elements(elements: list[dict], positions: dict[int, str]) -> dict[int, dict]:
    """Return fixed-scale Quality, Form, Overall, and Impact per element."""
    result: dict[int, dict] = {}
    for element in elements:
        position = positions.get(int(element.get("element_type", 0)), "MID")
        quality, driver = _quality_score(element, position)
        form = _form_score(element, position)
        overall = round(min(100.0, quality + form), 1)
        availability = _availability(element)
        expected_minutes = _expected_minutes(element, availability)
        impact = round(overall * availability * expected_minutes / 90.0, 1)
        result[int(element["id"])] = {
            "quality_rating": quality,
            "form_rating": form,
            "overall_rating": overall,
            "current_impact_rating": impact,
            "rating_driver": driver,
            "rating_expected_minutes": round(expected_minutes, 1),
            "rating_model_source": "role_aware_evidence_baseline",
        }
    return result


def evaluate_role_models(history: pd.DataFrame) -> pd.DataFrame:
    """Chronologically compare per-position FPL-points-per-90 models.

    This is intentionally an offline research entry point, never called by
    the API.  Every feature is already shifted by ``player_form`` before the
    target row is scored; the newest completed season is the holdout.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ..features.player_form import build_historical_player_form

    rows, feature_cols = build_historical_player_form(history)
    rows = rows.copy()
    rows["target_points_per90"] = rows["total_points"] / rows["minutes"] * 90
    latest = sorted(rows["season"].dropna().unique())[-1]
    report = []
    for position in ("GK", "DEF", "MID", "FWD"):
        role = rows[rows["position"] == position].dropna(subset=feature_cols + ["target_points_per90"])
        train, validation = role[role["season"] != latest], role[role["season"] == latest]
        if len(train) < 30 or validation.empty:
            continue
        baseline = float(train["target_points_per90"].mean())
        baseline_prediction = [baseline] * len(validation)
        model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
        model.fit(train[feature_cols], train["target_points_per90"])
        rich_prediction = model.predict(validation[feature_cols])
        baseline_mae = mean_absolute_error(validation["target_points_per90"], baseline_prediction)
        rich_mae = mean_absolute_error(validation["target_points_per90"], rich_prediction)
        baseline_rmse = mean_squared_error(validation["target_points_per90"], baseline_prediction) ** 0.5
        rich_rmse = mean_squared_error(validation["target_points_per90"], rich_prediction) ** 0.5
        coefficients = model.named_steps["ridge"].coef_
        driver = feature_cols[int(abs(coefficients).argmax())]
        report.append(
            {
                "position": position,
                "validation_season": latest,
                "baseline_mae": float(baseline_mae),
                "rich_mae": float(rich_mae),
                "baseline_rmse": float(baseline_rmse),
                "rich_rmse": float(rich_rmse),
                "selected_model": "rich" if rich_mae < baseline_mae and rich_rmse <= baseline_rmse else "baseline",
                "top_driver": driver,
                "n_train": len(train),
                "n_validation": len(validation),
            }
        )
    return pd.DataFrame(report)
