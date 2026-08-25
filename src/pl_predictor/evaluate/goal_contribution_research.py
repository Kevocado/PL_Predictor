"""Chronological experiments for player G+A and match-market features.

Nothing in this module changes a production model.  It makes the competing
baselines explicit and returns data frames suitable for the research notebook:
the current Poisson union, a prevalence baseline, and calibrated direct G+A
classifiers.  Every player feature comes from the shifted builders in
``features.player_form``; current-match minutes, goals, and assists are only
ever targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import fpl_history
from ..data import football_data
from ..features import player_form
from ..features.build import build_training_frame
from ..models import market_models, ml_scoreline
from ..models.manifest import chronological_split

BASE_FEATURES = [
    "goals_per90_last3", "goals_per90_last5", "goals_per90_last10",
    "assists_per90_last3", "assists_per90_last5", "assists_per90_last10",
    "expected_goals_per90_last3", "expected_goals_per90_last5", "expected_goals_per90_last10",
    "expected_assists_per90_last3", "expected_assists_per90_last5", "expected_assists_per90_last10",
    "starts_last3", "starts_last5", "starts_last10", "sub_rate_last3", "sub_rate_last5",
    "minutes_last3", "minutes_last5", "minutes_last10", "minutes_ema", "start_streak", "was_home",
]
ENHANCED_FEATURES = [
    "threat_last10", "creativity_last10", "ict_index_last10", "bps_last10", "bonus_last10",
    "expected_goal_involvements_per90_last10",
]


def build_goal_contribution_frame(seasons: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return one pre-match feature row per player-fixture and its G+A target."""
    raw = fpl_history.load_player_gw_history(seasons=seasons).sort_values(["season", "element", "kickoff_time"])
    starts, start_features = player_form.build_historical_start_features(raw)
    played, form_features = player_form.build_historical_player_form(raw)

    # ``played`` contains only positive-minute rows, while the lineup frame
    # keeps bench rows. Joining by original row index retains those legitimate
    # zero-contribution cases without using their realised minutes as a feature.
    rows = starts.join(played[form_features], how="left")
    rows["goal_contribution"] = ((rows["goals_scored"] + rows["assists"]) > 0).astype(int)
    rows["position"] = rows["position"].fillna("UNK")
    numeric_features = [feature for feature in BASE_FEATURES + ENHANCED_FEATURES if feature in rows]
    rows[numeric_features] = rows[numeric_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows["expected_minutes_pre_match"] = rows["minutes_ema"].clip(lower=0, upper=90).fillna(0.0)
    return rows, numeric_features + ["position"]


def _design_matrix(frame: pd.DataFrame, features: list[str], columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.get_dummies(frame[features], columns=["position"], dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is not None:
        matrix = matrix.reindex(columns=columns, fill_value=0.0)
    return matrix, matrix.columns.tolist()


def _ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    bucket = np.clip((probability * bins).astype(int), 0, bins - 1)
    total = len(actual)
    if total == 0:
        return float("nan")
    return float(sum(abs(actual[bucket == index].mean() - probability[bucket == index].mean()) * (bucket == index).sum() / total for index in range(bins) if (bucket == index).any()))


def _metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "average_precision": float(average_precision_score(actual, probability)),
        "ece": _ece(actual, probability),
    }


def _fit_platt(raw_probability: np.ndarray, actual: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(actual)) < 2:
        return None
    logits = np.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1))
    return LogisticRegression(max_iter=1000).fit(logits.reshape(-1, 1), actual)


def _apply_platt(model: LogisticRegression | None, raw_probability: np.ndarray) -> np.ndarray:
    if model is None:
        return raw_probability
    logits = np.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1))
    return model.predict_proba(logits.reshape(-1, 1))[:, 1]


def _poisson_union(frame: pd.DataFrame) -> np.ndarray:
    rate = frame["goals_per90_last10"].to_numpy() + frame["assists_per90_last10"].to_numpy()
    minutes = frame["expected_minutes_pre_match"].to_numpy()
    return 1 - np.exp(-np.clip(rate * minutes / 90, 0, None))


def _evaluate_fold(frame: pd.DataFrame, train_seasons: list[str], test_season: str) -> tuple[list[dict], pd.DataFrame]:
    train = frame[frame["season"].isin(train_seasons)]
    test = frame[frame["season"] == test_season]
    calibration_season = train_seasons[-1]
    fit = train[train["season"] != calibration_season]
    calibration = train[train["season"] == calibration_season]
    if fit.empty or calibration.empty or test.empty:
        return [], pd.DataFrame()

    y_fit = fit["goal_contribution"].to_numpy()
    y_calibration = calibration["goal_contribution"].to_numpy()
    y_test = test["goal_contribution"].to_numpy()
    rows = []
    importance = pd.DataFrame()

    prevalence = np.full(len(y_test), y_fit.mean())
    rows.append({"fold": test_season, "model": "prevalence", "n_test": len(test), **_metrics(y_test, prevalence)})

    union_fit = _poisson_union(fit)
    union_calibration = _poisson_union(calibration)
    union_test = _apply_platt(_fit_platt(union_calibration, y_calibration), _poisson_union(test))
    rows.append({"fold": test_season, "model": "poisson_union_calibrated", "n_test": len(test), **_metrics(y_test, union_test)})

    for name, features in (("direct_base", BASE_FEATURES), ("direct_enhanced", BASE_FEATURES + ENHANCED_FEATURES)):
        available = [feature for feature in features if feature in frame]
        feature_names = available + ["position"]
        X_fit, columns = _design_matrix(fit, feature_names)
        X_calibration, _ = _design_matrix(calibration, feature_names, columns)
        X_test, _ = _design_matrix(test, feature_names, columns)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X_fit, y_fit)
        calibration_prob = model.predict_proba(X_calibration)[:, 1]
        test_prob = _apply_platt(_fit_platt(calibration_prob, y_calibration), model.predict_proba(X_test)[:, 1])
        rows.append({"fold": test_season, "model": name, "n_test": len(test), **_metrics(y_test, test_prob)})
        if name == "direct_enhanced":
            coefficients = model[-1].coef_[0]
            importance = pd.DataFrame({"feature": columns, "coefficient": coefficients, "abs_coefficient": np.abs(coefficients)}).sort_values("abs_coefficient", ascending=False)

    return rows, importance


def evaluate_goal_contribution_models(seasons: list[str] | None = None, min_train_seasons: int = 2) -> dict:
    """Walk forward through seasons and compare calibrated G+A approaches.

    The return value is deliberately report-shaped rather than a fitted model:
    callers must inspect the metrics before considering any production change.
    """
    frame, _ = build_goal_contribution_frame(seasons)
    available_seasons = sorted(frame["season"].unique())
    all_rows: list[dict] = []
    importances: list[pd.DataFrame] = []
    for index in range(min_train_seasons, len(available_seasons)):
        rows, importance = _evaluate_fold(frame, available_seasons[:index], available_seasons[index])
        all_rows.extend(rows)
        if not importance.empty:
            importance["fold"] = available_seasons[index]
            importances.append(importance)
    metrics = pd.DataFrame(all_rows)
    summary = metrics.groupby("model", as_index=False)[["brier", "log_loss", "average_precision", "ece"]].mean() if not metrics.empty else pd.DataFrame()
    feature_importance = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient", "fold"])
    return {"metrics": metrics, "summary": summary, "feature_importance": feature_importance}


def build_projected_team_player_features(seasons: list[str] | None = None) -> pd.DataFrame:
    """Aggregate shifted player form into strictly pre-kickoff team features.

    The realised starter/minutes columns are never used.  Instead each player
    is weighted by their prior start rate, which makes this a valid candidate
    input to a historical scoreline experiment rather than a hindsight XI.
    """
    raw = fpl_history.load_player_gw_history(seasons=seasons)
    starts, _ = player_form.build_historical_start_features(raw)
    played, form_features = player_form.build_historical_player_form(raw)
    rows = starts.join(played[form_features], how="left")
    rows["start_weight"] = rows["starts_last5"].fillna(0).clip(0, 1)
    rows["kickoff_date"] = pd.to_datetime(rows["kickoff_time"], utc=True).dt.tz_localize(None).dt.normalize()

    definitions = {
        "projected_goal_rate": "goals_per90_last10",
        "projected_assist_rate": "assists_per90_last10",
        "projected_threat": "threat_last10",
        "projected_creativity": "creativity_last10",
    }
    for target, source in definitions.items():
        rows[target] = rows["start_weight"] * rows.get(source, 0).fillna(0)

    group_keys = ["kickoff_date", "fixture", "team", "was_home"]
    totals = rows.groupby(group_keys, as_index=False)[list(definitions)].sum()
    top = rows.groupby(group_keys, as_index=False)[list(definitions)].max()
    top = top.rename(columns={name: f"top_{name}" for name in definitions})
    totals = totals.merge(top, on=group_keys, how="left")

    aggregate_names = list(definitions) + [f"top_{name}" for name in definitions]
    home = totals[totals["was_home"]].drop(columns="was_home").rename(columns={"team": "team_home"})
    away = totals[~totals["was_home"]].drop(columns="was_home").rename(columns={"team": "team_away"})
    home = home.rename(columns={name: f"home_{name}" for name in aggregate_names})
    away = away.rename(columns={name: f"away_{name}" for name in aggregate_names})
    return home.merge(away, on=["kickoff_date", "fixture"], how="inner")


def evaluate_scoreline_player_aggregates(seasons: list[str] | None = None) -> pd.DataFrame:
    """Compare scoreline XGBoost with and without projected player aggregates.

    This is intentionally a research report.  A candidate cannot replace the
    live scoreline feature set until it also passes the existing multi-season
    walk-forward process.
    """
    matches = football_data.load_training_data(seasons=seasons)
    frame, feature_cols = build_training_frame(matches_df=matches)
    aggregates = build_projected_team_player_features(seasons=seasons)
    frame = frame.copy()
    frame["kickoff_date"] = pd.to_datetime(frame["date"]).dt.normalize()
    merged = frame.merge(aggregates, on=["kickoff_date", "team_home", "team_away"], how="left")
    aggregate_cols = [column for column in merged.columns if column.startswith(("home_projected_", "away_projected_", "home_top_", "away_top_"))]
    train, validation = chronological_split(merged)
    rows = []
    for name, columns in (("production_features", feature_cols), ("projected_player_aggregates", feature_cols + aggregate_cols)):
        X_train = train[columns].fillna(0)
        X_validation = validation[columns].fillna(0)
        home_model, away_model = ml_scoreline.train_goal_regressors(X_train, train["goals_home"], train["goals_away"])
        rows.append({"model": name, "n_features": len(columns), **ml_scoreline.evaluate_on_holdout(home_model, away_model, X_validation, validation)})
    return pd.DataFrame(rows)


def evaluate_market_probability_calibration(seasons: list[str] | None = None) -> pd.DataFrame:
    """Check count-model O/U probabilities, including shot-situation ablation.

    Corners/cards fair-odds guidance remains informational until this report
    shows a repeatable calibration improvement, not merely lower count MAE.
    """
    matches = football_data.load_training_data(seasons=seasons)
    frame, feature_cols = build_training_frame(matches_df=matches)
    train, validation = chronological_split(frame)
    situation_cols = [column for column in frame if "set_piece_xg_share" in column]
    rows = []
    for feature_set, columns in (("production_features", feature_cols), ("with_shot_situation", feature_cols + situation_cols)):
        for market, target, line in (("corners", "total_corners", 9.5), ("cards", "total_cards", 3.5)):
            model = market_models.train_lambda_regressor(train[columns].fillna(0), train[target])
            lambdas = model.predict(validation[columns].fillna(0))
            dispersion = market_models.check_overdispersion(train[target].to_numpy())
            probabilities = np.array([market_models.price_over_under(value, line, dispersion)["over"] for value in lambdas])
            actual = (validation[target].to_numpy() > line).astype(int)
            rows.append(
                {
                    "market": market,
                    "feature_set": feature_set,
                    "line": line,
                    "mae": float(np.mean(np.abs(lambdas - validation[target].to_numpy()))),
                    **_metrics(actual, probabilities),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    report = evaluate_goal_contribution_models()
    print(report["summary"].to_string(index=False))
    print("\nDirect G+A feature importance")
    print(report["feature_importance"].groupby("feature", as_index=False)["abs_coefficient"].mean().sort_values("abs_coefficient", ascending=False).head(20).to_string(index=False))
