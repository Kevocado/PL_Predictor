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


def _column(rows: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(rows.get(name, 0.0), errors="coerce").fillna(0.0) if name in rows else pd.Series(0.0, index=rows.index)


def _historical_role_quality(rows: pd.DataFrame) -> pd.Series:
    """Fixed-scale Quality built exclusively from shifted pre-match fields."""
    position = rows["position"].fillna("MID")
    xg = _column(rows, "expected_goals_per90_last10")
    xa = _column(rows, "expected_assists_per90_last10")
    xgi = _column(rows, "expected_goal_involvements_per90_last10")
    goals = _column(rows, "goals_per90_last10")
    saves = _column(rows, "saves_per90_last10")
    clean_sheets = _column(rows, "clean_sheets_per90_last10")
    bps = _column(rows, "bps_last10")
    defensive = _column(rows, "defensive_contribution_last10")
    # This is intentionally separate from the live rating implementation:
    # shifted historical fields have a different shape. Its independently
    # tuned caps were GK 48, DEF 53, MID 46, FWD 56, so scale every role down
    # to MID's smallest existing ceiling before it becomes a team unit.
    role_cap_scale = {"GK": 46.0 / 48.0, "DEF": 46.0 / 53.0, "MID": 1.0, "FWD": 46.0 / 56.0}
    raw = pd.Series(50.0, index=rows.index)
    raw += np.where(position == "GK", (np.minimum(18, saves * 4) + np.minimum(17, clean_sheets * 17) + np.minimum(13, bps * 1.5)) * role_cap_scale["GK"], 0)
    raw += np.where(position == "DEF", (np.minimum(18, clean_sheets * 18) + np.minimum(12, defensive * .12) + np.minimum(15, xgi * 25) + np.minimum(8, bps * .9)) * role_cap_scale["DEF"], 0)
    raw += np.where(position == "MID", (np.minimum(26, xgi * 25) + np.minimum(12, xa * 15) + np.minimum(8, goals * 9)) * role_cap_scale["MID"], 0)
    raw += np.where(position == "FWD", (np.minimum(28, xg * 38) + np.minimum(18, xgi * 20) + np.minimum(10, goals * 12)) * role_cap_scale["FWD"], 0)
    confidence = ((_column(rows, "starts_last10") * 10 / 10) + (_column(rows, "minutes_ema") / 90)) / 2
    return (50.0 + (raw.clip(upper=92.0) - 50.0) * confidence.clip(0, 1)).clip(0, 92)


def _historical_form_lift(rows: pd.DataFrame) -> pd.Series:
    """Capped form lift; no row can read its current-match outcome."""
    position = rows["position"].fillna("MID")
    starts = _column(rows, "starts_last5") * 5
    opportunity = ((_column(rows, "starts_last5") + _column(rows, "minutes_ema") / 90) / 2).clip(0, 1)
    xg = _column(rows, "expected_goals_per90_last3")
    xgi = _column(rows, "expected_goal_involvements_per90_last3")
    output = _column(rows, "goals_per90_last3") + _column(rows, "assists_per90_last3")
    underlying = np.where(position == "FWD", (xg / .75 + xgi / .95) / 2, xgi / .75)
    actual = np.where(position == "FWD", output / .85, output / .75)
    form = (6 * np.minimum(1, underlying) + 5 * np.minimum(1, actual) + 4 * opportunity).clip(0, 15)
    return pd.Series(np.where(starts >= 4, form, 0.0), index=rows.index)


def _legal_expected_xi_weight(rows: pd.DataFrame, group_keys: list[str]) -> pd.Series:
    """Select a legal, pre-match projected XI without realised lineups."""
    weights = pd.Series(0.0, index=rows.index)
    max_per_position = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
    minimum = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    for _, group in rows.groupby(group_keys, sort=False):
        candidates = group[group["position"].isin(max_per_position)].copy()
        candidates["selection_score"] = _column(candidates, "starts_last5") * _column(candidates, "minutes_ema").clip(0, 90)
        selected: list[int] = []
        counts = {position: 0 for position in max_per_position}
        for position, required in minimum.items():
            picks = candidates[candidates["position"] == position].sort_values("selection_score", ascending=False).head(required)
            selected.extend(picks.index.tolist())
            counts[position] += len(picks)
        remaining = candidates.drop(index=selected, errors="ignore").sort_values("selection_score", ascending=False)
        for index, row in remaining.iterrows():
            if len(selected) >= 11:
                break
            if counts[row["position"]] < max_per_position[row["position"]]:
                selected.append(index)
                counts[row["position"]] += 1
        selected_rows = candidates.loc[selected] if selected else candidates.iloc[0:0]
        weights.loc[selected_rows.index] = (
            _column(selected_rows, "starts_last5").clip(0, 1) * _column(selected_rows, "minutes_ema").clip(0, 90) / 90
        )
    return weights


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
    merged = home.merge(away, on=["kickoff_date", "fixture"], how="inner")

    # Research-only role-unit candidate.  All source columns are shifted;
    # legal expected-XI selection therefore cannot read a realised lineup,
    # minutes, availability, or outcome from the current match.
    rows["historical_quality"] = _historical_role_quality(rows)
    rows["historical_form"] = _historical_form_lift(rows)
    rows["historical_overall"] = (rows["historical_quality"] + rows["historical_form"]).clip(upper=100)
    rows["expected_xi_weight"] = _legal_expected_xi_weight(rows, group_keys)
    rows["weighted_unit_score"] = rows["historical_overall"] * rows["expected_xi_weight"]
    units = rows.groupby(group_keys + ["position"], as_index=False)["weighted_unit_score"].sum()
    units = units[units["position"].isin(["GK", "DEF", "MID", "FWD"])]
    home_units = units[units["was_home"]].pivot_table(index=["kickoff_date", "fixture"], columns="position", values="weighted_unit_score", fill_value=0).reset_index()
    away_units = units[~units["was_home"]].pivot_table(index=["kickoff_date", "fixture"], columns="position", values="weighted_unit_score", fill_value=0).reset_index()
    home_units = home_units.rename(columns={position: f"home_{position.lower()}_unit_strength" for position in ("GK", "DEF", "MID", "FWD")})
    away_units = away_units.rename(columns={position: f"away_{position.lower()}_unit_strength" for position in ("GK", "DEF", "MID", "FWD")})
    for frame, prefix in ((home_units, "home"), (away_units, "away")):
        for position in ("gk", "def", "mid", "fwd"):
            column = f"{prefix}_{position}_unit_strength"
            if column not in frame:
                frame[column] = 0.0
    return merged.merge(home_units, on=["kickoff_date", "fixture"], how="left").merge(away_units, on=["kickoff_date", "fixture"], how="left")


def summarise_team_unit_experiment(results: pd.DataFrame) -> dict:
    """Summarise candidate folds without allowing automatic promotion."""
    metrics = [metric for metric in ("rps", "brier", "ece", "log_loss", "coverage") if metric in results]
    baseline = results[results["model"] == "production_features"]
    candidate = results[results["model"] == "role_unit_strength"]
    return {
        "baseline_metrics": baseline[metrics].mean().to_dict(),
        "candidate_metrics": candidate[metrics].mean().to_dict(),
        "feature_importance": candidate.get("top_feature", pd.Series(dtype=str)).value_counts().to_dict(),
        "manual_review_required": True,
        "promotion_eligible": False,
        "promotion_gate": "Manual review requires lower mean RPS, no calibration regression, and a non-regressing recent fold.",
    }


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


def evaluate_scoreline_player_aggregates_walk_forward(
    seasons: list[str] | None = None, min_train_seasons: int = 3
) -> pd.DataFrame:
    """Re-test EXP-2026-03's projected player aggregates across every
    multi-season walk-forward fold, not just the one fixed 2025-26 holdout
    `evaluate_scoreline_player_aggregates` checks — this is the "expand to
    walk-forward folds first" follow-up the continuity log calls for before
    any promotion decision. One row per (fold, model); average across folds
    before deciding anything, and check for a material single-fold
    regression, same promotion bar as every other experiment here.

    `seasons` (if given) is `football_data`'s season format
    (`"2018-2019"`) and only governs the walk-forward fold range — player
    aggregates always pull the FPL history archive's own full season format
    (`"2018-19"`) independently, since the two data sources use different
    season-string conventions (mixing them here would silently produce an
    empty/wrong merge rather than an error). Folds outside FPL history's
    coverage simply get 0-filled aggregate columns via the existing left
    merge, same as `evaluate_scoreline_player_aggregates` already does for
    its own single holdout."""
    from ..data import fpl_history
    from . import walk_forward

    aggregates = build_projected_team_player_features(seasons=fpl_history.default_completed_seasons(n=8))
    aggregate_cols = [column for column in aggregates.columns if column.endswith("_unit_strength")]

    baseline_folds = walk_forward.prepare_folds(seasons=seasons, min_train_seasons=min_train_seasons)
    candidate_folds = walk_forward.prepare_folds(
        seasons=seasons,
        min_train_seasons=min_train_seasons,
        extra_feature_frame=aggregates,
        extra_feature_cols=aggregate_cols,
    )

    baseline = walk_forward.evaluate_folds(baseline_folds)
    baseline["model"] = "production_features"
    candidate = walk_forward.evaluate_folds(candidate_folds)
    candidate["model"] = "role_unit_strength"
    return pd.concat([baseline, candidate], ignore_index=True)


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
