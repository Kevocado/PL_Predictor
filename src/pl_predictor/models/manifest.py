"""manifest.py — train/save/load orchestration.

Mirrors FPL_Optimizer/model.py 1:1 in structure: chronological_split (hold
out the most recent season for validation, never a random split — avoids
temporal leakage), train everything, write one manifest.json aggregating
metadata/metrics across every model, and symmetric load_manifest()/
load_models() helpers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..config import MODELS_DIR
from ..data import football_data
from ..features.build import FixtureFeatureContext, build_training_frame
from . import market_models, ml_scoreline, scoreline

MANIFEST_PATH = MODELS_DIR / "manifest.json"
MANIFEST_HISTORY_PATH = MODELS_DIR / "manifest_history.jsonl"
CORNERS_MODEL_PATH = MODELS_DIR / "corners_xgb.json"
CARDS_MODEL_PATH = MODELS_DIR / "cards_xgb.json"
DIXON_COLES_PATH = MODELS_DIR / "dixon_coles.pkl"
BIVARIATE_POISSON_PATH = MODELS_DIR / "bivariate_poisson.pkl"
ML_HOME_MODEL_PATH = MODELS_DIR / "ml_scoreline_home.json"
ML_AWAY_MODEL_PATH = MODELS_DIR / "ml_scoreline_away.json"

RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def chronological_split(df: pd.DataFrame, val_season: str | None = None):
    """Hold out `val_season` as validation; train on everything else.

    `val_season` defaults to the most recent *fully completed* season (the
    one just before `CURRENT_SEASON_START_YEAR`) rather than "whatever season
    is chronologically latest in `df`" — those two used to be the same thing,
    but aren't once `train_all` starts folding the in-progress season's
    played-so-far matches into `df` too (see `train_all`'s docstring): a
    "pick the latest season" rule would make the still-incomplete current
    season the validation set instead, which is a small, moving-target,
    not-comparable-across-retrains holdout. Pinning `val_season` explicitly
    keeps the holdout fixed and fair while `df` (and therefore training data)
    grows through the season."""
    if val_season is None:
        val_season = football_data.season_str(football_data.CURRENT_SEASON_START_YEAR - 1)
    val_mask = df["season"] == val_season
    return df[~val_mask].copy(), df[val_mask].copy()


def _score_outcome_probs(model, val_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    preds = [scoreline.predict_fixture(model, h, a) for h, a in zip(val_df["team_home"], val_df["team_away"])]
    probs = np.array([[p["home_win"], p["draw"], p["away_win"]] for p in preds])
    fallback_rate = float(np.mean([p["fallback"] for p in preds]))
    return probs, fallback_rate


def _evaluate_scoreline_model(model, val_df: pd.DataFrame) -> dict:
    probs, fallback_rate = _score_outcome_probs(model, val_df)
    outcomes = val_df["ftr"].map(RESULT_CODE).to_numpy()
    return {
        "rps": pb.metrics.rps_average(probs, outcomes),
        "brier": pb.metrics.multiclass_brier_score(probs, outcomes),
        "ignorance": pb.metrics.ignorance_score(probs, outcomes),
        "fallback_rate": fallback_rate,
    }


def _evaluate_count_model(model, X_val, y_val) -> dict:
    preds = model.predict(X_val)
    return {
        "mae": float(mean_absolute_error(y_val, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
        "mean_actual": float(np.mean(y_val)),
        "mean_predicted": float(np.mean(preds)),
    }


def _feature_importance(model, X_val, y_val, feature_cols: list[str]) -> dict:
    """Gain (XGBoost's internal split-value metric) and permutation
    importance (how much held-out R^2 drops when a feature is shuffled —
    more trustworthy, since gain can overrate a feature the model merely
    splits on often without it actually driving accuracy). Same two-metric
    pattern already proven in FPL_Optimizer/model.py."""
    gain = dict(zip(feature_cols, model.feature_importances_.astype(float)))
    perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, scoring="r2")
    permutation = dict(zip(feature_cols, perm.importances_mean.astype(float)))
    return {"gain": gain, "permutation": permutation}


def train_all(seasons: list[str] | None = None, include_current_season: bool = True) -> Dict:
    """`include_current_season=True` (default) is what makes this an
    *updating* model rather than a fixed one refit on the same 8 completed
    seasons all year: it folds the in-progress season's played-so-far
    matches into training (never into validation — `chronological_split`
    keeps the holdout pinned to the last fully completed season regardless).
    Every match added this way goes through the exact same shift(1)
    rolling-form/Elo/xG pipeline as historical data, using prior seasons as
    each team's starting context, so there's no cold-start cliff at the
    season boundary and no leakage. Call this again (e.g. via the "Retrain
    models" button, weekly or after a gameweek) to pull in whatever's been
    played since the last retrain — that's the mechanism, there's no
    background scheduler."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    completed_df = football_data.load_training_data(seasons=seasons)
    current_partial = football_data.fetch_current_season_partial() if include_current_season else None
    n_current_season_matches = 0
    if current_partial is not None and not current_partial.empty:
        matches_df = pd.concat([completed_df, current_partial], ignore_index=True).sort_values("date").reset_index(drop=True)
        n_current_season_matches = len(current_partial)
    else:
        matches_df = completed_df

    df, feature_cols = build_training_frame(matches_df=matches_df)
    train_df, val_df = chronological_split(df)
    X_train, X_val = train_df[feature_cols].fillna(0), val_df[feature_cols].fillna(0)

    print(
        f"Training on {len(train_df)} matches ({n_current_season_matches} from the in-progress season), "
        f"validating on {len(val_df)} (held-out season)."
    )

    dc_model = scoreline.fit_dixon_coles(train_df)
    bp_model = scoreline.fit_bivariate_poisson(train_df)
    dc_metrics = _evaluate_scoreline_model(dc_model, val_df)
    bp_metrics = _evaluate_scoreline_model(bp_model, val_df)

    ml_home_model, ml_away_model = ml_scoreline.train_goal_regressors(X_train, train_df["goals_home"], train_df["goals_away"])
    ml_metrics = ml_scoreline.evaluate_on_holdout(ml_home_model, ml_away_model, X_val, val_df)
    ml_teams = sorted(set(train_df["team_home"]) | set(train_df["team_away"]))

    candidates = {"dixon_coles": dc_metrics["rps"], "bivariate_poisson": bp_metrics["rps"], "ml_scoreline": ml_metrics["rps"]}
    chosen = min(candidates, key=candidates.get)
    print(
        f"  > Dixon-Coles RPS={dc_metrics['rps']:.4f}  Bivariate-Poisson RPS={bp_metrics['rps']:.4f}  "
        f"ML-scoreline RPS={ml_metrics['rps']:.4f}  (chosen: {chosen})"
    )

    scoreline.save(dc_model, DIXON_COLES_PATH)
    scoreline.save(bp_model, BIVARIATE_POISSON_PATH)
    market_models.save_regressor(ml_home_model, ML_HOME_MODEL_PATH)
    market_models.save_regressor(ml_away_model, ML_AWAY_MODEL_PATH)

    corners_dispersion = market_models.check_overdispersion(train_df["total_corners"].to_numpy())
    cards_dispersion = market_models.check_overdispersion(train_df["total_cards"].to_numpy())

    corners_model = market_models.train_lambda_regressor(X_train, train_df["total_corners"])
    cards_model = market_models.train_lambda_regressor(X_train, train_df["total_cards"])
    corners_metrics = _evaluate_count_model(corners_model, X_val, val_df["total_corners"])
    cards_metrics = _evaluate_count_model(cards_model, X_val, val_df["total_cards"])
    print(f"  > Corners MAE={corners_metrics['mae']:.2f}  Cards MAE={cards_metrics['mae']:.2f}")

    corners_importance = _feature_importance(corners_model, X_val, val_df["total_corners"], feature_cols)
    cards_importance = _feature_importance(cards_model, X_val, val_df["total_cards"], feature_cols)

    market_models.save_regressor(corners_model, CORNERS_MODEL_PATH)
    market_models.save_regressor(cards_model, CARDS_MODEL_PATH)

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(df["season"].unique().tolist()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_current_season_matches": n_current_season_matches,
        "features": feature_cols,
        "scoreline": {
            "chosen_model": chosen,
            "dixon_coles": {"path": DIXON_COLES_PATH.name, "metrics": dc_metrics},
            "bivariate_poisson": {"path": BIVARIATE_POISSON_PATH.name, "metrics": bp_metrics},
            "ml_scoreline": {
                "home_path": ML_HOME_MODEL_PATH.name,
                "away_path": ML_AWAY_MODEL_PATH.name,
                "metrics": ml_metrics,
                "teams": ml_teams,
            },
        },
        "corners": {
            "path": CORNERS_MODEL_PATH.name,
            "metrics": corners_metrics,
            "dispersion": corners_dispersion,
            "importance": corners_importance,
        },
        "cards": {
            "path": CARDS_MODEL_PATH.name,
            "metrics": cards_metrics,
            "dispersion": cards_dispersion,
            "importance": cards_importance,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    _append_history(manifest)
    return manifest


def _append_history(manifest: Dict) -> None:
    """One line per retrain — how the *chosen* model's holdout metrics and
    training-set size evolve as the season progresses and more of it gets
    folded into training. Append-only (unlike manifest.json, which each
    retrain overwrites) so the Calibration page can chart a real trend
    instead of only ever showing the single latest snapshot."""
    chosen = manifest["scoreline"]["chosen_model"]
    entry = {
        "trained_at": manifest["trained_at"],
        "n_train": manifest["n_train"],
        "n_current_season_matches": manifest["n_current_season_matches"],
        "chosen_model": chosen,
        "rps": manifest["scoreline"][chosen]["metrics"]["rps"],
        "brier": manifest["scoreline"][chosen]["metrics"]["brier"],
    }
    with MANIFEST_HISTORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_manifest_history() -> list[Dict]:
    if not MANIFEST_HISTORY_PATH.exists():
        return []
    lines = MANIFEST_HISTORY_PATH.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "No trained models found. Run `python -m pl_predictor.models.manifest` "
            "or notebooks 01-04 first."
        )
    return json.loads(MANIFEST_PATH.read_text())


def load_models(matches_df: pd.DataFrame | None = None) -> Dict:
    """`matches_df` is only required if the chosen scoreline model is
    `ml_scoreline` (it needs a `FixtureFeatureContext` built from current
    data to serve live predictions) — callers that only need corners/cards/
    the manifest can omit it. `dixon_coles_for_rankings` is always loaded
    regardless of `chosen_model`, since Power Rankings reads its fitted
    attack/defence parameters directly and an ML-based scoreline model
    doesn't have an equivalent to show there."""
    manifest = load_manifest()
    chosen = manifest["scoreline"]["chosen_model"]

    dixon_coles_for_rankings = pb.models.DixonColesGoalModel.load(str(DIXON_COLES_PATH))

    if chosen == "dixon_coles":
        scoreline_model = dixon_coles_for_rankings
    elif chosen == "bivariate_poisson":
        scoreline_model = pb.models.BivariatePoissonGoalModel.load(str(BIVARIATE_POISSON_PATH))
    else:
        if matches_df is None:
            raise ValueError("load_models(matches_df=...) is required when chosen_model is 'ml_scoreline'.")
        context = FixtureFeatureContext(matches_df)
        scoreline_model = ml_scoreline.MLScorelineModel(
            home_model=market_models.load_regressor(ML_HOME_MODEL_PATH),
            away_model=market_models.load_regressor(ML_AWAY_MODEL_PATH),
            feature_cols=manifest["features"],
            teams=manifest["scoreline"]["ml_scoreline"]["teams"],
            context=context,
        )

    return {
        "scoreline": scoreline_model,
        "dixon_coles_for_rankings": dixon_coles_for_rankings,
        "corners": market_models.load_regressor(CORNERS_MODEL_PATH),
        "cards": market_models.load_regressor(CARDS_MODEL_PATH),
        "feature_cols": manifest["features"],
        "corners_dispersion": manifest["corners"]["dispersion"],
        "cards_dispersion": manifest["cards"]["dispersion"],
    }


if __name__ == "__main__":
    train_all()
