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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..config import MODELS_DIR
from ..features.build import build_training_frame
from . import market_models, scoreline

MANIFEST_PATH = MODELS_DIR / "manifest.json"
CORNERS_MODEL_PATH = MODELS_DIR / "corners_xgb.json"
CARDS_MODEL_PATH = MODELS_DIR / "cards_xgb.json"
DIXON_COLES_PATH = MODELS_DIR / "dixon_coles.pkl"
BIVARIATE_POISSON_PATH = MODELS_DIR / "bivariate_poisson.pkl"

RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def chronological_split(df: pd.DataFrame):
    """Hold out the most recent season as validation; train on the rest."""
    order = df["season"].str.slice(0, 4).astype(int)
    latest = order.max()
    val_mask = order == latest
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


def train_all(seasons: list[str] | None = None) -> Dict:
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    df, feature_cols = build_training_frame(seasons=seasons)
    train_df, val_df = chronological_split(df)
    X_train, X_val = train_df[feature_cols].fillna(0), val_df[feature_cols].fillna(0)

    print(f"Training on {len(train_df)} matches, validating on {len(val_df)} (held-out season).")

    dc_model = scoreline.fit_dixon_coles(train_df)
    bp_model = scoreline.fit_bivariate_poisson(train_df)
    dc_metrics = _evaluate_scoreline_model(dc_model, val_df)
    bp_metrics = _evaluate_scoreline_model(bp_model, val_df)
    chosen = "dixon_coles" if dc_metrics["rps"] <= bp_metrics["rps"] else "bivariate_poisson"
    print(f"  > Dixon-Coles RPS={dc_metrics['rps']:.4f}  Bivariate-Poisson RPS={bp_metrics['rps']:.4f}  (chosen: {chosen})")

    scoreline.save(dc_model, DIXON_COLES_PATH)
    scoreline.save(bp_model, BIVARIATE_POISSON_PATH)

    corners_dispersion = market_models.check_overdispersion(train_df["total_corners"].to_numpy())
    cards_dispersion = market_models.check_overdispersion(train_df["total_cards"].to_numpy())

    corners_model = market_models.train_lambda_regressor(X_train, train_df["total_corners"])
    cards_model = market_models.train_lambda_regressor(X_train, train_df["total_cards"])
    corners_metrics = _evaluate_count_model(corners_model, X_val, val_df["total_corners"])
    cards_metrics = _evaluate_count_model(cards_model, X_val, val_df["total_cards"])
    print(f"  > Corners MAE={corners_metrics['mae']:.2f}  Cards MAE={cards_metrics['mae']:.2f}")

    market_models.save_regressor(corners_model, CORNERS_MODEL_PATH)
    market_models.save_regressor(cards_model, CARDS_MODEL_PATH)

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(df["season"].unique().tolist()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "features": feature_cols,
        "scoreline": {
            "chosen_model": chosen,
            "dixon_coles": {"path": DIXON_COLES_PATH.name, "metrics": dc_metrics},
            "bivariate_poisson": {"path": BIVARIATE_POISSON_PATH.name, "metrics": bp_metrics},
        },
        "corners": {
            "path": CORNERS_MODEL_PATH.name,
            "metrics": corners_metrics,
            "dispersion": corners_dispersion,
        },
        "cards": {
            "path": CARDS_MODEL_PATH.name,
            "metrics": cards_metrics,
            "dispersion": cards_dispersion,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "No trained models found. Run `python -m pl_predictor.models.manifest` "
            "or notebooks 01-04 first."
        )
    return json.loads(MANIFEST_PATH.read_text())


def load_models() -> Dict:
    manifest = load_manifest()
    chosen = manifest["scoreline"]["chosen_model"]
    scoreline_path = DIXON_COLES_PATH if chosen == "dixon_coles" else BIVARIATE_POISSON_PATH
    scoreline_model = (
        pb.models.DixonColesGoalModel.load(str(scoreline_path))
        if chosen == "dixon_coles"
        else pb.models.BivariatePoissonGoalModel.load(str(scoreline_path))
    )
    return {
        "scoreline": scoreline_model,
        "corners": market_models.load_regressor(CORNERS_MODEL_PATH),
        "cards": market_models.load_regressor(CARDS_MODEL_PATH),
        "feature_cols": manifest["features"],
        "corners_dispersion": manifest["corners"]["dispersion"],
        "cards_dispersion": manifest["cards"]["dispersion"],
    }


if __name__ == "__main__":
    train_all()
