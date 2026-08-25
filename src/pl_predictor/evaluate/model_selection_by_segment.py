"""model_selection_by_segment.py — does a different scoreline model type
(Dixon-Coles, Bivariate-Poisson, or ml_scoreline) actually win on a
different market or a different, pre-registered fixture segment?

This is the evidence-gathering step Part 4 of the match-dominance plan
depends on: only build per-market/per-segment model *selection* into
`models/manifest.py`/`models/scoreline.py` if a real, multi-fold,
most-recent-season-corroborated edge shows up here for a specific model on
a specific segment — otherwise the honest conclusion is to keep the single
global `chosen_model`, same as every other rejected candidate this session.

Segments are pre-registered here, before looking at any result, to avoid
the segment-mining trap that already sank EXP-2026-02's blend:
  - Overall (every market, no segment) — the baseline comparison.
  - Cold-start-involved fixtures (either team's `confidence_home`/
    `confidence_away` is not `"current"` — i.e. the existing cold-start
    blend is doing real work for that side) vs. established-only fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..data import football_data
from ..features.build import build_training_frame
from ..models import ml_scoreline, scoreline
from ..models.scoreline import evaluate_grids_multi_market


def prepare_folds(seasons: list[str] | None = None, min_train_seasons: int = 3) -> list[dict]:
    seasons = seasons or football_data.default_completed_seasons(n=8)
    matches_df = football_data.load_training_data(seasons=seasons)
    df, feature_cols = build_training_frame(matches_df=matches_df)
    ml_feature_cols = [c for c in feature_cols if "fouls" not in c]

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_df = df[df["season"].isin(seasons[:i])]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append(
            {
                "val_season": val_season,
                "train_df": train_df,
                "val_df": val_df,
                "X_train": train_df[ml_feature_cols].fillna(0),
                "X_val": val_df[ml_feature_cols].fillna(0),
            }
        )
    return folds


def _dc_bp_grids(model, val_df: pd.DataFrame) -> list:
    """Per-row prediction, falling back to `FALLBACK_GOAL_EXPECTANCY` for a
    team the model never saw in training — the same fallback
    `scoreline.predict_fixture` uses, replicated here (rather than reused
    directly) because that function returns an already-extracted dict, not
    the raw grid object `evaluate_grids_multi_market` needs. Exactly the case a
    cold-start-fixture segment needs to handle gracefully rather than
    erroring; DC/BP have no batch-predict path (see `predict_fixtures_
    batch`'s own docstring), so a Python loop is already how these two
    models are evaluated everywhere else in this codebase."""
    grids = []
    for h, a in zip(val_df["team_home"], val_df["team_away"]):
        if scoreline.is_known_team(model, h) and scoreline.is_known_team(model, a):
            grids.append(model.predict(h, a, max_goals=10))
        else:
            grids.append(
                pb.models.create_dixon_coles_grid(
                    scoreline.FALLBACK_GOAL_EXPECTANCY, scoreline.FALLBACK_GOAL_EXPECTANCY, rho=0.0, max_goals=10
                )
            )
    return grids


def _is_cold_start_involved(val_df: pd.DataFrame) -> pd.Series:
    return (val_df["confidence_home"] != "current") | (val_df["confidence_away"] != "current")


def evaluate_models_by_segment(seasons: list[str] | None = None, min_train_seasons: int = 3) -> pd.DataFrame:
    """One row per (model, fold, segment): every scoreline market's
    metrics, for Dixon-Coles, Bivariate-Poisson, and ml_scoreline
    independently, split by the cold-start-involved segment."""
    folds = prepare_folds(seasons, min_train_seasons)
    rows = []
    for fold in folds:
        dc_model = scoreline.fit_dixon_coles(fold["train_df"])
        bp_model = scoreline.fit_bivariate_poisson(fold["train_df"])
        ml_home, ml_away = ml_scoreline.train_goal_regressors(
            fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
        )

        model_grids = {
            "dixon_coles": _dc_bp_grids(dc_model, fold["val_df"]),
            "bivariate_poisson": _dc_bp_grids(bp_model, fold["val_df"]),
            "ml_scoreline": ml_scoreline.predict_grids_batch(ml_home, ml_away, fold["X_val"]),
        }

        cold_start_mask = _is_cold_start_involved(fold["val_df"]).to_numpy()
        segments = {
            "overall": np.ones(len(fold["val_df"]), dtype=bool),
            "cold_start_involved": cold_start_mask,
            "established_only": ~cold_start_mask,
        }

        for model_name, grids in model_grids.items():
            for segment_name, mask in segments.items():
                segment_val_df = fold["val_df"][mask]
                segment_grids = [g for g, keep in zip(grids, mask) if keep]
                if segment_val_df.empty:
                    continue
                metrics = evaluate_grids_multi_market(segment_grids, segment_val_df)
                rows.append(
                    {
                        "model": model_name,
                        "segment": segment_name,
                        "val_season": fold["val_season"],
                        "n_fixtures": len(segment_val_df),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_fold.columns if c not in ("model", "segment", "val_season", "n_fixtures")]
    return per_fold.groupby(["segment", "model"], as_index=False)[metric_cols].mean()


if __name__ == "__main__":
    result = evaluate_models_by_segment()
    result.to_csv("reports/model_selection_by_segment.csv", index=False)
    print(result.to_string(index=False))
    print("\n--- summary ---")
    print(summarize(result).to_string(index=False))
