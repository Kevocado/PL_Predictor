"""scoreline_dominance_arms.py — Part 2 of the match-dominance research
plan: three leakage-safe scoreline arms evaluated on identical
chronological walk-forward folds, plus a per-market/segment breakdown on
the same arms, plus the same three arms repeated independently for
corners/cards. Nothing here changes a production model — see
docs/AI_CONTINUITY.md's non-negotiable research protocol and this entry's
own promotion-gate discussion once results exist.

Arms (same validation seasons for all three — 2021-22 through 2025-26 —
achieved by pairing each season window with the `min_train_seasons` that
lands on the same starting index):
  A. 8-season production-equivalent baseline (`min_train_seasons=3`).
  B. 12-season historical window, same features otherwise
     (`min_train_seasons=7`).
  C. 12-season window + the new `features.match_dominance` features,
     merged via `evaluate.walk_forward.prepare_folds`'s existing
     `extra_feature_frame` mechanism (same pattern EXP-2026-03's
     walk-forward follow-up already established).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.metrics import brier_score_loss, log_loss

from ..data import football_data, understat_shots
from ..features import match_dominance
from ..models import ml_scoreline, market_models
from . import walk_forward

ARM_SPECS = {
    "A_8_season_baseline": {"n_seasons": 8, "min_train_seasons": 3, "with_dominance": False},
    "B_12_season_window": {"n_seasons": 12, "min_train_seasons": 7, "with_dominance": False},
    "C_12_season_plus_dominance": {"n_seasons": 12, "min_train_seasons": 7, "with_dominance": True},
}

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}


def _multiclass_top_label_ece(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Standard top-label calibration error: bins predictions by the
    model's own confidence in its argmax class, compares bin-average
    accuracy to bin-average confidence. Generalizes
    `goal_contribution_research.py::_ece`'s binary version to the 1X2
    3-class case."""
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == outcomes).astype(float)
    bucket = np.clip((confidence * bins).astype(int), 0, bins - 1)
    total = len(outcomes)
    if total == 0:
        return float("nan")
    return float(
        sum(
            abs(correct[bucket == b].mean() - confidence[bucket == b].mean()) * (bucket == b).sum() / total
            for b in range(bins)
            if (bucket == b).any()
        )
    )


def _bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI over per-fixture values (e.g. per-row
    RPS contributions) within one fold."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _metrics_from_grids(grids: list, val_df: pd.DataFrame) -> dict:
    """Every scoreline market's metrics from an already-predicted list of
    grids (no refitting per market) — RPS/Brier/log-loss/ECE for 1X2,
    exact-scoreline log-loss, and BTTS / O-U-2.5 log-loss+Brier as separate
    markets. Model-agnostic: `grids` can come from any model whose
    `.predict`/batch path yields the same `FootballProbabilityGrid`-shaped
    objects — Dixon-Coles, Bivariate-Poisson, and `MLScorelineModel` all
    do (see `models/scoreline.py`'s uniform prediction interface). Used by
    both the data-window arms here and `model_selection_by_segment.py`'s
    DC/BP/ml_scoreline comparison, so this is the one place scoreline
    market metrics are computed."""
    outcomes = val_df["ftr"].map(_RESULT_CODE).to_numpy()
    probs_1x2 = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
    per_row_rps = np.array([pb.metrics.rps_average(p.reshape(1, -1), int(o)) for p, o in zip(probs_1x2, outcomes)])

    goals_total = (val_df["goals_home"] + val_df["goals_away"]).to_numpy()
    over_actual = (goals_total > 2.5).astype(int)
    over_probs = np.clip(np.array([g.total_goals("over", 2.5) for g in grids]), 1e-6, 1 - 1e-6)

    btts_actual = ((val_df["goals_home"] > 0) & (val_df["goals_away"] > 0)).astype(int).to_numpy()
    btts_probs = np.clip(np.array([g.btts_yes for g in grids]), 1e-6, 1 - 1e-6)

    exact_score_probs = np.array(
        [
            max(g.exact_score(int(h), int(a)), 1e-6)
            for g, h, a in zip(grids, val_df["goals_home"], val_df["goals_away"])
        ]
    )

    rps_ci_low, rps_ci_high = _bootstrap_ci(per_row_rps)

    return {
        "rps": float(per_row_rps.mean()),
        "rps_ci_low": rps_ci_low,
        "rps_ci_high": rps_ci_high,
        "brier_1x2": float(pb.metrics.multiclass_brier_score(probs_1x2, outcomes)),
        "log_loss_1x2": float(log_loss(outcomes, probs_1x2, labels=[0, 1, 2])),
        "ece_1x2": _multiclass_top_label_ece(probs_1x2, outcomes),
        "exact_scoreline_log_loss": float(-np.mean(np.log(exact_score_probs))),
        "over_2_5_log_loss": float(log_loss(over_actual, over_probs)),
        "over_2_5_brier": float(brier_score_loss(over_actual, over_probs)),
        "btts_log_loss": float(log_loss(btts_actual, btts_probs)),
        "btts_brier": float(brier_score_loss(btts_actual, btts_probs)),
    }


def _fold_metrics(home_model, away_model, X_val: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """`ml_scoreline`-specific convenience: predicts via the already-fitted
    goal regressors, then delegates to `_metrics_from_grids`."""
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, X_val)
    return _metrics_from_grids(grids, val_df)


def _dominance_extra_frame(seasons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Builds the match-dominance rolling features aligned to the exact
    same `matches_df` `walk_forward.prepare_folds` will independently
    rebuild from the same `seasons` list (deterministic — same cached
    season files, same sort), then reshapes it into the
    (kickoff_date, team_home, team_away) + feature-columns exact-merge
    shape `prepare_folds`'s `extra_feature_frame` already expects — the
    same pattern `evaluate_scoreline_player_aggregates_walk_forward`
    established for EXP-2026-03's follow-up."""
    matches_df = football_data.load_training_data(seasons=seasons)
    understat_seasons = sorted({season.split("-")[0] for season in seasons})
    dominance_data = understat_shots.load_match_dominance_data(seasons=understat_seasons)
    dominance_feats, dominance_cols = match_dominance.attach_dominance_features(matches_df, dominance_data)

    extra = pd.concat(
        [matches_df[["date", "team_home", "team_away"]].reset_index(drop=True), dominance_feats.reset_index(drop=True)],
        axis=1,
    )
    extra["kickoff_date"] = pd.to_datetime(extra["date"]).dt.normalize()
    return extra.drop(columns=["date"]), dominance_cols


def evaluate_scoreline_arms() -> pd.DataFrame:
    """One row per (arm, fold): every scoreline market's metrics. This is
    the primary Part 2 deliverable — run once, then slice for the
    per-market/segment breakdown rather than re-fitting per slice."""
    rows = []
    for arm_name, spec in ARM_SPECS.items():
        seasons = football_data.default_completed_seasons(n=spec["n_seasons"])
        extra_frame, extra_cols = (None, None)
        if spec["with_dominance"]:
            extra_frame, extra_cols = _dominance_extra_frame(seasons)

        folds = walk_forward.prepare_folds(
            seasons=seasons,
            min_train_seasons=spec["min_train_seasons"],
            extra_feature_frame=extra_frame,
            extra_feature_cols=extra_cols,
        )
        for fold in folds:
            home_model, away_model = ml_scoreline.train_goal_regressors(
                fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
            )
            metrics = _fold_metrics(home_model, away_model, fold["X_val"], fold["val_df"])
            rows.append({"arm": arm_name, "val_season": fold["val_season"], "n_train": len(fold["train_df"]), **metrics})

    return pd.DataFrame(rows)


def evaluate_count_market_arms(target: str) -> pd.DataFrame:
    """Same three arms, repeated independently for a count market
    (`"total_corners"` or `"total_cards"`) — MAE/RMSE plus probabilistic
    Brier/log-loss at the market's displayed line, per the project's
    'probability metrics before accuracy' protocol. Does NOT assume a
    scoreline-arm result transfers, per the plan's explicit instruction."""
    line = 9.5 if target == "total_corners" else 3.5
    rows = []
    for arm_name, spec in ARM_SPECS.items():
        seasons = football_data.default_completed_seasons(n=spec["n_seasons"])
        extra_frame, extra_cols = (None, None)
        if spec["with_dominance"]:
            extra_frame, extra_cols = _dominance_extra_frame(seasons)

        # Count markets train on the *full* feature_cols (fouls included),
        # unlike ml_feature_cols — prepare_folds always builds ml_feature_cols
        # (fouls-excluded) for X_train/X_val, so rebuild the full-feature
        # matrices here the same way manifest.py::train_all does for
        # corners/cards specifically.
        matches_df = football_data.load_training_data(seasons=seasons)
        from ..features.build import build_training_frame

        df, feature_cols = build_training_frame(matches_df=matches_df)
        if extra_frame is not None:
            df = df.copy()
            df["kickoff_date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.merge(extra_frame, on=["kickoff_date", "team_home", "team_away"], how="left")
            feature_cols = feature_cols + list(extra_cols)

        for i in range(spec["min_train_seasons"], len(seasons)):
            val_season = seasons[i]
            train_df = df[df["season"].isin(seasons[:i])]
            val_df = df[df["season"] == val_season]
            if train_df.empty or val_df.empty:
                continue
            X_train, X_val = train_df[feature_cols].fillna(0), val_df[feature_cols].fillna(0)

            model = market_models.train_lambda_regressor(X_train, train_df[target])
            preds = model.predict(X_val)
            dispersion = market_models.check_overdispersion(train_df[target].to_numpy())
            probs_over = np.array([market_models.price_over_under(p, line, dispersion)["over"] for p in preds])
            probs_over = np.clip(probs_over, 1e-6, 1 - 1e-6)
            actual_over = (val_df[target].to_numpy() > line).astype(int)

            rows.append(
                {
                    "arm": arm_name,
                    "val_season": val_season,
                    "target": target,
                    "mae": float(np.mean(np.abs(preds - val_df[target].to_numpy()))),
                    "rmse": float(np.sqrt(np.mean((preds - val_df[target].to_numpy()) ** 2))),
                    "over_under_log_loss": float(log_loss(actual_over, probs_over)),
                    "over_under_brier": float(brier_score_loss(actual_over, probs_over)),
                }
            )

    return pd.DataFrame(rows)


def summarize_arms(per_fold: pd.DataFrame, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean of every numeric metric column per arm, across folds — the
    headline table for a promotion decision. Always inspect the full
    per-fold table too (see the caller / AI_CONTINUITY.md write-up
    convention): an average can hide a single-fold regression."""
    metric_cols = metric_cols or [c for c in per_fold.columns if c not in ("arm", "val_season", "target", "n_train")]
    return per_fold.groupby("arm", as_index=False)[metric_cols].mean()


if __name__ == "__main__":
    scoreline = evaluate_scoreline_arms()
    scoreline.to_csv("reports/scoreline_dominance_arms.csv", index=False)
    print(scoreline.to_string(index=False))
    print("\n--- summary ---")
    print(summarize_arms(scoreline).to_string(index=False))

    for target in ("total_corners", "total_cards"):
        market = evaluate_count_market_arms(target)
        market.to_csv(f"reports/{target}_dominance_arms.csv", index=False)
        print(f"\n--- {target} ---")
        print(market.to_string(index=False))
        print(summarize_arms(market, ["mae", "rmse", "over_under_log_loss", "over_under_brier"]).to_string(index=False))
