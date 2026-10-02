"""manifest.py — train/save/load orchestration.

Mirrors FPL_Optimizer/model.py 1:1 in structure: chronological_split (hold
out the most recent season for validation, never a random split — avoids
temporal leakage), train everything, write one manifest.json aggregating
metadata/metrics across every model, and symmetric load_manifest()/
load_models() helpers.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import penaltyblog as pb
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..config import MODELS_DIR
from ..data import football_data
from ..features.build import FixtureFeatureContext, build_training_frame
from . import covariate_poisson, market_models, ml_scoreline, scoreline

MANIFEST_PATH = MODELS_DIR / "manifest.json"
# Written by `evaluate/scoreline_selection.py`, read here. Kept separate from
# `manifest.json` because computing it costs a full four-candidate walk-forward
# (minutes) and the manifest is rewritten on every retrain; see
# SELECTION_WALK_FORWARD_PATH's own entry in `manifest["scoreline"]["selection"]`
# for why the published number needs it at all.
SELECTION_WALK_FORWARD_PATH = MODELS_DIR / "selection_walk_forward.json"
MANIFEST_HISTORY_PATH = MODELS_DIR / "manifest_history.jsonl"
CORNERS_MODEL_PATH = MODELS_DIR / "corners_xgb.json"
CARDS_MODEL_PATH = MODELS_DIR / "cards_xgb.json"
HOME_SHOTS_MODEL_PATH = MODELS_DIR / "home_shots_xgb.json"
AWAY_SHOTS_MODEL_PATH = MODELS_DIR / "away_shots_xgb.json"
HOME_SHOTS_ON_TARGET_MODEL_PATH = MODELS_DIR / "home_shots_on_target_xgb.json"
AWAY_SHOTS_ON_TARGET_MODEL_PATH = MODELS_DIR / "away_shots_on_target_xgb.json"
DIXON_COLES_PATH = MODELS_DIR / "dixon_coles.pkl"
BIVARIATE_POISSON_PATH = MODELS_DIR / "bivariate_poisson.pkl"
ML_HOME_MODEL_PATH = MODELS_DIR / "ml_scoreline_home.json"
ML_AWAY_MODEL_PATH = MODELS_DIR / "ml_scoreline_away.json"
COVARIATE_POISSON_PATH = MODELS_DIR / "covariate_poisson.pkl"

# Which model actually serves a given non-1X2 market, when a real,
# multi-fold, most-recent-season-corroborated study found a different
# model beats whichever wins the overall 1X2 comparison for that specific
# market. Like MARKET_TRAINING_WINDOWS below, this is a fixed,
# research-decision constant — never re-derived from a single train_all()
# call's one holdout fold, which would risk flip-flopping on single-season
# noise (this project's own walk-forward studies repeatedly found average-
# vs-most-recent-season disagreements; see EXP-2026-11/13/14).
#
# EXP-2026-16 (docs/AI_CONTINUITY.md) found the covariate-Poisson model
# beats ml_scoreline on Over/Under 2.5 goals on the walk-forward average
# AND the single most recent completed season — clearing this project's
# two-part bar on paper. The margin was thin (<0.001 log-loss and Brier on
# both checks) and NOT monotonic: it won 3 of 5 folds (2021-22, 2022-23,
# 2025-26) but lost 2023-24 and 2024-25 by a comparable-or-larger margin
# than it won by elsewhere.
#
# REVERTED 2026-09-27. The evidence that put it there has decayed. On the
# live 2026-09-14 production manifest the two models are:
#
#     covariate_poisson  over_2_5_log_loss 0.689638   brier 0.248153
#     ml_scoreline       over_2_5_log_loss 0.689441   brier 0.248157
#
# so covariate_poisson is now marginally WORSE on log loss — the reverse of
# the EXP-2026-16 ordering — and better only by 0.000004 on Brier, which is
# noise at n=380. This override already failed the spirit of the two-gate
# rule even when it was promoted ("clears the letter of the bar, not a
# strong version of it"); a second retrain later failing the direction of
# the original evidence is exactly the condition the comment above said to
# revisit on. The chosen scoreline model now serves every market.
#
# The per-market override mechanism itself stays in place and stays tested —
# it is the durable value of EXP-2026-16, and this dict is now the single
# place a future override is added. Re-adding one requires a walk-forward
# mean win AND a most-recent-season win with a margin clear of the noise
# floor demonstrated here, not a decimal-place difference.
MARKET_MODEL_OVERRIDES: dict[str, str] = {}

RESULT_CODE = {"H": 0, "D": 1, "A": 2}

# Per-market historical training-window length in completed seasons.
# Corners specifically at 12 (vs. the 8-season default for scoreline and
# cards) per EXP-2026-11 (docs/AI_CONTINUITY.md): a 12-season window
# corroborated a real MAE improvement for corners on both the 5-fold
# walk-forward average AND the single most recent completed season
# (2.762->2.742 average, 2.699->2.656 on 2025-26 alone) — the same
# window measurably helped neither scoreline (regressed on 2025-26
# despite a better average) nor cards (mixed/inconclusive) in that same
# study, so only corners moved. Change a value here only after a new,
# equally corroborated result is logged the same way.
MARKET_TRAINING_WINDOWS = {"scoreline": 8, "corners": 12, "cards": 8}


def manifest_fingerprint() -> str | None:
    """Return the exact manifest content hash for live prediction lineage."""
    if not MANIFEST_PATH.exists():
        return None
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


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


def _selection_block(candidates: dict, chosen_market_metrics: dict, manifest_seasons: list[str] | None = None) -> dict:
    """Everything a reader needs to not over-trust the reported holdout RPS.

    The reported holdout figure is a best-of-four *on the fold it is reported on*:
    `chosen = min(candidates, key=candidates.get)` runs against this same
    `val_df`. That is a selection set, not a held-out one. The size of the
    resulting flattering is measured rather than asserted, in
    `evaluate/scoreline_selection.py` and the EXP-2026-26 ledger entry.

    The walk-forward diagnostics live in a cache file rather than being recomputed
    here, because they cost a full four-candidate walk-forward and this function
    runs on every retrain. So the cache must be checked for **provenance**, not
    merely for readability: a file that parses is not a file that is current. It
    records the seasons it was computed over, and a mismatch is reported as
    `stale` with no `headline_rps` -- because a stale headline presented as
    current is worse than an absent one, which is the whole reason this block
    exists.

    Non-finite values are coerced to `None` on the way through. JSON has no NaN,
    `json.dumps` will write the token anyway, and starlette's `JSONResponse`
    renders with `allow_nan=False` -- so one NaN here is a 500 on
    `GET /api/manifest`.
    """
    walk_forward = None
    status = "not_computed"
    if SELECTION_WALK_FORWARD_PATH.exists():
        try:
            cached = json.loads(SELECTION_WALK_FORWARD_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            cached = None
        if isinstance(cached, dict) and cached.get("walk_forward_mean"):
            cached_seasons = cached.get("seasons")
            # A subset test, not equality: the walk-forward is run over completed
            # seasons, while the manifest's window also folds in the in-progress
            # one, so the two are never equal. What must hold is that the
            # diagnostics were computed over seasons this manifest also knows --
            # a cache generated against a different data window is the case worth
            # catching, because its headline does not describe this model.
            #
            # This does NOT catch a cache that is merely old. `computed_at` is
            # published alongside so a reader can see it; refusing to compare
            # wall-clock against a data window would need a clock this manifest
            # does not otherwise keep.
            fresh = bool(
                cached_seasons
                and manifest_seasons
                and set(cached_seasons) <= set(manifest_seasons)
            )
            status = "cached" if fresh else "stale"
            if fresh:
                walk_forward = cached

    selection = {
        # Explicit, so no reader can mistake the holdout figure for an
        # out-of-sample estimate.
        "holdout_selected_on_this_fold": True,
        "candidates_on_this_holdout": candidates,
        "rps_ci": [
            chosen_market_metrics.get("rps_ci_low"),
            chosen_market_metrics.get("rps_ci_high"),
        ],
        "recommended_headline": "walk_forward_mean_rps",
        "walk_forward": _json_safe(walk_forward),
        "walk_forward_status": status,
        "walk_forward_computed_at": (walk_forward or {}).get("computed_at") if walk_forward else None,
    }
    if walk_forward is not None:
        means = walk_forward.get("walk_forward_mean") or {}
        best = walk_forward.get("best_by_walk_forward_mean")
        value = means.get(best) if best in means else None
        # `recommended_headline` is set unconditionally, so `headline_rps` is only
        # meaningful when there is a number to put beside it.
        selection["headline_rps"] = _json_safe(value)
    return selection


def _json_safe(value):
    """Recursively replace non-finite floats with None.

    `json.dumps` writes `NaN` by default and starlette refuses to read it back, so
    a single non-finite anywhere in the manifest becomes a 500 on the API rather
    than a visible null.
    """
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
    """Three views of the same question, "what does the model actually
    rely on":
    - gain: XGBoost's own internal split-value metric. Can overrate a
      feature the model merely splits on often without it truly driving
      accuracy.
    - permutation: how much held-out R^2 drops when a feature is shuffled.
      More trustworthy than gain, but only a magnitude, no sense of "why."
    - shap: mean *signed* SHAP value per feature — each prediction's output
      decomposed into exactly how much each feature pushed it up (positive)
      or down (negative) from the average, then averaged (keeping sign)
      across the holdout. Computed via XGBoost's own exact TreeSHAP
      (`pred_contribs=True`), not the separate `shap` package — same
      algorithm, no extra dependency. Signed (not just magnitude) so the
      frontend can render it the way SHAP output is normally read: colored
      by direction, ranked by |value|."""
    gain = dict(zip(feature_cols, model.feature_importances_.astype(float)))
    perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, scoring="r2")
    permutation = dict(zip(feature_cols, perm.importances_mean.astype(float)))

    dmatrix = xgb.DMatrix(X_val, feature_names=feature_cols)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    mean_signed_shap = contribs[:, :-1].mean(axis=0)  # last column is the bias/expected-value term
    shap = dict(zip(feature_cols, mean_signed_shap.astype(float)))

    return {"gain": gain, "permutation": permutation, "shap": shap}


def _build_frame(seasons: list[str], current_partial: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, int]:
    """One (df, feature_cols, train_df, val_df, n_current_season_matches)
    build for a given historical `seasons` window — factored out so
    `train_all` can build the corners-specific window (see
    `MARKET_TRAINING_WINDOWS`) without duplicating this logic, while still
    fetching `current_partial` only once regardless of how many windows
    are built from it."""
    completed_df = football_data.load_training_data(seasons=seasons)
    n_current_season_matches = 0
    if current_partial is not None and not current_partial.empty:
        matches_df = pd.concat([completed_df, current_partial], ignore_index=True).sort_values("date").reset_index(drop=True)
        n_current_season_matches = len(current_partial)
    else:
        matches_df = completed_df

    df, feature_cols = build_training_frame(matches_df=matches_df)
    train_df, val_df = chronological_split(df)
    return df, feature_cols, train_df, val_df, n_current_season_matches


def default_scoreline_seasons() -> list[str]:
    """The completed seasons the scoreline model is fitted on.

    Exposed so `api.routes.retrain(include_selection=True)` can run the selection
    diagnostics over exactly the window `train_all` will use. Defined here because
    the window is a decision about the scoreline model; a caller that re-derived it
    would silently drift the two apart, and the manifest's staleness check compares
    the diagnostics' seasons against its own.
    """
    return football_data.default_completed_seasons(n=MARKET_TRAINING_WINDOWS["scoreline"])


def train_all(seasons: list[str] | None = None, include_current_season: bool = True) -> Dict:
    """`include_current_season=True` (default) is what makes this an
    *updating* model rather than a fixed one refit on the same completed
    seasons all year: it folds the in-progress season's played-so-far
    matches into training (never into validation — `chronological_split`
    keeps the holdout pinned to the last fully completed season regardless).
    Every match added this way goes through the exact same shift(1)
    rolling-form/Elo/xG pipeline as historical data, using prior seasons as
    each team's starting context, so there's no cold-start cliff at the
    season boundary and no leakage. Call this again (e.g. via the "Retrain
    models" button, weekly or after a gameweek) to pull in whatever's been
    played since the last retrain — that's the mechanism, there's no
    background scheduler.

    `seasons`, if given explicitly, overrides `MARKET_TRAINING_WINDOWS` for
    every market uniformly (used by tests / one-off checks that want a
    smaller, faster, shared window) — production retrains leave it `None`
    so each market gets its own window length."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    default_seasons = seasons or default_scoreline_seasons()
    corners_seasons = seasons or football_data.default_completed_seasons(n=MARKET_TRAINING_WINDOWS["corners"])

    current_partial = football_data.fetch_current_season_partial() if include_current_season else None
    df, feature_cols, train_df, val_df, n_current_season_matches = _build_frame(default_seasons, current_partial)
    # `.fillna(0)` here is not a neutral default: it is the *encoding*, and it
    # is what teaches the boosters that a missing feature is 0.0. On this
    # project's data 46.88% of fitted rows carry a literal 0.0 in
    # `*_squad_continuity` (a promoted team has no prior-season squad data) and
    # every learned split on that column sits inside [0.585, 0.968] with none
    # below the observed minimum of 0.539 -- so the model has learned "0.0 here
    # means promoted", and serving a league mean instead reads as "established
    # squad" and measurably regresses RPS. The serving half of the same
    # contract is `ml_scoreline.MISSING_VALUE_ENCODING`; every value in it is
    # 0.0 precisely because of this line, and if this fill ever changes, that
    # dict has to change with it or train and serve will disagree silently.
    X_train, X_val = train_df[feature_cols].fillna(0), val_df[feature_cols].fillna(0)

    # fouls (`*_fouls_for`/`*_fouls_against`) are in `feature_cols` for
    # corners/cards, where they're a real signal (fouls -> cards). Measured
    # directly: including them in the scoreline (goals) regressors' shared
    # feature set made ml_scoreline's held-out RPS/Brier measurably worse
    # (0.2073->0.2088 RPS, 0.6159->0.6193 Brier) — 18 extra mostly-irrelevant
    # columns add variance on ~2280 training rows without adding goals
    # signal. So ml_scoreline trains on a fouls-excluded subset while
    # corners/cards keep the full set.
    ml_feature_cols = [c for c in feature_cols if "fouls" not in c]
    X_train_ml, X_val_ml = train_df[ml_feature_cols].fillna(0), val_df[ml_feature_cols].fillna(0)

    print(
        f"Training on {len(train_df)} matches ({n_current_season_matches} from the in-progress season), "
        f"validating on {len(val_df)} (held-out season)."
    )

    dc_model = scoreline.fit_dixon_coles(train_df)
    bp_model = scoreline.fit_bivariate_poisson(train_df)
    dc_metrics = _evaluate_scoreline_model(dc_model, val_df)
    bp_metrics = _evaluate_scoreline_model(bp_model, val_df)

    # NOTE: `dates=train_df["date"]` would add the same recency weighting
    # DC/BP use, but a direct experiment on this 8-season window showed it
    # makes ml_scoreline measurably worse (RPS 0.2088->0.2115, Brier
    # 0.6193->0.6247) — unlike DC/BP, ml_scoreline's rolling-form features
    # already encode recency directly, so the extra decay only shrinks
    # effective sample size. Left unweighted here; `dates` stays available
    # on train_goal_regressors for the historic-window experiment, where a
    # much wider (unweighted) window is the actual risk it guards against.
    ml_home_model, ml_away_model = ml_scoreline.train_goal_regressors(
        X_train_ml, train_df["goals_home"], train_df["goals_away"]
    )
    ml_metrics = ml_scoreline.evaluate_on_holdout(ml_home_model, ml_away_model, X_val_ml, val_df)
    ml_teams = sorted(set(train_df["team_home"]) | set(train_df["team_away"]))
    ml_importance_home = _feature_importance(ml_home_model, X_val_ml, val_df["goals_home"], ml_feature_cols)
    ml_importance_away = _feature_importance(ml_away_model, X_val_ml, val_df["goals_away"], ml_feature_cols)

    # Covariate-Poisson (EXP-2026-15): a 4th scoreline candidate, never
    # chosen for 1X2 (ml_scoreline wins that outright), but the source of
    # MARKET_MODEL_OVERRIDES' Over/Under 2.5 override. `market_metrics`
    # below is per-model per-market on *this* retrain's one holdout fold —
    # informational/auditable in manifest.json, never what decides
    # MARKET_MODEL_OVERRIDES itself (see that constant's own docstring for
    # why: a single fold isn't a safe basis for that decision).
    covariate_poisson_model = covariate_poisson.fit(train_df)
    dc_grids = scoreline.predict_grids_for_fixed_param_model(dc_model, val_df)
    bp_grids = scoreline.predict_grids_for_fixed_param_model(bp_model, val_df)
    ml_grids_for_markets = ml_scoreline.predict_grids_batch(ml_home_model, ml_away_model, X_val_ml)
    cp_grids = covariate_poisson.predict_grids_batch(covariate_poisson_model, val_df)
    market_metrics = {
        "dixon_coles": scoreline.evaluate_grids_multi_market(dc_grids, val_df),
        "bivariate_poisson": scoreline.evaluate_grids_multi_market(bp_grids, val_df),
        "ml_scoreline": scoreline.evaluate_grids_multi_market(ml_grids_for_markets, val_df),
        "covariate_poisson": scoreline.evaluate_grids_multi_market(cp_grids, val_df),
    }
    cp_metrics = market_metrics["covariate_poisson"]

    candidates = {
        "dixon_coles": dc_metrics["rps"],
        "bivariate_poisson": bp_metrics["rps"],
        "ml_scoreline": ml_metrics["rps"],
        "covariate_poisson": cp_metrics["rps"],
    }
    chosen = min(candidates, key=candidates.get)
    market_overrides = {
        market: model_name for market, model_name in MARKET_MODEL_OVERRIDES.items() if model_name != chosen
    }
    print(
        f"  > Dixon-Coles RPS={dc_metrics['rps']:.4f}  Bivariate-Poisson RPS={bp_metrics['rps']:.4f}  "
        f"ML-scoreline RPS={ml_metrics['rps']:.4f}  Covariate-Poisson RPS={cp_metrics['rps']:.4f}  (chosen: {chosen})"
    )
    if market_overrides:
        print(f"  > Market overrides: {market_overrides}")

    scoreline.save(dc_model, DIXON_COLES_PATH)
    scoreline.save(bp_model, BIVARIATE_POISSON_PATH)
    market_models.save_regressor(ml_home_model, ML_HOME_MODEL_PATH)
    market_models.save_regressor(ml_away_model, ML_AWAY_MODEL_PATH)
    covariate_poisson.save(covariate_poisson_model, COVARIATE_POISSON_PATH)

    # Corners trains on its own window (see MARKET_TRAINING_WINDOWS) only
    # when it actually differs from the default — reuses the already-built
    # frame otherwise rather than rebuilding identical data.
    if corners_seasons == default_seasons:
        corners_df, corners_feature_cols, corners_train_df, corners_val_df = df, feature_cols, train_df, val_df
    else:
        corners_df, corners_feature_cols, corners_train_df, corners_val_df, _ = _build_frame(corners_seasons, current_partial)
    # Serving (odds/value_bets.py::predict_market_models_for_fixture) reindexes
    # one shared feature row onto models["feature_cols"] (the default window's
    # list) for *both* corners and cards — safe only because build_training_
    # frame's feature_cols is a fixed list of column names independent of how
    # many seasons of data were loaded, never season-window-dependent. Assert
    # it rather than silently trust it: if this ever breaks, corners' XGBoost
    # model would be trained on one column order and served with another.
    assert corners_feature_cols == feature_cols, (
        "corners' training window produced a different feature_cols list than "
        "the default window — serving assumes these are identical"
    )
    X_train_corners = corners_train_df[corners_feature_cols].fillna(0)
    X_val_corners = corners_val_df[corners_feature_cols].fillna(0)

    corners_dispersion = market_models.check_overdispersion(corners_train_df["total_corners"].to_numpy())
    cards_dispersion = market_models.check_overdispersion(train_df["total_cards"].to_numpy())

    corners_model = market_models.train_lambda_regressor(X_train_corners, corners_train_df["total_corners"])
    cards_model = market_models.train_lambda_regressor(X_train, train_df["total_cards"])
    corners_metrics = _evaluate_count_model(corners_model, X_val_corners, corners_val_df["total_corners"])
    cards_metrics = _evaluate_count_model(cards_model, X_val, val_df["total_cards"])
    print(f"  > Corners MAE={corners_metrics['mae']:.2f}  Cards MAE={cards_metrics['mae']:.2f}")

    corners_importance = _feature_importance(corners_model, X_val_corners, corners_val_df["total_corners"], corners_feature_cols)
    cards_importance = _feature_importance(cards_model, X_val, val_df["total_cards"], feature_cols)

    market_models.save_regressor(corners_model, CORNERS_MODEL_PATH)
    market_models.save_regressor(cards_model, CARDS_MODEL_PATH)

    # Team shots: same default window/feature set as cards (no dedicated
    # MARKET_TRAINING_WINDOWS entry -- unlike corners, no research finding
    # yet justifies a different window length here). Two regressors, not
    # one match-total like corners/cards, since shot volume is naturally
    # asymmetric between a match's two sides.
    home_shots_dispersion = market_models.check_overdispersion(train_df["home_shots"].to_numpy())
    away_shots_dispersion = market_models.check_overdispersion(train_df["away_shots"].to_numpy())
    home_shots_model = market_models.train_lambda_regressor(X_train, train_df["home_shots"])
    away_shots_model = market_models.train_lambda_regressor(X_train, train_df["away_shots"])
    home_shots_metrics = _evaluate_count_model(home_shots_model, X_val, val_df["home_shots"])
    away_shots_metrics = _evaluate_count_model(away_shots_model, X_val, val_df["away_shots"])
    print(f"  > Home shots MAE={home_shots_metrics['mae']:.2f}  Away shots MAE={away_shots_metrics['mae']:.2f}")

    home_shots_importance = _feature_importance(home_shots_model, X_val, val_df["home_shots"], feature_cols)
    away_shots_importance = _feature_importance(away_shots_model, X_val, val_df["away_shots"], feature_cols)

    market_models.save_regressor(home_shots_model, HOME_SHOTS_MODEL_PATH)
    market_models.save_regressor(away_shots_model, AWAY_SHOTS_MODEL_PATH)

    home_sot_dispersion = market_models.check_overdispersion(train_df["home_shots_on_target"].to_numpy())
    away_sot_dispersion = market_models.check_overdispersion(train_df["away_shots_on_target"].to_numpy())
    home_sot_model = market_models.train_lambda_regressor(X_train, train_df["home_shots_on_target"])
    away_sot_model = market_models.train_lambda_regressor(X_train, train_df["away_shots_on_target"])
    home_sot_metrics = _evaluate_count_model(home_sot_model, X_val, val_df["home_shots_on_target"])
    away_sot_metrics = _evaluate_count_model(away_sot_model, X_val, val_df["away_shots_on_target"])
    print(f"  > Home SOT MAE={home_sot_metrics['mae']:.2f}  Away SOT MAE={away_sot_metrics['mae']:.2f}")

    home_sot_importance = _feature_importance(home_sot_model, X_val, val_df["home_shots_on_target"], feature_cols)
    away_sot_importance = _feature_importance(away_sot_model, X_val, val_df["away_shots_on_target"], feature_cols)

    market_models.save_regressor(home_sot_model, HOME_SHOTS_ON_TARGET_MODEL_PATH)
    market_models.save_regressor(away_sot_model, AWAY_SHOTS_ON_TARGET_MODEL_PATH)

    selection = _selection_block(
        candidates, market_metrics.get(chosen, {}),
        manifest_seasons=sorted(df["season"].unique().tolist()),
    )

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(df["season"].unique().tolist()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_current_season_matches": n_current_season_matches,
        "features": feature_cols,
        "market_training_windows": MARKET_TRAINING_WINDOWS,
        "scoreline": {
            "chosen_model": chosen,
            "market_overrides": market_overrides,
            "market_metrics": market_metrics,
            "selection": selection,
            "dixon_coles": {"path": DIXON_COLES_PATH.name, "metrics": dc_metrics},
            "bivariate_poisson": {"path": BIVARIATE_POISSON_PATH.name, "metrics": bp_metrics},
            "ml_scoreline": {
                "home_path": ML_HOME_MODEL_PATH.name,
                "away_path": ML_AWAY_MODEL_PATH.name,
                "feature_cols": ml_feature_cols,
                "metrics": ml_metrics,
                "teams": ml_teams,
                "importance_home": ml_importance_home,
                "importance_away": ml_importance_away,
            },
            "covariate_poisson": {"path": COVARIATE_POISSON_PATH.name, "metrics": cp_metrics},
        },
        "corners": {
            "path": CORNERS_MODEL_PATH.name,
            "metrics": corners_metrics,
            "dispersion": corners_dispersion,
            "importance": corners_importance,
            "seasons": sorted(corners_df["season"].unique().tolist()),
            "n_train": int(len(corners_train_df)),
        },
        "cards": {
            "path": CARDS_MODEL_PATH.name,
            "metrics": cards_metrics,
            "dispersion": cards_dispersion,
            "importance": cards_importance,
        },
        "shots": {
            "home_path": HOME_SHOTS_MODEL_PATH.name,
            "away_path": AWAY_SHOTS_MODEL_PATH.name,
            "home_metrics": home_shots_metrics,
            "away_metrics": away_shots_metrics,
            "home_dispersion": home_shots_dispersion,
            "away_dispersion": away_shots_dispersion,
            "home_importance": home_shots_importance,
            "away_importance": away_shots_importance,
        },
        "shots_on_target": {
            "home_path": HOME_SHOTS_ON_TARGET_MODEL_PATH.name,
            "away_path": AWAY_SHOTS_ON_TARGET_MODEL_PATH.name,
            "home_metrics": home_sot_metrics,
            "away_metrics": away_sot_metrics,
            "home_dispersion": home_sot_dispersion,
            "away_dispersion": away_sot_dispersion,
            "home_importance": home_sot_importance,
            "away_importance": away_sot_importance,
        },
    }
    # `allow_nan=False`: a non-finite anywhere in the manifest would otherwise be
    # written as the token `NaN`, which starlette's JSONResponse refuses to render
    # (`allow_nan=False` on its side too) -- a 500 on GET /api/manifest rather
    # than a visible null. Fail here instead, where the traceback points at the cause.
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    _append_history(manifest)
    return manifest


def _append_history(manifest: Dict) -> None:
    """One line per retrain — how the *chosen* model's holdout metrics and
    training-set size evolve as the season progresses and more of it gets
    folded into training. Append-only (unlike manifest.json, which each
    retrain overwrites) so the Calibration page can chart a real trend
    instead of only ever showing the single latest snapshot."""
    chosen = manifest["scoreline"]["chosen_model"]
    chosen_metrics = manifest["scoreline"][chosen]["metrics"]
    # `covariate_poisson`'s own `metrics` is `evaluate_grids_multi_market`'s
    # dict (`brier_1x2`, not `brier`) — a different shape than dixon_coles/
    # bivariate_poisson/ml_scoreline's `_evaluate_scoreline_model` output,
    # since it's normally only ever a MARKET_MODEL_OVERRIDES candidate, not
    # `chosen_model` itself. Both keys are the same underlying 1X2 Brier
    # score (`multiclass_brier_score` on the same probs/outcomes either
    # way), so this is a safe fallback for the rare retrain where it wins
    # the argmin outright rather than a guess.
    entry = {
        "trained_at": manifest["trained_at"],
        "n_train": manifest["n_train"],
        "n_current_season_matches": manifest["n_current_season_matches"],
        "chosen_model": chosen,
        "rps": chosen_metrics["rps"],
        "brier": chosen_metrics.get("brier", chosen_metrics.get("brier_1x2")),
    }
    with MANIFEST_HISTORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_manifest_history() -> list[Dict]:
    if not MANIFEST_HISTORY_PATH.exists():
        return []
    lines = MANIFEST_HISTORY_PATH.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score_change_history(history: list[Dict]) -> list[Dict]:
    """Keep the first validation point and meaningful held-out score moves.

    Retrains can refresh artefacts without changing the validation score.
    Showing every one of those as a chart point visually invents movement, so
    the Calibration surface consumes this compact, chronological series.
    """
    changes: list[Dict] = []
    previous: tuple[float | None, float | None] | None = None
    for entry in history:
        score = (entry.get("rps"), entry.get("brier"))
        if previous is None or score != previous:
            changes.append(entry)
            previous = score
    return changes


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "No trained models found. Run `python -m pl_predictor.models.manifest` "
            "or notebooks 01-04 first."
        )
    return json.loads(MANIFEST_PATH.read_text())


_OVERRIDE_MODEL_LOADERS = {"covariate_poisson": lambda: covariate_poisson.load(COVARIATE_POISSON_PATH)}


def load_models(matches_df: pd.DataFrame | None = None) -> Dict:
    """`matches_df` is required whenever a `FixtureFeatureContext` is
    needed for live serving — either because `chosen_model` is
    `ml_scoreline`, or because `scoreline.market_overrides` (see
    `MARKET_MODEL_OVERRIDES`) names a feature-driven override model
    (currently `covariate_poisson`); omit it only when neither applies.
    `dixon_coles_for_rankings` is always loaded regardless of
    `chosen_model`, since Power Rankings reads its fitted attack/defence
    parameters directly and an ML-based scoreline model doesn't have an
    equivalent to show there."""
    manifest = load_manifest()
    chosen = manifest["scoreline"]["chosen_model"]
    market_override_names = manifest["scoreline"].get("market_overrides", {})

    dixon_coles_for_rankings = pb.models.DixonColesGoalModel.load(str(DIXON_COLES_PATH))

    # `chosen == "covariate_poisson"` is currently only a theoretical case
    # (EXP-2026-16: it's never won the overall 1X2 comparison on real
    # production data, only specific markets it's an explicit override
    # for) but `train_all`'s `min(candidates, key=candidates.get)` doesn't
    # special-case it out, so a future retrain could reach it — handled
    # here for real rather than left to silently fall into the
    # `ml_scoreline` `else` branch below with the wrong model class.
    needs_context = (
        chosen in ("ml_scoreline", "covariate_poisson") or "covariate_poisson" in market_override_names.values()
    )
    context = None
    if needs_context:
        if matches_df is None:
            raise ValueError(
                "load_models(matches_df=...) is required when ml_scoreline/covariate_poisson is "
                "chosen_model or a market override needs a live feature context."
            )
        context = FixtureFeatureContext(matches_df)

    if chosen == "dixon_coles":
        scoreline_model = dixon_coles_for_rankings
    elif chosen == "bivariate_poisson":
        scoreline_model = pb.models.BivariatePoissonGoalModel.load(str(BIVARIATE_POISSON_PATH))
    elif chosen == "covariate_poisson":
        scoreline_model = covariate_poisson.load(COVARIATE_POISSON_PATH)
        scoreline_model.context = context
    else:
        scoreline_model = ml_scoreline.MLScorelineModel(
            home_model=market_models.load_regressor(ML_HOME_MODEL_PATH),
            away_model=market_models.load_regressor(ML_AWAY_MODEL_PATH),
            feature_cols=manifest["scoreline"]["ml_scoreline"]["feature_cols"],
            teams=manifest["scoreline"]["ml_scoreline"]["teams"],
            context=context,
        )

    # Resolve MARKET_MODEL_OVERRIDES' model *names* (what manifest.json
    # stores — plain strings, so the manifest stays a readable, diffable
    # record) into actual loaded model objects (what scoreline.predict_
    # fixture/predict_fixtures_batch's market_overrides parameter needs).
    scoreline_market_overrides = {}
    for market, model_name in market_override_names.items():
        override_model = _OVERRIDE_MODEL_LOADERS[model_name]()
        if hasattr(override_model, "context"):
            override_model.context = context
        scoreline_market_overrides[market] = override_model

    return {
        "scoreline": scoreline_model,
        "scoreline_market_overrides": scoreline_market_overrides,
        "dixon_coles_for_rankings": dixon_coles_for_rankings,
        "corners": market_models.load_regressor(CORNERS_MODEL_PATH),
        "cards": market_models.load_regressor(CARDS_MODEL_PATH),
        "home_shots": market_models.load_regressor(HOME_SHOTS_MODEL_PATH),
        "away_shots": market_models.load_regressor(AWAY_SHOTS_MODEL_PATH),
        "home_shots_on_target": market_models.load_regressor(HOME_SHOTS_ON_TARGET_MODEL_PATH),
        "away_shots_on_target": market_models.load_regressor(AWAY_SHOTS_ON_TARGET_MODEL_PATH),
        "feature_cols": manifest["features"],
        "corners_dispersion": manifest["corners"]["dispersion"],
        "cards_dispersion": manifest["cards"]["dispersion"],
        # The already-built FixtureFeatureContext, when one was needed (see
        # `needs_context` above) — callers building live feature rows
        # (features.build.build_features_for_fixtures) should reuse this
        # rather than constructing their own; see that function's docstring
        # for why (confirmed live: rebuilding it per-request both slowed
        # every request and leaked enough memory to OOM a small deployment).
        "context": context,
    }


if __name__ == "__main__":
    train_all()
