"""Leakage-safe seasonal retraining and consensus research.

Nothing in this module writes production model artifacts.  Historical rows are
always scored with their point-in-time feature values, and every challenger is
compared against the same future fixtures as its frozen pre-season control.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from ..data import football_data
from ..features.build import build_features_for_fixtures, build_training_frame
from ..models import ml_scoreline

RESULT_CODE = {"H": 0, "D": 1, "A": 2}
FIXED_CADENCES = {"every_10_matches": 10, "every_19_matches": 19}
PER_CLUB_CADENCE = "every_19_matches_per_club"
CORE_ARM = "core_frozen"
WEIGHTED_ARM = "season_weighted"
SEASON_ONLY_ARM = "season_only"
CONSENSUS_EQUAL_ARM = "consensus_equal"
CONSENSUS_WEIGHTED_ARM = "consensus_weighted"
SEASON_WEIGHT = 3.0


def _ece(outcomes: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    errors = []
    for class_index in range(3):
        predicted = probabilities[:, class_index]
        actual = (outcomes == class_index).astype(float)
        bucket = np.clip((predicted * bins).astype(int), 0, bins - 1)
        errors.append(
            sum(
                abs(actual[bucket == bucket_id].mean() - predicted[bucket == bucket_id].mean())
                * (bucket == bucket_id).mean()
                for bucket_id in range(bins)
                if (bucket == bucket_id).any()
            )
        )
    return float(np.mean(errors))


def _metrics(probabilities: np.ndarray, outcomes: np.ndarray, scoreline_probabilities: np.ndarray) -> dict:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return {
        "rps": float(pb.metrics.rps_average(probabilities, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probabilities, outcomes)),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
        "ece": _ece(outcomes, probabilities),
        "scoreline_log_loss": float(-np.log(np.clip(scoreline_probabilities, 1e-8, 1.0)).mean()),
        "n_fixtures": int(len(outcomes)),
    }


def _rps_delta_ci(candidate: np.ndarray, core: np.ndarray, outcomes: np.ndarray, draws: int = 250) -> tuple[float, float]:
    """Paired bootstrap interval: both arms resample the same fixtures."""
    if len(outcomes) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(draws):
        indexes = rng.integers(0, len(outcomes), len(outcomes))
        deltas.append(
            pb.metrics.rps_average(candidate[indexes], outcomes[indexes])
            - pb.metrics.rps_average(core[indexes], outcomes[indexes])
        )
    return (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))


def _prediction_arrays(home_model, away_model, evaluation: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, evaluation[feature_cols].fillna(0))
    probabilities = np.array([[grid.home_win, grid.draw, grid.away_win] for grid in grids])
    exact = np.array(
        [
            grid.grid[min(int(row.goals_home), grid.grid.shape[0] - 1), min(int(row.goals_away), grid.grid.shape[1] - 1)]
            for grid, row in zip(grids, evaluation.itertuples())
        ]
    )
    return probabilities, exact


def checkpoint_plan(target: pd.DataFrame, cadence: str, horizon_matches: int = 19) -> list[dict]:
    """Return date-safe checkpoints for a target season.

    A training cutoff never splits same-day fixtures: results on a given date
    first become usable only for fixtures on a later date.
    """
    if target.empty:
        return []
    target = target.sort_values("date").reset_index(drop=True)
    if cadence == "no_retrain":
        return [{"matches_seen": 0, "evaluation_start": 0, "evaluation_end": min(len(target), horizon_matches), "checkpoint": "preseason"}]
    by_date = list(target.groupby("date", sort=True))
    checkpoints = [{"matches_seen": 0, "evaluation_start": 0, "checkpoint": "preseason"}]
    seen = 0
    next_fixed = FIXED_CADENCES.get(cadence)
    clubs: dict[str, int] = defaultdict(int)
    completed_blocks = 0
    for _, day in by_date:
        seen += len(day)
        for row in day.itertuples():
            clubs[row.team_home] += 1
            clubs[row.team_away] += 1
        add = False
        label = None
        if next_fixed is not None and seen >= next_fixed:
            add, label = True, f"{next_fixed} league matches"
            next_fixed += FIXED_CADENCES[cadence]
        elif cadence == PER_CLUB_CADENCE and clubs:
            blocks = min(clubs.values()) // 19
            if blocks > completed_blocks:
                completed_blocks, add, label = blocks, True, f"all clubs reached {blocks * 19} matches"
        if add:
            evaluation_start = int(target.index[target["date"] > day["date"].max()].min()) if (target["date"] > day["date"].max()).any() else len(target)
            if evaluation_start < len(target):
                checkpoints.append({"matches_seen": seen, "evaluation_start": evaluation_start, "checkpoint": label})
    for checkpoint in checkpoints:
        checkpoint["evaluation_end"] = min(len(target), checkpoint["evaluation_start"] + horizon_matches)
    return [item for item in checkpoints if item["evaluation_start"] < item["evaluation_end"]]


def _fit_arm(arm: str, historical: pd.DataFrame, prior_target: pd.DataFrame, feature_cols: list[str]):
    if arm == CORE_ARM:
        train = historical
        weights = None
    elif arm == WEIGHTED_ARM:
        train = pd.concat([historical, prior_target], ignore_index=True)
        weights = np.where(train["season"].eq(prior_target["season"].iloc[0]), SEASON_WEIGHT, 1.0) if not prior_target.empty else None
    elif arm == SEASON_ONLY_ARM:
        if prior_target.empty:
            return None
        train, weights = prior_target, None
    else:
        raise ValueError(f"Unknown study arm: {arm}")
    if train.empty:
        return None
    return ml_scoreline.train_goal_regressors(
        train[feature_cols].fillna(0), train["goals_home"], train["goals_away"], sample_weight=weights
    )


def _blend_weights(probabilities: list[np.ndarray], outcomes: np.ndarray) -> np.ndarray:
    stacked = np.stack(probabilities, axis=1)
    actual = np.eye(3)[outcomes]

    def objective(weights):
        blended = np.tensordot(stacked, weights, axes=(1, 0))
        return np.mean(np.sum((blended - actual) ** 2, axis=1))

    result = minimize(
        objective,
        x0=np.full(len(probabilities), 1 / len(probabilities)),
        bounds=[(0.0, 1.0)] * len(probabilities),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        method="SLSQP",
    )
    return result.x if result.success else np.full(len(probabilities), 1 / len(probabilities))


def _append_record(records: list[dict], *, season: str, cadence: str, checkpoint: dict, arm: str, probabilities: np.ndarray, exact: np.ndarray, outcomes: np.ndarray, core: np.ndarray | None = None, weights: list[float] | None = None) -> None:
    metrics = _metrics(probabilities, outcomes, exact)
    ci_low, ci_high = _rps_delta_ci(probabilities, core, outcomes) if core is not None else (None, None)
    core_rps = float(pb.metrics.rps_average(core, outcomes)) if core is not None else None
    records.append(
        {
            "season": season,
            "cadence": cadence,
            "checkpoint": checkpoint["checkpoint"],
            "current_season_matches_seen": checkpoint["matches_seen"],
            "arm": arm,
            "rps_delta_vs_core": None if core_rps is None else metrics["rps"] - core_rps,
            "rps_delta_ci_low": ci_low,
            "rps_delta_ci_high": ci_high,
            "consensus_weights": weights,
            **metrics,
        }
    )


def run_seasonal_model_study(
    seasons: list[str] | None = None,
    min_history_seasons: int = 3,
    horizon_matches: int = 19,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Evaluate frozen, seasonal, and consensus arms over chronological seasons."""
    seasons = seasons or football_data.default_completed_seasons()
    matches = football_data.load_training_data(seasons=seasons)
    frame, feature_cols = build_training_frame(matches_df=matches)
    feature_cols = [column for column in feature_cols if "fouls" not in column]
    records: list[dict] = []
    prior_consensus: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)

    for season_index in range(min_history_seasons, len(seasons)):
        season = seasons[season_index]
        if progress is not None:
            progress(f"Evaluating {season} ({season_index - min_history_seasons + 1}/{len(seasons) - min_history_seasons})")
        target = frame[frame["season"] == season].sort_values("date").reset_index(drop=True)
        historical = frame[frame["season"].isin(seasons[:season_index])]
        if historical.empty or target.empty:
            continue
        frozen_models = _fit_arm(CORE_ARM, historical, target.iloc[:0], feature_cols)
        if frozen_models is None:
            continue
        season_consensus: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        for cadence in ("no_retrain", *FIXED_CADENCES, PER_CLUB_CADENCE):
            for checkpoint in checkpoint_plan(target, cadence, horizon_matches=horizon_matches):
                evaluation = target.iloc[checkpoint["evaluation_start"] : checkpoint["evaluation_end"]]
                cutoff = evaluation["date"].min()
                prior_target = target[target["date"] < cutoff]
                outcomes = evaluation["ftr"].map(RESULT_CODE).to_numpy()
                predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                arms = (CORE_ARM,) if cadence == "no_retrain" else (CORE_ARM, WEIGHTED_ARM, SEASON_ONLY_ARM)
                for arm in arms:
                    trained = frozen_models if arm == CORE_ARM else _fit_arm(arm, historical, prior_target, feature_cols)
                    if trained is None:
                        continue
                    predictions[arm] = _prediction_arrays(*trained, evaluation, feature_cols)
                core_probs, core_exact = predictions[CORE_ARM]
                _append_record(records, season=season, cadence=cadence, checkpoint=checkpoint, arm=CORE_ARM, probabilities=core_probs, exact=core_exact, outcomes=outcomes)
                for arm in (WEIGHTED_ARM, SEASON_ONLY_ARM):
                    if arm in predictions:
                        probs, exact = predictions[arm]
                        _append_record(records, season=season, cadence=cadence, checkpoint=checkpoint, arm=arm, probabilities=probs, exact=exact, outcomes=outcomes, core=core_probs)

                if WEIGHTED_ARM not in predictions or SEASON_ONLY_ARM not in predictions:
                    continue
                component_probs = [core_probs, predictions[WEIGHTED_ARM][0], predictions[SEASON_ONLY_ARM][0]]
                component_exact = [core_exact, predictions[WEIGHTED_ARM][1], predictions[SEASON_ONLY_ARM][1]]
                equal = np.mean(component_probs, axis=0)
                equal_exact = np.mean(component_exact, axis=0)
                _append_record(records, season=season, cadence=cadence, checkpoint=checkpoint, arm=CONSENSUS_EQUAL_ARM, probabilities=equal, exact=equal_exact, outcomes=outcomes, core=core_probs, weights=[1 / 3] * 3)
                prior = prior_consensus[cadence]
                if prior:
                    prior_probs = [np.concatenate([part[index] for part in prior]) for index in range(3)]
                    prior_outcomes = np.concatenate([part[3] for part in prior])
                    weights = _blend_weights(prior_probs, prior_outcomes)
                    weighted = np.tensordot(np.stack(component_probs, axis=1), weights, axes=(1, 0))
                    weighted_exact = np.tensordot(np.stack(component_exact, axis=1), weights, axes=(1, 0))
                    _append_record(records, season=season, cadence=cadence, checkpoint=checkpoint, arm=CONSENSUS_WEIGHTED_ARM, probabilities=weighted, exact=weighted_exact, outcomes=outcomes, core=core_probs, weights=weights.tolist())
                season_consensus[cadence].append((core_probs, predictions[WEIGHTED_ARM][0], predictions[SEASON_ONLY_ARM][0], outcomes))
        for cadence, entries in season_consensus.items():
            prior_consensus[cadence].extend(entries)

    metrics = pd.DataFrame(records)
    if metrics.empty:
        return {"horizon_matches": horizon_matches, "records": [], "summary": [], "status": "No eligible completed seasons."}
    summary = (
        metrics.groupby(["cadence", "arm"], as_index=False)
        .agg(
            rps=("rps", "mean"), brier=("brier", "mean"), log_loss=("log_loss", "mean"), ece=("ece", "mean"),
            scoreline_log_loss=("scoreline_log_loss", "mean"), rps_delta_vs_core=("rps_delta_vs_core", "mean"),
            n_fixtures=("n_fixtures", "sum"), folds=("season", "nunique"),
        )
        .sort_values(["rps", "brier"])
    )
    return {
        "horizon_matches": horizon_matches,
        "records": metrics.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "status": "Research only — production predictions and value bets are unchanged.",
    }


def _live_checkpoint(target: pd.DataFrame, cadence: str) -> tuple[int, pd.DataFrame]:
    """Return the last completed retrain boundary and its available results."""
    target = target.sort_values("date").reset_index(drop=True)
    if cadence in FIXED_CADENCES:
        seen = len(target) // FIXED_CADENCES[cadence] * FIXED_CADENCES[cadence]
        return seen, target.iloc[:seen]
    clubs: dict[str, int] = defaultdict(int)
    last_end = 0
    completed_blocks = 0
    for _, day in target.groupby("date", sort=True):
        for row in day.itertuples():
            clubs[row.team_home] += 1
            clubs[row.team_away] += 1
        blocks = min(clubs.values()) // 19 if clubs else 0
        if blocks > completed_blocks:
            completed_blocks = blocks
            last_end = int(day.index.max()) + 1
    return last_end, target.iloc[:last_end]


def build_live_study_predictions(matches_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Create research-only 1X2 snapshots without loading production artifacts.

    All arms see the same current feature rows. Their only difference is the
    results used to fit their weights, and these fitted models are returned to
    the caller rather than saved to ``models/``.
    """
    required = {"event_id", "team_home", "team_away", "commence_time"}
    if fixtures_df.empty or not required.issubset(fixtures_df.columns):
        return pd.DataFrame()
    frame, feature_cols = build_training_frame(matches_df=matches_df)
    feature_cols = [column for column in feature_cols if "fouls" not in column]
    current_season = football_data.season_str(football_data.CURRENT_SEASON_START_YEAR)
    target = frame[frame["season"] == current_season].sort_values("date").reset_index(drop=True)
    historical = frame[frame["season"] != current_season]
    if historical.empty:
        return pd.DataFrame()
    live_features = build_features_for_fixtures(fixtures_df, matches_df=matches_df)
    rows = []
    frozen = _fit_arm(CORE_ARM, historical, target.iloc[:0], feature_cols)
    if frozen is not None:
        probabilities, _ = _prediction_arrays(*frozen, live_features.assign(goals_home=0, goals_away=0), feature_cols)
        for cadence in ("no_retrain", *FIXED_CADENCES, PER_CLUB_CADENCE):
            for fixture, probs in zip(fixtures_df.itertuples(), probabilities):
                rows.append({"event_id": str(fixture.event_id), "team_home": fixture.team_home, "team_away": fixture.team_away, "commence_time": fixture.commence_time, "cadence": cadence, "arm": CORE_ARM, "current_season_matches_seen": 0, "home_win": probs[0], "draw": probs[1], "away_win": probs[2]})
    for cadence in (*FIXED_CADENCES, PER_CLUB_CADENCE):
        seen, prior_target = _live_checkpoint(target, cadence)
        for arm in (WEIGHTED_ARM, SEASON_ONLY_ARM):
            trained = _fit_arm(arm, historical, prior_target, feature_cols)
            if trained is None:
                continue
            probabilities, _ = _prediction_arrays(*trained, live_features.assign(goals_home=0, goals_away=0), feature_cols)
            for fixture, probs in zip(fixtures_df.itertuples(), probabilities):
                rows.append({"event_id": str(fixture.event_id), "team_home": fixture.team_home, "team_away": fixture.team_away, "commence_time": fixture.commence_time, "cadence": cadence, "arm": arm, "current_season_matches_seen": seen, "home_win": probs[0], "draw": probs[1], "away_win": probs[2]})
    snapshots = pd.DataFrame(rows)
    if snapshots.empty:
        return snapshots
    consensus_rows = []
    for (event_id, cadence), group in snapshots.groupby(["event_id", "cadence"]):
        components = group.set_index("arm")
        if not {CORE_ARM, WEIGHTED_ARM, SEASON_ONLY_ARM}.issubset(components.index):
            continue
        probabilities = components.loc[[CORE_ARM, WEIGHTED_ARM, SEASON_ONLY_ARM], ["home_win", "draw", "away_win"]].to_numpy(dtype=float).mean(axis=0)
        first = components.iloc[0]
        consensus_rows.append({"event_id": event_id, "team_home": first.team_home, "team_away": first.team_away, "commence_time": first.commence_time, "cadence": cadence, "arm": CONSENSUS_EQUAL_ARM, "current_season_matches_seen": int(group.current_season_matches_seen.max()), "home_win": probabilities[0], "draw": probabilities[1], "away_win": probabilities[2]})
    return pd.concat([snapshots, pd.DataFrame(consensus_rows)], ignore_index=True)


def run_in_season_learning_control(target_season: str | None = None, checkpoint_matches: int = 50, horizon_matches: int = 50) -> dict:
    """Backward-compatible single-season frozen-vs-rolling control."""
    if target_season is not None:
        matches = football_data.load_training_data()
        frame, _ = build_training_frame(matches_df=matches)
        if frame.empty or target_season not in set(frame["season"]):
            return {"target_season": target_season, "checkpoints": []}
    report = run_seasonal_model_study(horizon_matches=horizon_matches)
    records = [row for row in report["records"] if row["season"] == target_season] if target_season else report["records"]
    return {"target_season": target_season, "checkpoint_matches": checkpoint_matches, "horizon_matches": horizon_matches, "checkpoints": records}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline seasonal scoreline study.")
    parser.add_argument("--output", type=Path, help="JSON report path (default: reports/seasonal-model-study-<UTC>.json)")
    parser.add_argument("--horizon", type=int, default=19, help="Fixtures scored after each cadence checkpoint.")
    args = parser.parse_args()
    output = args.output or Path("reports") / f"seasonal-model-study-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = run_seasonal_model_study(horizon_matches=args.horizon, progress=print)
    output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    print(f"Saved seasonal study report to {output}")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


if __name__ == "__main__":
    _main()
