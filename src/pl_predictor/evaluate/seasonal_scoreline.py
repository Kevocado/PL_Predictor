"""A deliberately within-season scoreline challenger.

The model starts every target season at league-average scoring rates and
updates only from that season's completed matches. Team scoring and conceding
rates use Gamma-Poisson shrinkage, so a few extreme results cannot make a club
look permanently elite. This is an experiment, not a production model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.metrics import log_loss

from ..data import football_data
from ..features.build import build_training_frame
from ..models import ml_scoreline

RESULT_CODE = {"H": 0, "D": 1, "A": 2}
PRIOR_MATCHES = 12.0
MIN_LAMBDA = 0.2
MAX_LAMBDA = 3.5


def _rates(prior: pd.DataFrame) -> tuple[float, float]:
    return (float(prior["goals_home"].mean()), float(prior["goals_away"].mean()))


def _prediction(state: dict[str, dict[str, float]], home: str, away: str, home_base: float, away_base: float):
    def attack(team: str) -> float:
        values = state.get(team, {})
        return (values.get("goals_for", 0.0) + PRIOR_MATCHES * ((home_base + away_base) / 2)) / (
            values.get("matches", 0.0) + PRIOR_MATCHES
        ) / ((home_base + away_base) / 2)

    def defence(team: str) -> float:
        values = state.get(team, {})
        return (values.get("goals_against", 0.0) + PRIOR_MATCHES * ((home_base + away_base) / 2)) / (
            values.get("matches", 0.0) + PRIOR_MATCHES
        ) / ((home_base + away_base) / 2)

    home_lambda = float(np.clip(home_base * attack(home) * defence(away), MIN_LAMBDA, MAX_LAMBDA))
    away_lambda = float(np.clip(away_base * attack(away) * defence(home), MIN_LAMBDA, MAX_LAMBDA))
    return pb.models.create_dixon_coles_grid(home_lambda, away_lambda, rho=0.0, max_goals=10)


def _update(state: dict[str, dict[str, float]], row) -> None:
    for team, goals_for, goals_against in (
        (row.team_home, row.goals_home, row.goals_away),
        (row.team_away, row.goals_away, row.goals_home),
    ):
        values = state.setdefault(team, {"matches": 0.0, "goals_for": 0.0, "goals_against": 0.0})
        values["matches"] += 1
        values["goals_for"] += float(goals_for)
        values["goals_against"] += float(goals_against)


def _metrics(probabilities: np.ndarray, outcomes: np.ndarray, exact_probabilities: np.ndarray) -> dict:
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return {
        "rps": float(pb.metrics.rps_average(probabilities.copy(), outcomes.copy())),
        "brier": float(pb.metrics.multiclass_brier_score(probabilities.copy(), outcomes.copy())),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
        "scoreline_log_loss": float(-np.log(np.clip(exact_probabilities, 1e-8, 1.0)).mean()),
        "n_fixtures": int(len(outcomes)),
    }


def evaluate_within_season_model(target_season: str = "2025-2026") -> dict:
    """Compare the online seasonal challenger to frozen historical ML.

    Both arms predict every target fixture from data available before kickoff.
    The seasonal arm only updates once all fixtures on a calendar date finish,
    preventing same-day lookahead.
    """
    matches = football_data.load_training_data()
    frame, feature_cols = build_training_frame(matches_df=matches)
    feature_cols = [column for column in feature_cols if "fouls" not in column]
    target = frame[frame["season"] == target_season].sort_values("date").reset_index(drop=True)
    historical = frame[frame["date"] < target["date"].min()]
    if target.empty or historical.empty:
        return {"target_season": target_season, "status": "Target season is unavailable."}

    home_model, away_model = ml_scoreline.train_goal_regressors(
        historical[feature_cols].fillna(0), historical["goals_home"], historical["goals_away"]
    )
    frozen_grids = ml_scoreline.predict_grids_batch(home_model, away_model, target[feature_cols].fillna(0))
    frozen_probabilities = np.array([[grid.home_win, grid.draw, grid.away_win] for grid in frozen_grids])
    frozen_exact = np.array(
        [grid.grid[min(int(row.goals_home), 10), min(int(row.goals_away), 10)] for grid, row in zip(frozen_grids, target.itertuples())]
    )

    home_base, away_base = _rates(historical)
    state: dict[str, dict[str, float]] = {}
    seasonal_probabilities, seasonal_exact = [], []
    for _, day in target.groupby("date", sort=True):
        for row in day.itertuples():
            grid = _prediction(state, row.team_home, row.team_away, home_base, away_base)
            seasonal_probabilities.append([grid.home_win, grid.draw, grid.away_win])
            seasonal_exact.append(grid.grid[min(int(row.goals_home), 10), min(int(row.goals_away), 10)])
        for row in day.itertuples():
            _update(state, row)

    outcomes = target["ftr"].map(RESULT_CODE).to_numpy(dtype=int, copy=True)
    frozen = _metrics(frozen_probabilities, outcomes, frozen_exact)
    seasonal = _metrics(np.array(seasonal_probabilities), outcomes, np.array(seasonal_exact))
    return {
        "target_season": target_season,
        "model": "within_season_gamma_poisson",
        "prior_matches": PRIOR_MATCHES,
        "frozen_ml": frozen,
        "within_season": seasonal,
        "rps_delta_vs_frozen_ml": seasonal["rps"] - frozen["rps"],
        "brier_delta_vs_frozen_ml": seasonal["brier"] - frozen["brier"],
        "status": "Offline experiment only — does not affect production predictions or retraining.",
    }
