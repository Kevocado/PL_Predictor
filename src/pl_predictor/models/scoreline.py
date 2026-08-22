"""scoreline.py — the core goals/1X2/scoreline engine, built on
penaltyblog's DixonColesGoalModel (and BivariatePoissonGoalModel as a
comparison candidate).

One fitted model's `FootballProbabilityGrid` gives 1X2, exact scorelines,
BTTS, and over/under goals all at once — no separate model needed for those
markets.

Note: penaltyblog's Cython loss functions require *writable* float64 numpy
arrays, not pandas Series (a pandas Series' underlying buffer is read-only to
Cython memoryviews) — hence the explicit `.to_numpy().astype(np.float64)` /
`.astype(str)` conversions below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..config import MODELS_DIR


def _writable_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    goals_home = df["goals_home"].to_numpy().astype(np.float64)
    goals_away = df["goals_away"].to_numpy().astype(np.float64)
    team_home = df["team_home"].to_numpy().astype(str)
    team_away = df["team_away"].to_numpy().astype(str)
    return goals_home, goals_away, team_home, team_away


def fit_dixon_coles(df: pd.DataFrame, xi: float = 0.0018) -> pb.models.DixonColesGoalModel:
    goals_home, goals_away, team_home, team_away = _writable_inputs(df)
    weights = np.asarray(pb.models.dixon_coles_weights(df["date"], xi=xi), dtype=np.float64)
    model = pb.models.DixonColesGoalModel(goals_home, goals_away, team_home, team_away, weights=weights)
    model.fit()
    return model


def fit_bivariate_poisson(df: pd.DataFrame, xi: float = 0.0018) -> pb.models.BivariatePoissonGoalModel:
    goals_home, goals_away, team_home, team_away = _writable_inputs(df)
    weights = np.asarray(pb.models.dixon_coles_weights(df["date"], xi=xi), dtype=np.float64)
    model = pb.models.BivariatePoissonGoalModel(goals_home, goals_away, team_home, team_away, weights=weights)
    model.fit()
    return model


def _top_n_scorelines(grid: pb.models.FootballProbabilityGrid, n: int = 5, max_goals: int = 8) -> list[dict]:
    scores = [
        {"home": h, "away": a, "prob": grid.exact_score(h, a)}
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
    ]
    return sorted(scores, key=lambda s: s["prob"], reverse=True)[:n]


# League-average goals-per-team-per-match, used as a same-strength-as-average
# fallback for teams the model has never seen (newly-promoted clubs, or any
# club absent from the training seasons window). Dixon-Coles/Bivariate-Poisson
# can only score teams present at fit time; this keeps predictions available
# from matchday one rather than crashing, at the cost of not yet knowing
# anything specific about that team's actual strength.
FALLBACK_GOAL_EXPECTANCY = 1.35


def is_known_team(model, team: str) -> bool:
    return team in set(model.teams)


def predict_fixture(model, home: str, away: str, max_goals: int = 10) -> dict:
    known_home, known_away = is_known_team(model, home), is_known_team(model, away)
    if known_home and known_away:
        grid = model.predict(home, away, max_goals=max_goals)
    else:
        grid = pb.models.create_dixon_coles_grid(
            FALLBACK_GOAL_EXPECTANCY, FALLBACK_GOAL_EXPECTANCY, rho=0.0, max_goals=max_goals
        )

    return {
        "home_win": grid.home_win,
        "draw": grid.draw,
        "away_win": grid.away_win,
        "btts_yes": grid.btts_yes,
        "btts_no": grid.btts_no,
        "over_2_5": grid.total_goals("over", 2.5),
        "under_2_5": grid.total_goals("under", 2.5),
        "top_scorelines": _top_n_scorelines(grid),
        "grid": grid.grid,
        "fallback": not (known_home and known_away),
    }


def predict_many(model, fixtures_df: pd.DataFrame, max_goals: int = 10) -> pd.DataFrame:
    rows = []
    for _, row in fixtures_df.iterrows():
        pred = predict_fixture(model, row["team_home"], row["team_away"], max_goals=max_goals)
        rows.append(
            {
                "team_home": row["team_home"],
                "team_away": row["team_away"],
                "home_win": pred["home_win"],
                "draw": pred["draw"],
                "away_win": pred["away_win"],
                "btts_yes": pred["btts_yes"],
                "over_2_5": pred["over_2_5"],
                "top_scoreline": pred["top_scorelines"][0],
            }
        )
    return pd.DataFrame(rows)


def save(model, path: Path | None = None) -> Path:
    path = path or (MODELS_DIR / "dixon_coles.pkl")
    model.save(str(path))
    return path


def load(path: Path | None = None):
    path = path or (MODELS_DIR / "dixon_coles.pkl")
    return pb.models.DixonColesGoalModel.load(str(path))
