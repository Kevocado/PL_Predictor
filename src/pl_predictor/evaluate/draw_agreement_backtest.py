"""draw_agreement_backtest.py — walk-forward comparison of plain marginal-
argmax 1x2 accuracy against `outcomes.predicted_result` (argmax promoted to
"draw" via `outcomes.draw_agreement`), on real held-out seasons rather than
the ~30 fixtures currently tracked live. Answers: does crediting a
scoreline-model/percentage-model draw agreement actually net more correct
1x2 calls, or does it just relabel a few draws at the cost of others?

Deliberately not wired into `train_all`/auto-retrain -- same reasoning as
`walk_forward.py`: this is a periodic/manual confidence check, run directly
or from a notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..outcomes import DRAW_AGREEMENT_THRESHOLD, predicted_result
from ..models import ml_scoreline
from .walk_forward import prepare_folds

RESULT_CODE = {"H": "home_win", "D": "draw", "A": "away_win"}


def _fold_predictions(fold: dict) -> pd.DataFrame:
    """One row per validation fixture: marginal H/D/A probs, top scoreline,
    and actual result -- everything `evaluate_draw_agreement` needs, without
    re-running feature engineering (folds are pre-built by `prepare_folds`)."""
    home_model, away_model = ml_scoreline.train_goal_regressors(
        fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
    )
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, fold["X_val"])
    rows = []
    for grid, (_, val_row) in zip(grids, fold["val_df"].iterrows()):
        home_goals, away_goals = np.unravel_index(np.argmax(grid.goal_matrix), grid.goal_matrix.shape)
        rows.append(
            {
                "val_season": fold["val_season"],
                "home_win_prob": grid.home_win,
                "draw_prob": grid.draw,
                "away_win_prob": grid.away_win,
                "top_scoreline": f"{home_goals}-{away_goals}",
                "actual": RESULT_CODE[val_row["ftr"]],
            }
        )
    return pd.DataFrame(rows)


def evaluate_draw_agreement(
    seasons: list[str] | None = None, min_train_seasons: int = 3, threshold: float = DRAW_AGREEMENT_THRESHOLD
) -> pd.DataFrame:
    """One row per fold: fixture count, plain-argmax accuracy, draw-agreement
    accuracy, and the draw-specific breakdown (how many actual draws each
    rule calls correctly, and how many non-draws draw-agreement wrongly
    calls as a draw) -- the trade-off `predicted_result` is actually making."""
    folds = prepare_folds(seasons, min_train_seasons)
    rows = []
    for fold in folds:
        preds = _fold_predictions(fold)
        preds["argmax_pick"] = preds[["home_win_prob", "draw_prob", "away_win_prob"]].idxmax(axis=1).str.replace("_prob", "", regex=False)
        preds["agreement_pick"] = preds.apply(
            lambda r: predicted_result(r["top_scoreline"], r["home_win_prob"], r["draw_prob"], r["away_win_prob"], threshold),
            axis=1,
        )
        actual_draws = preds["actual"] == "draw"
        rows.append(
            {
                "val_season": fold["val_season"],
                "n_val": len(preds),
                "n_actual_draws": int(actual_draws.sum()),
                "argmax_accuracy": float((preds["argmax_pick"] == preds["actual"]).mean()),
                "agreement_accuracy": float((preds["agreement_pick"] == preds["actual"]).mean()),
                "argmax_draws_called_correctly": int(((preds["argmax_pick"] == "draw") & actual_draws).sum()),
                "agreement_draws_called_correctly": int(((preds["agreement_pick"] == "draw") & actual_draws).sum()),
                "agreement_false_draw_calls": int(((preds["agreement_pick"] == "draw") & ~actual_draws).sum()),
            }
        )
    return pd.DataFrame(rows)
