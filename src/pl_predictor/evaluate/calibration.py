"""calibration.py — how good are the model's probabilities, really?

Computes RPS/Brier/ignorance on held-out-season predictions, and the same
metrics for the de-vigged (Shin method) closing bookmaker odds as a
realistic baseline — the model should be in the same ballpark as (not
dramatically worse than) the market, which already prices in information
(injuries, team news) the model doesn't have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..models import scoreline

RESULT_CODE = {"H": 0, "D": 1, "A": 2}

# football-data.co.uk closing-odds columns for match result, in order
# (home, draw, away). b365 = Bet365; used as the reference bookmaker.
CLOSING_ODDS_COLS = ["b365_h", "b365_d", "b365_a"]


def model_calibration(model, val_df: pd.DataFrame) -> dict:
    if hasattr(model, "predict_many_from_rows"):
        # Batched: one XGBoost .predict() call for the whole holdout instead
        # of one call per row in a Python loop — the latter is what made
        # this endpoint take ~5-6s on every single page load (measured:
        # ~13ms/row XGBoost call overhead x 380 rows). See
        # ml_scoreline.predict_grids_batch's docstring.
        grids = model.predict_many_from_rows(val_df)
        probs = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
    else:
        # Dixon-Coles/Bivariate-Poisson: no live feature context, no
        # XGBoost overhead to batch away — feature_row=row lets an ML model
        # use this fixture's own point-in-time features instead of
        # re-deriving "current" form/Elo/xG (see scoreline.predict_fixture's
        # docstring); these two models just ignore the extra argument.
        preds = [
            scoreline.predict_fixture(model, row["team_home"], row["team_away"], feature_row=row)
            for _, row in val_df.iterrows()
        ]
        probs = np.array([[p["home_win"], p["draw"], p["away_win"]] for p in preds])
    outcomes = val_df["ftr"].map(RESULT_CODE).to_numpy()
    return {
        "rps": pb.metrics.rps_average(probs, outcomes),
        "brier": pb.metrics.multiclass_brier_score(probs, outcomes),
        "ignorance": pb.metrics.ignorance_score(probs, outcomes),
        "n_matches": len(val_df),
    }


def bookmaker_calibration(val_df: pd.DataFrame, odds_cols: list[str] = CLOSING_ODDS_COLS) -> dict | None:
    """Same metrics computed on de-vigged closing bookmaker odds, as a
    realistic baseline. Returns None if the odds columns aren't present for
    this data (older seasons / missing bookmaker)."""
    df = val_df.dropna(subset=odds_cols)
    if df.empty:
        return None

    probs = []
    for _, row in df.iterrows():
        implied = pb.implied.calculate_implied(
            [row[c] for c in odds_cols], method="shin", market_names=["H", "D", "A"]
        )
        probs.append([implied.get_probability_by_name(n) for n in ["H", "D", "A"]])

    probs = np.array(probs)
    outcomes = df["ftr"].map(RESULT_CODE).to_numpy()
    return {
        "rps": pb.metrics.rps_average(probs, outcomes),
        "brier": pb.metrics.multiclass_brier_score(probs, outcomes),
        "ignorance": pb.metrics.ignorance_score(probs, outcomes),
        "n_matches": len(df),
    }


def naive_favourite_baseline(val_df: pd.DataFrame) -> dict:
    """A naive 'always predict the home win with a fixed generic
    probability split' baseline (60/22/18, roughly the long-run EPL home
    W/D/L split) — the model should clearly beat this."""
    probs = np.tile([0.46, 0.25, 0.29], (len(val_df), 1))
    outcomes = val_df["ftr"].map(RESULT_CODE).to_numpy()
    return {
        "rps": pb.metrics.rps_average(probs, outcomes),
        "brier": pb.metrics.multiclass_brier_score(probs, outcomes),
        "n_matches": len(val_df),
    }
