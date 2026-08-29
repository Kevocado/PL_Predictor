"""Research-only closing-odds benchmark.

football-data.co.uk's historic closing prices are useful for measuring how
informative the fully mature market was, but are deliberately never exposed
as a live feature. The free live feed has no matching historic timestamp;
using a closing price to train an early forecast would leak late news.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.scoreline import multiclass_top_label_ece

CLOSING_COLUMN_SETS = [
    ("psch", "pscd", "psca"),
    ("avg_ch", "avg_cd", "avg_ca"),
    ("b365_ch", "b365_cd", "b365_ca"),
]


def _columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    for cols in CLOSING_COLUMN_SETS:
        if all(col in df and df[col].notna().any() for col in cols):
            return cols
    return None


def closing_odds_benchmark(df: pd.DataFrame) -> dict:
    cols = _columns(df)
    if cols is None:
        return {"available": False, "reason": "No complete closing 1X2 odds columns found.", "deployable": False}
    prices = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    valid = prices.notna().all(axis=1) & (prices > 1).all(axis=1) & df["ftr"].isin(["H", "D", "A"])
    prices = prices.loc[valid]
    implied = 1.0 / prices.to_numpy(dtype=float)
    probabilities = implied / implied.sum(axis=1, keepdims=True)
    outcomes = df.loc[valid, "ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    one_hot = np.eye(3)[outcomes]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "available": True,
        "deployable": False,
        "label": "research-only historical closing-odds benchmark",
        "columns": list(cols),
        "n_fixtures": int(len(prices)),
        "brier_1x2": brier,
        "ece_1x2": multiclass_top_label_ece(probabilities, outcomes),
        "warning": "Closing odds include information unavailable at the 24h/1h live decision points. They cannot train or validate a deployable early model.",
    }
