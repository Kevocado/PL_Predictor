"""market_models.py — corners and cards, the markets neither the goal model
nor The Odds API's free markets cover.

XGBoost regressors (Poisson objective — a natural fit for count targets)
predict expected corners/cards per match from the same rolling-form/rating
features as the scoreline model; `price_over_under` then generalizes
penaltyblog's documented "supply your own lambda, get market pricing"
pattern (see `penaltyblog.models.create_dixon_coles_grid`) to these counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import stats


def train_lambda_regressor(
    X_train,
    y_train,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
) -> xgb.XGBRegressor:
    """Trains a count regressor (Poisson objective) for a target like total
    corners or total cards in a match."""
    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def price_over_under(lam: float, line: float, dispersion: float | None = None) -> dict:
    """Prices P(over)/P(under) for a count total given expected value `lam`.
    Uses a Poisson survival function by default; if `dispersion` (the
    variance/mean ratio, from `check_overdispersion`) is materially > 1,
    switches to a negative-binomial with matching mean/variance instead —
    cards in particular tend to be overdispersed relative to a pure Poisson.
    """
    if dispersion is not None and dispersion > 1.2:
        variance = lam * dispersion
        p = lam / variance
        n = lam * p / (1 - p)
        over = float(stats.nbinom.sf(line, n, p))
    else:
        over = float(stats.poisson.sf(line, lam))
    return {"lambda": float(lam), "line": line, "over": over, "under": 1 - over}


def check_overdispersion(y: np.ndarray) -> float:
    """variance / mean of the target — a value materially above 1 indicates
    the count is overdispersed relative to a Poisson assumption."""
    return float(np.var(y) / np.mean(y))


def save_regressor(model: xgb.XGBRegressor, path: Path) -> Path:
    model.save_model(str(path))
    return path


def load_regressor(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model
