"""Leakage-safe walk-forward validation for the live value-bet rules."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import backtest, walk_forward
from ..models import ml_scoreline


def _games_played_from_train_df(train_df: pd.DataFrame) -> dict[str, int]:
    """Per-fold analogue of the live `FixtureFeatureContext.games_played`
    `scoreline._data_confidence` reads — how many matches this fold's own
    training window has seen for each team, so a newly-promoted side (few
    or zero rows) scores as low-confidence in validation exactly like it
    would in live serving, instead of `data_confidence` silently being
    `None` for every fold fixture."""
    counts: dict[str, int] = {}
    for column in ("team_home", "team_away"):
        for team, n in train_df[column].value_counts().items():
            counts[team] = counts.get(team, 0) + int(n)
    return counts


class _FoldScorelineModel:
    """Small adapter so the existing backtest can batch-score a fold."""

    def __init__(self, home_model, away_model, feature_cols: list[str], train_df: pd.DataFrame):
        self.home_model = home_model
        self.away_model = away_model
        self.feature_cols = feature_cols
        # `scoreline._data_confidence` only needs `.context.games_played` —
        # a plain namespace is enough, no live FixtureFeatureContext needed.
        self.context = SimpleNamespace(games_played=_games_played_from_train_df(train_df))

    def predict_many_from_rows(self, frame: pd.DataFrame):
        features = frame[self.feature_cols].fillna(0)
        return ml_scoreline.predict_grids_batch(self.home_model, self.away_model, features)


def _yield_summary(selections: list[dict]) -> dict:
    if not selections:
        return {"bets": 0, "wins": 0, "win_rate": None, "yield": None}
    returns = np.array([(bet["price"] - 1) if bet["won"] else -1 for bet in selections], dtype=float)
    wins = sum(bet["won"] for bet in selections)
    return {
        "bets": len(selections),
        "wins": int(wins),
        "win_rate": float(wins / len(selections) * 100),
        "yield": float(returns.mean() * 100),
    }


def _yield_interval(selections: list[dict], samples: int = 2_000) -> list[float] | None:
    if len(selections) < 2:
        return None
    returns = np.array([(bet["price"] - 1) if bet["won"] else -1 for bet in selections], dtype=float)
    generator = np.random.default_rng(42)
    bootstrap_means = generator.choice(returns, size=(samples, len(returns)), replace=True).mean(axis=1) * 100
    return [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])]


def _breakdown(selections: list[dict], key_fn) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for selection in selections:
        groups.setdefault(key_fn(selection), []).append(selection)
    return [{"label": label, **_yield_summary(group)} for label, group in sorted(groups.items())]


def _market_label(selection: dict) -> str:
    return "Match result" if selection["selection"] in {"home_win", "draw", "away_win"} else "Goals O/U 2.5"


def _odds_band(selection: dict) -> str:
    price = selection["price"]
    if price < 2:
        return "Under +100"
    if price < 3:
        return "+100 to +199"
    return "+200 or longer"


def run_walk_forward_value_bet_validation(min_train_seasons: int = 3) -> dict:
    """Evaluate the live single-bet rule across strictly future seasons.

    Every fold trains the production ML scoreline architecture only on
    earlier seasons, then replays archived Bet365 closing prices in the next
    season. This is intentionally more demanding than the single held-out
    replay and does not write into the live SQLite ledger.
    """
    folds = walk_forward.prepare_folds(min_train_seasons=min_train_seasons)
    fold_rows = []
    all_selections: list[dict] = []

    for fold in folds:
        train_df, val_df = fold["train_df"], fold["val_df"]
        home_model, away_model = ml_scoreline.train_goal_regressors(
            fold["X_train"], train_df["goals_home"], train_df["goals_away"]
        )
        model = _FoldScorelineModel(home_model, away_model, list(fold["X_train"].columns), train_df)
        selections: list[dict] = []
        start_date = str(val_df["date"].min().date())
        end_date = str(val_df["date"].max().date())
        backtest.build_value_bet_backtest(
            val_df,
            model,
            start_date,
            end_date,
            staking="flat",
            selections=selections,
        )
        for selection in selections:
            selection["season"] = fold["val_season"]
        all_selections.extend(selections)
        fold_rows.append({"season": fold["val_season"], "train_matches": len(train_df), **_yield_summary(selections)})

    summary = _yield_summary(all_selections)
    return {
        "model": "ml_scoreline",
        "min_train_seasons": min_train_seasons,
        "summary": {**summary, "yield_ci_95": _yield_interval(all_selections)},
        "folds": fold_rows,
        "by_market": _breakdown(all_selections, _market_label),
        "by_odds_band": _breakdown(all_selections, _odds_band),
    }
