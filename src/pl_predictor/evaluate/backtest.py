"""backtest.py — thin wrapper around penaltyblog.backtest.Backtest for a
simple edge-threshold value-bet simulation on the held-out season.

This is a sanity check, not a target to chase: a strongly positive backtested
ROI on a single season is a red flag for overfitting/leakage far more often
than it's a genuine edge. Expect roughly break-even-to-slightly-negative
before transaction costs.
"""

from __future__ import annotations

import pandas as pd
import penaltyblog as pb

from ..models import scoreline

RESULT_TO_SIDE = {"H": "home_win", "D": "draw", "A": "away_win"}
ODDS_COLS = {"home_win": "b365_h", "draw": "b365_d", "away_win": "b365_a"}


def build_value_bet_backtest(
    df_with_odds: pd.DataFrame,
    model,
    start_date: str,
    end_date: str,
    edge_threshold: float = 0.05,
    stake: float = 1.0,
    bankroll: float = 100.0,
) -> pb.backtest.Backtest:
    """Bets flat `stake` whenever the model's probability for a side exceeds
    the de-vigged bookmaker-implied probability by more than `edge_threshold`.
    `df_with_odds` must have `date`, `team_home`, `team_away`, `ftr`, and the
    Bet365 closing-odds columns (b365_h/b365_d/b365_a). Returns the run
    `Backtest` instance — call `.results()` for the summary dict, or inspect
    `.account.tracker` for the bankroll curve."""
    df = df_with_odds.dropna(subset=list(ODDS_COLS.values())).copy()
    bt = pb.backtest.Backtest(df, start_date, end_date)

    def logic(ctx):
        fixture = ctx.fixture
        pred = scoreline.predict_fixture(model, fixture["team_home"], fixture["team_away"])
        if pred["fallback"]:
            return  # no reliable edge estimate for an unseen team

        actual_side = RESULT_TO_SIDE[fixture["ftr"]]
        for side, odds_col in ODDS_COLS.items():
            odds = fixture[odds_col]
            implied = 1 / odds
            edge = pred[side] - implied
            if edge > edge_threshold:
                won = int(side == actual_side)
                ctx.account.place_bet(odds, stake, won)

    bt.start(bankroll=bankroll, logic=logic)
    return bt


def run_value_bet_backtest(*args, **kwargs) -> dict:
    """Convenience wrapper: same as `build_value_bet_backtest(...).results()`."""
    return build_value_bet_backtest(*args, **kwargs).results()
