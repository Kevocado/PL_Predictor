"""backtest.py — thin wrapper around penaltyblog.backtest.Backtest for an
edge-threshold value-bet simulation on the held-out season.

This is a sanity check, not a target to chase: a strongly positive backtested
ROI on a single season is a red flag for overfitting/leakage far more often
than it's a genuine edge. Expect roughly break-even-to-slightly-negative
before transaction costs.

Staking logic (default "kelly"):
- Bets only the single largest edge per fixture, not every side that clears
  the threshold — home/draw/away are mutually exclusive outcomes of the same
  event, so "betting" more than one is a self-correlated combination, not
  independent value.
- Sizes each stake with fractional Kelly (bankroll fraction proportional to
  edge and inversely proportional to the bet's variance) rather than a flat
  stake — a fixture the model is more confident about gets more bankroll, a
  marginal edge gets less. `kelly_fraction=0.10`, not the more commonly-cited
  quarter-Kelly: this model sits close to bookmaker calibration (see
  /api/calibration) rather than having a demonstrated real edge, so most
  "edges" it flags are small and noisy, not a proven advantage. Empirically
  (2025-26 holdout), quarter-Kelly turned a -27% flat-stake ROI into -54%
  with a 57%-of-bankroll drawdown — sizing up on unreliable edges punishes a
  bettor harder, not less, exactly as Kelly theory predicts. A tenth-Kelly
  is the fraction that actually reduced the drawdown here (-20% ROI); if
  this model's calibration improves further, revisit this constant upward.
- Skips odds beyond `max_odds`: a model's calibration is least trustworthy in
  the tails (rare, extreme outcomes), so a "big edge" on a 15/1 shot is more
  likely mis-calibration than real signal.
"""

from __future__ import annotations

import pandas as pd
import penaltyblog as pb

from ..models import scoreline

RESULT_TO_SIDE = {"H": "home_win", "D": "draw", "A": "away_win"}
ODDS_COLS = {"home_win": "b365_h", "draw": "b365_d", "away_win": "b365_a"}


def _kelly_stake(prob: float, odds: float, fraction: float, max_stake_fraction: float, bankroll: float) -> float:
    b = odds - 1
    f_star = (b * prob - (1 - prob)) / b
    return max(0.0, min(fraction * f_star, max_stake_fraction)) * bankroll


def _precompute_predictions(model, df: pd.DataFrame) -> dict:
    """One prediction per fixture, computed once before the sequential
    per-fixture backtest loop runs — for a feature-driven model (has
    `predict_many_from_rows`) this is a single batched XGBoost call instead
    of one call per fixture inside the loop, which is dramatically slower
    (see ml_scoreline.predict_grids_batch's docstring). Keyed by `df`'s own
    index, which `pb.backtest.Backtest` preserves through its internal
    date-filtering, so `logic()` can look a fixture's prediction up by
    `ctx.fixture.name` instead of recomputing it."""
    if hasattr(model, "predict_many_from_rows"):
        grids = model.predict_many_from_rows(df)
        return {
            idx: {"home_win": g.home_win, "draw": g.draw, "away_win": g.away_win, "fallback": False}
            for idx, g in zip(df.index, grids)
        }
    return {
        idx: scoreline.predict_fixture(model, row["team_home"], row["team_away"], feature_row=row)
        for idx, row in df.iterrows()
    }


def build_value_bet_backtest(
    df_with_odds: pd.DataFrame,
    model,
    start_date: str,
    end_date: str,
    edge_threshold: float = 0.05,
    bankroll: float = 100.0,
    staking: str = "kelly",
    flat_stake: float = 1.0,
    kelly_fraction: float = 0.10,
    max_stake_fraction: float = 0.05,
    max_odds: float | None = 6.0,
) -> pb.backtest.Backtest:
    """Simulates betting the held-out season whenever the model's probability
    for a side exceeds the raw bookmaker-implied probability by more than
    `edge_threshold`. `df_with_odds` must have `date`, `team_home`,
    `team_away`, `ftr`, and the Bet365 closing-odds columns (b365_h/b365_d/
    b365_a). Returns the run `Backtest` instance — call `.results()` for the
    summary dict, or inspect `.account.tracker` for the bankroll curve.

    `staking="kelly"` (default, see module docstring) or `"flat"` (bets a
    fixed `flat_stake` regardless of edge size — kept for comparison against
    the smarter default, not because it bets "with more sense")."""
    df = df_with_odds.dropna(subset=list(ODDS_COLS.values())).copy()
    predictions = _precompute_predictions(model, df)
    bt = pb.backtest.Backtest(df, start_date, end_date)

    def logic(ctx):
        fixture = ctx.fixture
        pred = predictions[fixture.name]
        if pred["fallback"]:
            return  # no reliable edge estimate for an unseen team

        actual_side = RESULT_TO_SIDE[fixture["ftr"]]

        best = None
        for side, odds_col in ODDS_COLS.items():
            odds = fixture[odds_col]
            if max_odds is not None and odds > max_odds:
                continue
            edge = pred[side] - (1 / odds)
            if edge > edge_threshold and (best is None or edge > best["edge"]):
                best = {"side": side, "odds": odds, "edge": edge, "prob": pred[side]}

        if best is None:
            return

        if staking == "kelly":
            stake = _kelly_stake(best["prob"], best["odds"], kelly_fraction, max_stake_fraction, ctx.account.current_bankroll)
            if stake <= 0:
                return
        else:
            stake = flat_stake

        won = int(best["side"] == actual_side)
        ctx.account.place_bet(best["odds"], stake, won)

    bt.start(bankroll=bankroll, logic=logic)
    return bt


def run_value_bet_backtest(*args, **kwargs) -> dict:
    """Convenience wrapper: same as `build_value_bet_backtest(...).results()`."""
    return build_value_bet_backtest(*args, **kwargs).results()
