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

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..models import scoreline

RESULT_TO_SIDE = {"H": "home_win", "D": "draw", "A": "away_win"}
ODDS_COLS = {
    "home_win": "b365_h",
    "draw": "b365_d",
    "away_win": "b365_a",
    "over_2_5": "b365>2.5",
    "under_2_5": "b365<2.5",
}


def _kelly_stake(prob: float, odds: float, fraction: float, max_stake_fraction: float, bankroll: float) -> float:
    b = odds - 1
    f_star = (b * prob - (1 - prob)) / b
    return max(0.0, min(fraction * f_star, max_stake_fraction)) * bankroll


def _precompute_predictions(model, df: pd.DataFrame, market_overrides: dict | None = None) -> dict:
    """One prediction per fixture, computed once before the sequential
    per-fixture backtest loop runs — for a feature-driven model (has
    `predict_many_from_rows`) this is a single batched XGBoost call instead
    of one call per fixture inside the loop, which is dramatically slower
    (see ml_scoreline.predict_grids_batch's docstring). Keyed by `df`'s own
    index, which `pb.backtest.Backtest` preserves through its internal
    date-filtering, so `logic()` can look a fixture's prediction up by
    `ctx.fixture.name` instead of recomputing it.

    `market_overrides` (see `models/scoreline.py::predict_fixture`) is
    applied afterward via each override model's own leakage-safe
    `feature_row`-based evaluation — this backtest should reflect what
    production actually serves, not just the primary scoreline model in
    isolation."""
    if hasattr(model, "predict_many_from_rows"):
        grids = model.predict_many_from_rows(df)
        results = {
            idx: {
                "home_win": g.home_win,
                "draw": g.draw,
                "away_win": g.away_win,
                "over_2_5": g.total_goals("over", 2.5),
                "under_2_5": g.total_goals("under", 2.5),
                "fallback": False,
            }
            for idx, g in zip(df.index, grids)
        }
        for idx, row in df.iterrows():
            results[idx]["data_confidence"] = scoreline._data_confidence(model, row["team_home"], row["team_away"])
    else:
        results = {
            idx: scoreline.predict_fixture(model, row["team_home"], row["team_away"], feature_row=row)
            for idx, row in df.iterrows()
        }

    if market_overrides:
        for market, override_model in market_overrides.items():
            for idx, row in df.iterrows():
                override_result = scoreline.predict_fixture(
                    override_model, row["team_home"], row["team_away"], feature_row=row
                )
                for field in scoreline.MARKET_FIELDS[market]:
                    results[idx][field] = override_result[field]
    return results


def _implied_probabilities(fixture: pd.Series) -> dict[str, float] | None:
    """De-vig archived Bet365 closing prices using the live Shin method."""
    try:
        h2h = pb.implied.calculate_implied(
            [fixture["b365_h"], fixture["b365_d"], fixture["b365_a"]],
            method="shin",
            market_names=["home_win", "draw", "away_win"],
        )
        totals = pb.implied.calculate_implied(
            [fixture["b365>2.5"], fixture["b365<2.5"]],
            method="shin",
            market_names=["over_2_5", "under_2_5"],
        )
    except ValueError:
        return None
    return {
        **{side: h2h.get_probability_by_name(side) for side in ["home_win", "draw", "away_win"]},
        **{side: totals.get_probability_by_name(side) for side in ["over_2_5", "under_2_5"]},
    }


def _actual_side(fixture: pd.Series, side: str) -> bool:
    if side in RESULT_TO_SIDE.values():
        return side == RESULT_TO_SIDE[fixture["ftr"]]
    total_goals = int(fixture["goals_home"]) + int(fixture["goals_away"])
    return total_goals > 2.5 if side == "over_2_5" else total_goals < 2.5


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
    selections: list[dict] | None = None,
    market_overrides: dict | None = None,
) -> pb.backtest.Backtest:
    """Simulates betting the held-out season whenever the model's probability
    for a side exceeds the de-vigged Bet365 closing probability by more than
    `edge_threshold`. The selection rule matches the live recommendation:
    one strongest independently priced 1X2 or O/U 2.5 market per fixture,
    with the same odds cap and no fallback predictions. `selections`, when
    supplied, is populated with an audit row for every simulated pick.

    `staking="kelly"` (default, see module docstring) or `"flat"` (bets a
    fixed `flat_stake` regardless of edge size — kept for comparison against
    the smarter default, not because it bets "with more sense"). Pass
    `market_overrides` (e.g. `models["scoreline_market_overrides"]`) so
    this reflects what production actually serves for a given market, not
    just `model` in isolation — see `_precompute_predictions`."""
    df = df_with_odds.dropna(subset=list(ODDS_COLS.values())).copy()
    predictions = _precompute_predictions(model, df, market_overrides=market_overrides)
    bt = pb.backtest.Backtest(df, start_date, end_date)

    def logic(ctx):
        fixture = ctx.fixture
        pred = predictions[fixture.name]
        if pred["fallback"]:
            return  # no reliable edge estimate for an unseen team

        implied = _implied_probabilities(fixture)
        if implied is None:
            return

        required_edge = edge_threshold * scoreline.required_edge_multiplier(pred.get("data_confidence"))
        best = None
        for side, odds_col in ODDS_COLS.items():
            odds = fixture[odds_col]
            if max_odds is not None and odds > max_odds:
                continue
            edge = pred[side] - implied[side]
            if edge > required_edge and (best is None or edge > best["edge"]):
                best = {"side": side, "odds": float(odds), "edge": float(edge), "prob": float(pred[side]), "implied": float(implied[side])}

        if best is None:
            return

        if staking == "kelly":
            stake = _kelly_stake(best["prob"], best["odds"], kelly_fraction, max_stake_fraction, ctx.account.current_bankroll)
            if stake <= 0:
                return
        else:
            stake = flat_stake

        won = int(_actual_side(fixture, best["side"]))
        ctx.account.place_bet(best["odds"], stake, won)
        if selections is not None:
            selections.append(
                {
                    "date": pd.Timestamp(fixture["date"]).date().isoformat(),
                    "fixture": f"{fixture['team_home']} {int(fixture['goals_home'])}-{int(fixture['goals_away'])} {fixture['team_away']}",
                    "selection": best["side"],
                    "price": best["odds"],
                    "model_probability": best["prob"],
                    "implied_probability": best["implied"],
                    "edge": best["edge"],
                    "won": bool(won),
                }
            )

    bt.start(bankroll=bankroll, logic=logic)
    return bt


def run_value_bet_backtest(*args, **kwargs) -> dict:
    """Convenience wrapper: same as `build_value_bet_backtest(...).results()`."""
    return build_value_bet_backtest(*args, **kwargs).results()


def bootstrap_drawdown_distribution(
    selections: list[dict],
    bankroll: float = 100.0,
    kelly_fraction: float = 0.10,
    max_stake_fraction: float = 0.05,
    staking: str = "kelly",
    flat_stake: float = 1.0,
    n_trials: int = 1000,
    seed: int = 0,
) -> dict:
    """`build_value_bet_backtest`'s `bankroll_curve` is a single realized
    path — one draw of which of the model's "should win ~60% of the time"
    bets actually won. That path's max drawdown (see this module's docstring
    for the -27%/-54%/-20% figures already measured) says as much about
    which particular bets happened to lose as it does about the strategy.

    This redraws `won` for every selection independently from a Bernoulli
    with the model's own `model_probability` (in the same chronological
    order the bets were placed, so streak/sequencing effects are preserved
    within a trial) `n_trials` times, replays the same staking rule as the
    real backtest via `pb.backtest.Account`, and reports the distribution of
    max drawdown and final ROI across trials — a calibration-conditional
    "how bad could this plausibly have gone" range, not a single path.
    Requires each `selections` row to have `price`, `model_probability`
    (matching `build_value_bet_backtest`'s `selections=` output)."""
    if not selections:
        return {
            "n_trials": 0,
            "max_drawdown_pct": {"mean": None, "median": None, "p95": None, "worst": None},
            "roi_pct": {"mean": None, "median": None, "p5": None},
        }

    rng = np.random.default_rng(seed)
    probs = np.array([float(s["model_probability"]) for s in selections])
    prices = np.array([float(s["price"]) for s in selections])

    drawdowns = np.empty(n_trials)
    rois = np.empty(n_trials)
    for trial in range(n_trials):
        draws = rng.random(len(selections)) < probs
        account = pb.backtest.Account(bankroll)
        for price, prob, won in zip(prices, probs, draws):
            if staking == "kelly":
                stake = _kelly_stake(float(prob), float(price), kelly_fraction, max_stake_fraction, account.current_bankroll)
                if stake <= 0:
                    continue
            else:
                stake = flat_stake
            account.place_bet(float(price), stake, int(won))

        # `Account.tracker` records post-bet bankroll only, with no entry for
        # the starting bankroll -- prepend it so a drawdown from the very
        # first bet is measured against the actual starting peak, not
        # against whatever the first recorded balance happens to be.
        curve = np.concatenate(([bankroll], account.tracker)) if account.tracker else np.array([bankroll])
        running_peak = np.maximum.accumulate(curve)
        drawdowns[trial] = float(((running_peak - curve) / running_peak).max() * 100) if running_peak.max() else 0.0
        rois[trial] = (account.current_bankroll - bankroll) / bankroll * 100

    return {
        "n_trials": n_trials,
        "max_drawdown_pct": {
            "mean": float(drawdowns.mean()),
            "median": float(np.median(drawdowns)),
            "p95": float(np.percentile(drawdowns, 95)),
            "worst": float(drawdowns.max()),
        },
        "roi_pct": {
            "mean": float(rois.mean()),
            "median": float(np.median(rois)),
            "p5": float(np.percentile(rois, 5)),
        },
    }
