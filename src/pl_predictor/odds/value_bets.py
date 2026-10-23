"""value_bets.py — join model predictions to live bookmaker odds and surface
edges (model probability - de-vigged implied probability) above a threshold.

Live edges are computed for h2h (1X2), totals (over/under 2.5 goals), and
BTTS — every market `data/sportsbook_api.py` prices live. Corners and cards
predictions are still included in the output table, but with no live market
to compare against (no provider used here prices them) — labeled as such
rather than silently omitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..models import scoreline
from . import kalshi_fees
try:
    from . import kalshi_consensus
except Exception:
    kalshi_consensus = None
from ..models.market_models import price_over_under

MAX_RECOMMENDATION_ODDS = 6.0
MAX_ODDS_AGE_SECONDS = 60 * 60

# A positive edge alone isn't enough to call something "value" -- measured
# directly against the offline backtest's own flagged bets, bucketed by the
# model's own probability for the flagged side: every bucket under 30% lost
# money on flat stakes (0-10% -100% ROI, 10-20% -31%, 20-30% -53%, n=44
# total), while 30-40% was already solidly profitable (+13% ROI, n=40) and
# 40-50% the best band measured (+43%, n=64). A heavy underdog can clear the
# edge threshold on paper while still being a bad bet in practice -- the
# model's calibration is least trustworthy exactly where it's making the
# boldest claim (see evaluate/backtest.py's own note on this). Applied to
# every market (1X2, O/U, BTTS), not just 1X2 -- the same tail-calibration
# risk applies to a lopsided Over/Under or BTTS price too.
MIN_VALUE_BET_PROBABILITY = 0.30

VALUE_BET_EDGE_THRESHOLD = 0.04

MARKET_CALIBRATION = {
    "match_result": {"calibrated": True, "reliability_slope": 1.33, "slope_ci": (0.55, 2.10), "n": 70, "calibration_error": 0.20},
    "totals_2_5": {"calibrated": True, "reliability_slope": 1.56, "slope_ci": (0.50, 6.78), "n": 70, "calibration_error": 0.25},
    "btts": {"calibrated": True, "reliability_slope": 3.28, "slope_ci": (0.58, 5.94), "n": 70, "calibration_error": 0.30},
    "corners": {"calibrated": False, "reliability_slope": None, "slope_ci": None, "n": 0, "calibration_error": 1.0},
    "cards": {"calibrated": False, "reliability_slope": None, "slope_ci": None, "n": 0, "calibration_error": 1.0},
}

def is_market_calibrated(market: str) -> bool:
    return bool(MARKET_CALIBRATION.get(market, {}).get("calibrated", False))

def market_calibration_error(market: str) -> float:
    return float(MARKET_CALIBRATION.get(market, {}).get("calibration_error", 1.0))

def side_market(side: str) -> str:
    if side in ("home_win", "draw", "away_win"):
        return "match_result"
    if side in ("over_2_5", "under_2_5"):
        return "totals_2_5"
    if side in ("btts_yes", "btts_no"):
        return "btts"
    if side.startswith("corner"):
        return "corners"
    if side.startswith("card"):
        return "cards"
    return side


def _best_quote(odds_df: pd.DataFrame, event_id, market: str, outcome_name: str, point: float | None = None) -> dict | None:
    rows = odds_df[
        (odds_df["event_id"] == event_id) & (odds_df["market"] == market) & (odds_df["outcome_name"] == outcome_name)
    ]
    if point is not None:
        rows = rows[rows["point"] == point]
    if rows.empty:
        return None
    quote = rows.loc[rows["price"].idxmax()]
    return {"price": float(quote["price"]), "bookmaker": str(quote["bookmaker"])}


def _best_price(odds_df: pd.DataFrame, event_id, market: str, outcome_name: str, point: float | None = None):
    quote = _best_quote(odds_df, event_id, market, outcome_name, point)
    return quote["price"] if quote is not None else None


def _devig_h2h(odds_df: pd.DataFrame, event_id, home: str, away: str) -> dict | None:
    home_price = _best_price(odds_df, event_id, "h2h", home)
    draw_price = _best_price(odds_df, event_id, "h2h", "Draw")
    away_price = _best_price(odds_df, event_id, "h2h", away)
    if None in (home_price, draw_price, away_price):
        return None
    try:
        implied = pb.implied.calculate_implied(
            [home_price, draw_price, away_price], method="shin", market_names=["home_win", "draw", "away_win"]
        )
    except ValueError:
        # penaltyblog's Shin solver can fail to converge on some real
        # live-odds combinations (root-finder needs opposite-signed
        # endpoints, which isn't guaranteed for every price triple) — treat
        # exactly like "no live odds for this fixture" rather than taking
        # down the whole fixtures response over one bad price set.
        return None
    return {k: implied.get_probability_by_name(k) for k in ["home_win", "draw", "away_win"]}


def _devig_totals(odds_df: pd.DataFrame, event_id, line: float = 2.5) -> dict | None:
    over_price = _best_price(odds_df, event_id, "totals", "Over", point=line)
    under_price = _best_price(odds_df, event_id, "totals", "Under", point=line)
    if None in (over_price, under_price):
        return None
    try:
        implied = pb.implied.calculate_implied(
            [over_price, under_price], method="shin", market_names=["over_2_5", "under_2_5"]
        )
    except ValueError:
        return None
    return {k: implied.get_probability_by_name(k) for k in ["over_2_5", "under_2_5"]}


def _devig_btts(odds_df: pd.DataFrame, event_id) -> dict | None:
    yes_price = _best_price(odds_df, event_id, "btts", "Yes")
    no_price = _best_price(odds_df, event_id, "btts", "No")
    if None in (yes_price, no_price):
        return None
    try:
        implied = pb.implied.calculate_implied(
            [yes_price, no_price], method="shin", market_names=["btts_yes", "btts_no"]
        )
    except ValueError:
        return None
    return {k: implied.get_probability_by_name(k) for k in ["btts_yes", "btts_no"]}


def _build_value_bet_table_redesign(
    fixtures_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    models: dict,
    edge_threshold: float = VALUE_BET_EDGE_THRESHOLD,
    kalshi_df=None,
    kalshi_lines=None,
    kalshi=None,
    **kwargs,
) -> pd.DataFrame:
    if kalshi_df is None:
        kalshi_df = kalshi_lines if kalshi_lines is not None else kalshi
    if kalshi_df is None:
        kalshi_df = kwargs.get("kalshi_data", None)

    rows = []
    preds = (
        scoreline.predict_fixtures_batch(
            models["scoreline"], fixtures_df, market_overrides=models.get("scoreline_market_overrides")
        )
        if not fixtures_df.empty
        else []
    )
    for (_, fixture), pred in zip(fixtures_df.iterrows(), preds):
        home, away, event_id = fixture["team_home"], fixture["team_away"], fixture["event_id"]

        row = {
            "event_id": event_id,
            "commence_time": fixture["commence_time"],
            "team_home": home,
            "team_away": away,
            "home_win_prob": pred["home_win"],
            "draw_prob": pred["draw"],
            "away_win_prob": pred["away_win"],
            "btts_yes_prob": pred["btts_yes"],
            "btts_no_prob": pred["btts_no"],
            "over_2_5_prob": pred["over_2_5"],
            "under_2_5_prob": pred["under_2_5"],
            "top_scoreline": f"{pred['top_scorelines'][0]['home']}-{pred['top_scorelines'][0]['away']}",
            "is_fallback_prediction": pred["fallback"],
            "data_confidence": pred["data_confidence"],
            "home_goal_expectation": pred["home_goal_expectation"],
            "away_goal_expectation": pred["away_goal_expectation"],
            "home_2plus_prob": pred["home_2plus_prob"],
            "away_2plus_prob": pred["away_2plus_prob"],
        }

        odds_timestamp = None
        if not odds_df.empty and "odds_fetched_at" in odds_df:
            timestamps = pd.to_datetime(odds_df["odds_fetched_at"], utc=True, errors="coerce").dropna()
            odds_timestamp = timestamps.max() if not timestamps.empty else None
        odds_age_seconds = (pd.Timestamp.now(tz="UTC") - odds_timestamp).total_seconds() if odds_timestamp is not None else None
        row["odds_fetched_at"] = odds_timestamp
        row["odds_age_seconds"] = odds_age_seconds
        row["odds_is_stale"] = odds_age_seconds is not None and odds_age_seconds > MAX_ODDS_AGE_SECONDS

        implied_h2h = _devig_h2h(odds_df, event_id, home, away) if not odds_df.empty else None
        implied_totals = _devig_totals(odds_df, event_id) if not odds_df.empty else None
        implied_btts = _devig_btts(odds_df, event_id) if not odds_df.empty else None
        implied = {**(implied_h2h or {}), **(implied_totals or {}), **(implied_btts or {})}

        quote = {
            "home_win": _best_quote(odds_df, event_id, "h2h", home) if not odds_df.empty else None,
            "draw": _best_quote(odds_df, event_id, "h2h", "Draw") if not odds_df.empty else None,
            "away_win": _best_quote(odds_df, event_id, "h2h", away) if not odds_df.empty else None,
            "over_2_5": _best_quote(odds_df, event_id, "totals", "Over", point=2.5) if not odds_df.empty else None,
            "under_2_5": _best_quote(odds_df, event_id, "totals", "Under", point=2.5) if not odds_df.empty else None,
            "btts_yes": _best_quote(odds_df, event_id, "btts", "Yes") if not odds_df.empty else None,
            "btts_no": _best_quote(odds_df, event_id, "btts", "No") if not odds_df.empty else None,
        }

        # Kalshi consensus
        kalshi_consensus = None
        if kalshi_df is not None:
            try:
                import pl_predictor.odds.kalshi_consensus as _kc
                kalshi_consensus = _kc.kalshi_consensus_for_fixture(kalshi_df, home, away, fixture["commence_time"])
            except Exception:
                kalshi_consensus = None

        _SIDES = ["home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no"]
        row["value_bet_tiers"] = {}
        row["value_bet_confidence"] = {}
        for side in _SIDES:
            side_implied = implied.get(side)
            row[f"{side}_implied"] = side_implied
            row[f"{side}_price"] = quote[side]["price"] if quote[side] is not None else None
            row[f"{side}_bookmaker"] = quote[side]["bookmaker"] if quote[side] is not None else None

            # consensus source
            kinfo = kalshi_consensus.get(side) if kalshi_consensus else None
            if kinfo is not None:
                consensus_prob = float(kinfo["prob"])
                consensus_source = "kalshi"
                consensus_price = float(kinfo.get("price", kinfo["prob"]))
            else:
                consensus_prob = side_implied
                consensus_source = "bookmakers" if side_implied is not None else None
                consensus_price = row[f"{side}_price"]
            row[f"{side}_consensus_source"] = consensus_source
            row[f"{side}_consensus_prob"] = consensus_prob
            row[f"{side}_consensus_price"] = consensus_price
            row[f"{side}_source"] = consensus_source

            model_prob = row[f"{side}_prob"] if f"{side}_prob" in row else None
            # model prob from pred
            prob_map = {
                "home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"],
                "over_2_5": pred["over_2_5"], "under_2_5": pred["under_2_5"],
                "btts_yes": pred["btts_yes"], "btts_no": pred["btts_no"],
            }
            model_prob = prob_map[side]

            # shrink
            mkt = side_market(side)
            err = market_calibration_error(mkt)
            shrink_factor = max(0.0, min(1.0, 1.0 - err))
            if consensus_prob is not None:
                shrunk_model = consensus_prob + (model_prob - consensus_prob) * shrink_factor
            else:
                shrunk_model = model_prob

            if consensus_prob is None:
                edge = None
            elif consensus_source == "kalshi":
                edge = kalshi_fees.net_edge_pct(shrunk_model * 100.0, consensus_prob * 100.0) / 100.0
            else:
                edge = shrunk_model - consensus_prob

            row[f"{side}_edge"] = edge
            row[f"{side}_shrunk_prob"] = shrunk_model

        required_edge = edge_threshold * scoreline.required_edge_multiplier(row["data_confidence"])
        flags = []
        if not row["odds_is_stale"]:
            for side in _SIDES:
                edge = row[f"{side}_edge"]
                prob = prob_map[side] if False else None
                # use original model prob for min prob check
                orig_prob_map = {
                    "home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"],
                    "over_2_5": pred["over_2_5"], "under_2_5": pred["under_2_5"],
                    "btts_yes": pred["btts_yes"], "btts_no": pred["btts_no"],
                }
                if edge is not None and edge > required_edge and orig_prob_map[side] >= MIN_VALUE_BET_PROBABILITY:
                    if is_market_calibrated(side_market(side)):
                        flags.append(side)
        row["value_bet_flags"] = flags

        # tiers
        for side in flags:
            e = row[f"{side}_edge"]
            tier = "Strong" if (e is not None and e >= max(required_edge * 2, required_edge + 0.04)) else "Lean"
            row["value_bet_tiers"][side] = tier
            row["value_bet_confidence"][side] = tier

        candidates = [
            side
            for side in row["value_bet_flags"]
            if not row["is_fallback_prediction"]
            and not row["odds_is_stale"]
            and row[f"{side}_price"] is not None
            and row[f"{side}_price"] <= MAX_RECOMMENDATION_ODDS
        ]
        best = max(candidates, key=lambda side: row[f"{side}_edge"]) if candidates else None
        row["recommended_market"] = best
        row["recommended_prob"] = row[f"{best}_prob"] if best else None
        row["recommended_implied"] = row[f"{best}_implied"] if best else None
        row["recommended_edge"] = row[f"{best}_edge"] if best else None
        row["recommended_price"] = row[f"{best}_price"] if best else None
        row["recommended_bookmaker"] = row[f"{best}_bookmaker"] if best else None

        # consensus source for recommended
        row["recommended_source"] = row[f"{best}_consensus_source"] if best else None

        row["corners_market_note"] = "No live market (Odds API doesn't cover corners)"
        row["cards_market_note"] = "No live market (Odds API doesn't cover cards)"

        rows.append(row)

    return pd.DataFrame(rows)

def predict_market_models_for_fixture(models: dict, feature_row: pd.Series, line_corners: float = 9.5, line_cards: float = 3.5) -> dict:
    """Corners/cards O/U pricing, plus each team's plain expected shots, for
    one fixture, given its feature row (from `features.build`). No live
    market to compare against — model-only. Shots are reported as a plain
    expected value per team (like `home_goal_expectation`/
    `away_goal_expectation`), not an O/U line — unlike corners/cards, shot
    volume is naturally asymmetric between a match's two sides, and there's
    no natural single threshold to quote per team the way 9.5/3.5 work for
    a match total."""
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    corners_lambda = float(models["corners"].predict(X)[0])
    cards_lambda = float(models["cards"].predict(X)[0])

    return {
        "corners": price_over_under(corners_lambda, line_corners, models.get("corners_dispersion")),
        "cards": price_over_under(cards_lambda, line_cards, models.get("cards_dispersion")),
        "home_shots": float(models["home_shots"].predict(X)[0]),
        "away_shots": float(models["away_shots"].predict(X)[0]),
        "home_shots_on_target": float(models["home_shots_on_target"].predict(X)[0]),
        "away_shots_on_target": float(models["away_shots_on_target"].predict(X)[0]),
    }


def _build_value_bet_table_legacy(
    fixtures_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    models: dict,
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []
    preds = (
        scoreline.predict_fixtures_batch(
            models["scoreline"], fixtures_df, market_overrides=models.get("scoreline_market_overrides")
        )
        if not fixtures_df.empty
        else []
    )
    for (_, fixture), pred in zip(fixtures_df.iterrows(), preds):
        home, away, event_id = fixture["team_home"], fixture["team_away"], fixture["event_id"]

        row = {
            "event_id": event_id,
            "commence_time": fixture["commence_time"],
            "team_home": home,
            "team_away": away,
            "home_win_prob": pred["home_win"],
            "draw_prob": pred["draw"],
            "away_win_prob": pred["away_win"],
            "btts_yes_prob": pred["btts_yes"],
            "btts_no_prob": pred["btts_no"],
            "over_2_5_prob": pred["over_2_5"],
            "under_2_5_prob": pred["under_2_5"],
            "top_scoreline": f"{pred['top_scorelines'][0]['home']}-{pred['top_scorelines'][0]['away']}",
            "is_fallback_prediction": pred["fallback"],
            "data_confidence": pred["data_confidence"],
            # Total goals / goal margin ("spread"), derived directly from
            # the scoreline model's own two goal-expectation numbers rather
            # than a dedicated regressor — measured directly (walk-forward,
            # 2026-08-28): a dedicated total-goals regressor lost to this on
            # every fold (log-loss 0.6917 vs 0.6829, Brier 0.2493 vs
            # 0.2449), and a dedicated margin regressor likewise lost on MAE
            # (1.384 vs 1.323) — the joint home/away fit already captures
            # this better than fitting either derived target directly.
            "home_goal_expectation": pred["home_goal_expectation"],
            "away_goal_expectation": pred["away_goal_expectation"],
            "home_2plus_prob": pred["home_2plus_prob"],
            "away_2plus_prob": pred["away_2plus_prob"],
        }

        odds_timestamp = None
        if not odds_df.empty and "odds_fetched_at" in odds_df:
            timestamps = pd.to_datetime(odds_df["odds_fetched_at"], utc=True, errors="coerce").dropna()
            odds_timestamp = timestamps.max() if not timestamps.empty else None
        odds_age_seconds = (pd.Timestamp.now(tz="UTC") - odds_timestamp).total_seconds() if odds_timestamp is not None else None
        row["odds_fetched_at"] = odds_timestamp
        row["odds_age_seconds"] = odds_age_seconds
        row["odds_is_stale"] = odds_age_seconds is not None and odds_age_seconds > MAX_ODDS_AGE_SECONDS

        implied_h2h = _devig_h2h(odds_df, event_id, home, away) if not odds_df.empty else None
        implied_totals = _devig_totals(odds_df, event_id) if not odds_df.empty else None
        implied_btts = _devig_btts(odds_df, event_id) if not odds_df.empty else None
        implied = {**(implied_h2h or {}), **(implied_totals or {}), **(implied_btts or {})}

        # Raw (non-devigged) best price per side — the implied probabilities
        # above have the bookmaker's margin stripped out, which is the right
        # thing to compare a model probability against for edge detection,
        # but it's not what a real bet actually pays out on. Kept alongside
        # the edge so a live value-bet tracker can compute real profit/ROI
        # from what was actually flagged, not just whether it "looked" good.
        quote = {
            "home_win": _best_quote(odds_df, event_id, "h2h", home) if not odds_df.empty else None,
            "draw": _best_quote(odds_df, event_id, "h2h", "Draw") if not odds_df.empty else None,
            "away_win": _best_quote(odds_df, event_id, "h2h", away) if not odds_df.empty else None,
            "over_2_5": _best_quote(odds_df, event_id, "totals", "Over", point=2.5) if not odds_df.empty else None,
            "under_2_5": _best_quote(odds_df, event_id, "totals", "Under", point=2.5) if not odds_df.empty else None,
            "btts_yes": _best_quote(odds_df, event_id, "btts", "Yes") if not odds_df.empty else None,
            "btts_no": _best_quote(odds_df, event_id, "btts", "No") if not odds_df.empty else None,
        }

        _SIDES = ["home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no"]
        for side in _SIDES:
            side_implied = implied.get(side)
            row[f"{side}_implied"] = side_implied
            row[f"{side}_edge"] = (row[f"{side}_prob"] - side_implied) if side_implied is not None else None
            row[f"{side}_price"] = quote[side]["price"] if quote[side] is not None else None
            row[f"{side}_bookmaker"] = quote[side]["bookmaker"] if quote[side] is not None else None

        required_edge = edge_threshold * scoreline.required_edge_multiplier(row["data_confidence"])
        row["value_bet_flags"] = [] if row["odds_is_stale"] else [
            side
            for side in _SIDES
            if row[f"{side}_edge"] is not None
            and row[f"{side}_edge"] > required_edge
            and row[f"{side}_prob"] >= MIN_VALUE_BET_PROBABILITY
        ]

        # A recommendation is intentionally one independently priced market,
        # never a parlay.  Individual model edges do not provide the joint
        # probability or bookmaker-specific price needed for a safe parlay.
        candidates = [
            side
            for side in row["value_bet_flags"]
            if not row["is_fallback_prediction"]
            and not row["odds_is_stale"]
            and row[f"{side}_price"] is not None
            and row[f"{side}_price"] <= MAX_RECOMMENDATION_ODDS
        ]
        best = max(candidates, key=lambda side: row[f"{side}_edge"]) if candidates else None
        row["recommended_market"] = best
        row["recommended_prob"] = row[f"{best}_prob"] if best else None
        row["recommended_implied"] = row[f"{best}_implied"] if best else None
        row["recommended_edge"] = row[f"{best}_edge"] if best else None
        row["recommended_price"] = row[f"{best}_price"] if best else None
        row["recommended_bookmaker"] = row[f"{best}_bookmaker"] if best else None

        row["corners_market_note"] = "No live market (Odds API doesn't cover corners)"
        row["cards_market_note"] = "No live market (Odds API doesn't cover cards)"




def build_value_bet_table(*args, **kwargs):
    redesign = kwargs.pop("redesign", False)
    if redesign:
        return _build_value_bet_table_redesign(*args, **kwargs)
    return _build_value_bet_table_legacy(*args, **kwargs)
