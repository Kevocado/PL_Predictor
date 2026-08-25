"""value_bets.py — join model predictions to live bookmaker odds and surface
edges (model probability - de-vigged implied probability) above a threshold.

Live edges are computed for h2h (1X2) and totals (over/under 2.5 goals) —
the two "core" markets The Odds API's bulk endpoint serves. BTTS, corners,
and cards predictions are still included in the output table, but with no
live market to compare against (those need The Odds API's per-event
"additional markets" endpoint, not implemented here) — labeled as such
rather than silently omitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..models import scoreline
from ..models.market_models import price_over_under

MAX_RECOMMENDATION_ODDS = 6.0
MAX_ODDS_AGE_SECONDS = 60 * 60


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


def build_value_bet_table(
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
            "over_2_5_prob": pred["over_2_5"],
            "under_2_5_prob": pred["under_2_5"],
            "top_scoreline": f"{pred['top_scorelines'][0]['home']}-{pred['top_scorelines'][0]['away']}",
            "is_fallback_prediction": pred["fallback"],
            "data_confidence": pred["data_confidence"],
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
        implied = {**(implied_h2h or {}), **(implied_totals or {})}

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
        }

        for side in ["home_win", "draw", "away_win", "over_2_5", "under_2_5"]:
            side_implied = implied.get(side)
            row[f"{side}_implied"] = side_implied
            row[f"{side}_edge"] = (row[f"{side}_prob"] - side_implied) if side_implied is not None else None
            row[f"{side}_price"] = quote[side]["price"] if quote[side] is not None else None
            row[f"{side}_bookmaker"] = quote[side]["bookmaker"] if quote[side] is not None else None

        row["value_bet_flags"] = [] if row["odds_is_stale"] else [
            side
            for side in ["home_win", "draw", "away_win", "over_2_5", "under_2_5"]
            if row[f"{side}_edge"] is not None and row[f"{side}_edge"] > edge_threshold
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

        rows.append(row)

    return pd.DataFrame(rows)


def predict_market_models_for_fixture(models: dict, feature_row: pd.Series, line_corners: float = 9.5, line_cards: float = 3.5) -> dict:
    """Corners/cards O/U pricing for one fixture, given its feature row (from
    `features.build`). No live market to compare against — model-only."""
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    corners_lambda = float(models["corners"].predict(X)[0])
    cards_lambda = float(models["cards"].predict(X)[0])

    return {
        "corners": price_over_under(corners_lambda, line_corners, models.get("corners_dispersion")),
        "cards": price_over_under(cards_lambda, line_cards, models.get("cards_dispersion")),
    }
