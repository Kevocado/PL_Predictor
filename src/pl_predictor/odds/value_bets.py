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


def _best_price(odds_df: pd.DataFrame, event_id, market: str, outcome_name: str, point: float | None = None):
    rows = odds_df[
        (odds_df["event_id"] == event_id) & (odds_df["market"] == market) & (odds_df["outcome_name"] == outcome_name)
    ]
    if point is not None:
        rows = rows[rows["point"] == point]
    if rows.empty:
        return None
    return float(rows["price"].max())  # best (highest) price across bookmakers


def _devig_h2h(odds_df: pd.DataFrame, event_id, home: str, away: str) -> dict | None:
    home_price = _best_price(odds_df, event_id, "h2h", home)
    draw_price = _best_price(odds_df, event_id, "h2h", "Draw")
    away_price = _best_price(odds_df, event_id, "h2h", away)
    if None in (home_price, draw_price, away_price):
        return None
    implied = pb.implied.calculate_implied(
        [home_price, draw_price, away_price], method="shin", market_names=["home_win", "draw", "away_win"]
    )
    return {k: implied.get_probability_by_name(k) for k in ["home_win", "draw", "away_win"]}


def _devig_totals(odds_df: pd.DataFrame, event_id, line: float = 2.5) -> dict | None:
    over_price = _best_price(odds_df, event_id, "totals", "Over", point=line)
    under_price = _best_price(odds_df, event_id, "totals", "Under", point=line)
    if None in (over_price, under_price):
        return None
    implied = pb.implied.calculate_implied(
        [over_price, under_price], method="shin", market_names=["over_2_5", "under_2_5"]
    )
    return {k: implied.get_probability_by_name(k) for k in ["over_2_5", "under_2_5"]}


def build_value_bet_table(
    fixtures_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    models: dict,
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for _, fixture in fixtures_df.iterrows():
        home, away, event_id = fixture["team_home"], fixture["team_away"], fixture["event_id"]
        pred = scoreline.predict_fixture(models["scoreline"], home, away)

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

        implied_h2h = _devig_h2h(odds_df, event_id, home, away) if not odds_df.empty else None
        implied_totals = _devig_totals(odds_df, event_id) if not odds_df.empty else None
        implied = {**(implied_h2h or {}), **(implied_totals or {})}

        for side in ["home_win", "draw", "away_win", "over_2_5", "under_2_5"]:
            side_implied = implied.get(side)
            row[f"{side}_implied"] = side_implied
            row[f"{side}_edge"] = (row[f"{side}_prob"] - side_implied) if side_implied is not None else None

        row["value_bet_flags"] = [
            side
            for side in ["home_win", "draw", "away_win", "over_2_5", "under_2_5"]
            if row[f"{side}_edge"] is not None and row[f"{side}_edge"] > edge_threshold
        ]

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
