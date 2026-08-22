"""value_bets.py — join model predictions to live bookmaker odds and surface
edges (model probability - de-vigged implied probability) above a threshold.

Corners/cards predictions are included in the output table but with no live
market (The Odds API's free markets don't cover them) — labeled as such
rather than silently omitted, per the plan's requirement to be transparent
about what can and can't be compared to a real market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..models import scoreline
from ..models.market_models import price_over_under

MARKET_OUTCOME_MAP = {
    "h2h": {"home_win": None, "draw": "Draw", "away_win": None},  # team names filled in per-fixture
    "totals": {"over_2_5": "Over", "under_2_5": "Under"},
    "btts": {"btts_yes": "Yes", "btts_no": "No"},
}


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
            "commence_time": fixture["commence_time"],
            "team_home": home,
            "team_away": away,
            "home_win_prob": pred["home_win"],
            "draw_prob": pred["draw"],
            "away_win_prob": pred["away_win"],
            "btts_yes_prob": pred["btts_yes"],
            "over_2_5_prob": pred["over_2_5"],
            "top_scoreline": f"{pred['top_scorelines'][0]['home']}-{pred['top_scorelines'][0]['away']}",
            "is_fallback_prediction": pred["fallback"],
        }

        implied_h2h = _devig_h2h(odds_df, event_id, home, away) if not odds_df.empty else None
        for side in ["home_win", "draw", "away_win"]:
            implied = implied_h2h.get(side) if implied_h2h else None
            row[f"{side}_implied"] = implied
            row[f"{side}_edge"] = (row[f"{side}_prob"] - implied) if implied is not None else None

        row["value_bet_flags"] = [
            side
            for side in ["home_win", "draw", "away_win"]
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
