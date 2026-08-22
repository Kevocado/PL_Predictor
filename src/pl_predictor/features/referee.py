"""referee.py — referee card-strictness as a feature for the cards model.

Research consensus: referee identity has a real, measurable effect on card
counts (some referees card far more readily than others), and the data is
already sitting unused in football-data.co.uk's CSVs (a `referee` column) —
no new data source needed.

**Live-serving limitation, stated plainly**: none of our fixture sources
(The Odds API, the FPL API) publish referee assignments in advance —
referees are typically only announced a few days before kickoff — so
`features.build.build_features_for_fixtures` has no way to know who's
reffing an upcoming match and always falls back to the league-average rate
there. This feature's value is mostly in improving the *historical* fit
(soaking up referee-driven variance in training so the model's other
coefficients are cleaner), not in differentiating between upcoming
fixtures — a real, if partial, benefit, not a wasted one.
"""

from __future__ import annotations

import pandas as pd

REFEREE_WINDOW = 20
FEATURE_COL = "referee_card_rate"


def _total_cards(df: pd.DataFrame) -> pd.Series:
    zeros = pd.Series(0, index=df.index)
    return df.get("hy", zeros).fillna(0) + df.get("ay", zeros).fillna(0) + df.get("hr", zeros).fillna(0) + df.get("ar", zeros).fillna(0)


def league_average_card_rate(matches_df: pd.DataFrame) -> float:
    """Fallback value for both new/rarely-seen referees in training and
    every upcoming fixture at serving time (referee unknown)."""
    total_cards = _total_cards(matches_df)
    return float(total_cards.mean()) if len(total_cards) else 3.5


def build_referee_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """One column, `referee_card_rate` — this referee's own rolling average
    total cards per match over their last 20 EPL matches, using only *prior*
    matches (shift(1)). Falls back to the league average for a referee's
    first appearance and for any match with no referee recorded."""
    df = matches_df.sort_values("date").copy()
    df["_total_cards"] = _total_cards(df)
    fallback = league_average_card_rate(matches_df)

    if "referee" not in df.columns:
        return pd.DataFrame({FEATURE_COL: fallback}, index=matches_df.index)

    grouped = df.groupby("referee", sort=False)
    df[FEATURE_COL] = grouped["_total_cards"].transform(
        lambda s: s.shift(1).rolling(REFEREE_WINDOW, min_periods=1).mean()
    )
    df[FEATURE_COL] = df[FEATURE_COL].fillna(fallback)
    return df[[FEATURE_COL]].reindex(matches_df.index)
