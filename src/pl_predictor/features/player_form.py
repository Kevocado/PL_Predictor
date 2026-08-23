"""player_form.py — rolling per-player form: goals/assists, plus the wider
FPL stat surface (ICT components, xG/xA, bps/bonus, defensive contribution,
clean sheets/saves).

Historical rolling features use the same shift(1).rolling(window) idiom as
team-level rolling_form.py, but computed as a genuine per-90 rate (summed
value over summed minutes in the window, times 90) rather than a
per-gameweek average — appearances of very different length (a 5-minute
substitute cameo vs a full 90) shouldn't count equally. That per-90
treatment fits "counting" stats (goals, assists, xG, xA, clean sheets,
saves — things that accumulate with more minutes played). ICT Index and its
Influence/Creativity/Threat components, bps, and bonus are FPL's own
per-match *scores*, not counts, so they get a plain rolling mean instead —
same distinction `xg_form.py` draws between rolling goals (a rate) and
rolling xG (also a rate, same treatment) at the team level.

Live "current form" blending ports FPL_Optimizer/features.py::
blended_form_features's three-tier confidence scheme (current season ->
prior season rate -> position-average prior, fading proportionally to
games_played) — adapted from that function's per-gameweek raw averages to
per-90 rates, to stay consistent with the historical features above.

Building this out is groundwork only — see `evaluate/player_stat_reliability.py`
for whether any of the extra stats beyond goals/assists actually predict a
player's *future* output better than the existing rolling-average baseline
before any of them get wired into `models/player_goals.py`'s live formula.
`defensive_contribution` is a stat FPL only introduced this (2025-26)
season — genuinely absent from older seasons' data, not a bug; handled the
same NaN-safe way as any other partial-coverage feature in this project
(e.g. Understat's own pre-2014-15 gap).
"""

from __future__ import annotations

import pandas as pd

WINDOWS = (5, 10)

# "Counting" stats — accumulate with playing time, so per-90 normalization
# makes sense the same way it does for goals/assists.
RATE_STATS = ["expected_goals", "expected_assists", "expected_goal_involvements", "clean_sheets", "saves"]
# FPL's own per-match *scores*, not counts — a rolling mean, not a per-90 rate.
MEAN_STATS = ["ict_index", "influence", "creativity", "threat", "bps", "bonus", "defensive_contribution"]


def build_historical_player_form(df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """`df` is FPL gameweek history (data.fpl_history.load_player_gw_history
    output). Returns rows for actual appearances only (minutes > 0) with
    rolling per-90 goal/assist-rate columns (plus the wider stat surface
    below, where the source column is present), using only *prior*
    appearances (shift(1)) so a row never sees its own outcome."""
    played = df[df["minutes"] > 0].sort_values(["element", "kickoff_time"]).copy()
    grouped = played.groupby("element", sort=False)

    feature_cols = []
    for w in windows:
        minutes_sum = grouped["minutes"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())

        goals_sum = grouped["goals_scored"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        assists_sum = grouped["assists"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        goals_col, assists_col = f"goals_per90_last{w}", f"assists_per90_last{w}"
        played[goals_col] = (goals_sum / minutes_sum * 90).where(minutes_sum > 0)
        played[assists_col] = (assists_sum / minutes_sum * 90).where(minutes_sum > 0)
        feature_cols += [goals_col, assists_col]

        for stat in RATE_STATS:
            if stat not in played.columns:
                continue
            stat_sum = grouped[stat].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
            col = f"{stat}_per90_last{w}"
            played[col] = (stat_sum / minutes_sum * 90).where(minutes_sum > 0)
            feature_cols.append(col)

        for stat in MEAN_STATS:
            if stat not in played.columns:
                continue
            col = f"{stat}_last{w}"
            played[col] = grouped[stat].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            feature_cols.append(col)

    return played, feature_cols


def position_rate_priors(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """League-wide per-90/mean rate by position — the last-resort fallback
    for a player with no FPL history at all (debutant, straight from the
    Championship)."""
    played = df[df["minutes"] > 0]
    if "position" not in played.columns or played.empty:
        return {}

    priors = {}
    for position, group in played.groupby("position"):
        total_minutes = group["minutes"].sum()
        if total_minutes == 0:
            continue
        prior = {
            "goals_per90": float(group["goals_scored"].sum() / total_minutes * 90),
            "assists_per90": float(group["assists"].sum() / total_minutes * 90),
            "avg_minutes": float(group["minutes"].mean()),
        }
        for stat in RATE_STATS:
            if stat in group.columns:
                prior[f"{stat}_per90"] = float(group[stat].sum() / total_minutes * 90)
        for stat in MEAN_STATS:
            if stat in group.columns:
                prior[stat] = float(group[stat].mean())
        priors[position] = prior
    return priors


def _season_prior_rate(prior_season: dict | None) -> dict | None:
    if not prior_season:
        return None
    minutes = prior_season.get("minutes") or 0
    if minutes < 90:  # barely played last season — no usable signal
        return None
    starts = prior_season.get("starts") or 0
    games_equiv = starts if starts > 0 else minutes / 90
    rates = {
        "goals_per90": float(prior_season.get("goals_scored", 0)) / minutes * 90,
        "assists_per90": float(prior_season.get("assists", 0)) / minutes * 90,
        "avg_minutes": float(minutes / games_equiv) if games_equiv > 0 else 90.0,
    }
    for stat in RATE_STATS:
        if stat in prior_season and prior_season[stat] is not None:
            rates[f"{stat}_per90"] = float(prior_season[stat]) / minutes * 90
    for stat in MEAN_STATS:
        # history_past is a season TOTAL, not an average — a mean-type stat
        # (e.g. bps) needs dividing by games played to be comparable to the
        # per-appearance rolling mean computed elsewhere.
        if stat in prior_season and prior_season[stat] is not None and games_equiv > 0:
            rates[stat] = float(prior_season[stat]) / games_equiv
    return rates


def blended_current_form(
    history_df: pd.DataFrame,
    prior_season: dict | None,
    position: str | None,
    position_priors: dict[str, dict[str, float]],
    windows: tuple[int, ...] = WINDOWS,
) -> tuple[dict, str]:
    """Live-serving equivalent of `build_historical_player_form`: this
    season's actual per-90/mean rate for every stat, blended toward last
    season's rate (or a position-average prior) proportionally to how many
    games have been played this season. Returns (rates_dict, confidence)
    where confidence is one of 'current', 'prior_season', 'position_avg',
    'none'."""
    played = history_df[history_df["minutes"] > 0].sort_values("GW") if not history_df.empty else history_df
    games_played = len(played)

    prior_rate = _season_prior_rate(prior_season)
    position_prior = position_priors.get(position) if position else None
    fallback = prior_rate or position_prior

    if games_played >= max(windows):
        confidence = "current"
    elif prior_rate is not None:
        confidence = "prior_season"
    elif position_prior is not None:
        confidence = "position_avg"
    elif games_played > 0:
        confidence = "current"
    else:
        confidence = "none"

    rates = {}
    max_w = max(windows)
    window_games = played.tail(max_w)
    weight = min(games_played / max_w, 1.0) if max_w else 0.0

    def _blend(current_val, fallback_val):
        if current_val is not None and fallback_val is not None:
            return weight * current_val + (1 - weight) * fallback_val
        if current_val is not None:
            return current_val
        return fallback_val

    for stat, col in [("goals_per90", "goals_scored"), ("assists_per90", "assists")]:
        fallback_rate = fallback.get(stat) if fallback else None
        current_rate = (
            float(window_games[col].sum() / window_games["minutes"].sum() * 90)
            if len(window_games) > 0 and window_games["minutes"].sum() > 0
            else None
        )
        rates[stat] = _blend(current_rate, fallback_rate) or 0.0

    for stat in RATE_STATS:
        key = f"{stat}_per90"
        fallback_rate = fallback.get(key) if fallback else None
        current_rate = (
            float(window_games[stat].sum() / window_games["minutes"].sum() * 90)
            if stat in window_games.columns and len(window_games) > 0 and window_games["minutes"].sum() > 0
            else None
        )
        blended = _blend(current_rate, fallback_rate)
        if blended is not None:
            rates[key] = blended

    for stat in MEAN_STATS:
        fallback_val = fallback.get(stat) if fallback else None
        current_val = (
            float(window_games[stat].mean())
            if stat in window_games.columns and len(window_games) > 0
            else None
        )
        blended = _blend(current_val, fallback_val)
        if blended is not None:
            rates[stat] = blended

    # Average minutes per recent appearance — used to discount rotation/
    # fringe players' expected goals by their typical playing time, since a
    # good per-90 rate over 20-minute cameos doesn't mean 90 minutes of
    # chances next match. Blended the same way as the rates above: early in
    # a season (or before it's started) there's no current-season minutes
    # data at all yet, so this must fall back too, not silently read as 0.
    fallback_minutes = fallback.get("avg_minutes") if fallback else None
    current_minutes = float(window_games["minutes"].mean()) if len(window_games) > 0 else None
    rates["avg_minutes"] = _blend(current_minutes, fallback_minutes) or 60.0  # unknown player, assume a typical sub/rotation cameo

    return rates, confidence
