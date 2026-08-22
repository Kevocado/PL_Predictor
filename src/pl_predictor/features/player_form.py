"""player_form.py — rolling per-player goal/assist rates.

Historical rolling features use the same shift(1).rolling(window) idiom as
team-level rolling_form.py, but computed as a genuine per-90 rate (summed
goals/assists over summed minutes in the window, times 90) rather than a
per-gameweek average — appearances of very different length (a 5-minute
substitute cameo vs a full 90) shouldn't count equally.

Live "current form" blending ports FPL_Optimizer/features.py::
blended_form_features's three-tier confidence scheme (current season ->
prior season rate -> position-average prior, fading proportionally to
games_played) — adapted from that function's per-gameweek raw averages to
per-90 rates, to stay consistent with the historical features above.
"""

from __future__ import annotations

import pandas as pd

WINDOWS = (5, 10)


def build_historical_player_form(df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """`df` is FPL gameweek history (data.fpl_history.load_player_gw_history
    output). Returns rows for actual appearances only (minutes > 0) with
    rolling per-90 goal/assist-rate columns, using only *prior* appearances
    (shift(1)) so a row never sees its own outcome."""
    played = df[df["minutes"] > 0].sort_values(["element", "kickoff_time"]).copy()
    grouped = played.groupby("element", sort=False)

    feature_cols = []
    for w in windows:
        goals_sum = grouped["goals_scored"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        assists_sum = grouped["assists"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        minutes_sum = grouped["minutes"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())

        goals_col, assists_col = f"goals_per90_last{w}", f"assists_per90_last{w}"
        played[goals_col] = (goals_sum / minutes_sum * 90).where(minutes_sum > 0)
        played[assists_col] = (assists_sum / minutes_sum * 90).where(minutes_sum > 0)
        feature_cols += [goals_col, assists_col]

    return played, feature_cols


def position_rate_priors(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """League-wide per-90 goal/assist rate by position — the last-resort
    fallback for a player with no FPL history at all (debutant, straight
    from the Championship)."""
    played = df[df["minutes"] > 0]
    if "position" not in played.columns or played.empty:
        return {}

    priors = {}
    for position, group in played.groupby("position"):
        total_minutes = group["minutes"].sum()
        if total_minutes == 0:
            continue
        priors[position] = {
            "goals_per90": float(group["goals_scored"].sum() / total_minutes * 90),
            "assists_per90": float(group["assists"].sum() / total_minutes * 90),
            "avg_minutes": float(group["minutes"].mean()),
        }
    return priors


def _season_prior_rate(prior_season: dict | None) -> dict | None:
    if not prior_season:
        return None
    minutes = prior_season.get("minutes") or 0
    if minutes < 90:  # barely played last season — no usable signal
        return None
    starts = prior_season.get("starts") or 0
    games_equiv = starts if starts > 0 else minutes / 90
    return {
        "goals_per90": float(prior_season.get("goals_scored", 0)) / minutes * 90,
        "assists_per90": float(prior_season.get("assists", 0)) / minutes * 90,
        "avg_minutes": float(minutes / games_equiv) if games_equiv > 0 else 90.0,
    }


def blended_current_form(
    history_df: pd.DataFrame,
    prior_season: dict | None,
    position: str | None,
    position_priors: dict[str, dict[str, float]],
    windows: tuple[int, ...] = WINDOWS,
) -> tuple[dict, str]:
    """Live-serving equivalent of `build_historical_player_form`: this
    season's actual per-90 rate, blended toward last season's rate (or a
    position-average prior) proportionally to how many games have been
    played this season. Returns (rates_dict, confidence) where confidence is
    one of 'current', 'prior_season', 'position_avg', 'none'."""
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
    for stat, col in [("goals_per90", "goals_scored"), ("assists_per90", "assists")]:
        fallback_rate = fallback.get(stat) if fallback else None
        current_rate = (
            float(window_games[col].sum() / window_games["minutes"].sum() * 90)
            if len(window_games) > 0 and window_games["minutes"].sum() > 0
            else None
        )
        if current_rate is not None and fallback_rate is not None:
            weight = min(games_played / max_w, 1.0)
            rates[stat] = weight * current_rate + (1 - weight) * fallback_rate
        elif current_rate is not None:
            rates[stat] = current_rate
        else:
            rates[stat] = fallback_rate or 0.0

    # Average minutes per recent appearance — used to discount rotation/
    # fringe players' expected goals by their typical playing time, since a
    # good per-90 rate over 20-minute cameos doesn't mean 90 minutes of
    # chances next match. Blended the same way as the rates above: early in
    # a season (or before it's started) there's no current-season minutes
    # data at all yet, so this must fall back too, not silently read as 0.
    fallback_minutes = fallback.get("avg_minutes") if fallback else None
    current_minutes = float(window_games["minutes"].mean()) if len(window_games) > 0 else None
    if current_minutes is not None and fallback_minutes is not None:
        weight = min(games_played / max_w, 1.0)
        rates["avg_minutes"] = weight * current_minutes + (1 - weight) * fallback_minutes
    elif current_minutes is not None:
        rates["avg_minutes"] = current_minutes
    else:
        rates["avg_minutes"] = fallback_minutes or 60.0  # unknown player, assume a typical sub/rotation cameo

    return rates, confidence
