"""head_to_head.py — recent head-to-head history between the specific pair
of teams in a fixture. Same strictly-before-this-match's-date discipline as
the rest of the feature layer."""

from __future__ import annotations

import pandas as pd

H2H_WINDOW = 5


def _pair_key(home: str, away: str) -> str:
    return "|".join(sorted([home, away]))


def build_h2h_features(matches_df: pd.DataFrame, window: int = H2H_WINDOW) -> pd.DataFrame:
    """Returns `h2h_home_goal_diff_avg`, `h2h_home_win_rate` aligned to
    `matches_df`'s index — computed from the last `window` meetings between
    that exact pair (regardless of which side was home in those meetings),
    using only matches strictly before the current one."""
    df = matches_df.sort_values("date").copy()
    df["pair"] = [
        _pair_key(h, a) for h, a in zip(df["team_home"], df["team_away"])
    ]

    goal_diff_avg, home_win_rate = [], []
    history: dict[str, list[tuple[str, int]]] = {}

    for _, row in df.iterrows():
        pair = row["pair"]
        home = row["team_home"]
        past = history.get(pair, [])[-window:]

        if past:
            diffs = [gd if team == home else -gd for team, gd in past]
            goal_diff_avg.append(sum(diffs) / len(diffs))
            home_win_rate.append(sum(1 for d in diffs if d > 0) / len(diffs))
        else:
            goal_diff_avg.append(None)
            home_win_rate.append(None)

        gd_this_match = row["goals_home"] - row["goals_away"]
        history.setdefault(pair, []).append((home, gd_this_match))

    out = pd.DataFrame(
        {"h2h_home_goal_diff_avg": goal_diff_avg, "h2h_home_win_rate": home_win_rate},
        index=df.index,
    )
    return out.reindex(matches_df.index)


def recent_meetings(matches_df: pd.DataFrame, home: str, away: str, n: int = H2H_WINDOW) -> list[dict]:
    """The last `n` actual meetings between this exact pair (most recent
    first), regardless of which side was home in those meetings — for
    display (a head-to-head list), not model features."""
    pair = _pair_key(home, away)
    df = matches_df.copy()
    df["pair"] = [_pair_key(h, a) for h, a in zip(df["team_home"], df["team_away"])]
    past = df[df["pair"] == pair].sort_values("date").tail(n).sort_values("date", ascending=False)

    return [
        {
            "date": row["date"].date().isoformat(),
            "team_home": row["team_home"],
            "team_away": row["team_away"],
            "goals_home": int(row["goals_home"]),
            "goals_away": int(row["goals_away"]),
        }
        for _, row in past.iterrows()
    ]
