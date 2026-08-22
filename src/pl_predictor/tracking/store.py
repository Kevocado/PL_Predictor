"""store.py — SQLite persistence for the live prediction track record.

Snapshots each fixture's core-market predictions *before* kickoff, then
reconciles them against actual results once matches are played. This is the
only way to honestly measure "how good are the predictions really" going
forward — a model re-evaluated after the fact against its own current
(possibly retrained) state would just be grading itself on data it may have
since trained on.

Markets tracked: the same core set already in `FixtureSummary` — 1X2, total
goals O/U 2.5, and BTTS. Corners/cards/player markets aren't tracked here
(no live line to compare against for most of them anyway — see
`odds/value_bets.py`).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import penaltyblog as pb

from ..config import TRACKING_DB_PATH

# (market, outcome_name, source probability column on the value-bet table)
MARKET_SPEC = [
    ("1x2", "home_win", "home_win_prob"),
    ("1x2", "draw", "draw_prob"),
    ("1x2", "away_win", "away_win_prob"),
    ("totals_2_5", "over", "over_2_5_prob"),
    ("totals_2_5", "under", "under_2_5_prob"),
    ("btts", "yes", "btts_yes_prob"),
]

_RESULT_TO_1X2 = {"H": "home_win", "D": "draw", "A": "away_win"}


def _naive(ts) -> pd.Timestamp:
    """Strip timezone info so every timestamp in this module compares
    cleanly against `data/football_data.py`'s naive `date` column — sources
    mix tz-aware (Odds API, UTC) and naive (FPL fallback) kickoff times."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            team_home TEXT NOT NULL,
            team_away TEXT NOT NULL,
            market TEXT NOT NULL,
            outcome_name TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            model_trained_at TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_outcome INTEGER,
            resolved_at TEXT,
            UNIQUE(event_id, market, outcome_name)
        )
        """
    )
    return conn


def record_predictions(table: pd.DataFrame, model_trained_at: str | None = None) -> int:
    """Snapshot current predictions for every fixture in `table` (the
    value-bet table already built by `odds/value_bets.py`). Already-logged
    (event_id, market, outcome_name) rows are left untouched — this is safe
    to call on every fixtures request."""
    if table.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, fixture in table.iterrows():
        for market, outcome_name, prob_col in MARKET_SPEC:
            rows.append(
                (
                    str(fixture["event_id"]),
                    fixture["team_home"],
                    fixture["team_away"],
                    market,
                    outcome_name,
                    float(fixture[prob_col]),
                    _naive(fixture["commence_time"]).isoformat(),
                    now,
                    model_trained_at,
                )
            )

    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO predictions
                (event_id, team_home, team_away, market, outcome_name, predicted_prob,
                 commence_time, snapshotted_at, model_trained_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def _actual_outcome(market: str, outcome_name: str, match: pd.Series) -> int:
    total_goals = match["goals_home"] + match["goals_away"]
    if market == "1x2":
        return int(_RESULT_TO_1X2[match["ftr"]] == outcome_name)
    if market == "totals_2_5":
        is_over = total_goals > 2.5
        return int(is_over if outcome_name == "over" else not is_over)
    if market == "btts":
        return int(match["goals_home"] > 0 and match["goals_away"] > 0)
    raise ValueError(f"Unknown market: {market}")


def reconcile_predictions(matches_df: pd.DataFrame, lookback_days: int = 3) -> int:
    """For every unresolved prediction whose kickoff has passed, look up the
    actual result in `matches_df` (matched on team names, within a few days
    of the predicted kickoff to absorb date/timezone slop) and fill in
    `actual_outcome`. Returns how many rows were resolved this call."""
    now = _naive(pd.Timestamp.now(tz="UTC"))

    with _connect() as conn:
        unresolved = pd.read_sql(
            "SELECT * FROM predictions WHERE resolved = 0", conn, parse_dates=["commence_time"]
        )
        if unresolved.empty:
            return 0

        due = unresolved[unresolved["commence_time"] < now]
        if due.empty:
            return 0

        resolved_count = 0
        for _, pred in due.iterrows():
            window_start = pred["commence_time"] - pd.Timedelta(days=lookback_days)
            window_end = pred["commence_time"] + pd.Timedelta(days=lookback_days)
            candidates = matches_df[
                (matches_df["team_home"] == pred["team_home"])
                & (matches_df["team_away"] == pred["team_away"])
                & (matches_df["date"] >= window_start)
                & (matches_df["date"] <= window_end)
            ]
            if candidates.empty:
                continue

            match = candidates.iloc[0]
            actual = _actual_outcome(pred["market"], pred["outcome_name"], match)
            conn.execute(
                "UPDATE predictions SET resolved = 1, actual_outcome = ?, resolved_at = ? WHERE id = ?",
                (actual, datetime.now(timezone.utc).isoformat(), int(pred["id"])),
            )
            resolved_count += 1

        return resolved_count


def get_track_record() -> dict:
    with _connect() as conn:
        resolved = pd.read_sql(
            "SELECT * FROM predictions WHERE resolved = 1", conn, parse_dates=["commence_time", "resolved_at"]
        )

    if resolved.empty:
        return {"n_resolved": 0, "rps": None, "brier": None, "weekly_trend": []}

    rps = None
    oneXtwo = resolved[resolved["market"] == "1x2"]
    if not oneXtwo.empty:
        pivot = oneXtwo.pivot_table(
            index="event_id", columns="outcome_name", values=["predicted_prob", "actual_outcome"]
        )
        pivot = pivot.dropna()
        if not pivot.empty:
            probs = pivot["predicted_prob"][["home_win", "draw", "away_win"]].to_numpy()
            outcomes = pivot["actual_outcome"][["home_win", "draw", "away_win"]].to_numpy().argmax(axis=1)
            rps = float(pb.metrics.rps_average(probs, outcomes))

    other = resolved[resolved["market"] != "1x2"]
    brier = float(((other["predicted_prob"] - other["actual_outcome"]) ** 2).mean()) if not other.empty else None

    resolved["squared_error"] = (resolved["predicted_prob"] - resolved["actual_outcome"]) ** 2
    resolved["week"] = resolved["commence_time"].dt.to_period("W").apply(lambda p: p.start_time.date().isoformat())
    weekly = resolved.groupby("week")["squared_error"].mean().reset_index()
    weekly_trend = [{"week": r["week"], "mean_squared_error": float(r["squared_error"])} for _, r in weekly.iterrows()]

    return {
        "n_resolved": int(len(resolved)),
        "n_fixtures": int(resolved["event_id"].nunique()),
        "rps": rps,
        "brier": brier,
        "weekly_trend": weekly_trend,
    }


def get_biggest_misses(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        resolved = pd.read_sql(
            "SELECT * FROM predictions WHERE resolved = 1", conn, parse_dates=["commence_time"]
        )
    if resolved.empty:
        return []

    resolved["squared_error"] = (resolved["predicted_prob"] - resolved["actual_outcome"]) ** 2
    worst = resolved.sort_values("squared_error", ascending=False).head(limit)
    return [
        {
            "team_home": r["team_home"],
            "team_away": r["team_away"],
            "commence_time": r["commence_time"].isoformat(),
            "market": r["market"],
            "outcome_name": r["outcome_name"],
            "predicted_prob": float(r["predicted_prob"]),
            "actual_outcome": int(r["actual_outcome"]),
        }
        for _, r in worst.iterrows()
    ]
