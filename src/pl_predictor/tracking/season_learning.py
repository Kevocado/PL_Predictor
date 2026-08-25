"""Prospective cohort report for models retrained during the current season."""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.metrics import log_loss

from .store import _connect


def _ensure_study_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seasonal_study_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            team_home TEXT NOT NULL,
            team_away TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            cadence TEXT NOT NULL,
            arm TEXT NOT NULL,
            current_season_matches_seen INTEGER NOT NULL,
            home_win REAL NOT NULL,
            draw REAL NOT NULL,
            away_win REAL NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_outcome TEXT,
            resolved_at TEXT,
            UNIQUE(event_id, cadence, arm)
        )
        """
    )


def record_study_predictions(table: pd.DataFrame) -> int:
    """Persist first pre-kickoff research snapshot for each fixture/arm.

    ``INSERT OR IGNORE`` is deliberate: later retrains must never overwrite
    the probability that the prospective study will eventually grade.
    """
    if table.empty:
        return 0
    now = pd.Timestamp.now(tz="UTC").isoformat()
    rows = [
        (
            str(row.event_id), row.team_home, row.team_away, pd.Timestamp(row.commence_time).tz_localize(None).isoformat(),
            row.cadence, row.arm, int(row.current_season_matches_seen), float(row.home_win), float(row.draw), float(row.away_win), now,
        )
        for row in table.itertuples()
    ]
    with _connect() as conn:
        _ensure_study_table(conn)
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO seasonal_study_predictions
              (event_id, team_home, team_away, commence_time, cadence, arm,
               current_season_matches_seen, home_win, draw, away_win, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def reconcile_study_predictions(matches_df: pd.DataFrame, lookback_days: int = 3) -> int:
    """Resolve research snapshots against confirmed results, never forecasts."""
    if matches_df.empty:
        return 0
    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    if isinstance(matches["date"].dtype, pd.DatetimeTZDtype):
        matches["date"] = matches["date"].dt.tz_localize(None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    with _connect() as conn:
        _ensure_study_table(conn)
        pending = pd.read_sql("SELECT * FROM seasonal_study_predictions WHERE resolved = 0", conn, parse_dates=["commence_time"])
        if pending.empty:
            return 0
        resolved = 0
        for row in pending[pending["commence_time"] < now].itertuples():
            candidates = matches[
                (matches["team_home"] == row.team_home)
                & (matches["team_away"] == row.team_away)
                & (matches["date"] >= row.commence_time - pd.Timedelta(days=lookback_days))
                & (matches["date"] <= row.commence_time + pd.Timedelta(days=lookback_days))
            ]
            if candidates.empty:
                continue
            outcome = {"H": "home_win", "D": "draw", "A": "away_win"}.get(candidates.iloc[0]["ftr"])
            if outcome is None:
                continue
            conn.execute(
                "UPDATE seasonal_study_predictions SET resolved = 1, actual_outcome = ?, resolved_at = ? WHERE id = ?",
                (outcome, pd.Timestamp.now(tz="UTC").isoformat(), row.id),
            )
            resolved += 1
        return resolved


def live_study_summary() -> list[dict]:
    """Return only settled, pre-kickoff research snapshots for calibration."""
    with _connect() as conn:
        _ensure_study_table(conn)
        rows = pd.read_sql("SELECT * FROM seasonal_study_predictions WHERE resolved = 1", conn)
    if rows.empty:
        return []
    outcome_codes = {"home_win": 0, "draw": 1, "away_win": 2}
    summary = []
    for (cadence, arm), group in rows.groupby(["cadence", "arm"]):
        probabilities = group[["home_win", "draw", "away_win"]].to_numpy(dtype=float, copy=True)
        outcomes = group["actual_outcome"].map(outcome_codes).to_numpy(copy=True)
        summary.append(
            {
                "cadence": cadence,
                "arm": arm,
                "n_fixtures": int(len(group)),
                "rps": float(pb.metrics.rps_average(probabilities, outcomes)),
                "brier": float(pb.metrics.multiclass_brier_score(probabilities, outcomes)),
                "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
            }
        )
    return sorted(summary, key=lambda item: (item["cadence"], item["rps"]))


def live_model_cohorts(manifest_history: list[dict]) -> list[dict]:
    """Score only resolved, genuinely live 1X2 snapshots by model version."""
    with _connect() as conn:
        rows = pd.read_sql(
            "SELECT event_id, outcome_name, predicted_prob, actual_outcome, model_trained_at, backfilled "
            "FROM predictions WHERE market = '1x2' AND resolved = 1 AND backfilled = 0",
            conn,
        )
    if rows.empty:
        return []
    current_matches = {entry["trained_at"]: entry.get("n_current_season_matches") for entry in manifest_history}
    cohorts = []
    for trained_at, group in rows.groupby("model_trained_at", dropna=False):
        fixtures = []
        for _, fixture in group.groupby("event_id"):
            probabilities = {row.outcome_name: row.predicted_prob for row in fixture.itertuples()}
            actual = fixture[fixture["actual_outcome"] == 1].iloc[0]["outcome_name"]
            if set(probabilities) != {"home_win", "draw", "away_win"}:
                continue
            fixtures.append(([probabilities["home_win"], probabilities["draw"], probabilities["away_win"]], {"home_win": 0, "draw": 1, "away_win": 2}[actual]))
        if not fixtures:
            continue
        probs = np.array([item[0] for item in fixtures])
        outcomes = np.array([item[1] for item in fixtures])
        cohorts.append({
            "model_trained_at": trained_at,
            "current_season_matches_seen": current_matches.get(trained_at),
            "n_fixtures": len(fixtures),
            "rps": float(pb.metrics.rps_average(probs, outcomes)),
            "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
        })
    return sorted(cohorts, key=lambda item: item["model_trained_at"] or "")
