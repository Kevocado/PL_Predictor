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

from ..config import TRACKING_DB_PATH
from ..models import scoreline

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
    # Migration: actual_goals_home/away didn't exist in the original schema
    # — added so recent-results views can show the real scoreline next to
    # the predicted probabilities, not just per-market hit/miss booleans.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    for col in ("actual_goals_home", "actual_goals_away"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} INTEGER")
    # backfilled: 0 for a prediction actually captured live before kickoff,
    # 1 for one computed after the fact by backfill_missing_predictions (see
    # its docstring for why this distinction stays visible rather than
    # blurring the two).
    if "backfilled" not in existing_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN backfilled INTEGER NOT NULL DEFAULT 0")
    # gameweek: only football-data.org's `matchday` carries this (football-
    # data.co.uk's CSVs have no equivalent column), so it's written in at
    # reconcile time rather than snapshot time, and stays NULL for anything
    # resolved via the football-data.co.uk fallback path. predicted_scoreline
    # is the model's single most-likely scoreline at prediction time (already
    # computed by odds/value_bets.py::build_value_bet_table's "top_scoreline"
    # for live predictions, and by backfill_missing_predictions's own
    # scoreline.predict_fixture call for backfilled ones) — stored so the
    # Recent Results view can show it next to the actual score without
    # recomputing anything.
    for col, col_type in (("gameweek", "INTEGER"), ("predicted_scoreline", "TEXT")):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
    return conn


def record_predictions(table: pd.DataFrame, model_trained_at: str | None = None, backfilled: bool = False) -> int:
    """Snapshot current predictions for every fixture in `table` (the
    value-bet table already built by `odds/value_bets.py`, or the
    equivalent shape `backfill_missing_predictions` builds). Already-logged
    (event_id, market, outcome_name) rows are left untouched — this is safe
    to call on every fixtures request."""
    if table.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, fixture in table.iterrows():
        predicted_scoreline = fixture.get("top_scoreline")
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
                    int(backfilled),
                    predicted_scoreline,
                )
            )

    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO predictions
                (event_id, team_home, team_away, market, outcome_name, predicted_prob,
                 commence_time, snapshotted_at, model_trained_at, backfilled, predicted_scoreline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    `actual_outcome`. Returns how many rows were resolved this call.

    `matches_df` carrying a `matchday` column (currently only
    football-data.org's `fetch_matches` provides one — football-data.co.uk
    has no equivalent) also fills in `gameweek` on resolve; without it,
    `gameweek` is left NULL rather than guessed."""
    has_matchday = "matchday" in matches_df.columns
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
            gameweek = int(match["matchday"]) if has_matchday and pd.notna(match["matchday"]) else None
            conn.execute(
                """
                UPDATE predictions
                SET resolved = 1, actual_outcome = ?, resolved_at = ?,
                    actual_goals_home = ?, actual_goals_away = ?, gameweek = ?
                WHERE id = ?
                """,
                (
                    actual,
                    datetime.now(timezone.utc).isoformat(),
                    int(match["goals_home"]),
                    int(match["goals_away"]),
                    gameweek,
                    int(pred["id"]),
                ),
            )
            resolved_count += 1

        return resolved_count


def _finished_matches_lookup(finished_matches: pd.DataFrame) -> set[tuple]:
    return {(r["team_home"], r["team_away"], _naive(r["date"]).date()) for _, r in finished_matches.iterrows()}


def has_unlogged_finished_matches(finished_matches: pd.DataFrame) -> bool:
    """Cheap pre-check for `backfill_missing_predictions`: same team+date
    matching it uses, but stops at the first gap instead of computing
    predictions. `finished_matches` needs `team_home`, `team_away`, `date`
    columns — pass whichever source has the freshest results (see
    `backfill_missing_predictions`'s docstring)."""
    if finished_matches.empty:
        return False

    with _connect() as conn:
        existing = pd.read_sql(
            "SELECT DISTINCT team_home, team_away, commence_time FROM predictions", conn, parse_dates=["commence_time"]
        )
    already_logged = {(r["team_home"], r["team_away"], _naive(r["commence_time"]).date()) for _, r in existing.iterrows()}

    return any(key not in already_logged for key in _finished_matches_lookup(finished_matches))


def backfill_missing_predictions(finished_matches: pd.DataFrame, model, model_trained_at: str | None = None) -> int:
    """One-time (and self-healing) catch-up for matches that finished
    before this app was polling `/api/fixtures` to snapshot them live —
    e.g. the handful of matches already played when prediction tracking
    first started this season, or any result that lands while the app is
    down. `finished_matches` needs `team_home`, `team_away`, `date`,
    `goals_home`, `goals_away`, `ftr` — pass whichever source has the
    freshest results (typically football-data.org, which tends to reflect
    a finished match faster than football-data.co.uk's own scrape).

    Computes what the model would have predicted via its ordinary
    live-serving `.predict(home, away)` path (the same one genuine
    upcoming fixtures use) rather than a point-in-time historical row —
    that's what makes this leakage-safe *specifically* for a match still
    missing from the backfill: the whole reason it needs backfilling is
    that football-data.co.uk (which the model's live context is built
    from) doesn't have this match yet, so "current state" at the moment
    this runs genuinely reflects "as of just before this match," the same
    as it would for any other fixture that hasn't been played yet. Once
    football-data.co.uk does catch up and the match's own stats flow into
    a future context rebuild, that's fine — this function has already
    logged the prediction it needs to for reconciliation, and later
    rebuilds don't retroactively change what was recorded here. Always
    marked `backfilled=True` (see `_connect`'s docstring note) so the
    Recent Results view can keep these visually distinct from predictions
    actually captured live before kickoff — a live-captured prediction
    proves no hindsight was even possible in principle; a backfilled one
    is honestly computed the same way but wasn't literally there before
    kickoff, and that distinction is worth keeping legible rather than
    blurring the two."""
    if finished_matches.empty:
        return 0

    with _connect() as conn:
        existing = pd.read_sql(
            "SELECT DISTINCT team_home, team_away, commence_time FROM predictions", conn, parse_dates=["commence_time"]
        )
    already_logged = {(r["team_home"], r["team_away"], _naive(r["commence_time"]).date()) for _, r in existing.iterrows()}

    prediction_rows, match_rows = [], []
    for _, row in finished_matches.iterrows():
        key = (row["team_home"], row["team_away"], _naive(row["date"]).date())
        if key in already_logged:
            continue
        pred = scoreline.predict_fixture(model, row["team_home"], row["team_away"])
        event_id = f"backfill-{row['team_home']}-{row['team_away']}-{key[2]}"
        top = pred["top_scorelines"][0]
        prediction_rows.append(
            {
                "event_id": event_id,
                "team_home": row["team_home"],
                "team_away": row["team_away"],
                "commence_time": row["date"],
                "home_win_prob": pred["home_win"],
                "draw_prob": pred["draw"],
                "away_win_prob": pred["away_win"],
                "over_2_5_prob": pred["over_2_5"],
                "under_2_5_prob": pred["under_2_5"],
                "btts_yes_prob": pred["btts_yes"],
                "top_scoreline": f"{top['home']}-{top['away']}",
            }
        )
        match_row = {
            "team_home": row["team_home"],
            "team_away": row["team_away"],
            # reconcile_predictions compares this directly against a
            # naive window (it reads commence_time back from SQLite,
            # which record_predictions always stores naive) — a
            # tz-aware "date" here (e.g. from football-data.org) would
            # raise "Invalid comparison between dtype=datetime64[us,
            # UTC] and Timestamp".
            "date": _naive(row["date"]),
            "goals_home": row["goals_home"],
            "goals_away": row["goals_away"],
            "ftr": row["ftr"],
        }
        if "matchday" in row.index:
            match_row["matchday"] = row["matchday"]
        match_rows.append(match_row)

    if not prediction_rows:
        return 0

    n = record_predictions(pd.DataFrame(prediction_rows), model_trained_at=model_trained_at, backfilled=True)
    reconcile_predictions(pd.DataFrame(match_rows), lookback_days=0)
    return n


def _fixture_hit_table() -> pd.DataFrame:
    """One row per resolved fixture (not per market row): team_home/away,
    commence_time, resolved_at, gameweek, predicted_scoreline, actual score,
    predicted H/D/A probabilities, actual_outcome (side name), hit (whether
    the model's highest-probability side matches what happened), backfilled.
    Shared by `get_track_record`/`get_results_by_gameweek`/
    `get_biggest_upsets` so "correct" is defined exactly once. Only fixtures
    with all three 1x2 rows resolved are included (matches the old
    `get_recent_results`'s behavior)."""
    with _connect() as conn:
        resolved = pd.read_sql(
            "SELECT * FROM predictions WHERE resolved = 1 AND market = '1x2'",
            conn,
            parse_dates=["commence_time", "resolved_at"],
        )
    if resolved.empty:
        return pd.DataFrame()

    rows = []
    for event_id, group in resolved.groupby("event_id"):
        probs = {r["outcome_name"]: float(r["predicted_prob"]) for _, r in group.iterrows()}
        if set(probs) != {"home_win", "draw", "away_win"}:
            continue
        first = group.iloc[0]
        hit_row = group[group["actual_outcome"] == 1]
        actual_side = hit_row.iloc[0]["outcome_name"] if not hit_row.empty else None
        predicted_side = max(probs, key=probs.get)
        rows.append(
            {
                "event_id": event_id,
                "team_home": first["team_home"],
                "team_away": first["team_away"],
                "commence_time": first["commence_time"],
                "resolved_at": first["resolved_at"],
                "gameweek": _none_if_nan_int(first["gameweek"]),
                "predicted_scoreline": first["predicted_scoreline"],
                "actual_goals_home": _none_if_nan_int(first["actual_goals_home"]),
                "actual_goals_away": _none_if_nan_int(first["actual_goals_away"]),
                "predicted_home_win": probs["home_win"],
                "predicted_draw": probs["draw"],
                "predicted_away_win": probs["away_win"],
                "predicted_prob_actual": probs.get(actual_side) if actual_side else None,
                "actual_outcome": actual_side,
                "hit": bool(actual_side is not None and predicted_side == actual_side),
                "backfilled": bool(first["backfilled"]),
            }
        )
    return pd.DataFrame(rows)


def get_track_record() -> dict:
    fixtures = _fixture_hit_table()
    if fixtures.empty:
        return {
            "n_resolved_fixtures": 0,
            "pct_correct_overall": None,
            "current_gameweek": None,
            "pct_correct_current_gameweek": None,
            "n_fixtures_current_gameweek": 0,
            "gameweek_trend": [],
        }

    n_resolved = int(len(fixtures))
    pct_correct_overall = float(fixtures["hit"].mean())

    with_gw = fixtures[fixtures["gameweek"].notna()]
    current_gameweek = int(with_gw["gameweek"].max()) if not with_gw.empty else None
    if current_gameweek is not None:
        this_gw = with_gw[with_gw["gameweek"] == current_gameweek]
        pct_correct_current_gameweek = float(this_gw["hit"].mean())
        n_fixtures_current_gameweek = int(len(this_gw))
    else:
        pct_correct_current_gameweek = None
        n_fixtures_current_gameweek = 0

    gameweek_trend = []
    if not with_gw.empty:
        grouped = with_gw.groupby("gameweek")["hit"].agg(["mean", "size"]).reset_index()
        gameweek_trend = [
            {"gameweek": int(r["gameweek"]), "pct_correct": float(r["mean"]), "n_fixtures": int(r["size"])}
            for _, r in grouped.sort_values("gameweek").iterrows()
        ]

    return {
        "n_resolved_fixtures": n_resolved,
        "pct_correct_overall": pct_correct_overall,
        "current_gameweek": current_gameweek,
        "pct_correct_current_gameweek": pct_correct_current_gameweek,
        "n_fixtures_current_gameweek": n_fixtures_current_gameweek,
        "gameweek_trend": gameweek_trend,
    }


def get_biggest_upsets(limit: int = 5) -> list[dict]:
    """Genuine 1x2 upsets only — the side that actually happened, ranked by
    how unlikely the model said it was. Deliberately not "biggest squared
    error across all markets": that would resurface confident-but-wrong
    goals/BTTS misses, which aren't what "upset" means and aren't tracked by
    this view anymore."""
    fixtures = _fixture_hit_table()
    fixtures = fixtures[fixtures["actual_outcome"].notna()] if not fixtures.empty else fixtures
    if fixtures.empty:
        return []

    worst = fixtures.sort_values("predicted_prob_actual", ascending=True).head(limit)
    return [
        {
            "team_home": r["team_home"],
            "team_away": r["team_away"],
            "commence_time": r["commence_time"].isoformat(),
            "gameweek": r["gameweek"],
            "actual_goals_home": r["actual_goals_home"],
            "actual_goals_away": r["actual_goals_away"],
            "actual_outcome": r["actual_outcome"],
            "predicted_prob": float(r["predicted_prob_actual"]),
        }
        for _, r in worst.iterrows()
    ]


def get_results_by_gameweek() -> list[dict]:
    """Resolved fixtures grouped by gameweek, most recent gameweek first
    (fixtures with no gameweek data — resolved via the football-data.co.uk
    fallback, which has no matchday column — bucketed last under `gameweek:
    null`). Each group also carries its own `pct_correct`/`n_fixtures` so the
    frontend can render a per-gameweek header without a second aggregation."""
    fixtures = _fixture_hit_table()
    if fixtures.empty:
        return []

    fixtures = fixtures.sort_values("commence_time", ascending=False)
    groups = []
    for gameweek, group in fixtures.groupby(fixtures["gameweek"].fillna(-1)):
        gw_value = None if gameweek == -1 else int(gameweek)
        groups.append(
            {
                "gameweek": gw_value,
                "pct_correct": float(group["hit"].mean()),
                "n_fixtures": int(len(group)),
                "fixtures": [
                    {
                        "event_id": r["event_id"],
                        "team_home": r["team_home"],
                        "team_away": r["team_away"],
                        "commence_time": r["commence_time"].isoformat(),
                        "predicted_scoreline": r["predicted_scoreline"],
                        "actual_goals_home": r["actual_goals_home"],
                        "actual_goals_away": r["actual_goals_away"],
                        "predicted_home_win": r["predicted_home_win"],
                        "predicted_draw": r["predicted_draw"],
                        "predicted_away_win": r["predicted_away_win"],
                        "actual_outcome": r["actual_outcome"],
                        "hit": r["hit"],
                        "backfilled": r["backfilled"],
                    }
                    for _, r in group.iterrows()
                ],
            }
        )

    groups.sort(key=lambda g: (g["gameweek"] is None, -(g["gameweek"] or 0)))
    return groups


def get_fixture_prediction(event_id: str) -> dict | None:
    """All markets' honestly pre-match-recorded probabilities for one
    fixture, by event_id. Used to show a finished fixture's prediction
    (e.g. reached via the gameweek view) without recomputing it live —
    for an already-played match, live recomputation risks that match's
    own result having fed back into "current model state," which would
    make the "prediction" shown partly hindsight. None if this event_id
    was never logged."""
    with _connect() as conn:
        rows = pd.read_sql(
            "SELECT * FROM predictions WHERE event_id = ?", conn, params=(event_id,), parse_dates=["commence_time"]
        )
    if rows.empty:
        return None

    probs = {r["outcome_name"]: float(r["predicted_prob"]) for _, r in rows.iterrows()}
    first = rows.iloc[0]
    return {
        "event_id": event_id,
        "team_home": first["team_home"],
        "team_away": first["team_away"],
        "commence_time": first["commence_time"].isoformat(),
        "home_win": probs.get("home_win", 0.0),
        "draw": probs.get("draw", 0.0),
        "away_win": probs.get("away_win", 0.0),
        "over_2_5": probs.get("over", 0.0),
        "under_2_5": probs.get("under", 0.0),
        "btts_yes": probs.get("yes", 0.0),
        "predicted_scoreline": first["predicted_scoreline"],
    }


def _none_if_nan_int(value) -> int | None:
    return None if pd.isna(value) else int(value)
