"""value_bet_ledger.py — SQLite persistence for the live value-bet track
record: did the fixtures actually flagged as "Value bet" in the app go on
to be profitable, not just whether the model's calibration looks fine in
the abstract?

Same honesty discipline as `tracking/store.py`'s match-prediction ledger:
every flagged (fixture, market) pair is snapshotted once, the first time
`odds/value_bets.py::build_value_bet_table` flags it — including the raw
bookmaker price at that moment, not just the de-vigged implied probability
used for edge detection, since price is what a real bet actually pays out
on — then reconciled against the real result once the match finishes.
Never re-flagged or recomputed after the fact.

Unlike `evaluate/backtest.py`'s synthetic held-out-season replay (one bet
per fixture — the single largest edge, since home/draw/away are mutually
exclusive outcomes of the same event), this tracks *every* flag the app
actually surfaced to the user, including cases where two independent
markets on the same fixture (e.g. home_win and under_2_5) were both
flagged — that's the honest question this answers: are the recommendations
actually shown to the user valuable, not "what's the single best bet per
match." The P&L simulation itself (staking, bankroll accounting) reuses
`penaltyblog.backtest.Account` directly and the same tenth-Kelly default as
the offline backtest, so the two numbers are directly comparable."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import penaltyblog as pb

from ..config import TRACKING_DB_PATH
from ..evaluate.backtest import _kelly_stake
from .store import _actual_outcome, _naive

# side (as used in odds/value_bets.py's value_bet_flags) -> (market,
# outcome_name), matching tracking/store.py's MARKET_SPEC vocabulary so
# _actual_outcome can be reused unmodified for reconciliation.
_SIDE_TO_MARKET = {
    "home_win": ("1x2", "home_win"),
    "draw": ("1x2", "draw"),
    "away_win": ("1x2", "away_win"),
    "over_2_5": ("totals_2_5", "over"),
    "under_2_5": ("totals_2_5", "under"),
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS value_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            team_home TEXT NOT NULL,
            team_away TEXT NOT NULL,
            market TEXT NOT NULL,
            outcome_name TEXT NOT NULL,
            model_prob REAL NOT NULL,
            implied_prob REAL NOT NULL,
            price REAL NOT NULL,
            edge REAL NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            won INTEGER,
            resolved_at TEXT,
            UNIQUE(event_id, market)
        )
        """
    )
    return conn


def record_value_bets(table: pd.DataFrame) -> int:
    """Snapshot every (fixture, market) pair `table["value_bet_flags"]`
    already flagged — `table` is the same value-bet table built by
    `odds/value_bets.py::build_value_bet_table` that `/api/fixtures` and
    `/api/fixtures/gameweek` already serve, so this costs nothing extra to
    call alongside `tracking_store.record_predictions`. Already-logged
    (event_id, market) pairs are left untouched — a flag's price/edge is
    whatever it was the *first* time it was seen, never updated as odds
    move, same idempotency as record_predictions."""
    if table.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, fixture in table.iterrows():
        flags = fixture.get("value_bet_flags") or []
        for side in flags:
            if side not in _SIDE_TO_MARKET:
                continue
            price = fixture.get(f"{side}_price")
            if price is None or pd.isna(price):
                continue  # shouldn't happen for a flagged side, but a flag with no price can't be tracked as a real bet
            market, outcome_name = _SIDE_TO_MARKET[side]
            rows.append(
                (
                    str(fixture["event_id"]),
                    fixture["team_home"],
                    fixture["team_away"],
                    market,
                    outcome_name,
                    float(fixture[f"{side}_prob"]),
                    float(fixture[f"{side}_implied"]),
                    float(price),
                    float(fixture[f"{side}_edge"]),
                    _naive(fixture["commence_time"]).isoformat(),
                    now,
                )
            )

    if not rows:
        return 0

    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO value_bets
                (event_id, team_home, team_away, market, outcome_name, model_prob, implied_prob, price, edge,
                 commence_time, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def reconcile_value_bets(matches_df: pd.DataFrame, lookback_days: int = 3) -> int:
    """Same team+date matching as `tracking_store.reconcile_predictions`,
    resolving `won` (1/0) for every flagged bet whose kickoff has passed.
    `matches_df` needs `team_home`, `team_away`, `date`, `goals_home`,
    `goals_away`, `ftr` — pass whichever source has the freshest results."""
    now = _naive(pd.Timestamp.now(tz="UTC"))

    # Same tz-aware-vs-naive fix as reconcile_predictions: football-data.org's
    # `date` column is tz-aware UTC, but commence_time is always stored naive
    # here (see record_value_bets), so comparing them directly raises.
    if isinstance(matches_df["date"].dtype, pd.DatetimeTZDtype):
        matches_df = matches_df.copy()
        matches_df["date"] = matches_df["date"].dt.tz_localize(None)

    with _connect() as conn:
        unresolved = pd.read_sql(
            "SELECT * FROM value_bets WHERE resolved = 0", conn, parse_dates=["commence_time"]
        )
        if unresolved.empty:
            return 0

        due = unresolved[unresolved["commence_time"] < now]
        if due.empty:
            return 0

        resolved_count = 0
        for _, bet in due.iterrows():
            window_start = bet["commence_time"] - pd.Timedelta(days=lookback_days)
            window_end = bet["commence_time"] + pd.Timedelta(days=lookback_days)
            candidates = matches_df[
                (matches_df["team_home"] == bet["team_home"])
                & (matches_df["team_away"] == bet["team_away"])
                & (matches_df["date"] >= window_start)
                & (matches_df["date"] <= window_end)
            ]
            if candidates.empty:
                continue

            match = candidates.iloc[0]
            won = _actual_outcome(bet["market"], bet["outcome_name"], match)
            conn.execute(
                "UPDATE value_bets SET resolved = 1, won = ?, resolved_at = ? WHERE id = ?",
                (won, datetime.now(timezone.utc).isoformat(), int(bet["id"])),
            )
            resolved_count += 1

        return resolved_count


def get_value_bet_track_record(
    staking: str = "kelly",
    bankroll: float = 100.0,
    kelly_fraction: float = 0.10,
    max_stake_fraction: float = 0.05,
    flat_stake: float = 1.0,
    max_odds: float | None = 6.0,
) -> dict:
    """Replays every *resolved* flagged value bet in chronological order
    (by kickoff), applying the same staking rule as the offline backtest
    (`evaluate/backtest.py` — Kelly or flat, same tenth-Kelly default, same
    max_odds cutoff for the tail-risk reason documented there) via
    `penaltyblog.backtest.Account` directly, so the two numbers are
    genuinely comparable. Pending (not-yet-played) flags are reported
    separately and excluded from the P&L — an open position, not a result."""
    with _connect() as conn:
        df = pd.read_sql("SELECT * FROM value_bets ORDER BY commence_time ASC", conn, parse_dates=["commence_time"])

    if df.empty:
        return {
            "n_flagged": 0,
            "n_resolved": 0,
            "n_pending": 0,
            "results": None,
            "bankroll_curve": [],
            "staking": staking,
        }

    pending = df[df["resolved"] == 0]
    resolved = df[df["resolved"] == 1]

    account = pb.backtest.Account(bankroll)
    for _, bet in resolved.iterrows():
        price = float(bet["price"])
        if max_odds is not None and price > max_odds:
            continue
        if staking == "kelly":
            stake = _kelly_stake(float(bet["model_prob"]), price, kelly_fraction, max_stake_fraction, account.current_bankroll)
            if stake <= 0:
                continue
        else:
            stake = flat_stake
        account.place_bet(price, stake, int(bet["won"]))

    total_bets = len(account.history)
    total_profit = account.current_bankroll - account.bankroll
    successful_bets = sum(h["outcome"] for h in account.history)
    results = {
        "Total Bets": total_bets,
        "Successful Bets": successful_bets,
        "Successful Bet %": (successful_bets / total_bets * 100) if total_bets else 0.0,
        "Max Bankroll": max(account.tracker) if account.tracker else None,
        "Min Bankroll": min(account.tracker) if account.tracker else None,
        "Profit": total_profit,
        "ROI": (total_profit / account.bankroll * 100) if total_bets else 0.0,
    }

    return {
        "n_flagged": int(len(df)),
        "n_resolved": int(len(resolved)),
        "n_pending": int(len(pending)),
        "results": results,
        "bankroll_curve": account.tracker,
        "staking": staking,
    }
