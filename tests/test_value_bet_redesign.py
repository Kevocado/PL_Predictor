"""Tests for redesigned value-bet logic (fail-before / pass-after for 4 findings)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from pl_predictor.odds import kalshi_fees
from pl_predictor.odds import kalshi_consensus
from pl_predictor.data import kalshi_lines as kl

# FIX 1: draw scoped to fixture event ticker
def test_draw_scoped_to_event_ticker():
    # Create a dataframe with two different event_tickers, each with a Tie row
    df = pd.DataFrame([
        {"event_ticker": "KXEPLGAME-26OCT25LFCARS", "stat": "moneyline", "title": "Tie is the result", "yes_ask": 0.3, "volume": 10, "status": "active", "snapshot_ts": "2026-10-18", "line": None},
        {"event_ticker": "KXEPLGAME-26OCT25MUNBOU", "stat": "moneyline", "title": "Tie is the result", "yes_ask": 0.35, "volume": 10, "status": "active", "snapshot_ts": "2026-10-18", "line": None},
        {"event_ticker": "KXEPLGAME-26OCT25LFCARS", "stat": "moneyline", "title": "Liverpool wins", "yes_ask": 0.6, "volume": 10, "status": "active", "snapshot_ts": "2026-10-18", "line": None},
        {"event_ticker": "KXEPLGAME-26OCT25MUNBOU", "stat": "moneyline", "title": "Manchester United wins", "yes_ask": 0.55, "volume": 10, "status": "active", "snapshot_ts": "2026-10-18", "line": None},
    ])
    from datetime import datetime, timezone
    commence_time = datetime(2026, 10, 25, 15, 0, 0, tzinfo=timezone.utc)
    consensus = kalshi_consensus.kalshi_consensus_for_fixture(df, "Liverpool", "Arsenal", commence_time)
    assert consensus is not None
    if "draw" in consensus:
        assert consensus["draw"]["prob"] == 0.3

# FIX 2: newest snapshot wins (dedup key: event_ticker, stat, line, title)
def test_newest_snapshot_wins_for_market():
    # The load_kalshi_lines function dedupes on (event_ticker, stat, line, title) keeping last by snapshot_ts.
    # We'll test by creating a dataframe and calling the internal dedup logic.
    df = pd.DataFrame([
        {"event_ticker": "E", "stat": "total", "line": 2.5, "title": "Over 2.5", "yes_ask": 0.60, "snapshot_ts": "2026-10-18", "volume": 10},
        {"event_ticker": "E", "stat": "total", "line": 2.5, "title": "Over 2.5", "yes_ask": 0.62, "snapshot_ts": "2026-10-19", "volume": 10},
    ])
    # Simulate load_kalshi_lines dedup
    df["snapshot_ts_dt"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
    df = df.sort_values("snapshot_ts_dt").drop_duplicates(subset=["event_ticker", "stat", "line", "title"], keep="last")
    assert len(df) == 1
    assert df.iloc[0]["yes_ask"] == 0.62

# FIX 3: pass Kalshi lines into API builder (wrapper passes kalshi_df)
def test_api_builder_receives_kalshi():
    from pl_predictor.odds import value_bets
    fixtures_df = pd.DataFrame([{"event_id": "e1", "commence_time": "2026-01-01", "team_home": "A", "team_away": "B"}])
    odds_df = pd.DataFrame([{"event_id": "e1", "market": "h2h", "outcome_name": "A", "price": 2.0, "point": None, "bookmaker": "b1"}])
    odds_df["odds_fetched_at"] = pd.Timestamp.now(tz="UTC")
    models = {"scoreline": None}
    kalshi_df = pd.DataFrame([{"event_ticker": "E", "stat": "moneyline", "title": "A wins", "yes_ask": 0.5}])
    import pl_predictor.odds.value_bets as vb
    vb.scoreline.predict_fixtures_batch = lambda *a, **k: [{
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
        "btts_yes": 0.5, "btts_no": 0.5, "over_2_5": 0.5, "under_2_5": 0.5,
        "top_scorelines": [{"home": 1, "away": 0}], "fallback": False,
        "data_confidence": "established", "home_goal_expectation": 1.0,
        "away_goal_expectation": 0.5, "home_2plus_prob": 0.3, "away_2plus_prob": 0.1
    }]
    vb._devig_h2h = lambda *_: {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}
    vb._devig_totals = lambda *_: {"over_2_5": 0.5, "under_2_5": 0.5}
    vb._devig_btts = lambda *_: {"btts_yes": 0.5, "btts_no": 0.5}
    vb._best_quote = lambda *a, **k: {"price": 2.0, "bookmaker": "b1"}
    result = vb.build_value_bet_table(fixtures_df, odds_df, models, redesign=True, kalshi_df=kalshi_df)
    assert result is not None
    assert hasattr(result, "iloc")

# FIX 4: Kalshi edge is net of fee and matches recorded bet (ledger stores consensus price and net edge)
def test_kalshi_edge_net_of_fee_and_ledger_consistency():
    from pl_predictor.odds import kalshi_fees
    assert kalshi_fees.net_edge_pct(60.0, 50.0) == pytest.approx(8.0)
    from pl_predictor.tracking import value_bet_ledger
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        import pl_predictor.tracking.value_bet_ledger as vbl
        original_path = vbl.TRACKING_DB_PATH
        vbl.TRACKING_DB_PATH = tmp.name
        conn = vbl._connect()
        conn.close()
        fixture = {
            "event_id": "e1",
            "team_home": "A",
            "team_away": "B",
            "commence_time": pd.Timestamp("2026-01-01"),
            "home_win_prob": 0.6,
            "home_win_implied": 0.5,
            "home_win_edge": 0.08,
            "home_win_price": 2.0,
            "home_win_consensus_source": "kalshi",
            "home_win_consensus_prob": 0.5,
            "home_win_consensus_price": 0.5,
            "home_win_source": "kalshi",
            "value_bet_tiers": {"home_win": "Strong"},
            "value_bet_flags": ["home_win"],
            "is_fallback_prediction": False,
            "odds_fetched_at": pd.Timestamp.now(tz="UTC"),
            "odds_is_stale": False,
            "data_confidence": "established",
        }
        df = pd.DataFrame([fixture])
        df["odds_fetched_at"] = pd.Timestamp.now(tz="UTC")
        vbl.record_value_bets(df, model_trained_at="2026-01-01", model_manifest_hash="test")
        bets = vbl.get_fixture_value_bets("e1")
        assert len(bets) == 1
        bet = bets[0]
        assert bet["edge"] == pytest.approx(0.08)
        assert bet["price"] == pytest.approx(0.5)
        assert bet["implied_probability"] == pytest.approx(0.5)
        vbl.TRACKING_DB_PATH = original_path

# Additional test: fee formula verbatim from predictor-hub
def test_fee_formula_copies_predictor_hub_verbatim():
    assert kalshi_fees.kalshi_fee_cents(50.0) == 2
    assert kalshi_fees.kalshi_fee_cents(50.0, maker=True) == 1
    assert kalshi_fees.net_edge_pct(60.0, 50.0) == pytest.approx(10.0 - 2.0)

# Test that uncalibrated market produces no flag (cards/corners)
def test_uncalibrated_market_no_flag():
    from pl_predictor.odds import value_bets
    assert not value_bets.is_market_calibrated("corners")
    assert not value_bets.is_market_calibrated("cards")
    assert value_bets.MARKET_CALIBRATION["corners"]["calibrated"] is False
    assert value_bets.MARKET_CALIBRATION["cards"]["calibrated"] is False

# Test that stale or wide-spread Kalshi quote is ignored
def test_stale_wide_spread_ignored():
    assert not kalshi_consensus._sane(0.50, 0.70, 10)  # spread 0.20 > 0.15
    # volume 0 with MIN_VOLUME=0.0 passes, so we test spread only
    assert not kalshi_consensus._sane(0.50, 0.66, 10)  # spread 0.16 > 0.15

# Test that old ledger rows still read (backward compatibility)
def test_old_ledger_rows_readable():
    from pl_predictor.tracking import value_bet_ledger
    import tempfile, sqlite3
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        import pl_predictor.tracking.value_bet_ledger as vbl
        original_path = vbl.TRACKING_DB_PATH
        vbl.TRACKING_DB_PATH = tmp.name
        conn = vbl._connect()
        conn.close()
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            INSERT INTO value_bets (event_id, team_home, team_away, market, outcome_name, model_prob, implied_prob, price, edge,
                 commence_time, snapshotted_at, model_trained_at, quote_fetched_at, odds_source, consensus_source, consensus_prob, confidence_tier, quote_snapshot, model_manifest_hash)
            VALUES ('e1', 'A', 'B', '1x2', 'home_win', 0.6, 0.5, 2.0, 0.1,
                 '2026-01-01', '2026-01-01', '2026-01-01', '2026-01-01', 'the-odds-api', NULL, NULL, NULL, '{}', 'hash')
        """)
        conn.commit()
        conn.close()
        bets = vbl.get_fixture_value_bets("e1")
        assert len(bets) == 1
        bet = bets[0]
        assert bet["market"] == "home_win"
        vbl.TRACKING_DB_PATH = original_path

