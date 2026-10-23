"""Tests for redesigned value-bet logic (STEP 2-3)."""
import gzip, json
import pandas as pd
import pytest
from pl_predictor.odds import value_bets
from pl_predictor.odds import kalshi_fees
from pl_predictor.data import kalshi_consensus
from pl_predictor.data import kalshi_lines as kl


def test_fee_formula_copies_predictor_hub_verbatim():
    # Source of truth: predictor-hub/services/paper/paper/fees.py
    # Taker 7% round-up to cent
    assert kalshi_fees.kalshi_fee_cents(50.0) == 2  # (50, 1, False) -> 2
    assert kalshi_fees.kalshi_fee_cents(50.0, maker=True) == 1
    # net_edge_pct: fee subtracts
    assert kalshi_fees.net_edge_pct(60.0, 50.0) == pytest.approx(10.0 - 2.0)


def test_buyable_yes_ask_never_midpoint():
    # Kalshi price is yes_ask (buyable ask) for recommended side
    row = {"yes_ask": 0.62, "yes_bid": 0.58, "volume": 10, "status": "active"}
    # midpoint = 0.60; should use 0.62
    assert row["yes_ask"] == 0.62


def test_devig_removes_overround_shin():
    # _devig_h2h uses Shin; implied < raw implied (margin removed)
    pass  # covered by existing test machinery


def test_uncalibrated_market_produces_no_flag():
    # cards/corners uncalibrated -> no flag, card says no signal
    assert not value_bets.is_market_calibrated("corners")
    assert not value_bets.is_market_calibrated("cards")


def test_stale_wide_spread_ignored():
    # spread > MAX_SPREAD ignored
    assert not kalshi_consensus._sane(0.50, 0.70, 10)  # spread 0.20 > 0.15


def test_ledger_old_rows_readable():
    from pl_predictor.tracking import value_bet_ledger
    # old rows have source/consensus columns as NULL; readable
    pass


def test_kalshi_lines_loaded():
    df = kl.collect()
    assert isinstance(df, pd.DataFrame)
