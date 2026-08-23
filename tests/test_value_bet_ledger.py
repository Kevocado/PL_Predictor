"""Round-trip check for the live value-bet track record ledger."""

import pandas as pd
import pytest

from pl_predictor.tracking import value_bet_ledger


@pytest.fixture
def clean_db(tmp_path, monkeypatch):
    # Patch the name as imported into value_bet_ledger, not config's — see
    # tests/test_tracking.py's clean_db for why (same real-DB-wipe bug class,
    # avoided here from the start by never touching the production path).
    monkeypatch.setattr(value_bet_ledger, "TRACKING_DB_PATH", tmp_path / "test_tracking.db")
    yield


def _flagged_table(**overrides) -> pd.DataFrame:
    row = {
        "event_id": "e1",
        "team_home": "Arsenal",
        "team_away": "Chelsea",
        "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"),
        "home_win_prob": 0.6,
        "home_win_implied": 0.5,
        "home_win_edge": 0.1,
        "home_win_price": 2.2,
        "under_2_5_prob": 0.6,
        "under_2_5_implied": 0.5,
        "under_2_5_edge": 0.1,
        "under_2_5_price": 1.9,
        "value_bet_flags": ["home_win", "under_2_5"],
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_record_is_idempotent(clean_db):
    table = _flagged_table()
    assert value_bet_ledger.record_value_bets(table) == 2  # one row per flagged market
    assert value_bet_ledger.record_value_bets(table) == 0  # already logged, no-op


def test_only_flagged_sides_are_recorded(clean_db):
    table = _flagged_table(value_bet_flags=["home_win"])
    assert value_bet_ledger.record_value_bets(table) == 1


def test_reconcile_and_replay_matches_offline_backtest_semantics(clean_db):
    table = _flagged_table()
    value_bet_ledger.record_value_bets(table)

    matches_df = pd.DataFrame(
        [
            {
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "date": pd.Timestamp("2020-01-01"),
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
            }
        ]
    )
    # home_win: H happened -> won. under_2_5: total goals = 3, over 2.5 -> lost.
    assert value_bet_ledger.reconcile_value_bets(matches_df) == 2

    record = value_bet_ledger.get_value_bet_track_record(staking="flat", bankroll=100.0, flat_stake=10.0)
    assert record["n_flagged"] == 2
    assert record["n_resolved"] == 2
    assert record["n_pending"] == 0
    assert record["results"]["Total Bets"] == 2
    assert record["results"]["Successful Bets"] == 1
    # home_win won at price 2.2 on a 10-unit flat stake: +12. under_2_5 lost: -10. Net +2.
    assert record["results"]["Profit"] == pytest.approx(2.0)
    assert record["bankroll_curve"][-1] == pytest.approx(102.0)


def test_reconcile_resolves_against_tz_aware_matches_df(clean_db):
    table = _flagged_table(value_bet_flags=["home_win"])
    value_bet_ledger.record_value_bets(table)

    matches_df = pd.DataFrame(
        [
            {
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "date": pd.Timestamp("2020-01-01T15:00:00", tz="UTC"),
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
            }
        ]
    )
    assert value_bet_ledger.reconcile_value_bets(matches_df) == 1


def test_pending_bets_excluded_from_pl(clean_db):
    table = _flagged_table(value_bet_flags=["home_win"])
    value_bet_ledger.record_value_bets(table)

    record = value_bet_ledger.get_value_bet_track_record()
    assert record["n_flagged"] == 1
    assert record["n_pending"] == 1
    assert record["n_resolved"] == 0
    assert record["results"]["Total Bets"] == 0
