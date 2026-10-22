"""Per-market confidence buckets for the track-record chart: a hit is judged on the market's own pick."""
import sqlite3

import pytest

from pl_predictor.tracking import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "t.db")
    store._connect().close()                       # creates the schema
    return tmp_path / "t.db"


def _row(db, event, market, outcome, prob, actual, gh=None, ga=None):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO predictions (event_id, team_home, team_away, market, outcome_name, predicted_prob, commence_time, snapshotted_at, "
        "resolved, actual_outcome, actual_goals_home, actual_goals_away) VALUES (?,?,?,?,?,?,?,?,1,?,?,?)",
        (event, "H", "A", market, outcome, prob, "2026-01-01T15:00:00Z", "2026-01-01T10:00:00Z", actual, gh, ga),
    )
    conn.commit()
    conn.close()


def test_a_fixture_is_one_pick_at_its_top_outcomes_probability(db):
    # home 62% / draw 23% / away 15%, home won: one pick, bucket 60-70%, right
    for outcome, p, a in (("home_win", 0.62, 1), ("draw", 0.23, 0), ("away_win", 0.15, 0)):
        _row(db, "e1", "1x2", outcome, p, a)
    # a second fixture: home 45% / draw 30% / away 25% -> below 50%, not a bucketed pick
    for outcome, p, a in (("home_win", 0.45, 0), ("draw", 0.30, 1), ("away_win", 0.25, 0)):
        _row(db, "e2", "1x2", outcome, p, a)
    assert store.get_calibration()["match_result"] == [{"bucket": "60-70%", "n": 1, "correct": 1}]


def test_over_under_and_btts_are_judged_on_the_side_the_model_favoured(db):
    for outcome, p, a in (("over", 0.72, 0), ("under", 0.28, 1)):        # model said over at 72%, it was under: wrong
        _row(db, "e1", "totals_2_5", outcome, p, a)
    _row(db, "e1", "btts", "yes", 0.30, 0, gh=1, ga=0)                     # model favoured NO at 70%, no BTTS: right
    cal = store.get_calibration()
    assert cal["over_under_2_5"] == [{"bucket": "70%+", "n": 1, "correct": 0}]
    assert cal["btts"] == [{"bucket": "70%+", "n": 1, "correct": 1}]


def test_an_incomplete_fixture_and_an_empty_store_say_nothing(db):
    _row(db, "e1", "1x2", "home_win", 0.6, 1)                              # only one of three outcomes stored: not a fixture pick
    assert store.get_calibration()["match_result"] == []
    assert store.get_calibration().get("btts") == []
