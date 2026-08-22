"""Round-trip check for the live prediction track record store."""

import pandas as pd
import pytest

from pl_predictor.config import TRACKING_DB_PATH
from pl_predictor.tracking import store


@pytest.fixture
def clean_db():
    if TRACKING_DB_PATH.exists():
        TRACKING_DB_PATH.unlink()
    yield
    if TRACKING_DB_PATH.exists():
        TRACKING_DB_PATH.unlink()


def test_record_is_idempotent(clean_db):
    table = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"),
                "home_win_prob": 0.6,
                "draw_prob": 0.25,
                "away_win_prob": 0.15,
                "over_2_5_prob": 0.7,
                "under_2_5_prob": 0.3,
                "btts_yes_prob": 0.55,
            }
        ]
    )
    assert store.record_predictions(table) == 6
    assert store.record_predictions(table) == 0  # already logged, no-op


def test_reconcile_resolves_against_actual_result(clean_db):
    table = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "commence_time": pd.Timestamp("2020-01-01T15:00:00Z"),
                "home_win_prob": 0.6,
                "draw_prob": 0.25,
                "away_win_prob": 0.15,
                "over_2_5_prob": 0.7,
                "under_2_5_prob": 0.3,
                "btts_yes_prob": 0.55,
            }
        ]
    )
    store.record_predictions(table)

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
    assert store.reconcile_predictions(matches_df) == 6

    record = store.get_track_record()
    assert record["n_resolved"] == 6
    assert record["rps"] is not None
    assert record["brier"] is not None

    misses = store.get_biggest_misses()
    assert len(misses) == 6
    assert all(m["actual_outcome"] in (0, 1) for m in misses)
    # sorted worst-first by squared error
    errors = [(m["predicted_prob"] - m["actual_outcome"]) ** 2 for m in misses]
    assert errors == sorted(errors, reverse=True)
