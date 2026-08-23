"""commence_time filtering — a fixture that's already kicked off must never
appear as "upcoming," regardless of what the upstream API returns."""

import pandas as pd

from pl_predictor.data.fixtures import _future_only


def test_drops_already_kicked_off_fixture():
    df = pd.DataFrame(
        [
            {"event_id": "past", "commence_time": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)},
            {"event_id": "future", "commence_time": pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)},
        ]
    )
    out = _future_only(df)
    assert list(out["event_id"]) == ["future"]


def test_handles_naive_timestamps_too():
    df = pd.DataFrame(
        [
            {"event_id": "past", "commence_time": pd.Timestamp.now() - pd.Timedelta(hours=1)},
            {"event_id": "future", "commence_time": pd.Timestamp.now() + pd.Timedelta(days=1)},
        ]
    )
    out = _future_only(df)
    assert list(out["event_id"]) == ["future"]


def test_empty_df_passthrough():
    df = pd.DataFrame(columns=["event_id", "commence_time"])
    assert _future_only(df).empty
