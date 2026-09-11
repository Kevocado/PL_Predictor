"""commence_time filtering — a fixture that's already kicked off must never
appear as "upcoming," regardless of what the upstream API returns."""

import pandas as pd
import requests

from pl_predictor.data import fixtures as fixtures_mod
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


def test_fpl_fallback_keeps_unplayed_fixtures_from_the_in_progress_gameweek(monkeypatch):
    """Real bug: FPL's `is_next` flag points at the gameweek *after* the one
    currently being played — filtering on `event >= is_next` silently
    dropped a current gameweek's own still-unplayed fixtures (e.g. GW2's
    remaining 5 matches disappeared entirely once GW2's first 5 had
    finished, only GW3+ showed). `not finished` alone is already the
    correct, per-fixture test; no gameweek-number filter belongs here."""
    bootstrap = {
        "teams": [{"id": 1, "name": "Team A"}, {"id": 2, "name": "Team B"}],
        "events": [
            {"id": 1, "is_previous": True, "is_current": False, "is_next": False},
            {"id": 2, "is_previous": False, "is_current": True, "is_next": False},
            {"id": 3, "is_previous": False, "is_current": False, "is_next": True},
        ],
    }
    fixtures = [
        {"id": 10, "event": 2, "finished": True, "kickoff_time": "2026-08-01T00:00:00Z", "team_h": 1, "team_a": 2},
        {"id": 11, "event": 2, "finished": False, "kickoff_time": "2026-08-31T00:00:00Z", "team_h": 2, "team_a": 1},
        {"id": 12, "event": 3, "finished": False, "kickoff_time": "2026-09-06T00:00:00Z", "team_h": 1, "team_a": 2},
    ]
    monkeypatch.setattr(fixtures_mod.fpl_api, "fetch_bootstrap", lambda: bootstrap)
    monkeypatch.setattr(fixtures_mod.fpl_api, "fetch_fixtures", lambda: fixtures)

    df = fixtures_mod._fixtures_from_fpl_api()

    assert sorted(df["gameweek"].tolist()) == [2, 3]


def test_falls_back_to_fpl_fixtures_when_odds_api_errors(monkeypatch):
    """Real incident: The Odds API returns 401 (not 429) once its monthly
    free-tier credit quota is exhausted -- indistinguishable, from here, from
    any other HTTP failure (a genuine rate limit, an outage). Only
    `OddsAPIKeyMissing` fell back before this; any other `requests` failure
    used to crash the whole Fixtures tab (and the public-snapshot build)
    instead of degrading to fixtures-with-no-odds the same way a missing key
    already does."""

    def _raise(*args, **kwargs):
        raise requests.HTTPError("401 quota exhausted")

    monkeypatch.setattr(fixtures_mod, "fetch_epl_odds_raw", _raise)
    monkeypatch.setattr(
        fixtures_mod,
        "_fixtures_from_fpl_api",
        lambda: pd.DataFrame(
            [
                {
                    "event_id": "fpl-1",
                    "commence_time": pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1),
                    "team_home": "Arsenal",
                    "team_away": "Chelsea",
                    "has_odds": False,
                }
            ]
        ),
    )

    df = fixtures_mod.get_upcoming_fixtures()

    assert list(df["event_id"]) == ["fpl-1"]
    assert df["has_odds"].tolist() == [False]
