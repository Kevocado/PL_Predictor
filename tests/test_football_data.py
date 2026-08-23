"""fetch_current_season_partial's pulselive.com fallback — football-data.co.uk's
publishing cadence for the in-progress season is unpredictable (confirmed
directly: it can show zero rows even after a full gameweek has been
played), so this needs test coverage for both branches: prefer
football-data.co.uk when it has data, fall back to pulselive.com when it
doesn't. Uses monkeypatch on each source's own public function (no HTTP
mocking infrastructure exists in this test suite) — same lightweight style
as the rest of this project's tests."""

import pandas as pd
import pytest

from pl_predictor.data import football_data, pulselive


def test_falls_back_to_pulselive_when_football_data_co_uk_has_nothing(monkeypatch):
    def _raise(season, force_refresh=False):
        raise RuntimeError("no file for this season yet")

    monkeypatch.setattr(football_data, "fetch_season", _raise)

    pulselive_df = pd.DataFrame(
        [{"team_home": "Hull", "team_away": "Man United", "goals_home": 2, "goals_away": 0}]
    )
    monkeypatch.setattr(pulselive, "fetch_current_season_matches", lambda: pulselive_df)

    result = football_data.fetch_current_season_partial()
    assert result is not None
    assert list(result["team_home"]) == ["Hull"]


def test_falls_back_to_pulselive_when_football_data_co_uk_is_empty(monkeypatch):
    """An empty (but not erroring) result — e.g. the season file exists but
    has no rows yet — should also trigger the fallback, not just a raised
    exception."""
    empty = pd.DataFrame(columns=["date", "team_home", "team_away", "goals_home", "goals_away"])
    monkeypatch.setattr(football_data, "fetch_season", lambda season, force_refresh=False: empty)

    pulselive_df = pd.DataFrame([{"team_home": "Arsenal", "team_away": "Coventry"}])
    monkeypatch.setattr(pulselive, "fetch_current_season_matches", lambda: pulselive_df)

    result = football_data.fetch_current_season_partial()
    assert result is not None
    assert list(result["team_home"]) == ["Arsenal"]


def test_prefers_football_data_co_uk_when_it_has_rows(monkeypatch):
    """Once football-data.co.uk actually publishes the season, it stays the
    preferred source (richer stats, already the historical baseline) —
    pulselive.com should not even be called."""
    co_uk_df = pd.DataFrame([{"team_home": "Brentford", "team_away": "Tottenham"}])
    monkeypatch.setattr(football_data, "fetch_season", lambda season, force_refresh=False: co_uk_df)

    def _fail_if_called():
        raise AssertionError("pulselive should not be called when football-data.co.uk has rows")

    monkeypatch.setattr(pulselive, "fetch_current_season_matches", _fail_if_called)

    result = football_data.fetch_current_season_partial()
    assert list(result["team_home"]) == ["Brentford"]


def test_returns_none_when_both_sources_have_nothing(monkeypatch):
    def _raise(season, force_refresh=False):
        raise RuntimeError("no file yet")

    monkeypatch.setattr(football_data, "fetch_season", _raise)
    empty = pd.DataFrame(columns=["team_home", "team_away"])
    monkeypatch.setattr(pulselive, "fetch_current_season_matches", lambda: empty)

    assert football_data.fetch_current_season_partial() is None


def test_returns_none_when_pulselive_itself_errors(monkeypatch):
    """A pulselive hiccup (network error, schema change) must never crash
    the caller — same discipline as every other best-effort data source in
    this project."""

    def _raise(season, force_refresh=False):
        raise RuntimeError("no file yet")

    monkeypatch.setattr(football_data, "fetch_season", _raise)

    def _pulselive_fails():
        raise ConnectionError("boom")

    monkeypatch.setattr(pulselive, "fetch_current_season_matches", _pulselive_fails)

    assert football_data.fetch_current_season_partial() is None
