"""A missing FOOTBALL_DATA_KEY isn't the only way football-data.org calls
fail -- a real incident (DNS resolution failure) took down
/api/projected-table with an unhandled ConnectionError because
_get_fd_org_standings only ever caught FootballDataOrgKeyMissing. Every
build() closure that calls into football_data_org must degrade the same
way on any requests failure, not just a missing key."""

import pandas as pd
import requests

from pl_predictor.api import routes
from pl_predictor.data.football_data_org import FootballDataOrgKeyMissing


def _clear():
    routes._clear_cache("fd_org_matches", "fd_org_standings", "remaining_fixtures_df")


def test_fd_org_matches_falls_back_on_request_exception(monkeypatch):
    _clear()
    monkeypatch.setattr(
        routes.football_data_org, "fetch_matches", lambda: (_ for _ in ()).throw(requests.ConnectionError("dns fail"))
    )
    result = routes._get_fd_org_matches()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_fd_org_standings_falls_back_on_request_exception(monkeypatch):
    _clear()
    monkeypatch.setattr(
        routes.football_data_org, "fetch_standings", lambda: (_ for _ in ()).throw(requests.ConnectionError("dns fail"))
    )
    result = routes._get_fd_org_standings()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_remaining_fixtures_falls_back_to_fpl_on_request_exception(monkeypatch):
    _clear()
    monkeypatch.setattr(
        routes.football_data_org, "fetch_matches", lambda: (_ for _ in ()).throw(requests.ConnectionError("dns fail"))
    )
    fallback = pd.DataFrame([{"event_id": "fpl-1", "team_home": "Arsenal", "team_away": "Chelsea"}])
    monkeypatch.setattr(routes.fixtures_mod, "get_all_remaining_fixtures", lambda: fallback)

    result = routes._get_remaining_fixtures_df()

    assert list(result["event_id"]) == ["fpl-1"]


def test_remaining_fixtures_falls_back_to_fpl_on_missing_key(monkeypatch):
    _clear()
    monkeypatch.setattr(
        routes.football_data_org, "fetch_matches", lambda: (_ for _ in ()).throw(FootballDataOrgKeyMissing("no key"))
    )
    fallback = pd.DataFrame([{"event_id": "fpl-1", "team_home": "Arsenal", "team_away": "Chelsea"}])
    monkeypatch.setattr(routes.fixtures_mod, "get_all_remaining_fixtures", lambda: fallback)

    result = routes._get_remaining_fixtures_df()

    assert list(result["event_id"]) == ["fpl-1"]
