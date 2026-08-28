"""other_competitions.py builds the fixture-congestion calendar (Champions
League, Europa League, Conference League, FA Cup, EFL Cup) that
`features/rest_days.py`/`features/fixture_congestion.py` need. Monkeypatches
each source's own public function (no HTTP mocking infrastructure exists in
this test suite — same lightweight style as `test_football_data.py`)."""

import pandas as pd
import requests

from pl_predictor.data import football_data_org, other_competitions


def test_to_team_rows_keeps_only_english_clubs():
    matches = pd.DataFrame(
        [
            {"team_home": "Arsenal", "team_away": "Real Madrid", "commence_time": "2025-09-16T16:45:00Z"},
            {"team_home": "Bayern Munich", "team_away": "PSG", "commence_time": "2025-09-17T19:00:00Z"},
        ]
    )
    result = other_competitions._to_team_rows(matches, "Champions League")
    assert list(result["team"]) == ["Arsenal"]
    assert list(result["competition"]) == ["Champions League"]


def test_to_team_rows_handles_empty_input():
    result = other_competitions._to_team_rows(pd.DataFrame(), "Champions League")
    assert result.empty
    assert list(result.columns) == ["team", "date", "competition"]


def test_fetch_champions_league_matches_degrades_when_key_missing(monkeypatch, tmp_path):
    # Cache is checked before any network call — point it at a fresh tmp
    # dir so this doesn't pick up real data already cached from a previous
    # live run.
    monkeypatch.setattr(other_competitions, "OTHER_COMPETITIONS_CACHE_DIR", tmp_path)

    def _raise(competition_id, season=None):
        raise football_data_org.FootballDataOrgKeyMissing("no key")

    monkeypatch.setattr(football_data_org, "fetch_matches", _raise)
    result = other_competitions.fetch_champions_league_matches()
    assert result.empty


def test_fetch_champions_league_matches_skips_seasons_beyond_free_tier_depth(monkeypatch, tmp_path):
    """The free tier 403s for a season too far back (confirmed live) — that
    must not abort the whole fetch, just that one season."""
    monkeypatch.setattr(other_competitions, "OTHER_COMPETITIONS_CACHE_DIR", tmp_path)

    def _fetch(competition_id, season=None):
        if season == other_competitions.football_data.CURRENT_SEASON_START_YEAR - 2:
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError(response=response)
        return pd.DataFrame(
            [{"event_id": 1, "team_home": "Arsenal", "team_away": "Real Madrid", "commence_time": "2025-09-16T16:45:00Z"}]
        )

    monkeypatch.setattr(football_data_org, "fetch_matches", _fetch)
    result = other_competitions.fetch_champions_league_matches()
    assert not result.empty
    assert set(result["team"]) == {"Arsenal"}


def test_fetch_cup_matches_degrades_when_espn_unreachable(monkeypatch):
    def _raise(label, slug):
        raise AssertionError("should never propagate a raw exception out of fetch_cup_matches")

    def _empty(label, slug):
        return other_competitions._EMPTY.copy()

    monkeypatch.setattr(other_competitions, "_fetch_espn_cup", _empty)
    result = other_competitions.fetch_cup_matches()
    assert result.empty
    assert list(result.columns) == ["team", "date", "competition"]


def test_get_team_fixture_calendar_combines_and_dedupes(monkeypatch):
    cl_df = pd.DataFrame(
        [
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-16"), "competition": "Champions League"},
            {"team": "Arsenal", "date": pd.Timestamp("2025-09-16"), "competition": "Champions League"},
        ]
    )
    cup_df = pd.DataFrame(
        [{"team": "Chelsea", "date": pd.Timestamp("2025-09-24"), "competition": "EFL Cup"}]
    )
    monkeypatch.setattr(other_competitions, "fetch_champions_league_matches", lambda: cl_df)
    monkeypatch.setattr(other_competitions, "fetch_cup_matches", lambda: cup_df)

    result = other_competitions.get_team_fixture_calendar()
    assert len(result) == 2
    assert set(result["team"]) == {"Arsenal", "Chelsea"}


def test_get_team_fixture_calendar_empty_when_all_sources_empty(monkeypatch):
    monkeypatch.setattr(other_competitions, "fetch_champions_league_matches", lambda: other_competitions._EMPTY.copy())
    monkeypatch.setattr(other_competitions, "fetch_cup_matches", lambda: other_competitions._EMPTY.copy())

    result = other_competitions.get_team_fixture_calendar()
    assert result.empty
    assert list(result.columns) == ["team", "date", "competition"]
