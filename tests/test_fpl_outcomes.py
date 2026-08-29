import pandas as pd
import requests

from pl_predictor.data import fpl_api


def test_fixture_player_outcomes_uses_fixture_explanations_for_double_gameweeks(monkeypatch):
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}],
        "elements": [{"id": 7, "team": 1}],
    }
    monkeypatch.setattr(fpl_api, "fetch_fixtures", lambda: [{
        "id": 44, "team_h": 1, "team_a": 2, "event": 3, "finished": True,
        "kickoff_time": "2026-08-22T15:00:00Z",
    }])
    monkeypatch.setattr(fpl_api, "fetch_event_live", lambda event: {"elements": [
        {"id": 7, "explain": [
            {"fixture": 44, "stats": [
                {"identifier": "goals_scored", "value": 1},
                {"identifier": "assists", "value": 0},
                {"identifier": "starts", "value": 1},
            ]},
            {"fixture": 99, "stats": [{"identifier": "goals_scored", "value": 2}]},
        ]},
    ]})

    outcomes = fpl_api.fixture_player_outcomes("Arsenal", "Chelsea", pd.Timestamp("2026-08-22T15:00:00Z"), bootstrap)

    assert outcomes == {7: {"goals": 1, "assists": 0, "started": True, "was_home": True}}


def test_fixture_player_outcomes_uses_available_event_data_before_fpl_marks_fixture_finished(monkeypatch):
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}],
        "elements": [{"id": 7, "team": 1}],
    }
    monkeypatch.setattr(fpl_api, "fetch_fixtures", lambda: [{
        "id": 44, "team_h": 1, "team_a": 2, "event": 3, "finished": False,
        "team_h_score": 2, "team_a_score": 1, "kickoff_time": "2026-08-22T15:00:00Z",
    }])
    monkeypatch.setattr(fpl_api, "fetch_event_live", lambda event: {"elements": [
        {"id": 7, "explain": [{"fixture": 44, "stats": [
            {"identifier": "goals_scored", "value": 1}, {"identifier": "starts", "value": 1},
        ]}]},
    ]})

    outcomes = fpl_api.fixture_player_outcomes("Arsenal", "Chelsea", pd.Timestamp("2026-08-22T15:00:00Z"), bootstrap)

    assert outcomes == {7: {"goals": 1, "assists": 0, "started": True, "was_home": True}}


def test_fixture_player_outcomes_falls_back_to_cached_player_history(monkeypatch, tmp_path):
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}],
        "elements": [{"id": 7, "team": 1}],
    }
    monkeypatch.setattr(fpl_api, "FPL_PLAYER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fpl_api, "fetch_fixtures", lambda: [{
        "id": 44, "team_h": 1, "team_a": 2, "event": 3, "finished": True,
        "kickoff_time": "2026-08-22T15:00:00Z",
    }])
    monkeypatch.setattr(fpl_api, "fetch_event_live", lambda event: (_ for _ in ()).throw(requests.RequestException("unavailable")))
    (tmp_path / "7.json").write_text('{"history": [{"fixture": 44, "goals_scored": 1, "assists": 1, "starts": 0, "minutes": 78}]}')

    outcomes = fpl_api.fixture_player_outcomes("Arsenal", "Chelsea", pd.Timestamp("2026-08-22T15:00:00Z"), bootstrap)

    assert outcomes == {7: {"goals": 1, "assists": 1, "started": True, "was_home": True}}


def test_fixture_player_outcomes_identifies_cached_fixture_without_fpl_fixture_lookup(monkeypatch, tmp_path):
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}],
        "elements": [{"id": 7, "team": 1}],
    }
    monkeypatch.setattr(fpl_api, "FPL_PLAYER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fpl_api, "fetch_fixtures", lambda: (_ for _ in ()).throw(requests.RequestException("offline")))
    (tmp_path / "7.json").write_text(
        '{"history": [{"fixture": 44, "kickoff_time": "2026-08-22T15:00:00Z", "opponent_team": 2, "was_home": true, "goals_scored": 1, "assists": 1, "starts": 1, "minutes": 90}]}'
    )

    outcomes = fpl_api.fixture_player_outcomes("Arsenal", "Chelsea", pd.Timestamp("2026-08-22T15:00:00Z"), bootstrap)

    assert outcomes == {7: {"goals": 1, "assists": 1, "started": True, "was_home": True}}


def test_fixture_player_outcomes_matches_cached_fixture_by_final_score_when_team_mapping_is_missing(monkeypatch, tmp_path):
    bootstrap = {"teams": [], "elements": []}
    monkeypatch.setattr(fpl_api, "FPL_PLAYER_CACHE_DIR", tmp_path)
    (tmp_path / "7.json").write_text(
        '{"history": [{"fixture": 44, "kickoff_time": "2026-08-22T15:00:00Z", "was_home": true, "goals_scored": 2, "assists": 1, "starts": 1, "minutes": 90}]}'
    )
    (tmp_path / "8.json").write_text(
        '{"history": [{"fixture": 44, "kickoff_time": "2026-08-22T15:00:00Z", "was_home": false, "goals_scored": 1, "assists": 0, "starts": 1, "minutes": 90}]}'
    )

    outcomes = fpl_api.fixture_player_outcomes(
        "Arsenal", "Chelsea", pd.Timestamp("2026-08-22T15:00:00Z"), bootstrap, expected_score=(2, 1)
    )

    assert outcomes == {
        7: {"goals": 2, "assists": 1, "started": True, "was_home": True},
        8: {"goals": 1, "assists": 0, "started": True, "was_home": False},
    }


def test_fetch_fixtures_uses_the_last_successful_cache_when_fpl_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(fpl_api, "FPL_EVENT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fpl_api.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    (tmp_path / "fixtures.json").write_text('[{"id": 44}]')

    assert fpl_api.fetch_fixtures() == [{"id": 44}]


def test_next_event_is_preferred_over_finished_current_event():
    bootstrap = {"events": [
        {"id": 2, "is_current": True, "finished": True},
        {"id": 3, "is_next": True, "finished": False},
    ]}
    assert fpl_api.get_current_event(bootstrap) == 3


def test_entry_picks_falls_back_from_unpublished_next_gameweek(monkeypatch):
    calls = []
    def fetch(entry_id, event_id):
        calls.append((entry_id, event_id))
        if event_id == 3:
            raise requests.HTTPError("not published")
        return {"picks": [{"element": i} for i in range(1, 16)]}

    monkeypatch.setattr(fpl_api, "fetch_entry_picks", fetch)
    payload, source_event = fpl_api.fetch_latest_entry_picks(
        42, 3, {"events": [{"id": 2, "is_current": True}, {"id": 1, "finished": True}]}
    )
    assert source_event == 2
    assert len(payload["picks"]) == 15
    assert calls == [(42, 3), (42, 2)]
