import pandas as pd

from pl_predictor.api import routes
from pl_predictor.api.routes import _resolve_current_gameweek


def _matches(rows):
    return pd.DataFrame(rows)


def test_stays_on_current_gameweek_while_it_still_has_unfinished_matches():
    matches = _matches([
        {"matchday": 3, "finished": True, "commence_time": "2026-09-05T15:00:00Z"},
        {"matchday": 3, "finished": False, "commence_time": "2026-09-06T15:00:00Z"},
        {"matchday": 4, "finished": False, "commence_time": "2026-09-12T15:00:00Z"},
    ])
    now = pd.Timestamp("2026-09-06T10:00:00Z")

    assert _resolve_current_gameweek(3, matches, now=now) == 3


def test_advances_to_next_gameweek_the_day_before_it_kicks_off():
    matches = _matches([
        {"matchday": 3, "finished": True, "commence_time": "2026-09-05T15:00:00Z"},
        {"matchday": 3, "finished": True, "commence_time": "2026-09-06T15:00:00Z"},
        {"matchday": 4, "finished": False, "commence_time": "2026-09-12T15:00:00Z"},
    ])
    # Gameweek 3 is fully finished; gameweek 4's first kickoff is under 24h away.
    now = pd.Timestamp("2026-09-11T20:00:00Z")

    assert _resolve_current_gameweek(3, matches, now=now) == 4


def test_advances_anytime_on_the_calendar_day_before_kickoff():
    # Real incident: kickoff is 2026-09-12T14:00Z; checking early on
    # 2026-09-11 (the calendar day before) is >24h away by a rolling
    # window, which used to wrongly keep this on gameweek 3 for most of
    # the day even though "the day before" had clearly arrived.
    matches = _matches([
        {"matchday": 3, "finished": True, "commence_time": "2026-09-05T15:00:00Z"},
        {"matchday": 3, "finished": True, "commence_time": "2026-09-06T15:00:00Z"},
        {"matchday": 4, "finished": False, "commence_time": "2026-09-12T14:00:00Z"},
    ])
    now = pd.Timestamp("2026-09-11T09:00:00Z")

    assert _resolve_current_gameweek(3, matches, now=now) == 4


def test_does_not_advance_while_more_than_a_day_before_next_kickoff():
    matches = _matches([
        {"matchday": 3, "finished": True, "commence_time": "2026-09-05T15:00:00Z"},
        {"matchday": 3, "finished": True, "commence_time": "2026-09-06T15:00:00Z"},
        {"matchday": 4, "finished": False, "commence_time": "2026-09-12T15:00:00Z"},
    ])
    now = pd.Timestamp("2026-09-08T09:00:00Z")

    assert _resolve_current_gameweek(3, matches, now=now) == 3


def test_does_not_advance_past_a_gameweek_with_no_matches_yet():
    matches = _matches([
        {"matchday": 3, "finished": True, "commence_time": "2026-09-05T15:00:00Z"},
        {"matchday": 3, "finished": True, "commence_time": "2026-09-06T15:00:00Z"},
    ])
    now = pd.Timestamp("2026-09-11T09:00:00Z")

    assert _resolve_current_gameweek(3, matches, now=now) == 3


def test_passes_through_none_and_empty_frame_unchanged():
    assert _resolve_current_gameweek(None, pd.DataFrame()) is None
    assert _resolve_current_gameweek(3, pd.DataFrame()) == 3


def test_gameweek_fallback_keeps_a_started_fpl_fixture_visible(monkeypatch):
    """A future-only fallback must not erase a current gameweek's live match."""
    live_fixture = pd.DataFrame(
        [
            {
                "event_id": 11,
                "gameweek": 2,
                "commence_time": pd.Timestamp("2026-08-28T19:00:00Z"),
                "team_home": "Crystal Palace",
                "team_away": "Man City",
                "has_odds": False,
            },
            {
                "event_id": 16,
                "gameweek": 2,
                "commence_time": pd.Timestamp("2026-08-30T13:00:00Z"),
                "team_home": "Chelsea",
                "team_away": "Brighton",
                "has_odds": False,
            },
        ]
    )
    monkeypatch.setattr(routes, "_value_bet_table", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_run_tracking_bookkeeping", lambda _table: None)
    monkeypatch.setattr(routes, "_get_fd_org_matches", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_get_remaining_fixtures_df", lambda: pd.DataFrame())
    monkeypatch.setattr(routes.fixtures_mod, "_fixtures_from_fpl_api", lambda: live_fixture)
    monkeypatch.setattr(routes.tracking_store, "get_track_record", lambda: {"current_gameweek": 2})
    monkeypatch.setattr(
        routes.tracking_store,
        "get_results_by_gameweek",
        lambda: [{
            "gameweek": 2,
            "fixtures": [{
                "event_id": "tracked-11", "team_home": "Crystal Palace", "team_away": "Man City",
                "commence_time": "2026-08-28T19:00:00Z", "actual_goals_home": 1, "actual_goals_away": 4,
                "predicted_home_win": 0.2, "predicted_draw": 0.2, "predicted_away_win": 0.6,
                "predicted_scoreline": "0-2", "hit": True, "backfilled": False,
            }],
        }],
    )
    monkeypatch.setattr(routes.tracking_store, "has_fixture_player_outcomes", lambda _event_id: True)
    monkeypatch.setattr(routes.tracking_store, "get_fixture_player_events", lambda _event_id, _bootstrap: {"home": [], "away": []})
    monkeypatch.setattr(routes, "_get_bootstrap", lambda: {"elements": []})
    monkeypatch.setattr(routes, "_get_models", lambda: {"scoreline": object()})
    monkeypatch.setattr(
        routes.scoreline,
        "predict_fixtures_batch",
        lambda _model, rows, market_overrides=None: [
            {"home_win": 0.5, "draw": 0.25, "away_win": 0.25, "top_scorelines": [{"home": 1, "away": 0}]}
            for _ in range(len(rows))
        ],
    )

    result = routes.current_gameweek_fixtures()

    assert [(fixture["team_home"], fixture["team_away"]) for fixture in result["fixtures"]] == [
        ("Crystal Palace", "Man City"),
        ("Chelsea", "Brighton"),
    ]
