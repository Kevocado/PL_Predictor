"""Proactive lineup-aware player-prediction refresh (see
routes.py::refresh_lineups_near_kickoff). Confirmed lineups are typically
posted ~60 minutes before kickoff; without this, a stale
fixture_players:{event_id} cache entry only rebuilds on the next request
that happens to arrive after its 5-minute TTL lapses -- which might be
never, if nobody opens that fixture again before kickoff."""

import pandas as pd

from pl_predictor.api import routes


def test_lineup_refresh_checkpoint_fires_once_per_offset_when_crossed():
    already_checked: set[tuple[str, int]] = set()

    # Not yet within 90 minutes -- no checkpoint due.
    assert routes._lineup_refresh_checkpoint(95.0, "e1", (90, 30), already_checked) is None

    # Crossed the 90-minute mark -- fires, and is recorded so it won't fire again.
    assert routes._lineup_refresh_checkpoint(88.0, "e1", (90, 30), already_checked) == 90
    assert ("e1", 90) in already_checked

    # Still within the 90-minute window on a later tick -- already fired, no repeat.
    assert routes._lineup_refresh_checkpoint(70.0, "e1", (90, 30), already_checked) is None

    # Crossed the 30-minute mark too -- fires once for that checkpoint.
    assert routes._lineup_refresh_checkpoint(28.0, "e1", (90, 30), already_checked) == 30
    assert ("e1", 30) in already_checked

    # Both checkpoints now spent -- nothing left to fire for this fixture.
    assert routes._lineup_refresh_checkpoint(5.0, "e1", (90, 30), already_checked) is None


def test_lineup_refresh_checkpoint_is_independent_per_fixture():
    already_checked: set[tuple[str, int]] = set()
    assert routes._lineup_refresh_checkpoint(88.0, "e1", (90, 30), already_checked) == 90
    # A different fixture crossing the same threshold still fires -- keyed
    # by (event_id, offset), not offset alone.
    assert routes._lineup_refresh_checkpoint(88.0, "e2", (90, 30), already_checked) == 90


def test_refresh_lineups_near_kickoff_force_rebuilds_due_fixtures(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    view = {
        "fixtures": [
            {
                "event_id": "due", "team_home": "Arsenal", "team_away": "Chelsea",
                "commence_time": (now + pd.Timedelta(minutes=85)).isoformat(), "finished": False,
            },
            {
                "event_id": "not-due", "team_home": "Fulham", "team_away": "Brentford",
                "commence_time": (now + pd.Timedelta(minutes=150)).isoformat(), "finished": False,
            },
            {
                "event_id": "already-played", "team_home": "Hull", "team_away": "Leeds",
                "commence_time": (now - pd.Timedelta(minutes=10)).isoformat(), "finished": True,
            },
        ]
    }
    monkeypatch.setattr(routes, "current_gameweek_fixtures", lambda: view)
    monkeypatch.setattr(routes, "_lineup_checks_done", set())

    rebuilt = []

    def fake_rank(event_id, home, away):
        rebuilt.append(event_id)
        return []

    monkeypatch.setattr(routes, "_rank_fixture_players", fake_rank)

    routes.refresh_lineups_near_kickoff()

    assert rebuilt == ["due"]
    assert ("due", 90) in routes._lineup_checks_done


def test_refresh_lineups_near_kickoff_skips_one_broken_fixture_without_blocking_others(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    view = {
        "fixtures": [
            {
                "event_id": "broken", "team_home": "A", "team_away": "B",
                "commence_time": (now + pd.Timedelta(minutes=85)).isoformat(), "finished": False,
            },
            {
                "event_id": "fine", "team_home": "C", "team_away": "D",
                "commence_time": (now + pd.Timedelta(minutes=25)).isoformat(), "finished": False,
            },
        ]
    }
    monkeypatch.setattr(routes, "current_gameweek_fixtures", lambda: view)
    monkeypatch.setattr(routes, "_lineup_checks_done", set())

    rebuilt = []

    def flaky_rank(event_id, home, away):
        if event_id == "broken":
            raise RuntimeError("espn down")
        rebuilt.append(event_id)
        return []

    monkeypatch.setattr(routes, "_rank_fixture_players", flaky_rank)

    routes.refresh_lineups_near_kickoff()

    assert rebuilt == ["fine"]
