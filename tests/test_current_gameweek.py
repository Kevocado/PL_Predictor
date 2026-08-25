import pandas as pd

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
