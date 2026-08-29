"""build_snapshot's reuse-from-previous-run logic: a fixture that was
already finished in the previous snapshot is permanently decided (its
result can't change again), so it should be copied over as-is rather than
recomputed. Still-upcoming fixtures, and any fixture that just finished
since the last run, must still go through fixture_detail/fixture_players
each time. This matters specifically because GitHub Actions checks out a
fresh environment on every scheduled run — data/public_snapshot.json is
the only thing carried forward between runs, so this reuse is the only
thing standing between "recompute the entire season every run" and
something that scales down as fixtures get decided."""

import pandas as pd

from pl_predictor import public_snapshot
from pl_predictor.api import routes


def _fake_fixtures_by_gameweek(finished_ids: set[str]) -> dict:
    def fixture(event_id: str, gw: int) -> dict:
        return {"event_id": event_id, "gameweek": gw, "finished": event_id in finished_ids}

    return {
        "1": {"fixtures": [fixture("gw1-a", 1), fixture("gw1-b", 1)]},
        "2": {"fixtures": [fixture("gw2-a", 2), fixture("gw2-b", 2)]},
    }


def _patch_common(monkeypatch, fixtures_by_gameweek: dict, detail_calls: list, players_calls: list, review_calls: list | None = None):
    monkeypatch.setattr(routes, "_get_fd_org_matches", lambda: pd.DataFrame({"matchday": [1, 2]}))
    monkeypatch.setattr(routes.tracking_store, "get_track_record", lambda: {"current_gameweek": 2})
    monkeypatch.setattr(routes, "_resolve_current_gameweek", lambda *_a, **_k: 2)
    monkeypatch.setattr(routes, "current_gameweek_fixtures", lambda gameweek: fixtures_by_gameweek[str(gameweek)])
    review_calls = review_calls if review_calls is not None else []

    def fake_detail(event_id, read_only=False):
        detail_calls.append(event_id)
        return {"event_id": event_id, "computed": "fresh"}

    def fake_players(event_id, read_only=False):
        players_calls.append(event_id)
        return {"event_id": event_id, "computed": "fresh"}

    def fake_review(event_id):
        review_calls.append(event_id)
        return {"event_id": event_id, "computed": "fresh"}

    monkeypatch.setattr(routes, "fixture_detail", fake_detail)
    monkeypatch.setattr(routes, "fixture_players", fake_players)
    monkeypatch.setattr(routes, "fixture_player_review", fake_review)
    monkeypatch.setattr(routes, "get_power_rankings", lambda: {})
    monkeypatch.setattr(routes, "get_projected_table", lambda: {})
    monkeypatch.setattr(routes, "get_hub_track_record", lambda: {})
    monkeypatch.setattr(routes, "get_team_hub", lambda: {})
    monkeypatch.setattr(routes, "get_player_hub", lambda: {})


def test_reuses_previously_finished_fixtures(monkeypatch):
    fixtures_by_gameweek = _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"})
    detail_calls: list[str] = []
    players_calls: list[str] = []
    _patch_common(monkeypatch, fixtures_by_gameweek, detail_calls, players_calls)

    previous = {
        "fixtures_by_gameweek": _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"}),
        "fixture_detail_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "computed": "previous-run"},
            "gw1-b": {"event_id": "gw1-b", "computed": "previous-run"},
        },
        "fixture_players_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "computed": "previous-run"},
        },
    }

    snapshot = public_snapshot.build_snapshot(previous)

    # Already-finished fixtures come back untouched from the previous run.
    assert snapshot["fixture_detail_by_event_id"]["gw1-a"]["computed"] == "previous-run"
    assert snapshot["fixture_detail_by_event_id"]["gw1-b"]["computed"] == "previous-run"
    assert "gw1-a" not in detail_calls
    assert "gw1-b" not in detail_calls

    # Still-upcoming fixtures (gw2) are always recomputed.
    assert snapshot["fixture_detail_by_event_id"]["gw2-a"]["computed"] == "fresh"
    assert snapshot["fixture_detail_by_event_id"]["gw2-b"]["computed"] == "fresh"
    assert set(detail_calls) == {"gw2-a", "gw2-b"}

    # A finished fixture missing from the previous run's players map (a
    # prior partial failure) is recomputed rather than silently dropped.
    assert snapshot["fixture_players_by_event_id"]["gw1-b"]["computed"] == "fresh"
    assert "gw1-b" in players_calls
    assert "gw1-a" not in players_calls


def test_recomputes_a_fixture_that_just_finished(monkeypatch):
    """A fixture finished *now* but not yet in the previous snapshot (it
    was still upcoming last run) must be recomputed once to capture its
    real result — never reused from a stale pre-match prediction."""
    fixtures_by_gameweek = _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"})
    detail_calls: list[str] = []
    players_calls: list[str] = []
    _patch_common(monkeypatch, fixtures_by_gameweek, detail_calls, players_calls)

    previous = {
        "fixtures_by_gameweek": _fake_fixtures_by_gameweek(finished_ids={"gw1-a"}),  # gw1-b was still upcoming
        "fixture_detail_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "computed": "previous-run"},
            "gw1-b": {"event_id": "gw1-b", "computed": "previous-run"},
        },
        "fixture_players_by_event_id": {},
    }

    snapshot = public_snapshot.build_snapshot(previous)

    assert snapshot["fixture_detail_by_event_id"]["gw1-a"]["computed"] == "previous-run"
    assert snapshot["fixture_detail_by_event_id"]["gw1-b"]["computed"] == "fresh"
    assert "gw1-b" in detail_calls
    assert "gw1-a" not in detail_calls


def test_retries_finished_detail_until_delayed_box_score_is_available(monkeypatch):
    fixtures_by_gameweek = _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"})
    detail_calls: list[str] = []
    players_calls: list[str] = []
    _patch_common(monkeypatch, fixtures_by_gameweek, detail_calls, players_calls)

    previous = {
        "fixtures_by_gameweek": _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"}),
        "fixture_detail_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "actual_stats": None},
            "gw1-b": {"event_id": "gw1-b", "actual_stats": {"home": {"Goals": 1}, "away": {"Goals": 0}}, "pre_match_value_bets": []},
        },
        "fixture_players_by_event_id": {},
    }

    snapshot = public_snapshot.build_snapshot(previous)

    assert snapshot["fixture_detail_by_event_id"]["gw1-a"]["computed"] == "fresh"
    assert "gw1-a" in detail_calls
    assert "gw1-b" not in detail_calls


def test_player_review_reused_and_backfilled_like_players(monkeypatch):
    """player_review_by_event_id must follow the exact same reuse rule as
    fixture_players_by_event_id: reused when the previous run already has
    it for a fixture that's finished in both runs, recomputed otherwise —
    this is what makes /fixtures/{id}/player-review serve real data
    instead of 404ing in PUBLIC_MODE (see routes.py)."""
    fixtures_by_gameweek = _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"})
    detail_calls: list[str] = []
    players_calls: list[str] = []
    review_calls: list[str] = []
    _patch_common(monkeypatch, fixtures_by_gameweek, detail_calls, players_calls, review_calls)

    previous = {
        "fixtures_by_gameweek": _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"}),
        "fixture_detail_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "computed": "previous-run"},
            "gw1-b": {"event_id": "gw1-b", "computed": "previous-run"},
        },
        "fixture_players_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "computed": "previous-run"},
            "gw1-b": {"event_id": "gw1-b", "computed": "previous-run"},
        },
        "player_review_by_event_id": {
            "gw1-a": {"event_id": "gw1-a", "computed": "previous-run"},
        },
    }

    snapshot = public_snapshot.build_snapshot(previous)

    assert snapshot["player_review_by_event_id"]["gw1-a"]["computed"] == "previous-run"
    assert "gw1-a" not in review_calls
    assert snapshot["player_review_by_event_id"]["gw1-b"]["computed"] == "fresh"
    assert "gw1-b" in review_calls
    # Still-upcoming fixtures (gw2) never get a review lookup at all — the
    # frontend only ever requests player-review once a fixture is finished
    # (gated on fixture_detail.post_match), so computing it for an
    # upcoming fixture would be pure waste (confirmed the hard way: doing
    # this unconditionally for the whole season is what turned one snapshot
    # run into a multi-hour hang the first time this field was added).
    assert "gw2-a" not in review_calls
    assert "gw2-b" not in review_calls
    assert "gw2-a" not in snapshot["player_review_by_event_id"]
    assert "gw2-b" not in snapshot["player_review_by_event_id"]


def test_no_previous_snapshot_recomputes_everything(monkeypatch):
    fixtures_by_gameweek = _fake_fixtures_by_gameweek(finished_ids={"gw1-a", "gw1-b"})
    detail_calls: list[str] = []
    players_calls: list[str] = []
    _patch_common(monkeypatch, fixtures_by_gameweek, detail_calls, players_calls)

    snapshot = public_snapshot.build_snapshot(None)

    assert set(detail_calls) == {"gw1-a", "gw1-b", "gw2-a", "gw2-b"}
    assert all(v["computed"] == "fresh" for v in snapshot["fixture_detail_by_event_id"].values())
