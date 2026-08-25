import pandas as pd

from pl_predictor.api import routes
from pl_predictor.api.schemas import FixturePlayers, PlayerPrediction


def _player(player_id: int, confirmed_starter: bool = True) -> PlayerPrediction:
    return PlayerPrediction(
        player_id=player_id,
        name="Saka",
        position="MID",
        anytime_goal_prob=0.35,
        anytime_assist_prob=0.25,
        anytime_goal_contribution_prob=0.50,
        status="a",
        news="",
        confidence="high",
        predicted_starter=True,
        confirmed_starter=confirmed_starter,
        expected_minutes=90,
        is_penalty_taker=False,
        is_set_piece_taker=False,
    )


def test_final_score_reconciles_player_outcomes_even_when_kickoff_is_future(monkeypatch):
    players = FixturePlayers(home_players=[_player(7, confirmed_starter=False)], away_players=[])
    captured = {}
    monkeypatch.setattr(routes, "_resolve_fixture_kickoff", lambda event_id: pd.Timestamp("2099-08-22T15:00:00Z"))
    monkeypatch.setattr(routes.tracking_store, "get_fixture_post_match", lambda event_id: {"final_score": "2-1"})
    monkeypatch.setattr(routes, "_get_bootstrap", lambda: {"teams": [], "elements": []})
    monkeypatch.setattr(
        routes.fpl_api,
        "fixture_player_outcomes",
        lambda *args, **kwargs: captured.setdefault("outcomes", {7: {"goals": 1, "assists": 0, "started": True}}),
    )
    monkeypatch.setattr(
        routes.tracking_store,
        "record_player_prediction_snapshots",
        lambda event_id, rows, provenance="snapshot": captured.setdefault("record", (event_id, rows, provenance)),
    )
    monkeypatch.setattr(
        routes.tracking_store,
        "reconcile_player_prediction_snapshots",
        lambda event_id, outcomes: captured.setdefault("reconcile", (event_id, outcomes)),
    )
    monkeypatch.setattr(
        routes.tracking_store,
        "record_fixture_player_outcomes",
        lambda event_id, outcomes: captured.setdefault("fixture_outcomes", []).append((event_id, outcomes)),
    )

    routes._snapshot_or_reconcile_player_predictions("event-1", "Arsenal", "Chelsea", players)

    assert captured["record"][0] == "event-1"
    assert captured["record"][2] == "reconstructed"
    assert captured["record"][1][0]["confirmed_starter"] is True
    assert captured["reconcile"] == ("event-1", {7: {"goals": 1, "assists": 0, "started": True}})
    assert captured["fixture_outcomes"] == [
        ("event-1", {7: {"goals": 1, "assists": 0, "started": True}}),
    ]


def test_backfill_fetches_official_outcomes_before_player_ranking(monkeypatch):
    captured = {}
    monkeypatch.setattr(routes.tracking_store, "resolved_fixtures_missing_player_snapshots", lambda gameweek: [{
        "event_id": "event-1", "team_home": "Arsenal", "team_away": "Chelsea", "commence_time": "2026-08-22T15:00:00Z",
    }])
    monkeypatch.setattr(
        routes,
        "_fetch_and_record_fixture_player_outcomes",
        lambda *args: captured.setdefault("outcomes", {7: {"goals": 1, "assists": 0, "started": True, "was_home": True}}),
    )
    monkeypatch.setattr(routes, "_rank_fixture_players", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ranking unavailable")))
    monkeypatch.setattr(routes.tracking_store, "has_player_prediction_snapshots", lambda event_id: False)
    monkeypatch.setattr(routes.tracking_store, "has_fixture_player_outcomes", lambda event_id: True)

    result = routes.backfill_completed_player_reviews(gameweek=1)

    assert captured["outcomes"][7]["goals"] == 1
    assert result == {"attempted": 1, "events_saved": 1, "completed": 0, "unresolved": 1}


def test_fixture_player_payload_is_cached_after_background_build(monkeypatch):
    players = FixturePlayers(home_players=[_player(7)], away_players=[])
    calls = []
    routes._clear_cache("fixture_players:event-1")
    monkeypatch.setattr(routes, "_rank_fixture_players", lambda *args: calls.append(args) or players)

    assert routes._get_cached_fixture_players("event-1", "Arsenal", "Chelsea") is players
    assert routes._get_cached_fixture_players("event-1", "Arsenal", "Chelsea") is players
    assert len(calls) == 1
