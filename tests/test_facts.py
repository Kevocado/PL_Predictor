"""Tests for the read-only PL /facts bundle the match explainer consumes.

Offline throughout: the public snapshot and the tracking store are injected,
so nothing reaches the Odds API, football-data.org or the model files.

The rule these tests exist to pin down: for a fixture that has STARTED, the
pick comes only from the stored pre-kickoff record. The public snapshot's
fixture DETAIL is recomputed from the current model (it disagrees with the
card for finished fixtures — confirmed against the real snapshot), so it may
never stand in for the stored pick.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, model_validator

from pl_predictor.api import facts as facts_mod
from pl_predictor.api.main import app

# The contract, copied from predictor-hub/services/explainer/explainer/facts.py.
class Market(BaseModel):
    market: str
    model_config = ConfigDict(extra="allow")


class Facts(BaseModel):
    sport: Literal["pl", "f1", "nfl", "cfb", "nba"]
    id: str
    title: str
    starts_at: str
    status: Literal["upcoming", "live", "final"]
    pick_timing: Literal["pre_kickoff", "rebuilt", "none"]
    pick: dict | None = None
    markets: list[Market] = []
    drivers: list[dict] = []
    context: dict = {}
    players: list[dict] = []
    record: dict | None = None
    result: dict | None = None

    @model_validator(mode="after")
    def _rebuilt_never_won(self) -> "Facts":
        if self.pick_timing == "rebuilt" and self.result and "pick_won" in self.result:
            raise ValueError("a rebuilt pick cannot carry result.pick_won")
        return self


NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
EVENT_ID = "100"


def _detail(**over):
    detail = {
        "event_id": EVENT_ID,
        "team_home": "Sunderland",
        "team_away": "Chelsea",
        "commence_time": "2026-11-08T14:00:00Z",
        "home_win": {"prob": 0.36, "implied": None, "edge": None},
        "draw": {"prob": 0.26, "implied": None, "edge": None},
        "away_win": {"prob": 0.38, "implied": None, "edge": None},
        "over_2_5": {"prob": 0.55, "implied": None, "edge": None},
        "under_2_5": {"prob": 0.45, "implied": None, "edge": None},
        "predicted_total_goals": 2.9,
        "btts_yes_prob": 0.57,
        "top_scoreline": "1-1",
        "has_live_odds": False,
        "value_bet_flags": [],
        "home_recent_form": ["W", "D", "L"],
        "away_recent_form": ["L", "W", "W"],
        "data_confidence": "established",
    }
    detail.update(over)
    return detail


def _card(**over):
    card = {
        "event_id": EVENT_ID,
        "team_home": "Sunderland",
        "team_away": "Chelsea",
        "commence_time": "2026-11-08T14:00:00Z",
        "finished": False,
        "actual_goals_home": None,
        "actual_goals_away": None,
        "predicted_home_win": 0.36,
        "predicted_draw": 0.26,
        "predicted_away_win": 0.38,
        "backfilled": False,
        "has_live_odds": False,
        "value_bet_flags": [],
    }
    card.update(over)
    return card


def _player(suffix, name, prob, team):
    return {
        "player_id": 1000 + int(suffix),
        "name": name,
        "position": "MID",
        "anytime_goal_prob": prob,
        "status": "a",
        "team": team,
    }


def _snapshot(detail=None, cards=None, players=None, gameweek=9):
    card_list = cards if cards is not None else [_card()]
    player_block = players if players is not None else {
        "home_players": [_player("1", "Wilson", 0.62, "Sunderland"), _player("2", "Jones", 0.31, "Sunderland")],
        "away_players": [_player("3", "Blue", 0.44, "Chelsea")],
    }
    return {
        "current_gameweek": gameweek,
        "fixtures_by_gameweek": {str(gameweek): {"gameweek": gameweek, "fixtures": card_list}},
        "fixture_detail_by_event_id": {EVENT_ID: detail if detail is not None else _detail()},
        "fixture_players_by_event_id": {EVENT_ID: player_block},
    }


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot())
    monkeypatch.setattr(
        facts_mod.routes.tracking_store, "get_track_record",
        lambda: {"n_resolved_fixtures": 50, "pct_correct_overall": 0.56, "n_rebuilt_fixtures": 4},
    )
    return TestClient(app)


# --- matchPick, ported ---------------------------------------------------

def test_match_pick_matches_the_frontend_rule():
    assert facts_mod.match_pick(0.36, 0.26, 0.38, "Tottenham", "Aston Villa") == {
        "side": "away_win", "label": "Aston Villa win", "prob": 0.38,
    }
    assert facts_mod.match_pick(0.57, 0.21, 0.22, "Brentford", "Chelsea")["label"] == "Brentford win"
    assert facts_mod.match_pick(0.3, 0.4, 0.3, "A", "B")["label"] == "Draw"
    # Ties break home, then draw, then away — like the API's Python max().
    assert facts_mod.match_pick(0.4, 0.4, 0.2, "A", "B")["side"] == "home_win"
    assert facts_mod.match_pick(0.2, 0.4, 0.4, "A", "B")["side"] == "draw"


# --- the contract -------------------------------------------------------

def test_bundle_validates_against_the_contract(api):
    body = api.get(f"/facts/{EVENT_ID}").json()

    Facts(**body)
    assert body["sport"] == "pl"
    assert body["id"] == EVENT_ID
    assert body["title"] == "Chelsea at Sunderland"
    assert body["starts_at"] == "2026-11-08T14:00:00Z"
    assert body["status"] == "upcoming"
    assert body["pick"] == {"side": "away_win", "label": "Chelsea win", "prob": 0.38}


def test_result_market_carries_all_three_probabilities(api):
    body = api.get(f"/facts/{EVENT_ID}").json()
    result = next(m for m in body["markets"] if m["market"] == "result")

    assert result["model"] == {
        "home_win": pytest.approx(0.36),
        "draw": pytest.approx(0.26),
        "away_win": pytest.approx(0.38),
    }


def test_total_goals_and_btts_markets_when_the_detail_has_them(api):
    body = api.get(f"/facts/{EVENT_ID}").json()
    by_market = {m["market"]: m for m in body["markets"]}

    assert by_market["total_goals"]["model_total"] == pytest.approx(2.9)
    assert by_market["total_goals"]["over_2_5"] == pytest.approx(0.55)
    assert by_market["btts"]["yes_prob"] == pytest.approx(0.57)


def test_total_goals_and_btts_are_skipped_when_absent(api, monkeypatch):
    detail = _detail()
    detail.pop("predicted_total_goals")
    detail.pop("btts_yes_prob")
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(detail=detail))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert [m["market"] for m in body["markets"]] == ["result"]


def test_live_odds_and_edge_appear_when_the_fixture_has_them(api, monkeypatch):
    detail = _detail(
        has_live_odds=True,
        home_win={"prob": 0.36, "implied": 0.42, "edge": -0.06},
        away_win={"prob": 0.38, "implied": 0.33, "edge": 0.05},
        value_bet_flags=["away value"],
    )
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(detail=detail))

    body = api.get(f"/facts/{EVENT_ID}").json()
    result = next(m for m in body["markets"] if m["market"] == "result")

    assert result["implied"] == {"Sunderland": pytest.approx(0.42), "Chelsea": pytest.approx(0.33)}
    assert result["edge"] == {"Sunderland": pytest.approx(-0.06), "Chelsea": pytest.approx(0.05)}


def test_players_are_the_top_three_scorers(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(players={
        "home_players": [_player("1", "A", 0.2, "Sunderland"), _player("2", "B", 0.7, "Sunderland")],
        "away_players": [_player("3", "C", 0.5, "Chelsea"), _player("4", "D", 0.1, "Chelsea")],
    }))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert [p["name"] for p in body["players"]] == ["B", "C", "A"]


def test_record_reports_pre_kickoff_hits_over_settled(api):
    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["record"] == {"label": "Picks made before kick-off", "hits": 28, "settled": 50}


def test_drivers_carry_recent_form_when_the_detail_has_it(api):
    body = api.get(f"/facts/{EVENT_ID}").json()

    assert any(d["name"] == "Recent form" for d in body["drivers"])


# --- pick_timing --------------------------------------------------------

def test_pick_timing_is_none_when_there_is_no_prediction_at_all(api, monkeypatch):
    # No stored card, and a detail with no usable probabilities: nothing to
    # claim, so nothing is claimed.
    detail = _detail()
    detail["home_win"] = {"prob": None, "implied": None, "edge": None}
    detail["draw"] = {"prob": None, "implied": None, "edge": None}
    detail["away_win"] = {"prob": None, "implied": None, "edge": None}
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(detail=detail, cards=[]))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["pick_timing"] == "none"
    assert body["pick"] is None


def test_upcoming_fixture_with_no_card_still_uses_the_current_read(api, monkeypatch):
    # An upcoming fixture's pick is a live forecast, made before kick-off, so
    # 'pre_kickoff' is the honest label even with no stored record yet.
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(cards=[]))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["status"] == "upcoming"
    assert body["pick"]["label"] == "Chelsea win"
    assert body["pick_timing"] == "pre_kickoff"


def test_pick_timing_is_rebuilt_when_the_fixture_is_backfilled(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(cards=[_card(backfilled=True)]))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["pick_timing"] == "rebuilt"
    assert body["pick"] is not None


# --- THE RULE: a started fixture uses the stored pre-kickoff record ------

def test_started_fixture_uses_the_stored_card_not_the_recomputed_detail(api, monkeypatch):
    started = _card(
        commence_time="2026-11-01T14:00:00Z", finished=True,
        actual_goals_home=2, actual_goals_away=1,
        predicted_home_win=0.57, predicted_draw=0.23, predicted_away_win=0.20,
    )
    # The detail disagrees: it is today's model, recomputed after the match.
    detail = _detail(
        commence_time="2026-11-01T14:00:00Z",
        home_win={"prob": 0.30, "implied": None, "edge": None},
        draw={"prob": 0.29, "implied": None, "edge": None},
        away_win={"prob": 0.41, "implied": None, "edge": None},
    )
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(detail=detail, cards=[started]))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["status"] == "final"
    # The stored card's Sunderland-at-0.57 pick wins, not the detail's 0.41 Chelsea.
    assert body["pick"] == {"side": "home_win", "label": "Sunderland win", "prob": 0.57}
    assert body["pick_timing"] == "pre_kickoff"
    # The result market carries the CARD's probabilities too: no recomputed
    # post-match number may appear anywhere in a started fixture's bundle.
    result_market = next(m for m in body["markets"] if m["market"] == "result")
    assert result_market["model"] == {
        "home_win": pytest.approx(0.57),
        "draw": pytest.approx(0.23),
        "away_win": pytest.approx(0.20),
    }
    assert "0.41" not in str(body["markets"])
    assert body["result"]["score"] == "Sunderland 2-1"
    # pick_won is judged on the STORED pick: Sunderland won, and it was the pick.
    assert body["result"]["pick_won"] is True
    Facts(**body)


def test_started_fixture_omits_pick_won_when_the_stored_pick_lost(api, monkeypatch):
    started = _card(
        commence_time="2026-11-01T14:00:00Z", finished=True,
        actual_goals_home=0, actual_goals_away=2,
        predicted_home_win=0.57, predicted_draw=0.23, predicted_away_win=0.20,
    )
    detail = _detail(
        commence_time="2026-11-01T14:00:00Z",
        home_win={"prob": 0.30, "implied": None, "edge": None},
        draw={"prob": 0.29, "implied": None, "edge": None},
        away_win={"prob": 0.41, "implied": None, "edge": None},
    )
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(detail=detail, cards=[started]))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["pick"] == {"side": "home_win", "label": "Sunderland win", "prob": 0.57}
    assert body["result"]["pick_won"] is False


def test_started_fixture_omits_pick_won_for_a_backfilled_card(api, monkeypatch):
    started = _card(
        commence_time="2026-11-01T14:00:00Z", finished=True,
        actual_goals_home=2, actual_goals_away=1,
        predicted_home_win=0.57, predicted_draw=0.23, predicted_away_win=0.20,
        backfilled=True,
    )
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(cards=[started]))

    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["status"] == "final"
    assert body["pick_timing"] == "rebuilt"
    assert "score" in body["result"]
    assert "pick_won" not in body["result"]
    Facts(**body)


def test_started_fixture_with_no_card_has_no_pick_and_no_verdict(api, monkeypatch):
    monkeypatch.setattr(
        facts_mod, "_snapshot",
        lambda: _snapshot(detail=_detail(commence_time="2026-11-01T14:00:00Z"), cards=[]),
    )

    body = api.get(f"/facts/{EVENT_ID}").json()

    # With no stored card there is no pre-kickoff pick, and no record of the
    # actual goals either, so nothing is claimed: no pick, no markets, and no
    # result. The status reads 'live' rather than 'final' precisely because
    # the stored record — the only thing that knows the score — is missing.
    assert body["pick"] is None
    assert body["pick_timing"] == "none"
    assert body["markets"] == []
    assert body["result"] is None
    assert "pick_won" not in (body["result"] or {})


def test_upcoming_fixture_uses_the_current_detail_read(api):
    body = api.get(f"/facts/{EVENT_ID}").json()

    assert body["status"] == "upcoming"
    assert body["pick"]["label"] == "Chelsea win"
    assert body["pick_timing"] == "pre_kickoff"


# --- /facts/upcoming ----------------------------------------------------

def test_upcoming_lists_only_fixtures_inside_the_window(api, monkeypatch):
    soon = _card(event_id="1", commence_time=(NOW + timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    later = _card(event_id="2", commence_time=(NOW + timedelta(hours=100)).isoformat().replace("+00:00", "Z"))
    past = _card(event_id="3", commence_time=(NOW - timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(cards=[soon, later, past]))

    body = api.get("/facts/upcoming?hours=72").json()

    assert body["ids"] == ["1"]


def test_unknown_event_id_is_404(api):
    assert api.get("/facts/does-not-exist").status_code == 404
