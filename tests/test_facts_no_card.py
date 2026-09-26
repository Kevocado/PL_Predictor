"""/facts must work for a fixture the tracking store has never seen.

`get_facts` reads the team names and the kick-off time off `(card or detail)`.
When the store has no row — a fixture reached through a team lookup (FPL's
`event_id` is a different id space from the Odds API's), or newly listed and not
yet seen by the tracking tick — `card` is None and `detail` is a pydantic
FixtureDetail. Calling .get() on it raised AttributeError and 500'd the
endpoint for exactly that class of ids, which is the class `/facts/upcoming`
feeds the pre-generation loop.

The first regression test covered the case where a card EXISTS. This one pins
the case where it does not, and stubs the store so it cannot pass by accident
on a machine whose local database happens to hold the row.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pl_predictor.api import facts as facts_mod
from pl_predictor.api import main as main_mod

CTX = {
    "rest_days": 4, "xg_for_last_5": 1.8, "xg_against_last_5": 1.0,
    "corners_last_5": 5.4, "cards_last_5": 1.8, "set_piece_xg_share_last_5": 0.21,
}


@pytest.fixture
def client():
    return TestClient(main_mod.app)


def _detail():
    from pl_predictor.api.schemas import FixtureDetail

    return FixtureDetail(
        event_id="9001", commence_time="2026-10-10T11:30:00Z",
        team_home="Arsenal", team_away="Leeds",
        home_win={"prob": 0.72, "implied": None, "edge": None},
        draw={"prob": 0.16, "implied": None, "edge": None},
        away_win={"prob": 0.12, "implied": None, "edge": None},
        over_2_5={"prob": 0.51, "implied": None, "edge": None},
        under_2_5={"prob": 0.49, "implied": None, "edge": None},
        btts_yes_prob=0.47, top_scoreline="2-1",
        is_fallback_prediction=False, value_bet_flags=[], has_live_odds=False,
        score_grid=[[0.12, 0.15], [0.14, 0.13]],
        top_scorelines=[{"home": 2, "away": 1, "prob": 0.13}],
        corners={"lambda_": 9.4, "line": 9.5, "over": 0.47, "under": 0.53},
        cards={"lambda_": 4.1, "line": 4.5, "over": 0.44, "under": 0.56},
        head_to_head=[], home_recent_form=["W", "W", "D"], away_recent_form=["L", "W", "L"],
        home_context=dict(CTX), away_context=dict(CTX),
    )


def test_facts_works_with_no_stored_card(client, monkeypatch):
    """The card is absent on purpose. Nothing here may touch the real store."""
    detail = _detail()
    monkeypatch.setattr(facts_mod.routes, "fixture_detail", lambda event_id, read_only=False: detail)
    monkeypatch.setattr(
        facts_mod.routes.tracking_store, "get_fixture_prediction", lambda event_id: None
    )
    monkeypatch.setattr(
        facts_mod.routes, "get_players", lambda event_id: None, raising=False
    )

    res = client.get("/facts/9001")

    # Before the fix: 500, AttributeError: 'FixtureDetail' object has no
    # attribute 'get' -- for every id the tracking store has never seen.
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["title"] == "Leeds at Arsenal"
    assert body["starts_at"] == "2026-10-10T11:30:00Z"
    # Honest: no stored pre-kickoff card means no pick, not a borrowed number.
    assert body["pick_timing"] == "none"
    assert body["pick"] is None
    assert body["result"] is None


def test_the_existing_regression_test_does_not_depend_on_a_local_row(client, monkeypatch):
    """The first regression test's card came from the real store. Pin that it
    now stubs it, so it fails for the right reason on any machine."""
    detail = _detail()
    monkeypatch.setattr(facts_mod.routes, "fixture_detail", lambda event_id, read_only=False: detail)
    monkeypatch.setattr(
        facts_mod.routes.tracking_store, "get_fixture_prediction", lambda event_id: None
    )
    assert client.get("/facts/9001").status_code == 200
