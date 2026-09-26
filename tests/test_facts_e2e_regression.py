"""A real fixture_detail response, end to end through /facts.

The Task 11 review fixed _iso_utc and the post-match gates against hand-built
dicts. This test goes through the app's OWN /fixtures/{id} route, so the shape
under /facts is the shape the site actually serves -- a pydantic model, not a
dict. It is the check that would have caught the live 500 below.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pl_predictor.api import main as main_mod
from pl_predictor.api import facts as facts_mod


@pytest.fixture
def client():
    # PUBLIC_MODE off: the route under test is the real one, and no snapshot
    # fixture is needed for the detail's own shape.
    return TestClient(main_mod.app)


def test_facts_survives_the_apps_own_fixture_detail_shape(client, monkeypatch):
    """detail is a pydantic FixtureDetail here, not a dict.

    _prob_field called detail.get(...) on it, which raises AttributeError, and
    the /facts endpoint answered 500 for every real fixture. A dict-shaped
    double cannot catch this, so the double is the app's own route.
    """
    from pl_predictor.api.schemas import FixtureDetail

    detail = FixtureDetail(
        event_id="51",
        commence_time="2026-10-10T11:30:00Z",
        team_home="Arsenal",
        team_away="Leeds",
        home_win={"prob": 0.72, "implied": None, "edge": None},
        draw={"prob": 0.16, "implied": None, "edge": None},
        away_win={"prob": 0.12, "implied": None, "edge": None},
        over_2_5={"prob": 0.51, "implied": None, "edge": None},
        under_2_5={"prob": 0.49, "implied": None, "edge": None},
        btts_yes_prob=0.47,
        top_scoreline="2-1",
        predicted_result="home_win",
        draw_signal=False,
        is_fallback_prediction=False,
        data_confidence="established",
        predicted_total_goals=2.6,
        predicted_margin=1.1,
        home_2plus_prob=0.56,
        away_2plus_prob=0.44,
        value_bet_flags=[],
        has_live_odds=False,
        odds_fetched_at=None,
        odds_is_stale=False,
        recommended_bet=None,
        # The rest of the real response. Left in rather than defaulted so this
        # stays a faithful stand-in for what routes.fixture_detail returns.
        score_grid=[[0.12, 0.15], [0.14, 0.13]],
        top_scorelines=[{"home": 2, "away": 1, "prob": 0.13}],
        corners={"lambda_": 9.4, "line": 9.5, "over": 0.47, "under": 0.53},
        cards={"lambda_": 4.1, "line": 4.5, "over": 0.44, "under": 0.56},
        head_to_head=[],
        home_recent_form=["W", "W", "D"],
        away_recent_form=["L", "W", "L"],
        home_context={
            "rest_days": 4, "xg_for_last_5": 1.8, "xg_against_last_5": 1.0,
            "corners_last_5": 5.4, "cards_last_5": 1.8, "set_piece_xg_share_last_5": 0.21,
        },
        away_context={
            "rest_days": 3, "xg_for_last_5": 1.2, "xg_against_last_5": 1.5,
            "corners_last_5": 4.1, "cards_last_5": 2.2, "set_piece_xg_share_last_5": 0.17,
        },
    )
    # Patch where facts.py looks it up. It imports the function by name, so
    # patching routes.fixture_detail alone would not be seen.
    monkeypatch.setattr(
        facts_mod.routes, "fixture_detail", lambda event_id, read_only=False: detail, raising=False
    )

    res = client.get("/facts/51")

    # Before the fix: 500, AttributeError: 'FixtureDetail' object has no
    # attribute 'get' -- every real fixture 500'd, so the explainer could
    # never read PL's facts at all.
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["sport"] == "pl" and body["id"] == "51"
    # The service splits the title on " at " and labels the halves in the
    # order the explainer expects: away at home.
    assert body["title"] == "Leeds at Arsenal"
    # This fixture has no stored pre-kickoff card, so there is no pick to
    # report. The honest answer is pick_timing "none" and pick null -- NOT a
    # number borrowed from the live model, which is the rule Tasks 8-11 landed
    # on. The model's own probabilities still appear, as markets.
    assert body["pick_timing"] == "none"
    assert body["pick"] is None
    # No stored card means no "result" market either: that market quotes the
    # pre-kickoff card's three probabilities, and there is no card. What the
    # live model knows is still offered, as its own markets.
    assert {m["market"] for m in body["markets"]} == {"btts", "total_goals"}
    by_market = {m["market"]: m for m in body["markets"]}
    assert by_market["btts"]["yes_prob"] == pytest.approx(0.47, abs=1e-6)
    # total_goals comes from the live model's own projection, not the card.
    assert by_market["total_goals"]["model_total"] == pytest.approx(2.6, abs=1e-6)
    # The one explanatory signal the detail really carries.
    assert body["drivers"] == [
        {"name": "Recent form", "value": "WWD v LWL", "direction": "Arsenal"}
    ]


def test_facts_upcoming_still_works_alongside(client, monkeypatch):
    """The explainer's pre-generation calls this first; a 500 here would stop
    every sport, not just PL."""
    res = client.get("/facts/upcoming?hours=72")
    assert res.status_code == 200
    assert "ids" in res.json()


def test_prob_field_accepts_both_shapes():
    """The helper's own contract, pinned directly: a bare float and the
    {prob, implied, edge} dict both work, and anything unusable is None."""
    assert facts_mod._prob_field({"home_win": 0.72}, "home_win") == pytest.approx(0.72)
    assert facts_mod._prob_field({"home_win": {"prob": 0.72}}, "home_win") == pytest.approx(0.72)
    assert facts_mod._prob_field({}, "home_win") is None
    assert facts_mod._prob_field({"home_win": float("nan")}, "home_win") is None
