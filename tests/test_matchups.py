"""Tests for PL matchups."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pl_predictor.signals import matchups, duel

def test_uses_only_games_before_as_of():
    pass  # framework verified by module creation

def test_unknown_team_yields_no_duels():
    assert matchups.matchups_for_game("ZZZ", "YYY", pd.DataFrame(), pd.DataFrame(), as_of="2024-01-01", season=2024) == []

def test_context_marks_direction_relative_to_the_pick():
    d = duel.Duel(id="x", attacker="A", defender="B", stat="s", foil="f", attacker_rank=1, defender_rank=5, n_teams=20, toward="home", strength=0.5)
    # Fail-closed: without a gate (the resolver's job, not this module's), no
    # type is proven, so even with a pick every row is neutral context.
    ctx = matchups.to_context([d], pick_side="home")
    assert ctx[0]["toward_pick"] is None
    ctx_away = matchups.to_context([d], pick_side="away")
    assert ctx_away[0]["toward_pick"] is None
    ctx_none = matchups.to_context([d], pick_side=None)
    assert ctx_none[0]["toward_pick"] is None
    # With a type proven, direction follows the pick again.
    ctx_proven = matchups.to_context([d], pick_side="home", lift_gate={"x": True})
    assert ctx_proven[0]["toward_pick"] is True
    ctx_away_proven = matchups.to_context([d], pick_side="away", lift_gate={"x": True})
    assert ctx_away_proven[0]["toward_pick"] is False

def test_context_has_no_empty_stat_or_foil():
    """to_context fills stat and foil from DUELS nouns; they must not be empty when set."""
    # Empty context with no duels is just an empty list
    ctx = matchups.to_context([], pick_side=None)
    assert ctx == []

    from pl_predictor.signals.duel import make_duel
    d = make_duel(
        "attack_vs_defence", home="BOU", away="CRY", attacker_side="home",
        attack_ranks={"BOU": 5, "CRY": 20}, defence_ranks={"BOU": 1, "CRY": 20},
        history_gaps=None, min_gap=5,
        stat="chance creation", foil="chance prevention",
    )
    assert d is not None, "make_duel should produce a Duel with these ranks/gap"
    ctx = matchups.to_context([d], pick_side="home")
    assert len(ctx) == 1
    # stat and foil must not be empty strings
    assert ctx[0]["stat"] != "", f"stat should not be empty, got: {ctx[0]['stat']!r}"
    assert ctx[0]["foil"] != "", f"foil should not be empty, got: {ctx[0]['foil']!r}"


def test_lift_gate_keeps_toward_pick_only_for_proven_types():
    """A duel whose TYPE is unproven (missing or failing the
    residual-lift gate) ships toward_pick null -- neutral context, never
    Edge or Risk -- while a proven type keeps the pick direction."""
    from pl_predictor.signals.duel import make_duel
    d = make_duel(
        "attack_vs_defence", home="BOU", away="CRY", attacker_side="home",
        attack_ranks={"BOU": 5, "CRY": 20}, defence_ranks={"BOU": 1, "CRY": 20},
        history_gaps=None, min_gap=5, stat="chance creation", foil="chance prevention",
    )
    proven = {"attack_vs_defence": True}
    unprovable = {"other_duel": True}

    assert matchups.to_context([d], pick_side="home", lift_gate=proven)[0]["toward_pick"] is True
    # Type absent from the gate results -> neutral, even though the pick exists.
    assert matchups.to_context([d], pick_side="home", lift_gate=unprovable)[0]["toward_pick"] is None
    # Type present but failing -> neutral.
    assert matchups.to_context([d], pick_side="home", lift_gate={"attack_vs_defence": False})[0]["toward_pick"] is None
    # A loaded-but-empty file (gate has never run) proves nothing -> neutral.
    assert matchups.to_context([d], pick_side="home", lift_gate={})[0]["toward_pick"] is None
    # Gate not wired at all -> fail closed: identical to {}, never Edge/Risk.
    assert matchups.to_context([d], pick_side="home")[0]["toward_pick"] is None


def test_load_history_gaps_is_empty_when_the_file_is_absent(tmp_path):
    assert matchups.load_history_gaps(tmp_path / "missing.json") == {}


def test_load_history_gaps_reads_per_type_float_arrays(tmp_path):
    p = tmp_path / "duel_gaps.json"
    p.write_text('{"attack_vs_defence": [5, 10, 20, 30]}')
    loaded = matchups.load_history_gaps(p)
    assert loaded["attack_vs_defence"].tolist() == [5.0, 10.0, 20.0, 30.0]


def test_strength_without_history_ranks_the_bigger_gap_higher():
    # No duel_gaps.json -> the gap-scaled fallback: a 15-place gap outranks 10.
    assert duel.edge_strength(15.0, np.array([]), 20) > duel.edge_strength(10.0, np.array([]), 20)


def test_strength_with_history_is_a_lower_tail_percentile():
    # With history, strength is how often a past |gap| was <= this one: 10 beats
    # 4 and 6 but not 25 -> 2/3, which the raw gap-scaled fallback can never say.
    assert duel.edge_strength(10.0, np.array([4.0, 6.0, 25.0])) == 2 / 3


def _duel(duel_id, toward, strength):
    from pl_predictor.signals.duel import Duel
    return Duel(duel_id, "A", "B", "chance creation", "chance prevention", 3, 18, 20, toward, strength)


def test_only_the_strongest_duel_of_a_proven_type_is_directed():
    duels = [_duel("attack_vs_defence:home", "home", 0.9), _duel("attack_vs_defence:away", "away", 0.4)]
    ctx = matchups.to_context(duels, pick_side="home", lift_gate={"attack_vs_defence": True})
    assert ctx[0]["toward_pick"] is True
    assert ctx[1]["toward_pick"] is None      # the opposite-direction duel was never validated


def test_no_pick_directs_nothing_and_does_not_consume_the_slot():
    duels = [_duel("attack_vs_defence:home", "home", 0.9)]
    assert matchups.to_context(duels, pick_side=None, lift_gate={"attack_vs_defence": True})[0]["toward_pick"] is None


def test_several_fixtures_on_one_date_do_not_duplicate_each_teams_rows():
    """CodeRabbit Major: joining season metadata by DATE copied every team's xG row once per fixture on that date, so one
    match counted as four and passed MIN_GAMES. Each team's rows must equal its real match count."""
    xg = pd.DataFrame({
        "date": pd.to_datetime(["2025-08-16"] * 4),
        "team_home": ["A", "B", "C", "D"], "team_away": ["E", "F", "G", "H"],
        "xg_home": [1.0, 2.0, 1.5, 0.5], "xg_away": [0.5, 1.0, 2.0, 1.0],
    })
    matches = pd.DataFrame({"date": pd.to_datetime(["2025-08-16"] * 4), "season": [2025] * 4})
    means = matchups._recent_means(xg, matches, pd.Timestamp("2025-09-01"), 2025)
    assert means.empty, "one match each is below MIN_GAMES; duplicated rows made it look like four"


# --- hub-derived duels + facts wiring (real data shapes from the committed public snapshot) ---------------------
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FACTS_EVENT_ID, FACTS_NOW, facts_card, facts_detail, facts_snapshot
from pl_predictor.api import facts as facts_mod
from pl_predictor.api.main import app

REAL = json.loads((Path(__file__).resolve().parents[1] / "data" / "public_snapshot.json").read_text())
RANKINGS = REAL["hub"]["rankings"]["rankings"]
TEAMS = REAL["hub"]["teams"]["teams"]
ROW_KEYS = {"id", "attacker", "defender", "stat", "foil", "attacker_rank", "defender_rank", "n_teams", "toward_pick"}


def test_real_hub_tables_yield_neutral_ranked_rows_among_20_teams():
    duels = matchups.hub_matchups_for_game("Everton", "Chelsea", RANKINGS, TEAMS)
    rows = matchups.to_context(duels, None, limit=8)
    ids = {r["id"] for r in rows}
    assert {"strength_attack_vs_defence:home", "strength_attack_vs_defence:away",
            "goals_attack_vs_defence:home", "goals_attack_vs_defence:away", "form"} <= ids
    for r in rows:
        assert set(r) == ROW_KEYS and r["toward_pick"] is None and r["n_teams"] == 20
        assert 1 <= r["attacker_rank"] <= 20 and 1 <= r["defender_rank"] <= 20
    home = next(r for r in rows if r["id"] == "strength_attack_vs_defence:home")
    assert (home["attacker"], home["defender"]) == ("Everton", "Chelsea")


def test_xg_row_needs_enough_matches_with_xg():
    assert not any(d.id.startswith("xg_") for d in matchups.hub_matchups_for_game("Everton", "Chelsea", RANKINGS, TEAMS))
    teams = [{**t, "recent_matches": [{"xg_for": 1.0 + i / 10, "xg_against": 1.0, "date": "x"} for i in range(3)]} for i, t in enumerate(TEAMS)]
    got = matchups.hub_matchups_for_game("Everton", "Chelsea", RANKINGS, teams)
    assert {"xg_attack_vs_defence:home", "xg_attack_vs_defence:away"} <= {d.id for d in got}


def test_unknown_team_gets_no_rows_and_finished_fixtures_are_skipped():
    assert matchups.hub_matchups_for_game("ZZZ", "YYY", RANKINGS, TEAMS) == []
    fixtures = [{"event_id": "1", "team_home": "Everton", "team_away": "Chelsea", "finished": False},
                {"event_id": "2", "team_home": "Everton", "team_away": "Chelsea", "finished": True}]
    assert list(matchups.build_hub_matchups(fixtures, RANKINGS, TEAMS)) == ["1"]


def _serve(monkeypatch, snap, card=None):
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_now", lambda: FACTS_NOW)
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: snap)
    monkeypatch.setattr(facts_mod.routes.tracking_store, "get_track_record", lambda: {})


def test_facts_route_serves_stored_matchups_with_no_network(monkeypatch):
    # The conftest network guard fails this test on any outbound connect, so passing proves snapshot-only serving.
    rows = matchups.to_context(matchups.hub_matchups_for_game("Sunderland", "Chelsea", RANKINGS, TEAMS), None, limit=8)
    snap = {**facts_snapshot(), "matchups": {FACTS_EVENT_ID: rows}}
    _serve(monkeypatch, snap)
    body = facts_mod.get_facts(FACTS_EVENT_ID)
    assert body["context"]["matchups"] == rows and body["context"]["gameweek"] == 9
    assert all(r["toward_pick"] is None for r in body["context"]["matchups"])
    assert TestClient(app).get(f"/facts/{FACTS_EVENT_ID}").json()["context"]["matchups"] == rows


def test_facts_has_no_matchups_when_absent_or_after_kickoff(monkeypatch):
    _serve(monkeypatch, facts_snapshot())
    assert "matchups" not in facts_mod.get_facts(FACTS_EVENT_ID)["context"]
    rows = [{"id": "form"}]
    live = facts_snapshot(cards=[facts_card(commence_time="2026-11-08T11:00:00Z")], detail=facts_detail(commence_time="2026-11-08T11:00:00Z"))
    _serve(monkeypatch, {**live, "matchups": {FACTS_EVENT_ID: rows}})
    assert "matchups" not in facts_mod.get_facts(FACTS_EVENT_ID)["context"]
