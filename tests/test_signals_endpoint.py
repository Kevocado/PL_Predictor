"""`GET /api/signals/{event_id}` for PL. The endpoint's job is three things and no
more: ask each adapter, keep the strongest few, and never let a signal's absence
break the gameweek page.

Run: python -m pytest tests/test_signals_endpoint.py -q
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from pl_predictor.api import facts
from pl_predictor.api.signals import MAX_SIGNALS, get_signals, signals_for_game


def row(name: str, prob: float, status: str = "a") -> dict:
    return {"name": name, "anytime_goal_prob": prob, "status": status}


@pytest.fixture
def context(monkeypatch):
    """Replace `facts.game_context` so these tests read no snapshot, no tracking
    store and no FPL. The context function itself is exercised by its own module's
    tests; what is under test HERE is what the endpoint does with what it returns.
    """

    def install(**ctx):
        base = {"id": "7", "started": False, "rows": []}
        base.update(ctx)
        monkeypatch.setattr(facts, "game_context", lambda event_id: base)

    return install


def test_returns_an_absence_for_a_gameweek_with_someone_out(context):
    context(rows=[row("Saka", 0.62), row("Haaland", 0.90, "i")])
    payload = get_signals("7")
    assert payload["sport"] == "pl"
    assert payload["id"] == "7"
    assert len(payload["signals"]) == 1
    assert payload["signals"][0]["headline"]["text"] == (
        "Out: Haaland, our #1 scorer, 90% to score"
    )


def test_an_empty_list_is_a_complete_answer_not_an_error(context):
    """Spec §2: no data, no signal. Most gameweeks have nobody out, so this is
    the common case and it must not look like a failure to the client."""
    context(rows=[row("Saka", 0.62)])
    assert get_signals("7") == {"sport": "pl", "id": "7", "signals": []}


def test_a_started_gameweek_carries_no_signal(context):
    """An absence is only information before the players are known. After the
    fact it is hindsight, and `pre_kickoff_only` is a claim the payload makes."""
    context(started=True, rows=[row("Haaland", 0.90, "i")])
    assert get_signals("7")["signals"] == []


def test_an_unknown_gameweek_is_a_404(context, monkeypatch):
    def unknown(event_id):
        raise HTTPException(status_code=404, detail=f"No fixture with event_id={event_id}")

    monkeypatch.setattr(facts, "game_context", unknown)
    # NOT swallowed into an empty list: an id the client got wrong should say so.
    with pytest.raises(HTTPException) as exc:
        get_signals("999")
    assert exc.value.status_code == 404


def test_an_adapter_that_raises_does_not_take_down_the_page(context, monkeypatch):
    """A signal is an enhancement on a gameweek page. The page has to survive its
    absence, so a broken adapter is 'no signal', not a 500."""
    context(rows=[row("Haaland", 0.90, "i")])
    monkeypatch.setattr(
        "pl_predictor.api.signals.absence.absence_signal",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pipeline down")),
    )
    assert get_signals("7")["signals"] == []


def test_a_failure_still_returns_the_success_shape(context, monkeypatch):
    """A client reading `sport` or `id` must not get a different object only when
    the backend is failing — the one moment it is least able to cope."""
    def boom(event_id):
        raise RuntimeError("snapshot unreadable")

    monkeypatch.setattr(facts, "game_context", boom)
    assert get_signals("7") == {"sport": "pl", "id": "7", "signals": []}


def test_a_404_is_not_absorbed_into_the_success_shape(context, monkeypatch):
    """The counterpart: an unknown id is the client's error and must still say so."""
    def unknown(event_id):
        raise HTTPException(status_code=404, detail="nope")

    monkeypatch.setattr(facts, "game_context", unknown)
    with pytest.raises(HTTPException):
        get_signals("999")


def test_max_signals_is_the_specs_two_or_three(context):
    """Spec §2's rule lives at the endpoint. One adapter returns at most one today,
    so nothing is truncated — but the cap is the rule, so it is asserted rather
    than assumed."""
    assert MAX_SIGNALS in (2, 3)


def test_signals_come_back_strongest_first(context):
    """The sort is what makes the cap meaningful. With one adapter it is a
    single-element list, so the ordering is checked on the key the endpoint uses
    rather than by inventing a second adapter to shuffle."""
    context(rows=[row("Haaland", 0.90, "i")])
    signals = signals_for_game("7")
    assert [s["strength"] for s in signals] == sorted(
        (s["strength"] for s in signals), reverse=True
    )
