"""PL's `absence` signal. The tests that matter here are the ones about what the
row REFUSES to say, because an absence row has three easy ways to be wrong: it can
rank a player by a pool that has already dropped the out ones, it can call a
doubtful player out, and it can invent a name.

Run: python -m pytest tests/test_absence_signal.py -q
"""
from __future__ import annotations

import pytest

from pl_predictor.signals import absence
from pl_predictor.signals.absence import absence_signal, headline


def row(name: str, prob: float, status: str = "a", **extra) -> dict:
    return {
        "name": name,
        "anytime_goal_prob": prob,
        "status": status,
        **extra,
    }


# A pool with one injured player who is NOT the model's best scorer, so the rank
# and the strength are different numbers and a test can only pass by reading both.
POOL = [
    row("Saka", 0.62, "a"),
    row("Toney", 0.55, "a"),
    row("Gordon", 0.41, "a"),
    row("Haaland", 0.90, "i"),   # out, and the model's top scorer
    row("Doku", 0.20, "a"),
]


def test_names_the_out_player_and_states_the_figure_it_draws():
    sig = absence_signal("7", POOL, as_of="2026-10-03T11:00:00Z")
    assert sig is not None
    assert sig["kind"] == "absence"
    assert sig["visual"] == "absence_strip"
    assert sig["headline"]["text"] == "Out: Haaland, our #1 scorer, 90% to score"
    # The figure in the words is the figure the marker draws. The shared component
    # THROWS on a mismatch, so these must be one number.
    assert sig["headline"]["figures"]["projection"] == 90


def test_rank_is_over_every_player_including_the_out_ones():
    """The trap: ranking the pool with the out player REMOVED promotes everyone
    below them, so Haaland at 0.90 becomes '#2' if he is dropped and the two
    below him slide up. He is our #1 and the row has to say so."""
    sig = absence_signal("7", POOL)
    assert sig is not None
    assert "#1" in sig["headline"]["text"]


def test_doubtful_is_not_out():
    """`availability_multiplier` gives `d` a chance/100, so a doubtful player is
    still expected to play. Calling them out asserts the opposite."""
    sig = absence_signal("7", [row("Salah", 0.80, "d")])
    assert sig is None


def test_suspended_and_unavailable_are_out_too():
    for status in ("s", "u"):
        sig = absence_signal("7", [row("Bellerin", 0.5, status)])
        assert sig is not None, status
        assert "Bellerin" in sig["headline"]["text"]


def test_a_missing_status_is_not_out():
    """An older cached snapshot has no `status` field at all. Reading that as
    'everyone is out' is the opposite of the truth, not a smaller claim."""
    sig = absence_signal("7", [{"name": "Ghost", "anytime_goal_prob": 0.9}])
    assert sig is None


def test_returns_none_when_nobody_is_out():
    assert absence_signal("7", [row("Saka", 0.62), row("Toney", 0.55)]) is None


def test_returns_none_on_an_empty_pool():
    assert absence_signal("7", []) is None


def test_picks_the_biggest_absence_not_the_first_one_found():
    pool = [row("Small", 0.10, "i"), row("Big", 0.71, "i")]
    sig = absence_signal("7", pool)
    assert sig is not None
    assert "Big" in sig["headline"]["text"]
    assert sig["strength"] == pytest.approx(0.71)


def test_n_counts_players_and_is_not_a_graded_sample():
    pool = [row("A", 0.6, "i"), row("B", 0.3, "i"), row("C", 0.1, "a")]
    sig = absence_signal("7", pool)
    assert sig is not None
    # 2 injured players. A spec floor of n >= 30 would refuse this row, which is
    # the category error the shared component's exemption exists to prevent.
    assert sig["n"] == 2


def test_strength_rises_with_the_projection_and_stays_bounded():
    low = absence_signal("7", [row("L", 0.05, "i")])
    high = absence_signal("7", [row("H", 0.95, "i")])
    assert low is not None and high is not None
    assert low["strength"] < high["strength"]
    assert 0.0 <= low["strength"] <= 1.0 and 0.0 <= high["strength"] <= 1.0


def test_refuses_to_name_a_player_it_cannot_name():
    """'Out: , our #2 scorer' is not a degraded version of the claim."""
    sig = absence_signal("7", [{"name": "  ", "anytime_goal_prob": 0.9, "status": "i"}])
    assert sig is None


def test_ignores_a_row_with_no_projection():
    """A player with no projection is not in the model's ranking, so calling them
    '#1' states a place in a table they are not in."""
    pool = [{"name": "Nobody", "anytime_goal_prob": None, "status": "i"}, row("Real", 0.3, "a")]
    sig = absence_signal("7", pool)
    assert sig is None


def test_gameweek_id_is_carried_as_a_string():
    """PL's fixture ids are gameweek numbers. `1` and `"1"` are different keys to
    the client, and the client reads `id` off the response envelope."""
    sig = absence_signal(7, POOL)
    assert sig is not None
    assert sig["game_id"] == "7"


def test_pre_kickoff_only_is_true():
    """An absence is only usable before the fixture starts. After it, the same
    payload would be quoting a lineup that was already known."""
    sig = absence_signal("7", POOL)
    assert sig is not None
    assert sig["pre_kickoff_only"] is True


def test_source_counts_players_and_survives_the_singular():
    one = absence_signal("7", [row("Only", 0.4, "i")])
    two = absence_signal("7", [row("A", 0.4, "i"), row("B", 0.3, "i")])
    assert one is not None and two is not None
    assert "1 player out" in one["source"]
    assert "2 players out" in two["source"]


def test_ranks_do_not_depend_on_the_order_the_rows_arrive_in():
    shuffled = list(reversed(POOL))
    a = absence_signal("7", POOL)
    b = absence_signal("7", shuffled)
    assert a is not None and b is not None
    assert a["headline"]["text"] == b["headline"]["text"]


def test_out_statuses_are_exactly_the_zero_availability_ones():
    """Guard against someone adding 'd' here. `fpl_api.availability_multiplier`
    returns 0.0 for these three and `chance/100` for 'd'."""
    assert absence.OUT_STATUSES == frozenset({"i", "s", "u"})
    assert "d" not in absence.OUT_STATUSES


def test_headline_is_within_the_wordspec_limit():
    """Spec §4 caps a headline at 12 words so it cannot become a paragraph."""
    text = headline("Van Dijk", 3, 71)
    assert len(text.split()) <= 12


def test_uses_no_sportsbook_or_certainty_vocabulary():
    """The rendered page is checked for these words; an absence row quoting none
    of them is what keeps that check green."""
    text = headline("Haaland", 1, 90).lower()
    for word in ("lock", "guaranteed", "best bet", "edge", "value", "sure"):
        assert word not in text
