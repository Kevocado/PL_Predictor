"""The reversal of 2026-10-01: the track record counts every recorded pick.

This file used to be `test_track_record_pre_kickoff_only.py` and asserted the
OPPOSITE of everything below, which is the point worth stating before the
numbers: under the pre-2026-10-01 rule the headline was the pre-kickoff record
and a pick recorded after its own kickoff never counted toward it. PL's
headline was empty for exactly that reason — the shipped payload read
`n_resolved_fixtures: 0, n_rebuilt_fixtures: 50, pct_correct_overall: null`.

Kevin's reason for reversing it, which is the right one: the models are re-run
constantly, so under the old rule a re-run on an already-played game stopped
counting and the record emptied out on every model change. Recorded stays
recorded, and every recorded pick counts.

Four properties are pinned here, and none of them is "the headline got bigger":

1. **The headline counts every counted pick, whenever it was made.** The empty
   headline is the failure this change exists to end.
2. **One counted pick per (fixture, market) — the EARLIEST recorded one.** A
   later rerun is history: it neither displaces the counted pick nor counts a
   second time. Without this, re-running until the model is right is free.
3. **`pre_kickoff` beside it is the subset made before kickoff, with its own n.**
   It is the same summariser over the counted picks whose own timestamps prove
   they were made before their own kickoff, so its `n` is the exact size of that
   subset rather than a figure reconciled by hand.
4. **Every per-pick row carries `made_before_kickoff` and its timestamp**, both
   derived from the row's own two timestamps compared as UTC instants. Nothing
   after the start may be labelled "made before kickoff", and nothing the
   timestamps do not prove may be labelled so either.

**The counting key is NOT one key.** The decision doc's "(game, market)" is
destructive for player picks, because many players share a market: collapsing a
pick to game level deletes the player record entirely. So the two record types
are keyed apart and the split is pinned here:

- fixture-level markets (match result, exact score, O/U 2.5, BTTS) -> one counted
  pick per `(fixture, market)`, which reduces to one row per fixture because all
  four markets live as columns on that row;
- player picks -> one counted pick per `(fixture, player_id, market)`.
"""
import pandas as pd
import pytest

from pl_predictor.tracking import store


def _frame(hits, snapshotted_at, kickoff="2026-09-13T14:00:00Z", gameweeks=None, event_ids=None):
    """A resolved-fixture frame in the shape `get_track_record` reads.

    `snapshotted_at` and `commence_time` are the ONLY inputs to
    `made_before_kickoff` — there is no `backfilled` column here at all, on
    purpose. The rule is derived from timestamps, so a test that supplied a
    flag could pass while the derivation was wrong.

    Team names are derived from `event_id`, so two rows carrying the same
    `event_id` are one fixture (the rerun case) and different ids are different
    fixtures. Teams and not ids because `_fixture_hit_table` dedupes on
    `(team_home, team_away, kickoff date)` — the same match is legitimately
    snapshotted under two event_ids (Odds API hex vs FPL numeric).
    """
    n = len(hits)
    ids = event_ids if event_ids is not None else [f"e{i}" for i in range(n)]
    return pd.DataFrame({
        "hit": hits,
        "snapshotted_at": snapshotted_at,
        "commence_time": [kickoff] * n,
        "event_id": ids,
        "team_home": [f"Home {i}" for i in ids],
        "team_away": [f"Away {i}" for i in ids],
        "gameweek": gameweeks if gameweeks is not None else [1] * n,
        "exact_score_hit": hits,
        "over_under_hit": hits,
        "btts_hit": hits,
        "actual_outcome": ["home_win"] * n,
        "predicted_prob_actual": [0.6] * n,
    })


BEFORE = "2026-09-13T10:00:00Z"
AFTER = "2026-09-13T15:00:00Z"


def _pinned(monkeypatch, frame):
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: frame)
    return store.get_track_record()


# --- THE CASE THAT PROVES THE DECISION: PL's shipped, EMPTY headline --------

def test_PLs_shipped_state_becomes_a_populated_headline_and_an_empty_secondary(monkeypatch):
    """The whole reversal, on PL's actual numbers.

    Measured from `data/public_snapshot.json` on `origin/main` (generated
    2026-10-01T15:24Z), the shipped record is 50 resolved fixtures, every one of
    them recorded after its own kickoff: `n_resolved_fixtures: 0,
    n_rebuilt_fixtures: 50, pct_correct_overall: null`, with the 50 picks sitting
    in `all_picks` at 0.52. `data/tracking_picks.jsonl` is 0 lines.

    So the defect is not hypothetical and the fix is measurable: the headline
    goes from an absence to 50 picks at 52%, and the secondary figure becomes the
    absence. Both halves matter — a change that populated the headline by
    relabelling post-kickoff picks as pre-kickoff ones would pass every
    "headline is non-null" test and be the exact dishonesty this reversal is
    about.
    """
    n, hits = 50, 26  # 26 of 50 correct in the shipped `all_picks` block
    frame = _frame(
        [True] * hits + [False] * (n - hits),
        [AFTER] * n,
        gameweeks=[1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10,
    )
    out = _pinned(monkeypatch, frame)

    # BEFORE: n_resolved_fixtures 0, pct_correct_overall null, n_rebuilt 50.
    assert out["n_resolved_fixtures"] == 50
    assert out["pct_correct_overall"] == pytest.approx(26 / 50, abs=1e-9)
    assert out["n_rebuilt_fixtures"] == 50, "all 50 were made after their own kickoff"
    assert out["all_picks"]["n_resolved"] == 50
    assert out["all_picks"]["pct_correct"] == pytest.approx(0.52, abs=1e-9)

    # AFTER: the secondary figure is the absence it should have been, and it is
    # EMPTY rather than 0%. This is the half a "just include everything" change
    # would get wrong.
    assert out["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert out["pre_kickoff"]["pct_correct_overall"] is None, (
        "not one of the 50 picks was captured before kickoff, so the honest live figure is an "
        "absence. Reporting 0.0% would be a rate computed over nothing."
    )

    # And the count is exactly auditable: 50 listed rows, every one labelled.
    rows = out["per_pick"]
    assert len(rows) == 50, f"per_pick lists {len(rows)} rows, so a reader cannot tally the headline"
    assert all(r["made_before_kickoff"] is False for r in rows), (
        "no pick's own timestamps place it before its kickoff, so none may be labelled that way"
    )
    assert all(r["snapshotted_at"] for r in rows), "every row carries the time its pick was made"


# --- 1. the headline counts every recorded pick -------------------------------

def test_the_headline_counts_picks_made_after_kickoff(monkeypatch):
    """The rule this file exists for, stated as one assertion.

    Every pick here was recorded after its own kickoff. Under the old rule the
    headline was `None` with `n_resolved_fixtures == 0` — which is precisely
    the state PL shipped in, and why the page said "no honest score to show"
    while 50 graded picks sat in the same payload.
    """
    frame = _frame([True] * 6 + [False] * 4, [AFTER] * 10)
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 10, (
        f"headline counts {out['n_resolved_fixtures']} picks, expected all 10. A pick recorded "
        f"after its own kickoff is still a recorded pick, and it counts."
    )
    assert out["pct_correct_overall"] == 0.6, (
        f"headline is {out['pct_correct_overall']!r}. 6 of 10 is the record; None is what the "
        f"pre-reversal rule produced here and is what emptied the record on every model change."
    )


def test_the_headline_is_never_null_while_a_counted_pick_exists(monkeypatch):
    """`None` means "no picks", not "no picks made in time".

    0 out of 0 is an absence and must read as one; 6 out of 10 is a rate and
    must not be withheld because all ten picks were late.
    """
    out = _pinned(monkeypatch, _frame([True] * 3, [AFTER] * 3))
    assert out["pct_correct_overall"] is not None
    assert out["n_resolved_fixtures"] == 3


# --- 2. one counted pick per unit: the EARLIEST recorded one -----------------

def test_a_later_rerun_neither_replaces_nor_doubles_the_counted_pick(monkeypatch):
    """Otherwise re-running until the model is right is free.

    Two rows for one fixture, graded opposite ways. Only the EARLIEST counts:
    the later rerun is kept in the table as history and is in no figure and no
    list.
    """
    frame = _frame(
        [True, False],
        [BEFORE, AFTER],
        event_ids=["e1", "e1"],
    )
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 1, (
        f"headline counts {out['n_resolved_fixtures']} picks for one fixture's one market. The "
        f"key is one counted pick per (fixture, market); a rerun must not grade it twice."
    )
    assert out["pct_correct_overall"] == 1.0, (
        f"headline is {out['pct_correct_overall']!r}. The EARLIEST recorded pick is the counted "
        f"one (a hit here), so a later rerun that missed must not displace it."
    )
    assert len(out["per_pick"]) == 1, (
        f"per_pick lists {len(out['per_pick'])} rows. It must list counted picks only, so a "
        f"reader tallying the list arrives at the headline."
    )


def test_the_counted_pick_is_the_earliest_by_INSTANT_not_by_string(monkeypatch):
    """Which row wins must not depend on how its timestamp is spelled.

    Both rows are the same fixture. The earlier one carries a `-05:00` offset:
    10:00 at UTC-5 is 15:00 UTC, so as an INSTANT it is the LATER row and must
    lose. A string comparison of `10:00:00-05:00` against `2026-09-13T15:00:00Z`
    says the opposite for most of the day.
    """
    frame = pd.DataFrame({
        "hit": [False, True],
        # Instant 2026-09-13T15:00:00Z — later than the `Z` row below.
        "snapshotted_at": ["2026-09-13T10:00:00-05:00", "2026-09-13T14:00:00Z"],
        "commence_time": ["2026-09-13T14:00:00Z", "2026-09-13T14:00:00Z"],
        # Same fixture, same teams, same kickoff: one unit, two recorded rows.
        "event_id": ["e1", "e1"],
        "team_home": ["Arsenal", "Arsenal"],
        "team_away": ["Chelsea", "Chelsea"],
        "gameweek": [1, 1],
        "exact_score_hit": [False, True],
        "over_under_hit": [False, True],
        "btts_hit": [False, True],
        "actual_outcome": ["home_win", "home_win"],
        "predicted_prob_actual": [0.6, 0.6],
    })
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 1
    assert out["pct_correct_overall"] == 1.0, (
        "the counted pick is the one at 14:00Z. The row stamped 10:00-05:00 is 15:00Z, which "
        "is later, so the earlier INSTANT wins — the opposite of what comparing the strings says."
    )


def test_an_unparseable_timestamp_never_displaces_one_carrying_a_real_instant(monkeypatch):
    """It cannot be proven earliest, so it sorts last.

    Failing this way is about ORDER, not exclusion: the unparseable row is
    still recorded and still counted if it is the only one for its key.
    """
    # The unreadable row is the MISS; the row carrying a real instant is the HIT,
    # so a headline of 1.0 can only come from the latter being the counted pick.
    frame = _frame([False, True], ["not-a-date", BEFORE], event_ids=["e1", "e1"])
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 1
    assert out["pct_correct_overall"] == 1.0, (
        "the row with a real instant is the earliest one that can be proven, so it is counted; "
        "the unparseable row cannot be shown to precede it."
    )


def test_a_fixture_with_only_an_unparseable_timestamp_is_still_counted(monkeypatch):
    """Recorded stays recorded. The label fails closed; the pick does not vanish.

    This is the asymmetry the reversal introduces and it is deliberate: under
    the old rule this function's answer decided whether a pick counted at all,
    and an unprovable timestamp meant a silent hole in the record.
    """
    out = _pinned(monkeypatch, _frame([True], ["not-a-date"]))

    assert out["n_resolved_fixtures"] == 1
    assert out["pre_kickoff"]["n_resolved_fixtures"] == 0, (
        "its own timestamps cannot prove it was made before kickoff, so it must not be in the "
        "pre-kickoff subset. Never backfill a true the timestamps do not support."
    )


# --- 3. the counting key differs BY RECORD TYPE ------------------------------

def test_fixture_markets_are_keyed_by_fixture_and_market(monkeypatch):
    """All four graded markets live as columns on ONE row per fixture.

    So one counted row carries all four, and `_per_pick_rows` still emits one
    row per (fixture, market) off it. The dedup must therefore NOT collapse the
    row to a single market — which is the specific bug a naive "(fixture, market)"
    drop_duplicates would introduce here.
    """
    frame = _frame([True], [BEFORE])
    frame["actual_goals_home"] = 1
    frame["actual_goals_away"] = 0
    frame["predicted_scoreline"] = "1-0"
    out = _pinned(monkeypatch, frame)

    # `per_pick` is one row per fixture in PL, not per market: all four graded
    # markets are columns on the counted row. So the assertion that the dedup
    # did not collapse a market is on the by_market block, where all four must
    # survive off that single counted row.
    assert len(out["per_pick"]) == 1, (
        f"per_pick emitted {len(out['per_pick'])} rows for one fixture. One counted pick per "
        f"(fixture, market) with the markets as columns is ONE row, not four and not one per "
        f"market-collapsed unit."
    )
    assert set(out["by_market"]) == {"exact_score", "match_result", "over_under_2_5", "btts"}, (
        f"by_market has keys {sorted(out['by_market'])}. A counted row that lost a market to the "
        f"dedup would still leave four keys — but with the wrong n, which is the next assertion."
    )
    assert all(m["n_resolved"] == 1 for m in out["by_market"].values()), (
        f"a market lost its row to the dedup: {out['by_market']}. All four must be rated off the "
        f"one counted fixture row."
    )


def test_player_picks_are_keyed_by_fixture_player_and_market_not_by_fixture_alone():
    """The destructive-key check, pinned where it can be got wrong.

    Many players share one market in one fixture. Keyed by `(fixture, market)`,
    every player but one is dropped and the player record is deleted outright.
    PL's scorer grader keys player picks separately, on
    `(fixture, player_id, market)`; this asserts it stays that way, so a future
    "just use one counting key everywhere" refactor cannot silently delete it.
    """
    counted = store._counted_player_picks(pd.DataFrame({
        "event_id": ["e1", "e1", "e1"],
        "player_id": [1, 2, 1],
        "market": ["goal", "goal", "assist"],
        "snapshotted_at": [BEFORE, BEFORE, BEFORE],
        "hit": [True, True, True],
    }))

    keys = set(zip(counted["event_id"], counted["player_id"], counted["market"]))
    assert keys == {("e1", 1, "goal"), ("e1", 2, "goal"), ("e1", 1, "assist")}, (
        f"player picks collapsed to {sorted(keys)}. Two players share the `goal` market in one "
        f"fixture and both must survive — the player record is the record."
    )


def test_a_player_rerun_also_counts_once_and_the_earliest_wins():
    frame = pd.DataFrame({
        "event_id": ["e1", "e1"],
        "player_id": [1, 1],
        "market": ["goal", "goal"],
        "snapshotted_at": [BEFORE, AFTER],
        "hit": [True, False],
    })
    counted = store._counted_player_picks(frame)

    assert len(counted) == 1
    assert bool(counted.iloc[0]["hit"]) is True, "the earliest recorded player pick is the counted one"


def test_the_two_record_types_are_counted_separately_not_by_one_summariser():
    """PL's grader mixes both populations in one payload, so the keys must differ.

    `get_track_record` summarises fixtures; `get_scorer_accuracy` summarises
    player picks. Asserted as two distinct entry points with distinct keys
    because a single shared key applied to both is exactly the destructive case.
    """
    assert store._counted_picks(pd.DataFrame()).empty
    assert store._counted_player_picks(pd.DataFrame()).empty
    assert store.get_scorer_accuracy.__doc__, "the player grader is a separate entry point"


# --- 4. pre_kickoff is the honest read, and its n is exact -------------------

def test_pre_kickoff_is_the_subset_with_its_own_n(monkeypatch):
    """Given different accuracies, so the two figures cannot be confused.

    The pre-kickoff picks are all hits and the late ones all misses, so a
    `pre_kickoff` that accidentally read the headline would report 60% instead
    of 100%.
    """
    frame = _frame([True] * 4 + [False] * 6, [BEFORE] * 4 + [AFTER] * 6)
    out = _pinned(monkeypatch, frame)
    pre = out["pre_kickoff"]

    assert pre["n_resolved_fixtures"] == 4, (
        f"pre_kickoff.n_resolved_fixtures is {pre['n_resolved_fixtures']}, expected 4 — the "
        f"exact size of the subset whose own timestamps prove they were made before kickoff."
    )
    assert pre["pct_correct_overall"] == 1.0
    assert out["pct_correct_overall"] == 0.4, "the headline is the mixed 40%"
    assert out["n_rebuilt_fixtures"] == 6, (
        "n_rebuilt_fixtures is retained and now means counted picks made at or after their own "
        "kickoff, so headline.n == pre_kickoff.n + n_rebuilt_fixtures."
    )


def test_the_two_figures_reconcile_exactly(monkeypatch):
    frame = _frame([True, False, True, True, False], [BEFORE, AFTER, BEFORE, AFTER, AFTER])
    out = _pinned(monkeypatch, frame)

    assert (
        out["n_resolved_fixtures"]
        == out["pre_kickoff"]["n_resolved_fixtures"] + out["n_rebuilt_fixtures"]
    ), "the difference between the two figures is exactly the size of the late-pick inflation"


def test_pre_kickoff_is_null_when_no_counted_pick_was_made_in_time(monkeypatch):
    """PL's shipped state: 50 graded picks, none made before kickoff.

    The secondary figure must be an absence, not 0%. The HEADLINE is not — it
    is 50 picks — which is the whole difference this reversal makes.
    """
    out = _pinned(monkeypatch, _frame([True] * 7, [AFTER] * 7))

    assert out["n_resolved_fixtures"] == 7, "the headline counts all seven"
    assert out["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert out["pre_kickoff"]["pct_correct_overall"] is None, (
        f"pre_kickoff reports {out['pre_kickoff']['pct_correct_overall']!r} over zero picks. "
        f"0.0% would be a rate computed over nothing."
    )


def test_by_market_follows_the_headline_and_pre_kickoff_gets_its_own_breakdown(monkeypatch):
    """`by_market` sits inside the headline block, so it is the same population.

    A headline from every counted pick beside a per-market table computed over
    the pre-kickoff subset is the same defect one level down, and it is the
    kind that survives review because each number is individually defensible.
    """
    frame = _frame([True] * 6 + [False] * 4, [BEFORE] * 6 + [AFTER] * 4)
    out = _pinned(monkeypatch, frame)

    assert out["by_market"]["match_result"]["n_resolved"] == 10, (
        f"by_market reports n_resolved={out['by_market']['match_result']['n_resolved']}, "
        f"expected 10. It sits inside the headline block, so it is read as the same population."
    )
    assert out["pre_kickoff"]["by_market"]["match_result"]["n_resolved"] == 6


def test_gameweek_trend_and_current_gameweek_follow_the_headline(monkeypatch):
    """Otherwise the trend chart and the gameweek tile silently keep the old rule."""
    frame = _frame([True] * 3 + [False] * 2, [BEFORE] * 3 + [AFTER] * 2, gameweeks=[5] * 5)
    out = _pinned(monkeypatch, frame)

    assert out["n_fixtures_current_gameweek"] == 5, (
        "the current-gameweek tile counts every counted pick, like the headline. Scoping it to "
        "the pre-kickoff subset is the same defect as a pre-kickoff-only headline."
    )
    assert out["pct_correct_current_gameweek"] == 0.6
    assert out["gameweek_trend"] == [{"gameweek": 5, "pct_correct": 0.6, "n_fixtures": 5}]


# --- 5. made_before_kickoff is DERIVED, on every read, failing closed --------

def test_every_per_pick_row_carries_its_own_flag_and_timestamp(monkeypatch):
    """The disclosure: labelled per pick, with the time beside it.

    A row that says only "hit: true" cannot be checked against the claim that
    the headline counts picks made after the start; a row that carries the
    timestamp can.
    """
    frame = _frame([True, False], [BEFORE, AFTER])
    rows = _pinned(monkeypatch, frame)["per_pick"]

    assert len(rows) == 2
    for row in rows:
        assert "made_before_kickoff" in row, f"row carries no made_before_kickoff: {row}"
        assert "snapshotted_at" in row, f"row carries no timestamp to check the flag against: {row}"
    flags = {bool(r["made_before_kickoff"]) for r in rows}
    assert flags == {True, False}, (
        f"every row reads made_before_kickoff={sorted(flags)}; each row must say when its own "
        f"pick was made, judged from its own timestamps."
    )


def test_a_pick_at_the_kickoff_instant_is_not_made_before_kickoff(monkeypatch):
    """At kickoff is not before it: a pick written in the same second the match
    starts had no chance to be made on the night."""
    frame = _frame([True], ["2026-09-13T14:00:00Z"])
    out = _pinned(monkeypatch, frame)

    assert out["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert out["n_rebuilt_fixtures"] == 1, "it counts; it is just not labelled as made in time"


@pytest.mark.parametrize("bad", [None, "", "not-a-date"])
def test_an_unparseable_timestamp_fails_closed_to_false(monkeypatch, bad):
    frame = _frame([True], [bad])
    rows = _pinned(monkeypatch, frame)["per_pick"]

    assert rows[0]["made_before_kickoff"] is False, (
        "an unprovable claim is not a verified one. Never backfill a true the timestamps do not "
        "support."
    )


def test_the_flag_is_not_read_from_a_stored_column(monkeypatch):
    """The strongest form of the rule: the input does not exist.

    A `backfilled` column on the frame is deliberately absent from `_frame`, and
    a row is added here carrying `backfilled = False` on a pick made after
    kickoff. If any code path read the flag, this row would be labelled
    pre-kickoff.
    """
    frame = _frame([True], [AFTER])
    frame["backfilled"] = False
    rows = _pinned(monkeypatch, frame)["per_pick"]

    assert rows[0]["made_before_kickoff"] is False, (
        "a stored flag said this pick was made in time; its own timestamps say otherwise and the "
        "timestamps win."
    )


def test_a_pick_made_after_kickoff_is_never_labelled_made_before_it(monkeypatch):
    """Rule 5, asserted directly rather than through the aggregate."""
    frame = _frame([True, True], ["2026-09-13T14:00:00Z", "2026-09-13T23:59:59Z"])
    rows = _pinned(monkeypatch, frame)["per_pick"]

    assert all(r["made_before_kickoff"] is False for r in rows)


# --- 6. THE STRADDLE TEST: adversarial, in both directions ------------------

def test_a_pick_whose_STRINGS_read_pre_kickoff_and_INSTANTS_read_post_is_called_post(monkeypatch):
    """The failure this guards is invisible: both answers are defensible.

    `10:00:00-05:00` sorts BEFORE `15:00:00Z` as a string and is 15:00 UTC, i.e.
    exactly AT the `15:00:00Z` kickoff. So the string order says pre-kickoff, the
    instant order says at-kickoff (not before), and the truth is the second.
    A `derived on every read` rule that compared strings would call this
    pre-kickoff and put a post-kickoff pick in the honest figure.

    The string order is asserted FIRST, as a precondition, so this test cannot
    quietly stop straddling: if a future edit changes either literal, the
    precondition fails and says why, rather than the test passing vacuously.
    """
    pick, kickoff = "2026-09-13T10:00:00-05:00", "2026-09-13T15:00:00Z"
    assert pick < kickoff, (
        f"precondition: {pick!r} must sort before {kickoff!r} as a string for this test to "
        f"straddle at all. Change the literals, not the claim."
    )
    frame = _frame([True], [pick], kickoff=kickoff)
    rows = _pinned(monkeypatch, frame)["per_pick"]

    assert rows[0]["made_before_kickoff"] is False, (
        "the instants say this pick was made AT kickoff, not before it. Comparing the strings "
        "would have called it pre-kickoff."
    )
    assert _pinned(monkeypatch, frame)["pre_kickoff"]["n_resolved_fixtures"] == 0


def test_the_mirror_pick_whose_STRINGS_read_post_and_INSTANTS_read_pre_is_called_pre(monkeypatch):
    """The opposite direction, because a one-sided straddle test is half a test.

    `23:00:00+09:00` sorts AFTER `13:00:00Z` as a string and is 14:00 UTC, half
    an hour before the `15:00:00Z` kickoff. String order says post-kickoff, the
    truth is pre-kickoff — and a string comparison would drop a genuinely
    in-time pick out of the honest figure.
    """
    pick, kickoff = "2026-09-13T23:00:00+09:00", "2026-09-13T15:00:00Z"
    assert pick > kickoff, (
        f"precondition: {pick!r} must sort AFTER {kickoff!r} as a string for this test to "
        f"straddle at all. Change the literals, not the claim."
    )
    frame = _frame([True], [pick], kickoff=kickoff)
    out = _pinned(monkeypatch, frame)

    assert out["per_pick"][0]["made_before_kickoff"] is True, (
        "the instants say this pick was made at 14:00 UTC, 30 minutes before the 15:00 UTC "
        "kickoff. Comparing the strings would have called it post-kickoff."
    )
    assert out["pre_kickoff"]["n_resolved_fixtures"] == 1, (
        "a pick genuinely made in time belongs in the honest figure, whatever its spelling."
    )


def test_a_naive_kickoff_is_read_as_utc_not_stripped(monkeypatch):
    """Sources mix tz-aware (Odds API) and naive (FPL fallback) kickoffs.

    13:30 at UTC-5 is 18:30 UTC — four and a half hours AFTER a naive 14:00
    kickoff. Stripping the offset compares 13:30 with 14:00 and calls it
    pre-kickoff, which is the bug `pick_record`'s own tests were written for.
    """
    frame = _frame([True], ["2026-09-13T13:30:00-05:00"], kickoff="2026-09-13T14:00:00")
    rows = _pinned(monkeypatch, frame)["per_pick"]
    assert rows[0]["made_before_kickoff"] is False


def test_a_tz_aware_pick_against_a_naive_kickoff_is_not_rejected_as_an_artefact(monkeypatch):
    frame = _frame([True], ["2026-09-13T10:00:00Z"], kickoff="2026-09-13T14:00:00")
    out = _pinned(monkeypatch, frame)

    assert out["pre_kickoff"]["n_resolved_fixtures"] == 1, (
        "10:00 UTC against a naive 14:00 kickoff is before it; a mixed-precision comparison must "
        "not reject a correct pick."
    )


# --- 7. the payload keeps answering every key a site reads ------------------

def test_field_names_are_kept_and_the_renames_are_the_ones_named(monkeypatch):
    """Nothing a site reads today changes meaning or disappears.

    `all_picks` is RETAINED under its published name and still means every
    counted pick; `n_rebuilt_fixtures` is RETAINED and now means counted picks
    made at or after their own kickoff. `pre_kickoff` is the only NEW key.
    """
    out = _pinned(monkeypatch, _frame([True] * 3, [BEFORE] * 2 + [AFTER]))

    for key in ("n_resolved_fixtures", "n_rebuilt_fixtures", "pct_correct_overall",
                "current_gameweek", "pct_correct_current_gameweek",
                "n_fixtures_current_gameweek", "gameweek_trend", "by_market",
                "all_picks", "per_pick", "pre_kickoff"):
        assert key in out, f"{key} is missing from the record: {sorted(out)}"
    assert out["all_picks"]["n_resolved"] == out["n_resolved_fixtures"], (
        "all_picks keeps its published meaning — every pick — which under the reversal is the "
        "same population as the headline. It is not silently pointed at a subset."
    )


def test_an_empty_record_still_answers_every_key(monkeypatch):
    """The frontend reads these keys unconditionally, so a missing one is a
    crash on the page rather than a null on a field."""
    out = _pinned(monkeypatch, pd.DataFrame())

    for key in ("n_resolved_fixtures", "n_rebuilt_fixtures", "pct_correct_overall",
                "current_gameweek", "pct_correct_current_gameweek",
                "n_fixtures_current_gameweek", "gameweek_trend", "by_market",
                "all_picks", "per_pick", "pre_kickoff"):
        assert key in out, f"{key} is missing from the empty record: {sorted(out)}"
    assert out["pct_correct_overall"] is None
    assert out["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert out["pre_kickoff"]["pct_correct_overall"] is None
    assert out["all_picks"]["n_resolved"] == 0
    assert out["per_pick"] == []
