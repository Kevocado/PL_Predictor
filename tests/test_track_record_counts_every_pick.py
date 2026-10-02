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
destructive for player picks, because many players share a fixture: collapsing a
pick to game level deletes the player record entirely. So the two record types
are keyed apart and the split is pinned here:

- fixture-level markets (match result, exact score, O/U 2.5, BTTS) -> one counted
  pick per `(fixture, market)`, which reduces to one row per fixture's
  `event_id` because all four markets live as columns on that row;
- player picks -> one counted pick per `(event_id, player_id)`.

Neither key names a `market`, and both tables have no such column: the markets
are columns on the row. Both keys are also the id keys NFL/CFB/NBA use, so
"which fixture is this" is answered the same way in every sport.

**A rule is only enforced where it is called.** Both dedupes used to be
asserted here and reachable from serving in one case only:
`_counted_player_picks` was called from no production path at all, and
`get_scorer_accuracy` graded every resolved row. The tests below now go through
the entry points.
"""
import sqlite3

import pandas as pd
import pytest

from pl_predictor.tracking import store


def _frame(hits, snapshotted_at, kickoff="2026-09-13T14:00:00Z", gameweeks=None,
           event_ids=None, teams=None):
    """A resolved-fixture frame in the shape `get_track_record` reads.

    `snapshotted_at` and `commence_time` are the ONLY inputs to
    `made_before_kickoff` — there is no `backfilled` column here at all, on
    purpose. The rule is derived from timestamps, so a test that supplied a
    flag could pass while the derivation was wrong.

    Team names are derived from `event_id`, so two rows carrying the same
    `event_id` are one fixture (the rerun case) and different ids are different
    fixtures. Pass `teams` to override that and pin the case where identity
    must NOT come from team names.
    """
    n = len(hits)
    ids = event_ids if event_ids is not None else [f"e{i}" for i in range(n)]
    pairs = teams if teams is not None else [(f"Home {i}", f"Away {i}") for i in ids]
    return pd.DataFrame({
        "hit": hits,
        "snapshotted_at": snapshotted_at,
        "commence_time": [kickoff] * n,
        "event_id": ids,
        "team_home": [p[0] for p in pairs],
        "team_away": [p[1] for p in pairs],
        "gameweek": gameweeks if gameweeks is not None else [1] * n,
        # `_fixture_market_hit_table`'s real column set, so a payload read off
        # this frame exercises the same field accesses a real one does.
        "predicted_scoreline": [f"{2 if h else 1}-0" for h in hits],
        "actual_goals_home": [2 if h else 0 for h in hits],
        "actual_goals_away": [0 if h else 1 for h in hits],
        "exact_score_hit": hits,
        "over_under_hit": hits,
        "btts_hit": hits,
        "predicted_home_win": [0.6] * n,
        "predicted_draw": [0.25] * n,
        "predicted_away_win": [0.15] * n,
        "actual_outcome": ["home_win"] * n,
        "predicted_prob_actual": [0.6] * n,
        "resolved_at": ["2026-09-13T18:00:00"] * n,
    })


BEFORE = "2026-09-13T10:00:00Z"
AFTER = "2026-09-13T15:00:00Z"


def _pinned(monkeypatch, frame):
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: frame)
    return store.get_track_record()


def _player_frame(rows, order_by_instant=None):
    """A resolved-player frame in the shape `get_scorer_accuracy` really reads.

    Columns are exactly `player_prediction_snapshots`'s — `event_id`,
    `player_id`, `name`, `team`, the three probabilities, `confirmed_starter`,
    `qualifies_call`, `is_recommended`, `provenance`, the two actuals and
    `resolved`. Built here rather than against a live table so a primary-key
    violation can be staged, which is the only way to exercise a reader-side
    dedupe: `INSERT OR IGNORE` on `(event_id, player_id)` makes duplicates
    unreachable through the writer.

    `rows` are `(event_id, player_id, goal_probability, actual_goals)`.
    `order_by_instant` lists the rows as latest-first, so an entry at index `i`
    is stamped BEFORE when `True` and AFTER when False.
    """
    order = order_by_instant or [True] * len(rows)
    return pd.DataFrame({
        "event_id": [r[0] for r in rows],
        "player_id": [r[1] for r in rows],
        "name": [f"P{r[1]}" for r in rows],
        "team": ["Home"] * len(rows),
        "goal_probability": [r[2] for r in rows],
        "assist_probability": [0.10] * len(rows),
        "contribution_probability": [max(0.30, r[2]) for r in rows],
        "confirmed_starter": [1] * len(rows),
        "qualifies_call": [1] * len(rows),
        "is_recommended": [1] * len(rows),
        "provenance": ["snapshot"] * len(rows),
        "actual_goals": [r[3] for r in rows],
        "actual_assists": [0] * len(rows),
        "resolved": [1] * len(rows),
        # The table carries no `snapshotted_at`, so this column is what
        # `_instants` orders on when present and is absent in production. Kept
        # here so "earliest wins" is testable.
        "snapshotted_at": [BEFORE if early else AFTER for early in order],
    })


def _unkeyed_player_db(tmp_path, frame):
    """A real SQLite connection holding `frame`, with the table's PRIMARY KEY
    deliberately left off.

    `_connect` is monkeypatched over this, so `get_scorer_accuracy` runs its own
    `SELECT * FROM player_prediction_snapshots WHERE resolved = 1` and its own
    grading logic against it. Only the schema is arranged: the columns are
    `frame`'s, and the constraint is not, because the point of these tests is
    rows the table's own key forbids. A reader-side dedupe has nothing to defend
    against when the writer already makes duplicates unreachable, which is
    exactly why the gap between the two went unnoticed.
    """
    conn = sqlite3.connect(str(tmp_path / "tracking.db"))
    sql_types = {
        "object": "TEXT",
        "int64": "INTEGER",
        "int32": "INTEGER",
        "float64": "REAL",
        "bool": "INTEGER",
    }
    schema = ", ".join(
        f"{name} {sql_types.get(str(dtype), 'REAL')}" for name, dtype in frame.dtypes.items()
    )
    conn.execute(f"CREATE TABLE player_prediction_snapshots ({schema})")
    placeholders = ",".join("?" * len(frame.columns))
    conn.executemany(
        f"INSERT INTO player_prediction_snapshots VALUES ({placeholders})",
        [tuple(row) for row in frame.itertuples(index=False)],
    )
    conn.commit()
    return conn


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
        f"key is one counted pick per (fixture, market), which reduces to one row per "
        f"`event_id`; a rerun must not grade it twice."
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


# --- 2b. the fixture key is the event_id, like every other sport ------------

def test_the_fixture_counting_key_is_the_event_id_and_nothing_derived_from_it(monkeypatch):
    """Two fixtures are two fixtures, even when they share a team pair.

    The key used to be `(team_home, team_away, kickoff date)` — a guess about
    fixture identity re-derived from team names and a date. Both halves of that
    guess are lossy, and this test is the case where they lose a fixture
    outright: an UNREADABLE `commence_time` yields `kickoff_date=None`, and
    pandas' `drop_duplicates` treats every null as equal, so two genuinely
    distinct matches between the same two teams collapsed into one counted pick.

    Same teams, no readable kickoff, two event_ids, opposite verdicts. If the
    key is `event_id`, both count.
    """
    frame = _frame(
        [True, False],
        [BEFORE, BEFORE],
        kickoff="not-a-date",
        event_ids=["e1", "e2"],
        teams=[("Arsenal", "Chelsea"), ("Arsenal", "Chelsea")],
    )
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 2, (
        f"two distinct event_ids between the same teams collapsed to {out['n_resolved_fixtures']} "
        f"counted pick. Identity is the event_id; a null kickoff date is not a shared identity."
    )
    assert out["pct_correct_overall"] == 0.5
    assert len(out["per_pick"]) == 2, (
        "per_pick must list what produced the headline, so a reader tallying it gets 2"
    )


def test_two_event_ids_straddling_a_UTC_midnight_are_two_fixtures(monkeypatch):
    """The other half of the same guess, stated so the choice is visible.

    One kickoff at 23:50 UTC and another at 00:10 UTC are different matches. If
    identity were still partly date-derived they would land on adjacent calendar
    dates and the earlier one would swallow the later one; keyed on `event_id`
    they are simply two ids.

    This is a deliberate choice, not an oversight: PL is the only sport that
    treated two ids for one match as one fixture, and the cost was that a real
    double-booking — the same two teams twice in a day, or a replay — became one
    counted pick, or none at all when the kickoff could not be read at all. The
    upstream fix for genuinely-duplicated fixtures is to de-duplicate the fixture
    feed, which is a place where the two ids can be compared with their odds and
    kickoff in hand. A reader-side date heuristic has neither.
    """
    frame = pd.DataFrame({
        "hit": [True, False],
        "snapshotted_at": [BEFORE, BEFORE],
        "commence_time": ["2026-09-13T23:50:00Z", "2026-09-14T00:10:00Z"],
        "event_id": ["e_late_night", "e_after_midnight"],
        "team_home": ["Arsenal", "Arsenal"],
        "team_away": ["Chelsea", "Chelsea"],
        "gameweek": [1, 1],
        "exact_score_hit": [True, False],
        "over_under_hit": [True, False],
        "btts_hit": [True, False],
        "actual_outcome": ["home_win", "away_win"],
        "predicted_prob_actual": [0.6, 0.6],
    })
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 2, (
        f"a kickoff at 23:50Z and one at 00:10Z the next day are two matches, and they count "
        f"as {out['n_resolved_fixtures']}."
    )
    assert [r["event_id"] for r in out["per_pick"]] == ["e_late_night", "e_after_midnight"]


def test_a_rerun_under_one_event_id_still_counts_once_after_the_key_change(monkeypatch):
    """Aligning the key must not have cost the rerun rule.

    This is the invariant the event_id key exists to preserve, so it is asserted
    against the key itself rather than inferred: two rows, one id, opposite
    verdicts, and the EARLIEST is the counted one.
    """
    frame = _frame([True, False], [BEFORE, AFTER], event_ids=["e1", "e1"])
    out = _pinned(monkeypatch, frame)

    assert out["n_resolved_fixtures"] == 1
    assert out["pct_correct_overall"] == 1.0, (
        "the earliest row is the hit; the later rerun that missed must not displace it"
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


def test_player_picks_are_keyed_by_fixture_and_player_not_by_fixture_alone():
    """The destructive-key check, pinned against the real schema's columns.

    Many players share one fixture. Keyed by `event_id` alone, every player but
    one is dropped and the player record is deleted outright.

    The frame below carries exactly the columns a real
    `SELECT * FROM player_prediction_snapshots` returns — no `market`, because
    that table has PRIMARY KEY `(event_id, player_id)` and keeps goal/assist/G+A
    as COLUMNS. This was written against a hand-built frame that had a `market`
    column, so `_counted_player_picks`'s `drop_duplicates(subset=[..., "market"])`
    passed in tests and raised `KeyError: Index(['market'])` on every real read.
    A test whose fixture invents a column the table does not have cannot catch
    that, which is why the columns here come from the schema.
    """
    frame = _player_frame([
        ("e1", 1, 0.30, 1),
        ("e1", 2, 0.45, 1),
    ])

    counted = store._counted_player_picks(frame)

    keys = set(zip(counted["event_id"], counted["player_id"]))
    assert keys == {("e1", 1), ("e1", 2)}, (
        f"player picks collapsed to {sorted(keys)}. Two players in one fixture must both "
        f"survive — the player record is the record."
    )
    assert set(counted.columns).issuperset({"goal_probability", "assist_probability"}), (
        "keying on a market column, or collapsing a player row to one market, would have "
        "dropped the other markets off the row. They are columns; all three must survive."
    )


def test_a_player_rerun_counts_once_and_the_earliest_row_wins():
    """Otherwise a re-keyed or restored history grades one pick twice."""
    # Listed latest-first, so a dedupe that simply kept the first row it saw
    # would keep the 0.60 miss. The 0.25 row is stamped BEFORE, so it must win.
    counted = store._counted_player_picks(_player_frame([
        ("e1", 1, 0.60, 0),  # stamped AFTER -- a missed goal call
        ("e1", 1, 0.25, 1),  # stamped BEFORE -- the counted pick, a hit
    ], order_by_instant=[False, True]))

    assert len(counted) == 1
    assert float(counted.iloc[0]["goal_probability"]) == 0.25, (
        f"the counted row is the 0.60 one, stamped AFTER. The earliest recorded player pick is "
        f"the counted one, whatever order the rows arrived in."
    )


def test_the_player_grader_deduplicates_on_the_real_schema(monkeypatch, tmp_path):
    """The dead-code defect, as behaviour rather than as a docstring.

    `_counted_player_picks` was asserted in this file and called from NOTHING in
    serving: `get_scorer_accuracy` read every resolved row and never deduped. So
    the key it enforced did not exist as far as any consumer was concerned, and
    it happened to be a key the table has no column for.

    Two rows for one `(event_id, player_id)` cannot happen today — the primary
    key forbids it — which is precisely why the rule has to be written in the
    reader and precisely why a silent gap here goes unnoticed. So they are
    staged through a frame that bypasses the constraint, and the assertion is on
    what `get_scorer_accuracy` RETURNS: one call graded, not two.
    """
    rows = _player_frame([
        ("e1", 1, 0.60, 1),   # late rerun: a MISSED goal call
        ("e1", 1, 0.25, 1),   # earliest: a HIT
        ("e1", 2, 0.40, 1),   # a different player in the same fixture: also HIT
    ], order_by_instant=[False, True, True])
    monkeypatch.setattr(store, "_connect", lambda: _unkeyed_player_db(tmp_path, rows))

    out = store.get_scorer_accuracy()

    snapshot = out["snapshot"]
    assert snapshot["calls"] == 2, (
        f"the grader reports {snapshot['calls']} calls for 2 counted picks out of 3 rows. "
        f"The duplicate (event_id, player_id) must be deduped to the EARLIEST row."
    )
    assert snapshot["call_hits"] == 2, (
        "the earliest row for player 1 is the HIT. Grading the later 0.60 rerun instead "
        "would report a miss and quietly understate the model."
    )
    assert snapshot["call_hit_rate"] == 1.0
    # The calibration buckets are graded over the same counted rows.
    assert sum(bucket["n"] for bucket in snapshot["calibration"]) == 2, (
        "calibration must describe the same counted population as `calls`, or the two "
        f"disagree about how many picks exist: {snapshot['calibration']}"
    )


def test_the_player_grader_survives_a_table_that_lacks_the_market_column(monkeypatch, tmp_path):
    """The KeyError, asserted directly against real columns.

    Before the fix, `drop_duplicates(subset=["event_id", "player_id", "market"])`
    raised `KeyError: Index(['market'])` on the frame this grader actually
    reads. Asserted on the same `SELECT *` column set `_connect` produces.
    """
    rows = _player_frame([("e1", 1, 0.30, 1)])
    assert "market" not in rows.columns, (
        "precondition: the real table has no `market` column. If this fails the fixture no "
        "longer reproduces the schema the defect lived in."
    )

    counted = store._counted_player_picks(rows)

    assert len(counted) == 1
    monkeypatch.setattr(store, "_connect", lambda: _unkeyed_player_db(tmp_path, rows))
    assert store.get_scorer_accuracy()["snapshot"]["calls"] == 1


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


def test_the_gameweek_group_and_the_track_record_agree_on_pre_kickoff(monkeypatch):
    """One derivation, three surfaces — asserted across all three at once.

    `get_track_record`'s `pre_kickoff.n_resolved_fixtures`, the gameweek
    group's `n_made_before_kickoff`, and each fixture row's `made_before_kickoff`
    are three readings of the same question. `get_results_by_gameweek` used to
    answer it from the stored `backfilled` flag, so a gameweek could report a
    `n_rebuilt` that disagreed with the headline printed right above it, and the
    gameweek card rendered that as "picks made before kickoff correct" on a
    surface nobody thought to check against the record.

    `backfilled` is deliberately set INCONSISTENTLY with the timestamps below:
    the pre-kickoff pick is flagged `backfilled=True` and the post-kickoff pick
    `backfilled=False`, which is exactly what a live tracking tick produces. A
    reading of the flag gets both answers wrong.
    """
    frame = _frame(
        [True, True, False],
        [BEFORE, AFTER, BEFORE],
        gameweeks=[4, 4, 4],
    )
    frame["backfilled"] = [True, False, False]
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: frame)

    record = store.get_track_record()
    groups = store.get_results_by_gameweek()
    group = next(g for g in groups if g["gameweek"] == 4)

    # 2 picks made before their own kickoff, 1 after.
    assert record["pre_kickoff"]["n_resolved_fixtures"] == 2
    assert group["n_made_before_kickoff"] == 2, (
        f"the gameweek group reports {group['n_made_before_kickoff']} picks made in time against "
        f"the track record's {record['pre_kickoff']['n_resolved_fixtures']} for the same fixtures. "
        f"`n_rebuilt` used to count the `backfilled` flag instead, so the two could disagree."
    )
    assert group["n_rebuilt"] == 1
    assert group["n_rebuilt"] == record["n_rebuilt_fixtures"], (
        "same meaning on both surfaces: counted picks made at or after their own kickoff"
    )
    assert group["n_rebuilt"] + group["n_made_before_kickoff"] == group["n_fixtures"], (
        "the group's own arithmetic must reconcile, or the split is not a split"
    )

    # And every fixture row carries its own derived label, checkable against the
    # timestamp beside it.
    labels = {r["event_id"]: r["made_before_kickoff"] for r in group["fixtures"]}
    assert labels == {"e0": True, "e1": False, "e2": True}, (
        f"per-fixture labels are {labels}. They come from each pick's own two timestamps, and "
        f"the `backfilled` flags on this frame disagree with them on purpose."
    )
    for row in group["fixtures"]:
        assert row["snapshotted_at"], (
            "a timing label nobody can check is a timing label nobody can audit"
        )


def test_an_unreadable_timestamp_does_not_invent_a_gameweek_pre_kickoff_figure(monkeypatch):
    """Fails closed, at group level.

    One pick in this gameweek has an unreadable `snapshotted_at`. It cannot be
    placed in time, so it is counted (rule 1) but not in the pre-kickoff figure,
    and the group's arithmetic still has to reconcile — a figure invented to
    make the numbers balance would be worse than the absence.
    """
    frame = _frame([True, True], ["not-a-date", "also-not-a-date"], gameweeks=[2, 2])
    # `_frame` omits `backfilled` on purpose everywhere else; the gameweek
    # payload carries it through, so this test supplies it explicitly — and
    # deliberately sets it to False on both rows, which is the case a reader of
    # the flag would get wrong in the optimistic direction.
    frame["backfilled"] = [False, False]
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: frame)

    record = store.get_track_record()
    group = next(g for g in store.get_results_by_gameweek() if g["gameweek"] == 2)

    assert record["n_resolved_fixtures"] == 2, "recorded stays recorded"
    assert record["pre_kickoff"]["n_resolved_fixtures"] == 0
    assert record["pre_kickoff"]["pct_correct_overall"] is None, (
        "0.0% over zero provable picks is a rate computed over nothing"
    )
    assert group["n_made_before_kickoff"] == 0
    assert group["n_rebuilt"] == 2, (
        "an unprovable pick is reported under the late side rather than quietly counted as in-time"
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
