"""B8 on PL: the headline is pre-kickoff picks only, and nothing is hidden.

The live payload said `n_resolved_fixtures: 50, n_rebuilt_fixtures: 50,
pct_correct_overall: 0.52` — the headline was computed entirely from picks
rebuilt after kickoff, and the "52%" carried look-forward bias the product
exists to avoid. NFL already does this (NFL#23 / Sports#13): headline from
pre-kickoff picks, all-picks figure beside it, per-pick rows carrying
`rebuilt`. This is the same ruling applied to PL, and these tests are what
makes the two comparable rather than merely similar.

Two things are being pinned, and they are not the same thing:

* the headline excludes rebuilt picks. Asserted by giving the two populations
  different accuracies, so a headline computed over the wrong one cannot pass.
* nothing is discarded. The all-picks figure and the per-pick rows cover every
  resolved fixture including rebuilt ones, so a reader can still see them.
  Without this second half, "fix the headline" is satisfiable by hiding
  everything, which is the opposite of the ruling.
"""
import pandas as pd
import pytest

from pl_predictor.tracking import store


def _frame(hits, rebuilt, gameweeks=None):
    """A resolved-fixture frame in the shape `get_track_record` reads."""
    n = len(hits)
    return pd.DataFrame({
        "hit": hits,
        "backfilled": rebuilt,
        "gameweek": gameweeks if gameweeks is not None else [1] * n,
        "exact_score_hit": hits,
        "over_under_hit": hits,
        "btts_hit": hits,
    })


@pytest.fixture
def patched(monkeypatch):
    """Pin the two frame sources, so the test is about the two populations."""
    def _set(pre_kickoff, all_resolved):
        monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: all_resolved)
    return _set


def test_the_headline_ignores_rebuilt_picks(monkeypatch):
    """Different accuracies, so a mixed headline cannot pass by accident.

    The pre-kickoff picks are all hits; the rebuilt ones are all misses. A
    headline over both is 50%, which is a real number and a wrong one. The
    headline here must be 100%, and `None` if there were no pre-kickoff picks.
    """
    pre = _frame([True] * 10, [False] * 10, gameweeks=[1] * 10)
    rebuilt = _frame([False] * 10, [True] * 10, gameweeks=[1] * 10)
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: pd.concat([pre, rebuilt]))

    out = store.get_track_record()

    assert out["pct_correct_overall"] == 1.0, (
        f"headline is {out['pct_correct_overall']}, which is the mixed 50%. The headline "
        f"must come from pre-kickoff picks only."
    )
    assert out["n_resolved_fixtures"] == 10, (
        "the headline count must be the pre-kickoff count, not every resolved fixture"
    )
    assert out["n_rebuilt_fixtures"] == 10, "rebuilt picks must still be counted and reported"


def test_a_null_headline_while_there_are_no_pre_kickoff_picks(monkeypatch):
    """0 out of 0 is not 0%. It is an absence, and it must read as one.

    The live site has exactly this state in practice: every stored pick is
    rebuilt. Reporting 0.0% there would be a claim about accuracy with no picks
    behind it, which is the specific dishonesty B8 exists to prevent.
    """
    only_rebuilt = _frame([True] * 7, [True] * 7, gameweeks=[1] * 7)
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: only_rebuilt)

    out = store.get_track_record()

    assert out["pct_correct_overall"] is None, (
        f"headline is {out['pct_correct_overall']!r} with zero pre-kickoff picks. There is "
        f"no accuracy to report, and 0.0% would be a rate computed over nothing."
    )
    assert out["n_resolved_fixtures"] == 0
    assert out["n_rebuilt_fixtures"] == 7, "the rebuilt picks must remain visible"


def test_the_all_picks_figure_covers_rebuilt_too(monkeypatch):
    """The other half of B8, and the one a "fix" can quietly skip.

    The headline is scoped; nothing is deleted. This figure is computed over the
    UNFILTERED frame, so a reader who wants the mixed number can still have it —
    just labelled, instead of passed off as the honest one.
    """
    pre = _frame([True] * 10, [False] * 10, gameweeks=[1] * 10)
    rebuilt = _frame([False] * 10, [True] * 10, gameweeks=[1] * 10)
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: pd.concat([pre, rebuilt]))

    out = store.get_track_record()
    all_picks = out["all_picks"]

    assert all_picks["n_resolved"] == 20, (
        f"all_picks covers {all_picks['n_resolved']} fixtures, expected all 20. Scoping the "
        f"headline must not also drop the rebuilt picks from the record."
    )
    assert all_picks["pct_correct"] == 0.5, "and the all-picks figure is the mixed rate"


def test_per_pick_rows_say_which_picks_were_rebuilt(monkeypatch):
    """Per-pick rows, each labelled.

    This is what lets a reader audit the headline instead of trusting it: a row
    that does not say whether it was made in time cannot be checked against the
    claim that only in-time picks count.
    """
    pre = _frame([True] * 4, [False] * 4, gameweeks=[1] * 4)
    rebuilt = _frame([False] * 4, [True] * 4, gameweeks=[1] * 4)
    frame = pd.concat([pre, rebuilt])
    for col, val in (("team_home", "Arsenal"), ("team_away", "Chelsea"),
                     ("actual_outcome", "home_win"), ("predicted_prob_actual", 0.6)):
        frame[col] = val
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: frame)

    rows = store.get_track_record()["per_pick"]

    assert len(rows) == 8, f"expected 8 per-pick rows, got {len(rows)}"
    flags = {bool(r["rebuilt"]) for r in rows}
    assert flags == {True, False}, (
        f"every row carries rebuilt={sorted(flags)}; each row must say whether its pick "
        f"was made before kickoff"
    )


def test_the_market_breakdown_follows_the_same_population(monkeypatch):
    """by_market sits inside the headline block, so it must be the same subset.

    A headline from pre-kickoff picks beside a per-market table computed over
    all of them is the same defect one level down, and it is the kind that
    survives a review because each number is individually defensible.
    """
    pre = _frame([True] * 6, [False] * 6, gameweeks=[1] * 6)
    rebuilt = _frame([False] * 6, [True] * 6, gameweeks=[1] * 6)
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: pd.concat([pre, rebuilt]))

    out = store.get_track_record()
    n = out["by_market"]["match_result"]["n_resolved"]

    assert n == 6, (
        f"by_market reports n_resolved={n}, expected 6 (the pre-kickoff subset). It sits "
        f"inside the headline block, so it is read as the same population."
    )


def test_an_empty_record_still_answers_every_key(monkeypatch):
    """No fixtures at all: every key present, headline null, all-picks zero.

    The frontend reads these keys unconditionally, so a missing one is a
    crash on the page rather than a null on a field.
    """
    monkeypatch.setattr(store, "_fixture_market_hit_table", lambda: pd.DataFrame())

    out = store.get_track_record()

    for key in ("n_resolved_fixtures", "n_rebuilt_fixtures", "pct_correct_overall",
                "current_gameweek", "pct_correct_current_gameweek",
                "n_fixtures_current_gameweek", "gameweek_trend", "by_market",
                "all_picks", "per_pick"):
        assert key in out, f"{key} is missing from the empty record: {sorted(out)}"
    assert out["pct_correct_overall"] is None
    assert out["all_picks"]["n_resolved"] == 0
    assert out["per_pick"] == []
