"""The AI explainer's record strip must keep reading the PRE-KICKOFF figure.

The reversal changed what the track record's HEADLINE means: it is now every
counted pick, whenever it was made (spec 2026-10-01-track-record-counts-every-
pick). It did not change what the explainer may say.

`facts._record()` builds a block labelled **"Picks made before kick-off"**, and
that label is a claim. A started fixture's bundle quotes the stored,
earliest-recorded pick — a pick made on the night — beside it. If the block read
the new headline, it would put picks made after the start under a heading that
says they were made before it, on the surface a reader trusts most.

So this reads `pre_kickoff`, the same way NFL's facts bundle reads it, and the
difference is asserted with two DIFFERENT numbers so a change back cannot pass.
"""
from fastapi.testclient import TestClient

from pl_predictor.api import facts as facts_mod
from pl_predictor.api.main import app

# The shared builders come from `conftest`, not from `tests.test_facts`.
# `from tests.test_facts import ...` needs the repo root on sys.path so that
# `tests` resolves as a package, and there is no `tests/__init__.py`, no
# pythonpath and no rootdir config -- so it resolved on a developer's machine
# and failed CI collection with `ModuleNotFoundError: No module named 'tests'`.
# conftest is loaded by pytest for every test in this directory, so importing
# from it cannot depend on cwd or on any path the repo does not declare.
from conftest import (  # noqa: E402  (path set by pytest's rootdir insertion)
    FACTS_EVENT_ID as EVENT_ID,
    facts_card as _card,
    facts_snapshot as _snapshot,
)


def _client(monkeypatch, record_payload):
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot())
    monkeypatch.setattr(
        facts_mod.routes.tracking_store, "get_track_record", lambda: record_payload
    )
    return TestClient(app)


# Headline: every counted pick, 50 of them, 56% correct. Pre-kickoff subset:
# 12 of them, 50% correct. Two different numbers on purpose.
PAYLOAD = {
    "n_resolved_fixtures": 50,
    "pct_correct_overall": 0.56,
    "n_rebuilt_fixtures": 38,
    "pre_kickoff": {"n_resolved_fixtures": 12, "pct_correct_overall": 0.5},
}


def test_the_record_block_reads_the_pre_kickoff_figure(monkeypatch):
    """Not the headline.

    The headline is 50 picks at 56%; the pre-kickoff subset is 12 at 50%. A
    block labelled "Picks made before kick-off" that reported 28/50 would be
    quoting 38 picks made after the start as though they were made before it.
    """
    client = _client(monkeypatch, PAYLOAD)
    body = client.get(f"/facts/{EVENT_ID}").json()

    assert body["record"] == {"label": "Picks made before kick-off", "hits": 6, "settled": 12}, (
        f"the record block read {body['record']}. Its label says 'made before kick-off', so it "
        f"must summarise the pre_kickoff subset — 6 of 12 — and not the headline's 28 of 50."
    )


def test_the_block_is_absent_when_no_pick_was_made_in_time(monkeypatch):
    """PL's shipped state: 50 graded picks, none made before kickoff.

    Under the reversal the headline is 50/56% and NOT absent — so this test is
    also what proves `_record` did not simply start reading the headline. The
    explainer still declines to quote a pre-game record it does not have.
    """
    client = _client(monkeypatch, {
        "n_resolved_fixtures": 50, "pct_correct_overall": 0.56, "n_rebuilt_fixtures": 50,
        "pre_kickoff": {"n_resolved_fixtures": 0, "pct_correct_overall": None},
    })
    body = client.get(f"/facts/{EVENT_ID}").json()

    assert body["record"] is None, (
        f"the record block reported {body['record']} with zero picks made before kickoff. A rate "
        f"over nothing is not a record, and the headline's 50 late picks do not make one."
    )


def test_a_null_rate_over_zero_picks_does_not_become_zero(monkeypatch):
    """`None`, not 0.0: the count is the thing that can be zero."""
    client = _client(monkeypatch, {
        "n_resolved_fixtures": 3, "pct_correct_overall": 0.33, "n_rebuilt_fixtures": 3,
        "pre_kickoff": {"n_resolved_fixtures": 0, "pct_correct_overall": None},
    })
    body = client.get(f"/facts/{EVENT_ID}").json()
    assert body["record"] is None


def test_the_record_block_and_the_fixtures_pick_timing_agree_on_one_derivation(monkeypatch):
    """Two claims about the same subset, both in one payload.

    `record` says how many picks were made before kick-off, and `pick_timing`
    says whether THIS fixture's pick was one of them. Both used to read the
    stored `backfilled` flag, which is provenance — a column whose writer takes
    it as an argument — so the two could describe different populations under one
    heading, and neither had to agree with `get_track_record`'s `pre_kickoff`.

    Asserted with a flag that disagrees with the derived timing in both
    directions: the card is `backfilled=True` (the backfill job wrote it) yet its
    timestamps put it before kickoff. `pick_timing` follows the timestamps, and
    the record block still reports the pre-kickoff subset the summary declared.
    """
    client = _client(monkeypatch, PAYLOAD)
    card = _card(commence_time="2026-11-01T14:00:00Z", finished=True, backfilled=True)
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(cards=[card]))

    body = client.get(f"/facts/{EVENT_ID}").json()

    assert body["pick_timing"] == "pre_kickoff", (
        f"pick_timing is {body['pick_timing']!r}. This card is flagged backfilled, but "
        f"made_before_kickoff says its own timestamps put the pick before the start, and that "
        f"is the question the field answers."
    )
    assert body["record"] == {"label": "Picks made before kick-off", "hits": 6, "settled": 12}
    assert body["record"]["settled"] == PAYLOAD["pre_kickoff"]["n_resolved_fixtures"], (
        "the block and the summary must be the same population, so a fixture labelled "
        "pre_kickoff here is counted among the picks `record` says were made in time"
    )


def test_a_late_pick_is_not_counted_in_the_block_it_contradicts(monkeypatch):
    """The other direction: a graded-looking pick must not sit under a pre-kickoff heading."""
    client = _client(monkeypatch, PAYLOAD)
    card = _card(commence_time="2026-11-01T14:00:00Z", finished=True, backfilled=False)
    card["made_before_kickoff"] = False
    monkeypatch.setattr(facts_mod, "_snapshot", lambda: _snapshot(cards=[card]))

    body = client.get(f"/facts/{EVENT_ID}").json()

    assert body["pick_timing"] == "rebuilt"
    assert "pick_won" not in (body["result"] or {}), (
        "this pick was not made before kickoff, so it is shown and never judged"
    )
