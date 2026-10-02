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
