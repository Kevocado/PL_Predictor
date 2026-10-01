"""The durable pick record: what it guarantees, and what it must refuse.

The bug this exists to end: `tracking.db` is gitignored and was kept across CI
runs only by an `actions/cache` entry, which is not storage. Caches are evicted
after seven days of no access and are not written at all when the job that would
write them fails. So the picks — the evidence behind a published accuracy figure
— were one eviction or one failed job away from gone, and the site went back to a
headline computed entirely from picks made after kickoff. That is the state this
repo actually shipped in (`n_rebuilt` 50 of 50).

**What this file asserts, after the 2026-10-01 reversal:** the record holds
EVERY pick, labelled with when it was made. It no longer refuses a post-kickoff
pick — refusing it is what kept the file empty (0 lines) and the headline null.
`test_pick_record_records_every_pick.py` is where that half lives; this file
holds the properties that were already true and had to SURVIVE the reversal.

1. it is append-only and idempotent — a re-run appends nothing, so a no-op
   workflow run is an empty diff rather than a row of noise;
2. **it never rewrites a recorded pick.** Once a line exists it is evidence of
   what the model said and when, and a later run cannot amend it even if it would
   have written a different number. A record that can be corrected is not a
   record;
3. it never stores a derived verdict (`resolved`, `actual_outcome`).

Timing derivation — the offsets, the failure-closed label, and what is refused —
is in `test_pick_record_records_every_pick.py` and `test_pick_record_hardening.py`.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pl_predictor.tracking import pick_record

KICKOFF = "2026-09-13T14:00:00Z"


def row(event_id="e1", market="match_result", outcome="home_win", prob=0.6,
        snapshotted_at="2026-09-13T10:00:00Z", commence_time=KICKOFF):
    return {
        "event_id": event_id, "team_home": "Arsenal", "team_away": "Chelsea",
        "market": market, "outcome_name": outcome, "predicted_prob": prob,
        "commence_time": commence_time, "snapshotted_at": snapshotted_at,
        # Present in the database, deliberately not recorded: grading is
        # derived from the result at read time and must not be frozen here.
        "resolved": 1, "actual_outcome": 1,
    }


# --- 1. append-only and idempotent --------------------------------------------

def test_a_rerun_appends_nothing(tmp_path):
    """Why the file is safe to commit on every run: an unchanged workflow
    produces an empty diff, so a no-op run is invisible in history."""
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row()], path)
    first = path.read_text()
    assert pick_record.append_picks([row()], path) == 0
    assert path.read_text() == first
    assert len(path.read_text().splitlines()) == 1


def test_a_new_pick_is_appended_to_the_existing_file(tmp_path):
    """Appending, not rewriting — a rebuild must not drop what is already
    recorded, which is the whole property."""
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row(event_id="e1")], path)
    pick_record.append_picks([row(event_id="e2")], path)
    ids = [json.loads(l)["event_id"] for l in path.read_text().splitlines()]
    assert ids == ["e1", "e2"]


def test_a_malformed_line_does_not_stop_later_runs(tmp_path):
    """A half-written line from an interrupted run must not make every
    subsequent run fail."""
    path = tmp_path / "picks.jsonl"
    path.write_text('{"broken\n')
    assert pick_record.append_picks([row(event_id="e2")], path) == 1


# --- 2. the guarantee that matters --------------------------------------------

def test_a_recorded_pick_is_never_amended_by_a_later_different_value(tmp_path):
    """Once written, a pick is evidence of what the model said.

    A later run that would have recorded a different probability must not
    rewrite history. This is asserted on the FILE, not on the helper's
    return value, because a helper that reports "0 new" while having
    overwritten the line would satisfy every other test in this file.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row(prob=0.61)], path)
    pick_record.append_picks([row(prob=0.33)], path)
    recorded = json.loads(path.read_text().strip())
    assert recorded["predicted_prob"] == 0.61, "a recorded pick was amended"


def test_the_record_excludes_the_derived_verdict(tmp_path):
    """`resolved` and `actual_outcome` are deliberately not stored.

    Grading is derived from the fixture result at read time; baking it in would
    freeze a verdict that later data may correct, which would make the record
    self-confirming.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row()], path)
    stored = json.loads(path.read_text().strip())
    assert "resolved" not in stored
    assert "actual_outcome" not in stored


# --- 3. it is the record, not just a cache --------------------------------------

def test_the_record_is_tracked_by_git_and_the_cache_is_not(tmp_path):
    """The distinction the whole change rests on.

    `actions/cache` is not storage. This asserts the two artefacts are in
    different states on purpose, because a future editor adding
    `data/tracking_picks.jsonl` to `.gitignore` — to "tidy up" alongside the
    other `data/` entries — would reintroduce the original bug with no test
    failing anywhere else.

    `git ls-files` rather than `check-ignore`: a file can be ignored AND
    tracked (an ignore rule does not apply to tracked paths), and the failure
    that matters is "not tracked".
    """
    root = Path(pick_record.picks_path()).parents[1]
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch",
         "data/tracking_picks.jsonl"],
        capture_output=True, text=True,
    )
    assert tracked.returncode == 0, (
        "data/tracking_picks.jsonl is not tracked by git. The durable record is the "
        "committed file; actions/cache is an accelerator and is not storage."
    )


def test_a_gitignored_tracking_db_is_why_this_is_needed(tmp_path):
    """Stated as a test because it is the premise, and the premise can change.

    If `data/tracking.db` ever becomes committed, this file is belt-and-braces.
    If it is still ignored — and it is — the committed JSONL is the only copy
    that survives an eviction or a failed job.
    """
    root = Path(pick_record.picks_path()).parents[1]
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "data/tracking.db"],
        capture_output=True, text=True,
    )
    assert ignored.returncode == 0, (
        "data/tracking.db is no longer gitignored, so the cache is no longer the only "
        "thing carrying picks between runs. Update pick_record's module docstring."
    )


def test_the_python_version_in_use_is_the_one_under_test():
    """The CI heredoc runs under whatever `python` is on PATH, and this file is
    loaded through it. A syntax error here fails only in CI, minutes later, in a
    step whose output nobody reads closely — so it is compiled here instead."""
    compile(Path(pick_record.__file__).read_text(), pick_record.__file__, "exec")
