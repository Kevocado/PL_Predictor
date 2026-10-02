"""The durable record now holds EVERY pick, labelled with when it was made.

This is the `pick_record.py` half of the 2026-10-01 reversal, and it changed the
file's contract, so the contract is stated here rather than left to the
docstring the change rewrites.

Before: the file recorded only picks whose `snapshotted_at` preceded their own
`commence_time`, because under the old rule a post-kickoff pick was not part of
the record at all — there was nothing to keep it for. PL shipped an EMPTY
`data/tracking_picks.jsonl` (0 lines) as a direct consequence: not one pick was
ever captured before a kickoff, which is exactly why the headline was empty.

After: every recorded pick goes in, each carrying its own two timestamps and
the `made_before_kickoff` derived from them. The record's job is no longer "the
evidence for the pre-kickoff figure" — it is "what was recorded, and when", and
the pre-kickoff subset is computed from that at read time.

Three properties are pinned, and the first is the reversal:

1. **Every pick is recorded**, whatever moment it was made. A rebuilt pick is a
   pick; excluding it from the durable file is the exclusion this change ended.
2. **`made_before_kickoff` is derived from the row's own timestamps**, never
   from a flag, and never a `true` the timestamps do not prove.
3. **Append-only and idempotent.** A re-run appends nothing, so a no-op workflow
   run is an empty diff; and a line that exists is never amended, so a later run
   with a different probability cannot rewrite what the model said.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from pl_predictor.tracking import pick_record

KICKOFF = "2026-09-13T14:00:00Z"
BEFORE = "2026-09-13T10:00:00Z"
AFTER = "2026-09-13T15:00:00Z"


def row(event_id="e1", market="match_result", outcome="home_win", prob=0.6,
        snapshotted_at=BEFORE, commence_time=KICKOFF):
    return {
        "event_id": event_id, "team_home": "Arsenal", "team_away": "Chelsea",
        "market": market, "outcome_name": outcome, "predicted_prob": prob,
        "commence_time": commence_time, "snapshotted_at": snapshotted_at,
        # Present in the database, deliberately not recorded: grading is derived
        # from the fixture result at read time and must not be frozen here.
        "resolved": 1, "actual_outcome": 1,
        "backfilled": True,  # present in the DB; a flag must never decide timing
    }


def lines(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --- 1. every pick is recorded ----------------------------------------------

def test_a_pick_made_before_kickoff_is_recorded(tmp_path):
    path = tmp_path / "picks.jsonl"
    assert pick_record.append_picks([row()], path) == 1
    assert lines(path)[0]["event_id"] == "e1"


def test_a_pick_made_AFTER_kickoff_is_also_recorded(tmp_path):
    """The reversal, as one assertion.

    This is the case the old rule refused outright — `append_picks` returned 0
    and wrote no file — and it is the case that made PL's durable record empty
    and the headline null. The pick is a pick; it is recorded, and labelled.
    """
    path = tmp_path / "picks.jsonl"
    assert pick_record.append_picks([row(snapshotted_at=AFTER)], path) == 1, (
        "a pick recorded after its own kickoff is still a recorded pick and belongs in the "
        "record. Excluding it is the exclusion the 2026-10-01 reversal ended."
    )
    stored = lines(path)[0]
    assert stored["snapshotted_at"] == AFTER
    assert stored["made_before_kickoff"] is False, (
        "recorded AND labelled: a pick made after the start may never be presented as one made "
        "before it."
    )


def test_a_pick_at_the_kickoff_instant_is_recorded_and_labelled_not_pre_kickoff(tmp_path):
    """At kickoff is not before it: a pick written in the same second the match
    starts had no chance to be made on the night."""
    path = tmp_path / "picks.jsonl"
    assert pick_record.append_picks([row(snapshotted_at=KICKOFF)], path) == 1
    assert lines(path)[0]["made_before_kickoff"] is False


def test_the_record_mixes_both_and_labels_each_one(tmp_path):
    """One file, both kinds, each labelled — that is the disclosure.

    A reader can now see the whole record and, per pick, when it was made. Before
    this change the file could only ever hold the in-time subset, so the late
    picks were in no durable artefact anywhere.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks(
        [row(event_id="e1", snapshotted_at=BEFORE), row(event_id="e2", snapshotted_at=AFTER)],
        path,
    )
    stored = {r["event_id"]: r for r in lines(path)}

    assert len(stored) == 2
    assert stored["e1"]["made_before_kickoff"] is True
    assert stored["e2"]["made_before_kickoff"] is False


# --- 2. the flag is derived from timestamps, never stored or trusted --------

def test_the_recorded_flag_comes_from_the_timestamps_not_the_row_s_flag(tmp_path):
    """`row()` carries `backfilled: True` on every pick, pre-kickoff included.

    If the label were read from it, the in-time pick would be recorded as
    post-kickoff. There is no flag column in the schema and none is added.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row(snapshotted_at=BEFORE)], path)
    assert lines(path)[0]["made_before_kickoff"] is True, (
        "backfilled=True was on the row and the label ignored it, because the question is when "
        "the pick was made and a flag is only as honest as whatever set it."
    )


@pytest.mark.parametrize("bad", [None, "", "not-a-date"])
def test_an_unparseable_timestamp_is_recorded_but_never_labelled_pre_kickoff(tmp_path, bad):
    """Recorded, and unlabelled. The asymmetry is deliberate.

    Rule 1 (recorded stays recorded) has no exception for an unreadable
    timestamp; rule "never backfill a true the timestamps do not prove" does.
    """
    path = tmp_path / "picks.jsonl"
    assert pick_record.append_picks([row(snapshotted_at=bad)], path) == 1, (
        "an unprovable timing claim is a reason to withhold the LABEL, not to drop the pick — "
        "dropping it is how the record ends up empty."
    )
    assert lines(path)[0]["made_before_kickoff"] is False


def test_a_bad_kickoff_is_recorded_but_never_labelled_pre_kickoff(tmp_path):
    path = tmp_path / "picks.jsonl"
    assert pick_record.append_picks([row(commence_time="not-a-date")], path) == 1
    assert lines(path)[0]["made_before_kickoff"] is False


def test_the_derivation_compares_instants_honouring_offsets(tmp_path):
    """13:30 at UTC-5 is 18:30 UTC: four and a half hours AFTER a 14:00 UTC
    kickoff. Stripping the offset compares 13:30 with 14:00 and calls it
    pre-kickoff."""
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks(
        [row(event_id="late", snapshotted_at="2026-09-13T13:30:00-05:00"),
         row(event_id="early", snapshotted_at="2026-09-13T08:30:00-05:00")],
        path,
    )
    stored = {r["event_id"]: r["made_before_kickoff"] for r in lines(path)}
    assert stored == {"late": False, "early": True}


# --- 3. append-only and idempotent -----------------------------------------

def test_a_rerun_appends_nothing(tmp_path):
    """Why the file is safe to commit on every run: an unchanged workflow
    produces an empty diff, so a no-op run is invisible in history."""
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row()], path)
    first = path.read_text()
    assert pick_record.append_picks([row()], path) == 0
    assert path.read_text() == first


def test_a_recorded_pick_is_never_amended_by_a_later_different_value(tmp_path):
    """Once written, a pick is evidence of what the model said.

    Asserted on the FILE, not on the helper's return value, because a helper
    that reports "0 new" while having overwritten the line would satisfy every
    other test in this file.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row(prob=0.61)], path)
    pick_record.append_picks([row(prob=0.33)], path)
    assert lines(path)[0]["predicted_prob"] == 0.61, "a recorded pick was amended"


def test_a_later_rerun_of_the_same_pick_is_not_appended_as_a_second_record(tmp_path):
    """The record holds one line per (event_id, market, outcome_name).

    This is the counting key on disk: the EARLIEST recorded pick for a unit is
    the one in the file, and a rerun cannot add a second line that would let it
    be graded twice.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row(snapshotted_at=BEFORE)], path)
    assert pick_record.append_picks([row(snapshotted_at=AFTER, prob=0.9)], path) == 0
    assert len(lines(path)) == 1


def test_the_record_still_excludes_the_derived_verdict(tmp_path):
    """`resolved` and `actual_outcome` are deliberately not stored.

    Grading is derived from the fixture result at read time; baking it in would
    freeze a verdict that later data may correct, which would make the record
    self-confirming.
    """
    path = tmp_path / "picks.jsonl"
    pick_record.append_picks([row()], path)
    stored = lines(path)[0]
    assert "resolved" not in stored
    assert "actual_outcome" not in stored


def test_a_malformed_line_does_not_stop_later_runs(tmp_path):
    """A half-written line from an interrupted run must not make every
    subsequent run fail."""
    path = tmp_path / "picks.jsonl"
    path.write_text('{"broken\n')
    assert pick_record.append_picks([row(event_id="e2")], path) == 1


def test_an_unterminated_final_line_does_not_corrupt_the_next_append(tmp_path):
    """A file whose last line has no "\\n" would otherwise have its final record
    joined to the first new one, producing a line that is not JSON and silently
    dropping BOTH from every later load."""
    path = tmp_path / "picks.jsonl"
    first = row(event_id="e1")
    path.write_text(json.dumps({k: first[k] for k in pick_record.PICK_FIELDS}, sort_keys=True))
    assert pick_record.append_picks([row(event_id="e2")], path) == 1
    assert [r["event_id"] for r in lines(path)] == ["e1", "e2"]


def test_lines_that_are_valid_json_but_not_records_are_skipped(tmp_path):
    path = tmp_path / "picks.jsonl"
    good = {"event_id": "e1", "market": "match_result", "outcome_name": "A"}
    path.write_text("null\n[]\n42\n\"text\"\n" + json.dumps(good) + "\nnot json at all\n")
    assert pick_record.load_recorded_picks(path) == {("e1", "match_result", "A")}


# --- 4. it is still the record, not just a cache ---------------------------

def test_the_record_is_tracked_by_git_and_the_cache_is_not(tmp_path):
    """The distinction the whole change rests on.

    `actions/cache` is not storage. `git ls-files` rather than `check-ignore`: a
    file can be ignored AND tracked, and the failure that matters is "not
    tracked".
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
    """Stated as a test because it is the premise, and the premise can change."""
    root = Path(pick_record.picks_path()).parents[1]
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "data/tracking.db"],
        capture_output=True, text=True,
    )
    assert ignored.returncode == 0, (
        "data/tracking.db is no longer gitignored, so the cache is no longer the only "
        "thing carrying picks between runs. Update pick_record's module docstring."
    )


def test_the_workflow_step_records_every_pick_not_only_pre_kickoff_ones():
    """The CI step is what actually calls `append_picks`.

    If it pre-filtered to pre-kickoff rows before calling, the function would
    record every pick and the file would still only ever hold the in-time ones —
    the reversal silently undone one layer up, with every unit test green.
    """
    workflow = Path(pick_record.picks_path()).parents[1] / ".github/workflows/refresh-public-snapshot.yml"
    text = workflow.read_text()
    step = text[text.index("append_picks"):]
    assert "is_pre_kickoff" not in step and "pre-kickoff" not in step.split("\n\n")[0], (
        "the workflow filters rows before handing them to append_picks. Every recorded pick "
        "must be handed over; append_picks derives the label per row."
    )
    assert "SELECT event_id" in step, "the step must select the columns append_picks records"


# --- 5. the rename is documented with its call sites ------------------------

def test_the_rename_from_load_pre_kickoff_picks_is_complete():
    """`load_pre_kickoff_picks` returned the KEYS already recorded, and it now
    returns the keys of a file that holds late picks too, so the name lies.

    Asserted rather than documented because the risk is a partial rename: the
    helper renamed and one caller left behind, which raises at runtime in CI
    only.
    """
    assert not hasattr(pick_record, "load_pre_kickoff_picks"), (
        "the old name is back; it says 'pre_kickoff' about a file that now records every pick."
    )
    assert hasattr(pick_record, "load_recorded_picks")
    # The old name appears in docstrings, which is how the rename is documented.
    # It may not appear as a CALL, which is what raises at runtime — in the CI
    # heredoc, where the traceback is one line of a long build nobody reads.
    src = Path(pick_record.__file__).read_text()
    calls = re.findall(r"\bload_pre_kickoff_picks\s*\(", src)
    assert not calls, (
        f"a call site still uses the old name ({len(calls)} call(s)). It raises in the CI "
        f"heredoc, where the output is a single line of a long build."
    )


def test_is_pre_kickoff_is_still_the_name_the_derivation_is_exported_under():
    """Kept, because it is the honest name for the LABEL and three call sites
    depend on it. What changed is that it no longer decides whether a pick is
    kept."""
    assert pick_record.is_pre_kickoff(row(snapshotted_at=BEFORE)) is True
    assert pick_record.is_pre_kickoff(row(snapshotted_at=AFTER)) is False


@pytest.mark.parametrize("call_site", ["tests/test_pick_record.py", "tests/test_pick_record_hardening.py"])
def test_the_call_sites_of_the_changed_helpers_exist(call_site):
    root = Path(pick_record.picks_path()).parents[1]
    assert (root / call_site).exists(), f"{call_site} was removed but its helpers changed"
