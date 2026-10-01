"""Three ways the durable pick record could be wrong without any test noticing.

Found by CodeRabbit review of PL#29 and verified against the real functions
before this file was written. The first is an honesty defect: the record's claim
is about WHEN a pick was made, so the comparison must be between INSTANTS.

Nothing here was relaxed by the 2026-10-01 reversal — a post-kickoff pick is now
recorded rather than refused, but the LABEL on it is derived exactly as strictly
as before, and never more permissively.
"""
import json

from pl_predictor.tracking import pick_record


def _row(snap, kick, **kw):
    base = {"event_id": "e1", "team_home": "A", "team_away": "B", "market": "match_result",
            "outcome_name": "A", "predicted_prob": 0.5,
            "commence_time": kick, "snapshotted_at": snap}
    base.update(kw)
    return base


# --- 1. instants, not wall-clock digits ---------------------------------------

def test_a_pick_made_after_kickoff_in_another_offset_is_not_pre_kickoff():
    # 13:30 at UTC-5 is 18:30 UTC: 4.5 hours AFTER a 14:00 UTC kickoff. Stripping
    # the offset compares 13:30 with 14:00 and calls it pre-kickoff.
    assert not pick_record.is_pre_kickoff(_row("2026-09-13T13:30:00-05:00", "2026-09-13T14:00:00Z"))


def test_a_pick_made_before_kickoff_in_another_offset_is_pre_kickoff():
    # 08:30 at UTC-5 is 13:30 UTC: 30 minutes before a 14:00 UTC kickoff.
    assert pick_record.is_pre_kickoff(_row("2026-09-13T08:30:00-05:00", "2026-09-13T14:00:00Z"))


def test_a_kickoff_in_another_offset_is_compared_as_an_instant():
    # Kickoff 16:00 at UTC+2 is 14:00 UTC; a 13:30 UTC pick is 30 minutes early.
    assert pick_record.is_pre_kickoff(_row("2026-09-13T13:30:00Z", "2026-09-13T16:00:00+02:00"))
    # ... and a 14:30 UTC pick is 30 minutes late, though its digits (14:30) < 16:00.
    assert not pick_record.is_pre_kickoff(_row("2026-09-13T14:30:00Z", "2026-09-13T16:00:00+02:00"))


def test_naive_timestamps_are_still_read_as_utc():
    assert pick_record.is_pre_kickoff(_row("2026-09-13T13:00:00", "2026-09-13T14:00:00"))
    assert not pick_record.is_pre_kickoff(_row("2026-09-13T14:00:00", "2026-09-13T14:00:00"))


def test_the_offset_case_is_labelled_correctly_when_recorded_too(tmp_path):
    """Since the reversal this row IS recorded, so the label is all there is.

    Before, the assertion was that the append was refused (n == 0, no file). Now
    the row is in the record and its `made_before_kickoff` must be False — a
    post-kickoff pick that gets written down must never be written down as a
    pre-kickoff one.
    """
    p = tmp_path / "picks.jsonl"
    n = pick_record.append_picks([_row("2026-09-13T13:30:00-05:00", "2026-09-13T14:00:00Z")], path=p)
    assert n == 1
    assert json.loads(p.read_text().strip())["made_before_kickoff"] is False


def test_mixed_timeszone_sources_compare_correctly(tmp_path):
    """Sources mix tz-aware (Odds API, UTC) and naive (FPL fallback) kickoffs.
    A pick at 10:00 UTC against a 14:00 naive kickoff is before it, and must
    not be rejected for a comparison artefact."""
    assert pick_record.is_pre_kickoff(_row(
        snap="2026-09-13T10:00:00Z", kick="2026-09-13T14:00:00")) is True


# --- 2. a torn final line must not swallow the next record ---------------------

def test_an_unterminated_final_line_does_not_corrupt_the_next_append(tmp_path):
    p = tmp_path / "picks.jsonl"
    first = _row("2026-09-13T10:00:00Z", "2026-09-13T14:00:00Z", event_id="e1")
    p.write_text(json.dumps({k: first[k] for k in pick_record.PICK_FIELDS}, sort_keys=True))  # no "\n"
    assert pick_record.append_picks(
        [_row("2026-09-13T10:00:00Z", "2026-09-13T14:00:00Z", event_id="e2")], path=p) == 1
    lines = [ln for ln in p.read_text().split("\n") if ln.strip()]
    assert len(lines) == 2
    assert [json.loads(ln)["event_id"] for ln in lines] == ["e1", "e2"]   # both lines valid JSON
    # Renamed from load_pre_kickoff_picks on 2026-10-01; it loads the keys of a
    # file that now holds late picks too.
    assert pick_record.load_recorded_picks(p) == {("e1", "match_result", "A"), ("e2", "match_result", "A")}


def test_an_existing_recorded_line_is_never_rewritten(tmp_path):
    p = tmp_path / "picks.jsonl"
    r = _row("2026-09-13T10:00:00Z", "2026-09-13T14:00:00Z")
    pick_record.append_picks([r], path=p)
    before = p.read_bytes()
    pick_record.append_picks([_row("2026-09-13T11:00:00Z", "2026-09-13T14:00:00Z", predicted_prob=0.9)], path=p)
    assert p.read_bytes() == before


# --- 3. valid JSON that is not a record ---------------------------------------

def test_lines_that_are_valid_json_but_not_records_are_skipped(tmp_path):
    p = tmp_path / "picks.jsonl"
    good = {"event_id": "e1", "market": "match_result", "outcome_name": "A"}
    p.write_text("null\n[]\n42\n\"text\"\n" + json.dumps(good) + "\nnot json at all\n")
    assert pick_record.load_recorded_picks(p) == {("e1", "match_result", "A")}
