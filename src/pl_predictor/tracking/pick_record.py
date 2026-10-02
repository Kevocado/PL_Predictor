"""A durable, append-only record of every recorded pick, and when it was made.

**Why this exists when `tracking.db` already holds the same rows.** It does, and
it is the store the product reads. The problem is that it does not survive:
`tracking.db` is gitignored, and the only thing keeping it across CI runs was an
`actions/cache` entry, which is not storage. Caches are evicted after seven
days of no access, and none is written at all when the job that would write it
fails. Either way the picks are gone and the record reverts to whatever the next
run rebuilds — the exact state this repo shipped in, with `n_rebuilt` 50 of 50 and
a headline computed entirely from picks made after kickoff.

So: the cache is now explicitly an accelerator, and this file is the record.

**Every recorded pick, not only the pre-kickoff ones.** This changed on
2026-10-01, when Kevin reversed the rule that only a pick made before the start
counts: the models are re-run constantly, so under that rule a re-run on an
already-played game stopped counting and the record emptied out on every model
change. PL's shipped `data/tracking_picks.jsonl` was EMPTY — 0 lines — because
not one pick had ever been captured before a kickoff, which is exactly why the
headline was null.

So the file's job is no longer "the evidence for the pre-kickoff figure". It is
"what was recorded, and when". Each line carries its own two timestamps and the
`made_before_kickoff` DERIVED from them, so the pre-kickoff subset is computed
from the record at read time rather than being the only thing kept. Honesty moves
from exclusion to disclosure: a pick made after kickoff is recorded and LABELLED,
never recorded as one made before it.

**Append-only, and only ever appended by the CI job.** A recorded pick is a claim
about what the model said, and when. Once written it must not change, and the only
writer is the scheduled workflow. Nothing in the live application writes here — a
running app can rebuild a pick at any moment, and letting one append would defeat
the entire point.

**Committed, not published.** The file is the evidence behind a published
accuracy figure, so it belongs in the repository rather than in an artefact
store nobody can diff.

**Format:** one JSON object per line, one line per `predictions` row, keyed
`(event_id, market, outcome_name)` — the row's own `UNIQUE` constraint, which
is the finest key this store has. De-duplication is on that key, so a re-run
appends nothing and the EARLIEST recorded pick for a unit is the one on disk.
That is NOT the same key `store._counted_picks` uses: the track record counts
one row per FIXTURE (`event_id`, with all four markets as columns on it),
because a row of `predictions` is a market outcome and there are six of them per
fixture. The two keys deliberately disagree — the file is an append-only record
of what was written, the track record is one counted pick per fixture — and
neither one is derived from the other. Loading the file's keys is
`load_recorded_picks()`, below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

#: Columns that define a pick. Deliberately the minimum that makes a row
#: meaningful on its own — `resolved` and `actual_outcome` are deliberately
#: absent, because grading is derived from the fixture result at read time and
#: baking it in would freeze a verdict that later data may correct.
#:
#: `made_before_kickoff` IS recorded, unlike the grading columns: it is derived
#: from `snapshotted_at` and `commence_time`, both of which are already on the
#: line, so it is re-derivable rather than a new claim — and it is what makes the
#: file usable as the disclosure the 2026-10-01 reversal asks for, instead of a
#: wall of picks with no indication of when any of them was made.
PICK_FIELDS = (
    "event_id",
    "team_home",
    "team_away",
    "market",
    "outcome_name",
    "predicted_prob",
    "commence_time",
    "snapshotted_at",
)


def picks_path() -> Path:
    """Where the record lives. Overridable so a test can point it elsewhere."""
    override = os.getenv("PL_PICK_RECORD_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "data" / "tracking_picks.jsonl"


def _utc(value) -> "pd.Timestamp | None":
    """A timestamp as a naive UTC instant, so two of them compare as INSTANTS.

    `store._naive` strips an offset without converting, which is right for that
    module's own purpose (matching a source's wall-clock column) and wrong here:
    13:30 at UTC-5 is 18:30 UTC, 4.5 hours after a 14:00 UTC kickoff, and
    stripping the offset compares 13:30 with 14:00 and calls it pre-kickoff. A
    naive value is read as UTC, which is what every writer of these columns
    emits.

    `store._utc_instant` is this same rule on the read side of the track record.
    They are kept as two small functions rather than one shared import because
    this module is loaded by the CI job through a bare `python -` heredoc with no
    package context to share a helper through.
    """
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def is_pre_kickoff(row: dict) -> bool:
    """Whether a stored row was written before its fixture started.

    **What this decides changed on 2026-10-01; what it MEANS did not.** It is
    still the honest name for the label, and it is still derived from the row's
    own two timestamps, never from a flag: the question is when the pick was made
    relative to the match, and `record_predictions` takes `backfilled=` as an
    argument, so a flag is only as honest as whatever set it.

    Before, this function also decided whether the pick was RECORDED at all — a
    post-kickoff pick was refused, which is why this file shipped empty and the
    headline shipped null. It no longer does. An unparseable or absent timestamp
    is NOT pre-kickoff — an unprovable claim is not a verified one — but the pick
    is still recorded, because rule 1 (recorded stays recorded) has no exception
    for a timestamp we cannot read.

    A pick stamped exactly at kickoff is not before it: a pick written in the
    same second the match starts had no chance to be made on the night.
    """
    snap = _utc(row.get("snapshotted_at"))
    kick = _utc(row.get("commence_time"))
    if snap is None or kick is None:
        return False
    return snap < kick


def append_picks(rows: Iterable[dict], path: Path | None = None) -> int:
    """Append every row not already recorded, labelled with when it was made.
    Returns how many were new.

    **Records all of them, whatever moment each pick was made** — the 2026-10-01
    reversal. A post-kickoff pick used to be refused outright, and refusing it is
    what kept this file empty and the headline null.

    De-duplication is on the row's own key — `(event_id, market, outcome_name)`,
    the same `UNIQUE` constraint the database uses — so a re-run of the workflow
    appends nothing, AND a rerun of the model on an already-recorded pick cannot
    add a second line that would let it be graded twice. The first line written
    for a key is the earliest recorded pick for that unit, which is exactly the
    counted pick the track record's reader chooses. That is what makes the file
    safe to commit on every run: an unchanged file produces an empty diff, so a
    no-op run is invisible in history rather than a row of noise.

    `made_before_kickoff` is written onto each line rather than left to the
    reader, because the file is read by things that do not import the derivation
    — and because the derivation is over the two timestamps already on the line,
    so it is reproducible rather than a new fact.
    """
    target = path or picks_path()
    seen = load_recorded_picks(target)
    lines: list[str] = []
    for row in rows:
        key = (row.get("event_id"), row.get("market"), row.get("outcome_name"))
        if key in seen:
            continue
        seen.add(key)
        record = {k: row.get(k) for k in PICK_FIELDS}
        record["made_before_kickoff"] = is_pre_kickoff(row)
        lines.append(json.dumps(record, sort_keys=True))
    if not lines:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    # A file whose last line has no "\n" (a crash mid-write, a hand edit) would
    # otherwise have its final record joined to the first new one, producing a
    # line that is not JSON and silently dropping BOTH from every later load.
    # Only ever add a separator; an existing line is never touched.
    needs_separator = False
    if target.exists() and target.stat().st_size:
        with target.open("rb") as fh:
            fh.seek(-1, 2)
            needs_separator = fh.read(1) != b"\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(("\n" if needs_separator else "") + "\n".join(lines) + "\n")
    return len(lines)


def load_recorded_picks(path: Path | None = None) -> set[tuple[Any, Any, Any]]:
    """The keys already recorded, for de-duplication.

    **Renamed from `load_pre_kickoff_picks` on 2026-10-01.** It returned exactly
    what it returns now — a set of `(event_id, market, outcome_name)` — but the
    file it reads now holds late picks too, so the old name describes a subset
    the file no longer corresponds to. Call sites: `append_picks` above, and the
    three tests in `tests/test_pick_record*.py` that call it directly.

    A malformed line is skipped rather than raising: a half-written line from an
    interrupted run must not make every later run fail, and a duplicate would
    be rejected by the row's own key anyway.
    """
    target = path or picks_path()
    keys: set[tuple[Any, Any, Any]] = set()
    if not target.exists():
        return keys
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        keys.add((row.get("event_id"), row.get("market"), row.get("outcome_name")))
    return keys
