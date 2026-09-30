"""A durable, append-only record of every pre-kickoff pick.

**Why this exists when `tracking.db` already holds the same rows.** It does, and
it is the store the product reads. The problem is that it does not survive:
`tracking.db` is gitignored, and the only thing keeping it across CI runs was
an `actions/cache` entry, which is not storage. Caches are evicted after seven
days of no access, and none is written at all when the job that would write it
fails. Either way the pre-kickoff picks are gone and the headline goes back to
null — the exact state this repo shipped in, with `n_rebuilt` 50 of 50 and a
headline computed entirely from picks made after kickoff.

So: the cache is now explicitly an accelerator, and this file is the record.

**Append-only, and only ever appended by the CI job.** A pre-kickoff pick is a
claim about what the model said before the match. Once written it must not
change, and the only writer is the scheduled workflow, which runs at a fixed
time well before any fixture it records has kicked off. Nothing in the live
application writes here — a running app can rebuild a pick at any moment, and
letting one append would defeat the entire point.

**Committed, not published.** The file is the evidence behind a published
accuracy figure, so it belongs in the repository rather than in an artefact
store nobody can diff.

**Format:** one JSON object per line, one line per `predictions` row that was
written with a `snapshotted_at` at or before its fixture's `commence_time`.
Reconstructing from it is `load_pre_kickoff_picks()`, below.
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
    """
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def is_pre_kickoff(row: dict) -> bool:
    """Whether a stored row was written before its fixture started.

    Read from the row's own two timestamps, never from a flag: the question is
    when the pick was made relative to the match, and `record_predictions`
    takes `backfilled=` as an argument, so a flag is only as honest as whatever
    set it. Unparseable or absent timestamps are NOT pre-kickoff — an
    unprovable claim is not a verified one, and treating it as one is how this
    bug looked healthy in the first place.
    """
    try:
        snap = _utc(row.get("snapshotted_at"))
        kick = _utc(row.get("commence_time"))
    except (TypeError, ValueError):
        return False
    if snap is None or kick is None:
        return False
    # At kickoff is not before it: a pick written in the same second the match
    # starts had no chance to be made on the night.
    return snap < kick


def append_picks(rows: Iterable[dict], path: Path | None = None) -> int:
    """Append pre-kickoff rows not already recorded. Returns how many were new.

    De-duplication is on the row's own key — `(event_id, market, outcome_name)`,
    the same `UNIQUE` constraint the database uses — so a re-run of the workflow
    appends nothing. That is what makes the file safe to commit on every run:
    an unchanged file produces an empty diff, so a no-op run is invisible in
    history rather than a row of noise.
    """
    target = path or picks_path()
    seen = load_pre_kickoff_picks(target)
    lines: list[str] = []
    for row in rows:
        if not is_pre_kickoff(row):
            continue
        key = (row.get("event_id"), row.get("market"), row.get("outcome_name"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(json.dumps({k: row.get(k) for k in PICK_FIELDS}, sort_keys=True))
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


def load_pre_kickoff_picks(path: Path | None = None) -> set[tuple[Any, Any, Any]]:
    """The keys already recorded, for de-duplication.

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
