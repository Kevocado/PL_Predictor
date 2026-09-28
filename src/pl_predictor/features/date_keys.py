"""One definition of "the same instant", for every date-keyed join in `features/`.

A pandas `merge` or `merge_asof` on a datetime column needs both sides at the same
**resolution**, not merely the same values. Otherwise:

    pandas.errors.MergeError: incompatible merge keys [1]
    dtype('<M8[us]') and dtype('<M8[s]'), must be the same type

...even though both columns are midnight on the same days, and the join is otherwise
entirely correct.

That resolution is not a property of the data. It depends on which pandas wrote a
given parquet cache, and on what strings each loader was handed:

- `load_training_data` reads `data/cache/*.parquet`, and parquet round-trips whatever
  dtype wrote it, so the same logical `date` column arrives as `datetime64[s]` from
  one cache and `datetime64[us]` from another.
- Understat and football-data.co.uk dates are parsed by independent loaders, and
  `pd.to_datetime` infers `s` or `us` from the strings it is given.
- Rolling-form joins are between a frame and a `set_index`/`reset_index` round trip
  of itself, which is a third path.

The full suite passed on a warm local cache and failed on CI's cold one, in
`test_no_lookahead_in_training_frame` and `test_current_season_check`. Nothing about
the model or the tests changed; only which machine built the cache.

Pinning pandas would make CI and a laptop agree, and leave the next cache written by a
different version free to disagree again. Coercing the key is the fix that holds
regardless of who wrote the cache, so every date-keyed join goes through here.

## The part that is easy to get wrong: the two join types disagree about null keys

`errors="coerce"` turns an unparseable date into `NaT`, and **what happens next depends
on which join you use**:

- **`merge(..., how="left")`** — `NaT` **matches `NaT`**. A coerced key is not inert: it
  joins against every other coerced key, and duplicate `(NaT, team)` keys on the right
  *multiply* the left rows. Measured: 3 input rows in, 5 out. That breaks the
  one-row-per-match contract `build_training_frame` documents, and the frame is then
  `concat`'d positionally against ten other feature blocks, so duplicated rows misalign
  all of them.
- **`merge_asof`** — rejects null keys outright: `ValueError: Merge keys contain null
  values on left side`. No match, no coercion; it raises.

So "coerce and move on" is wrong in one direction and "coerce and hope" is wrong in the
other, and the correct handling differs per call site. Hence three helpers, not one:

| helper | for | a bad date becomes |
|---|---|---|
| `as_date_key` | the **left** side of a `how="left"` merge | a retained row with `NaN` features |
| `drop_unmatchable` | the **right** side of a `how="left"` merge | no key, so it cannot match anything |
| `as_asof_key` | **both** sides of a `merge_asof` | no key, because a null key is fatal there |

`errors="coerce"` is deliberate: raising would take out a whole training run over one
bad cell in a cached upstream CSV. The trade is that a match whose date cannot be read
loses its features rather than the run. The tests drive the real call sites so the
choice cannot be changed silently in either direction.

## What this does not do

That trade is narrower than it looks, and the first version of this docstring read as
though it were total. "One bad cell" covers a cell that cannot be *parsed*. It does not
cover a cell that parses cleanly and then does not fit `datetime64[ns]`, which is
1677-09-21 to 2262-04-11:

    pd.to_datetime("2500-01-01", errors="coerce")   # -> Timestamp('2500-01-01'), dtype datetime64[us]
    ...astype("datetime64[ns]")                     # -> OutOfBoundsDatetime

The parse succeeds and infers microseconds. The failure is the *narrowing cast* to the
dtype above, and `errors="coerce"` governs the parse only, so it cannot cover the cast
without a `try`. So `as_date_key` and `as_asof_key` raise `OutOfBoundsDatetime` on a
far-future or far-past date, and `drop_unmatchable` -- which never casts -- does not.
Raising is the right behaviour: a timestamp that cannot be represented at the common
resolution is exactly the case where silently wrapping, truncating, or coercing to
`NaT` would attach a real fixture to a different real fixture. But it is a raise, so
"one bad cell never takes out the run" is true of unparseable cells only, and this
module is not total over its input. A loader that cannot rule out far-future dates has
to handle that exception itself.

For football data this is hypothetical rather than live -- a fixture is not dated in
the year 2500 -- which is exactly why it went unnoticed, and also why it should be
pinned rather than left to a future loader's one bad cell.
"""

from __future__ import annotations

import pandas as pd

# Nanoseconds are the resolution every historical pandas version agrees on, so this is
# the safest common denominator rather than whichever one is newest.
#
# It is a *common denominator*, not a superset: `datetime64[ns]` spans
# 1677-09-21 to 2262-04-11, so it is the widest range of the resolutions in play
# (s < us < ms < ns) while being the narrowest of the two 128-bit ones pandas
# also offers. `DATE_KEY_DTYPE` is therefore where this module stops being
# total, and where it is narrowest rather than widest is a deliberate trade --
# see the "What this does not do" section in the module docstring.
DATE_KEY_DTYPE = "datetime64[ns]"

# The `astype` below is a cast, and casts do not honour `errors="coerce"`.
# `pd.to_datetime("2500-01-01", errors="coerce")` *succeeds* -- it infers
# `datetime64[us]` for that series -- and the overflow only happens one line
# later, when the result is narrowed to nanoseconds. That is why the contract
# below is stated as "raises" rather than "coerces".
DATE_KEY_RANGE = (pd.Timestamp.min, pd.Timestamp.max)


def as_date_key(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """A copy of `frame` whose `column` is a datetime at `DATE_KEY_DTYPE`.

    Rows are kept, including any whose date **could not be parsed** (those become
    `NaT`). Use on the **left** side of a `how="left"` merge, where a retained
    row that matches nothing is the desired outcome.

    **Total, except for dates outside the `ns` range.** A cell that *parses* but
    falls after 2262-04-11 (or before 1677-09-21) raises
    `pandas.errors.OutOfBoundsDatetime` rather than becoming `NaT`, because the
    failure is in the narrowing cast to `DATE_KEY_DTYPE` and not in the parse.
    `errors="coerce"` does not cover it and cannot, as written. That is the
    right way for it to fail -- a silently wrapped or truncated timestamp would
    join a real fixture to the wrong one -- but it means this function is not
    total, and the module docstring used to imply that it was.

    Never use it alone on the right side of a `how="left"` merge — `NaT` matches
    `NaT` and duplicate keys fan out. Pair it with `drop_unmatchable` there.
    """
    if column not in frame.columns:
        return frame
    out = frame.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").astype(DATE_KEY_DTYPE)
    return out


def drop_unmatchable(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """`frame` without the rows whose `column` could not be read as a timestamp.

    For the **right** side of a `how="left"` merge, so a `NaT` key cannot match another
    `NaT` key and multiply rows.
    """
    if column not in frame.columns:
        return frame
    return frame[pd.to_datetime(frame[column], errors="coerce").notna()]


def as_asof_key(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """`frame` normalised *and* stripped of unreadable keys, for `merge_asof`.

    `merge_asof` raises `ValueError: Merge keys contain null values on left side` on a
    null key, so for these joins a bad date is dropped from **both** sides rather than
    left to fail the run. The affected matches lose their features entirely instead of
    keeping the row with `NaN`s — a limitation of `merge_asof`, not a free choice, and
    still better than not finishing.
    """
    return drop_unmatchable(as_date_key(frame, column), column)
