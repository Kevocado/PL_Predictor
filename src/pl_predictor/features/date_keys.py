"""One definition of "the same instant", for every date-keyed join in `features/`.

A pandas `merge` or `merge_asof` on a datetime column needs both sides at the same
**resolution**, not merely the same values. Otherwise:

    pandas.errors.MergeError: incompatible merge keys [1]
    dtype('<M8[us]') and dtype('<M8[s]'), must be the same type

...even though both columns are midnight on the same days, and the join is
otherwise entirely correct.

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

Pinning pandas would make CI and a laptop agree, and leave the next cache written by
a different version free to disagree again. Coercing the key is the fix that holds
regardless of who wrote the cache, so every date-keyed join goes through here.

## What this deliberately does not do

`errors="coerce"`. A cell that cannot be read as a timestamp becomes NaT and fails to
match, which under a `how="left"` merge degrades one row's features to NaN -- and
XGBoost handles NaN natively. Raising instead would take out a whole training run
over one bad cell in a cached upstream CSV. The trade is pinned by a test so it
cannot be changed silently in either direction.
"""

from __future__ import annotations

import pandas as pd

# Nanoseconds are the resolution every historical pandas version agrees on, so this
# is the safest common denominator rather than whichever one is newest.
DATE_KEY_DTYPE = "datetime64[ns]"


def as_date_key(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """A copy of `frame` whose `column` is a datetime at `DATE_KEY_DTYPE`.

    Pass the result of this on *both* sides of a join. Normalising one side is
    useless, and is the usual way this goes wrong a second time.
    """
    if column not in frame.columns:
        return frame
    out = frame.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").astype(DATE_KEY_DTYPE)
    return out
