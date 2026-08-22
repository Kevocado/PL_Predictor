"""cold_start.py — confidence-tiered fallback for teams with little or no
history in the loaded data window (newly-promoted teams, or a team's first
few matches back in the league after a spell in the Championship).

Direct port of the blending idea in FPL_Optimizer/features.py::
blended_form_features: as a team accumulates real matches, its rolling-form
value is trusted more and the league-wide average fallback fades out
smoothly (weight = games_played / window), rather than a hard cutoff.
"""

from __future__ import annotations

import re

import pandas as pd

_WINDOW_RE = re.compile(r"last_(\d+)_")


def _window_of(col: str) -> int | None:
    m = _WINDOW_RE.search(col)
    return int(m.group(1)) if m else None


def apply_cold_start_fallback(long_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Blends each `last_{w}_*` feature column toward the league-wide average
    for teams with fewer than `w` prior matches in the loaded data. Adds a
    `confidence` column: 'current' (enough real data), 'blended' (partial),
    or 'none' (zero prior matches, pure league average / still NaN if no
    league average exists for that stat).

    Returns (long_df with columns blended in place, confidence series).
    """
    long_df = long_df.sort_values(["team", "date"]).copy()
    games_played = long_df.groupby("team").cumcount()

    league_avg = {col: long_df[col].mean() for col in feature_cols}

    for col in feature_cols:
        window = _window_of(col)
        if window is None or col not in long_df.columns:
            continue
        avg = league_avg[col]
        if pd.isna(avg):
            continue

        weight = (games_played / window).clip(upper=1.0)
        current = long_df[col]
        long_df[col] = weight * current.fillna(avg) + (1 - weight) * avg

    max_window = max((w for w in (_window_of(c) for c in feature_cols) if w), default=1)
    confidence = pd.Series("current", index=long_df.index)
    confidence[games_played < max_window] = "blended"
    confidence[games_played == 0] = "none"

    return long_df, confidence.reindex(long_df.index)
