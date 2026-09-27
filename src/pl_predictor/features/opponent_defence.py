"""Opponent-defensive-strength features for player rates.

Every player-rate feature in `player_form` is opponent-blind. A striker who
scores four against a relegated side is recorded identically to one who scores
four against the league leaders, so the model cannot learn that the two
performances mean different things. This module supplies the missing
information: how leaky and how weak the *opponent* has been, measured over
prior matches only.

**Everything is derived from the FPL archive's own fixture results**, so there
is no second data source and therefore no cross-source name reconciliation to
get wrong. `team_h_score`/`team_a_score` ship with every player-gameweek row and
are always written from the fixture's own perspective, so `was_home` attributes
them to the right club. This is a deliberate departure from joining
football-data.co.uk team names, where a single unmapped club silently drops
every player facing it.

`opponent_team` is an integer FPL id that is season-local and appears in the
archive *only* as somebody's opponent, never as a player's own team. The id is
recovered without a network call: within a season every club meets every other
exactly once, so the club an id names is the one club that never meets it. Where
that rule is ambiguous the season is skipped rather than guessed, because a
guessed mapping is indistinguishable from a correct one until it quietly
corrupts a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (3, 5, 10)

_REQUIRED = (
    "season", "team", "opponent_team", "kickoff_time",
    "was_home", "team_h_score", "team_a_score",
)


def _coerce_bool(series: pd.Series) -> pd.Series:
    """The archive ships `was_home` as the strings "True"/"False"."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _require(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED if column not in df.columns]
    if missing:
        return pd.DataFrame()
    out = df[["season", "team", "opponent_team", "kickoff_time", "was_home",
              "team_h_score", "team_a_score"]].copy()
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], errors="coerce")
    out["is_home"] = _coerce_bool(out["was_home"])
    out["home_goals"] = pd.to_numeric(out["team_h_score"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["team_a_score"], errors="coerce")
    return out.dropna(subset=["kickoff_time", "home_goals", "away_goals"])


def build_team_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team, kickoff) with that team's goals for and against.

    The archive holds ~25 player rows per fixture, so this collapses them; a
    fixture counted 25 times would compress every rolling window to the same
    match repeated.
    """
    rows = _require(df)
    if rows.empty:
        return pd.DataFrame(columns=["season", "team", "kickoff_time", "goals_for", "goals_against", "is_home"])

    rows = rows.drop_duplicates(["season", "team", "kickoff_time"])
    rows["goals_for"] = np.where(rows["is_home"], rows["home_goals"], rows["away_goals"])
    rows["goals_against"] = np.where(rows["is_home"], rows["away_goals"], rows["home_goals"])
    return (
        rows[["season", "team", "kickoff_time", "goals_for", "goals_against", "is_home"]]
        .sort_values(["season", "team", "kickoff_time"])
        .reset_index(drop=True)
    )


def solve_team_id_names(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Map FPL `opponent_team` ids to club names, per season.

    Every club meets every other exactly once in a season, so the club named by
    an id is the sole club that never meets it. A season where that is not
    uniquely true for every id is omitted: a partial map would look like a
    complete one and only reveal itself as a silent feature failure.
    """
    rows = _require(df)
    mapping: dict[str, dict[str, str]] = {}
    if rows.empty:
        return mapping

    for season, group in rows.groupby("season", sort=True):
        met = group.groupby("team")["opponent_team"].agg(lambda values: set(values.astype(str)))
        names = set(met.index)
        ids = set(group["opponent_team"].astype(str))
        if not names or not ids:
            continue
        resolved: dict[str, str] = {}
        for team_id in ids:
            never_met = [name for name in names if team_id not in met[name]]
            if len(never_met) != 1:
                resolved = {}
                break
            resolved[team_id] = never_met[0]
        if resolved and len(set(resolved.values())) == len(resolved):
            mapping[season] = resolved
    return mapping


def _defence_ratings(matches: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Attach a per-window defensive rating to each team-match, prior matches only.

    `goals_against - goals_for` over prior matches: positive means a defence
    that leaks more than it scores, which is the state a scorer is choosing
    between. Normalising against the team's own attack keeps the scale
    league-relative without needing a league mean, so promoted sides are not
    flattered by a weak attack for.
    """
    columns = ["season", "team", "kickoff_time"]
    if matches.empty:
        return pd.DataFrame(index=matches.index, columns=columns)

    ordered = matches.sort_values(columns).copy()
    grouped = ordered.groupby(["season", "team"], sort=False)
    for window in windows:
        conceded = grouped["goals_against"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        scored = grouped["goals_for"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        ordered[f"opponent_defence_last{window}"] = conceded - scored
    return ordered[columns + [f"opponent_defence_last{w}" for w in windows]]


def build_opponent_defence_features(
    df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS
) -> tuple[pd.DataFrame, list[str]]:
    """Return opponent-defensive-strength columns aligned row-for-row with `df`.

    Rows whose opponent cannot be resolved — a season the id map skipped, or a
    fixture the archive does not describe — come back as NaN rather than a
    default, so a broken join is visible as missing data instead of arriving as
    a plausible constant the model then leans on.
    """
    names = [f"opponent_defence_last{window}" for window in windows]
    empty = pd.DataFrame(index=df.index, columns=names, dtype=float)
    if df.empty:
        return empty, names

    matches = build_team_match_history(df)
    if matches.empty:
        return empty, names

    rated = _defence_ratings(matches, windows)
    id_names = solve_team_id_names(df)
    if not id_names:
        return empty, names

    opponent_name = pd.Series(index=df.index, dtype=object)
    for season, mapping in id_names.items():
        selector = (df["season"] == season) if "season" in df.columns else pd.Series(True, index=df.index)
        opponent_name.loc[selector] = df.loc[selector, "opponent_team"].astype(str).map(mapping)

    lookup = rated.set_index(["season", "team", "kickoff_time"])
    keys = pd.MultiIndex.from_arrays([
        df["season"] if "season" in df.columns else pd.Series(index=df.index, dtype=object),
        opponent_name,
        pd.to_datetime(df["kickoff_time"], errors="coerce"),
    ])
    joined = lookup.reindex(keys)
    joined.index = df.index

    return joined.astype(float), names
