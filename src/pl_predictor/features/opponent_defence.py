"""Opponent-defensive-strength features for player rates.

Every player-rate feature in `player_form` is opponent-blind. A striker who
scores four against a relegated side is recorded identically to one who scores
four against the league leaders, so the model cannot learn that the two
performances mean different things. This module supplies the missing
information: how leaky and how weak the *opponent* has been, measured over
prior matches only.

**One rating function, two adapters.** `rate_team_matches` is the only place a
rating is computed, and it takes a generic team-match frame. Fitting feeds it the
FPL archive; serving feeds it `rolling_form.to_team_perspective`. That is
deliberate. An earlier version of this module derived the rating from the
archive alone, which meant serving would have had to reimplement it -- and two
implementations of one feature is exactly how NFL came to have two tests
asserting 82 and 81.5 for the same player and the same week (commit `0628c6d`).
`test_the_serving_adapter_agrees_with_the_training_adapter` exists to catch that
drift if the two paths ever diverge again.

**Fitting needs no second data source.** `team_h_score`/`team_a_score` ship with
every player-gameweek row and are always written from the fixture's own
perspective, so `was_home` attributes them to the right club. The obvious
alternative -- joining football-data.co.uk club names through
`data/team_names.py::to_canonical` -- was rejected: a single unmapped club
silently drops the feature for every player facing it, and that coverage has to
be measured rather than assumed.

`opponent_team` is an integer FPL id that is season-local and appears in the
archive *only* as somebody's opponent, never as a player's own team. The id is
recovered without a network call: within a season every club meets every other
exactly once, so the club an id names is the one club that never meets it. Where
that rule is ambiguous, or where the result is not one club per id, the season is
skipped rather than guessed -- a partial map is indistinguishable from a correct
one until it quietly corrupts a feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOWS = (3, 5, 10)

_ARCHIVE_COLUMNS = (
    "season", "team", "opponent_team", "kickoff_time",
    "was_home", "team_h_score", "team_a_score",
)


def _coerce_bool(series: pd.Series) -> pd.Series:
    """The archive ships `was_home` as the strings "True"/"False"."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _require_archive_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _ARCHIVE_COLUMNS if column not in df.columns]
    if missing:
        return pd.DataFrame()
    out = df[["season", "team", "opponent_team", "kickoff_time", "was_home",
              "team_h_score", "team_a_score"]].copy()
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], errors="coerce")
    out["is_home"] = _coerce_bool(out["was_home"])
    out["home_goals"] = pd.to_numeric(out["team_h_score"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["team_a_score"], errors="coerce")
    return out.dropna(subset=["kickoff_time", "home_goals", "away_goals"])


def rate_team_matches(
    team_matches: pd.DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    date_column: str = "date",
) -> pd.DataFrame:
    """The one place a defensive rating is computed.

    `goals_against - goals_for` over **prior** matches: positive means a defence
    that leaks more than it scores, which is the state a scorer is choosing
    between. Normalising against the team's own attack keeps the scale
    league-relative without needing a league mean, so promoted sides are not
    flattered for a weak attack for.

    Rows with a missing `goals_for`/`goals_against` are treated as matches that
    have not been played: they carry a rating built from real prior matches but
    contribute nothing themselves. That is what lets the serving path ask "how
    strong is this defence *before* an upcoming fixture" by appending a
    placeholder row for the fixture rather than by duplicating the rolling logic.
    """
    names = [f"opponent_defence_last{window}" for window in windows]
    if team_matches.empty:
        return pd.DataFrame(index=team_matches.index, columns=names, dtype=float)

    group_keys = ["team"] if "season" not in team_matches.columns else ["season", "team"]
    ordered = team_matches.sort_values(group_keys + [date_column]).copy()
    grouped = ordered.groupby(group_keys, sort=False)
    for window in windows:
        conceded = grouped["goals_against"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        scored = grouped["goals_for"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        ordered[f"opponent_defence_last{window}"] = conceded - scored
    return ordered[names]


def team_matches_from_matches_df(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Serving adapter: a `matches_df` in football-data.co.uk shape becomes the
    generic team-match frame `_rate_team_matches` expects.

    Returns `season`, `team`, `date`, `goals_for`, `goals_against`. Only completed
    matches carry a score, and an unplayed fixture yields a NaN row, which the
    rating function treats as a match that has not happened -- exactly what an
    upcoming fixture should be.
    """
    from .rolling_form import to_team_perspective

    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["season", "team", "date", "goals_for", "goals_against"])
    long_df = to_team_perspective(matches_df)
    keep = ["team", "date", "goals_for", "goals_against"]
    if "season" in long_df.columns:
        keep = ["season"] + keep
    out = long_df[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "team"])
    # Dedupe, mirroring `build_team_match_history`. Without this the serving
    # adapter is strictly less defended than the fitting one against the same
    # input, and `rate_opponents` raises "cannot handle a non-unique multi-index"
    # on the duplicated key -- which `routes._opponent_defence_for_fixture` then
    # swallows, so every fixture silently gets the league prior.
    if "season" in out.columns:
        return out.drop_duplicates(["season", "team", "date"], keep="first").reset_index(drop=True)
    return out.drop_duplicates(["team", "date"], keep="first").reset_index(drop=True)


def rate_opponents(
    team_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    date_column: str = "date",
) -> pd.DataFrame:
    """Rate the *opponent* of each fixture, using only matches before it kicks off.

    `fixtures` needs `opponent` (the club being faced) and `date`; `season` is
    used to group when present. The result is aligned row-for-row with
    `fixtures`.

    Implemented by appending a placeholder row for the opponent at the fixture's
    date and letting `rate_team_matches` rate it, rather than by rating "up to
    date D" separately. Sharing the one function is the entire point: a second
    implementation is how the training and serving definitions would drift.
    """
    names = [f"opponent_defence_last{window}" for window in windows]
    if fixtures.empty or team_matches.empty:
        return pd.DataFrame(index=fixtures.index, columns=names, dtype=float)

    if "season" in team_matches.columns and "season" not in fixtures.columns:
        # Not an `and`. With a one-sided `and`, fixtures lacking `season` made the
        # placeholders season-NaN, so `pd.concat` gave them no season and
        # `rate_team_matches` grouped them as singletons with no history -- every
        # rating came back NaN, i.e. the league prior, with no error. A silent
        # no-op is worse than a refusal.
        raise ValueError(
            "fixtures must carry `season` when team_matches does, otherwise the "
            "placeholder rows cannot be grouped with the club's real history and "
            "every rating silently comes back NaN"
        )

    dates = pd.to_datetime(fixtures["date"], errors="coerce")
    if dates.isna().any():
        # `rate_team_matches` sorts with pandas' default `na_position="last"`, so a
        # NaT placeholder sorts to the END of its club and its `shift(1)` window
        # becomes the club's *entire* history -- including matches played after the
        # fixture being priced. Measured: a club conceding 1,1,1,1 then 0,0,0,0
        # returns an honest 1.0 for a real date and a leaked 0.0 for a NaT one.
        # `na_position="first"` is not the fix: that would return the rating as of
        # the season start, a different wrong answer. Refuse instead.
        return pd.DataFrame(np.nan, index=fixtures.index, columns=names, dtype=float)

    has_season = "season" in team_matches.columns
    key_columns = (["season"] if has_season else []) + ["team", "date"]
    real = team_matches.copy()
    real["date"] = pd.to_datetime(real["date"], errors="coerce")
    # Defence in depth: `team_matches_from_matches_df` already dedupes, but this is
    # public API and a caller can hand it anything. `set_index(...).reindex(...)`
    # raises "cannot handle a non-unique multi-index" on a duplicated
    # (club, date), and at the one production call site that raise is swallowed --
    # so an un-deduped frame meant every fixture silently got the league prior.
    before = len(real)
    real = real.drop_duplicates(group_keys_real := ((["season"] if has_season else []) + ["team", "date"]))
    if len(real) != before:
        logger.warning("dropped %d duplicate (team, date) rows from team_matches", before - len(real))

    wanted = pd.DataFrame({
        "team": fixtures["opponent"].to_numpy(),
        "date": dates.to_numpy(),
    })
    if has_season:
        wanted["season"] = fixtures["season"].to_numpy()

    # Only invent a placeholder for a (club, date) the history does not already
    # describe. A question about a match that has already been played must
    # return that match's own prior-matches rating, and a placeholder dated the
    # same day would sort *after* the real row and see it -- turning a lookup of
    # history into a leak of the very match being described.
    unknown = ~pd.MultiIndex.from_frame(wanted[key_columns]).isin(pd.MultiIndex.from_frame(real[key_columns]))
    combined = real
    if unknown.any():
        placeholders = wanted[unknown].copy()
        placeholders["goals_for"] = np.nan
        placeholders["goals_against"] = np.nan
        combined = pd.concat([real, placeholders], ignore_index=True)

    rated = combined.copy()
    ratings = rate_team_matches(combined, windows=windows, date_column="date")
    rated[names] = ratings.reindex(combined.index).to_numpy()

    joined = rated.set_index(key_columns).reindex(pd.MultiIndex.from_frame(wanted[key_columns]))
    joined.index = fixtures.index
    return joined[names].astype(float)


def build_team_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team, kickoff) with that team's goals for and against.

    The archive holds ~25 player rows per fixture, so this collapses them; a
    fixture counted 25 times would compress every rolling window to the same
    match repeated.
    """
    rows = _require_archive_columns(df)
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
    uniquely true for every id, or where two ids land on the same club, is
    omitted: a partial map would look like a complete one and only reveal itself
    as a silent feature failure.

    **The two checks are belt and braces, and the first is provably redundant.**
    If any id has two or more candidate clubs then the total number of candidates
    exceeds the number of clubs, so by pigeonhole two ids name the same club and
    the bijectivity check below rejects the season regardless. Relaxing
    `len(never_met) != 1` to `not never_met` therefore changes no observable
    behaviour, which is why no test distinguishes the two -- a property of the
    logic, not a coverage gap. The per-id check is kept because it names the
    actual failure (an incomplete meeting graph) and fails at the right place.
    """
    rows = _require_archive_columns(df)
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


def build_opponent_defence_features(
    df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS
) -> tuple[pd.DataFrame, list[str]]:
    """Fitting adapter: opponent-defensive-strength columns for player-gameweek rows.

    Rows whose opponent cannot be resolved -- a season the id map skipped, or a
    fixture the archive does not describe -- come back as NaN rather than a
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

    id_names = solve_team_id_names(df)
    if not id_names:
        return empty, names

    rated = matches.copy()
    ratings = rate_team_matches(matches, windows=windows, date_column="kickoff_time")
    rated[names] = ratings.reindex(matches.index).to_numpy()

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

    return joined[names].astype(float), names
