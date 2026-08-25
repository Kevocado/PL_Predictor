"""promoted_team_prior.py — a ClubElo-derived Elo prior for teams with zero
match history in the loaded football-data window (EXP-2026-05 slot in
docs/AI_CONTINUITY.md).

`features/cold_start.py::apply_cold_start_fallback` already blends a
brand-new team's *rolling-form* columns toward the league average as real
matches accumulate — but the project's own Elo/Pi ratings
(`features/ratings.py`) have no equivalent fallback at all: a team this
project has never seen starts at penaltyblog's Elo/Pi default, which
carries no information about how strong that club actually is. ClubElo
covers England's second tier, so a promoted club already has a real, dated
rating from the division it just came from — this module bridges that
rating onto this project's own Elo scale.

STATUS: research-only, not wired into `features/build.py`. It has not been
evaluated against real data — see `data/clubelo.py`'s status note and
docs/AI_CONTINUITY.md's EXP-2026-05 entry. Do not import this from
`build_training_frame` or `FixtureFeatureContext` until a walk-forward
evaluation (isolated to promoted-team fixtures, per the project's
promotion gate) shows a real improvement.

Cross-scale bridge: rather than assume ClubElo's rating scale is directly
comparable to this project's own Elo (fit independently, different K-factor
and home-field constant), a promoted team's ClubElo rating is expressed as
a z-score against the *other* current top-flight teams' ClubElo ratings on
the same date, then that z-score is mapped onto this project's own Elo
distribution for those same established teams on that date. This only
assumes ClubElo ranks teams sensibly relative to each other, not that the
two systems share a scale — a much weaker, more defensible assumption.
"""

from __future__ import annotations

import pandas as pd

from ..data import clubelo
from . import ratings


def zscore_bridge(
    reference_source_ratings: pd.Series, reference_target_ratings: pd.Series, team_source_rating: float
) -> float:
    """Express `team_source_rating` as a z-score against
    `reference_source_ratings`, then map that z-score onto
    `reference_target_ratings`'s distribution (mean + z * std). Both
    reference series must be aligned to the same set of established teams
    on the same date. Falls back to `reference_target_ratings`'s mean (the
    existing cold-start convention — see `cold_start.py`) if either
    reference series has fewer than 2 teams or zero variance."""
    if len(reference_source_ratings) < 2 or len(reference_target_ratings) < 2:
        return float(reference_target_ratings.mean()) if len(reference_target_ratings) else float("nan")
    source_std = reference_source_ratings.std()
    if not source_std or pd.isna(source_std):
        return float(reference_target_ratings.mean())
    z = (team_source_rating - reference_source_ratings.mean()) / source_std
    return float(reference_target_ratings.mean() + z * reference_target_ratings.std())


def cold_start_teams(matches_df: pd.DataFrame) -> set[str]:
    """Teams making their historically-first appearance somewhere in
    `matches_df` — i.e. every team that would otherwise hit the flat Elo/Pi
    default the moment they appear. For identifying genuine debut fixtures
    (e.g. to isolate promoted-team fixtures for an evaluation), always pass
    the *complete*, untruncated match history: a team's first row in an
    arbitrarily truncated slice is not necessarily their first-ever match,
    so slicing first and calling this second silently over-counts. This is
    NOT the right helper for "does team X already have history as of date
    D" within a truncated window — for that, just check `X in
    (set(history["team_home"]) | set(history["team_away"]))`, as
    `clubelo_elo_prior` below does."""
    from . import rolling_form

    long_df = rolling_form.to_team_perspective(matches_df)
    games_played = long_df.groupby("team").cumcount()
    return set(long_df.loc[games_played == 0, "team"])


def clubelo_elo_prior(
    matches_df: pd.DataFrame, team: str, as_of_date, established_teams: set[str] | None = None
) -> float | None:
    """This team's Elo prior, bridged from its ClubElo rating as of the day
    before `as_of_date`, onto the current Elo distribution of
    `established_teams` (teams already in `matches_df` before `as_of_date`
    with real match history — defaults to every team that isn't itself
    cold-start as of that date). Returns None if ClubElo has no rating for
    this team on that date (unmapped name, or genuinely not in England's
    top two tiers) — callers should fall back to the existing flat-average
    behavior in that case, never raise."""
    lookup_date = (pd.Timestamp(as_of_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    team_rating = clubelo.team_rating_asof(team, lookup_date)
    if team_rating is None:
        return None

    history = matches_df[matches_df["date"] < pd.Timestamp(as_of_date)]
    if established_teams is None:
        # Any team that appears at least once in `history` has a real
        # (if noisy) Elo rating by `as_of_date` — cold-start is "zero
        # matches," not "fewer than some window," so no further filtering
        # of `history`'s own teams is needed here.
        established_teams = set(history["team_home"]) | set(history["team_away"])
    established_teams = established_teams - {team}
    if not established_teams:
        return None

    elo_model = ratings.fit_elo(history)
    local_ratings = pd.Series({t: elo_model.get_team_rating(t) for t in established_teams})
    clubelo_ratings = pd.Series(
        {t: rating for t in established_teams if (rating := clubelo.team_rating_asof(t, lookup_date)) is not None}
    )
    common = local_ratings.index.intersection(clubelo_ratings.index)
    if len(common) < 2:
        return None

    return zscore_bridge(clubelo_ratings.loc[common], local_ratings.loc[common], team_rating)
