"""build.py — the single feature-construction entry point.

Every consumer (training, backtesting, the Streamlit app, and upcoming-
fixture prediction) calls `build_training_frame()` / `build_features_for`
rather than reimplementing feature logic — this is what keeps notebooks from
going stale the way FPL_Optimizer/exploration.ipynb did (it duplicated logic
inline instead of importing the shared module).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import football_data, other_competitions, understat, understat_shots
from ..models.projected_table import compute_standings
from . import (
    cold_start,
    fixture_congestion,
    head_to_head,
    ratings,
    referee,
    rest_days,
    rolling_form,
    shot_situation,
    squad_change,
    streaks,
    table_context,
    xg_form,
)
from .date_keys import as_asof_key, as_date_key, drop_unmatchable

# Targets / raw-outcome columns that must never appear in the feature list
# (that would be leaking the match's own result into its own features).
TARGET_COLS = [
    "goals_home", "goals_away", "ftr", "total_corners", "total_cards",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
]


def _league_goals_per_match(matches_df: pd.DataFrame) -> float | None:
    """League-average goals scored per team per match — the fallback expected-
    goals rate used when Understat xG is unavailable. Returns None only if
    `matches_df` carries no goals at all (nothing to fall back to).

    Derived from `matches_df` (actual goals) rather than from Understat: this
    is the one path that must still work when the Understat cache is cold,
    which is exactly when it is needed. Actual goals per team per match is
    the same quantity xG estimates, and the two agree closely (measured over
    2022-23..2025-26: 1.533 from `matches_df` vs 1.544 mean of the training
    frame's `home_xg_for_last_5`).

    Note this is the *league* rate and is used only as a last resort — see
    `FixtureFeatureContext.build_row`'s xG loop, which prefers the team's own
    rolling goals because that measured better on every fold (see that
    comment for the numbers)."""
    if matches_df is None or matches_df.empty:
        return None
    if "goals_home" not in matches_df.columns or "goals_away" not in matches_df.columns:
        return None
    goals = pd.concat([matches_df["goals_home"], matches_df["goals_away"]], ignore_index=True).dropna()
    if goals.empty:
        return None
    return float(goals.mean())


#: The stat/window keys the xG league average is keyed by — the keys
#: `xg_form.latest_xg_form` returns, the keys `FixtureFeatureContext`'s blend
#: looks up, and (after the side prefix) the columns
#: `xg_form.attach_xg_features` produces.
XG_STAT_COLS = tuple(f"{stat}_last_{w}" for stat in ("xg_for", "xg_against") for w in xg_form.WINDOWS)


def _xg_league_avg_rates(xg_current: pd.DataFrame, matches_df: pd.DataFrame) -> pd.Series:
    """The league-average expected-goals rate per `XG_STAT_COLS`, keyed the way
    `resolve_missing_xg` and the cold-start blend look it up.

    Understat's own cross-team mean when it has data; otherwise the
    actual-goals rate from `matches_df` (`_league_goals_per_match`), so a cold
    Understat still leaves both ends of the pipeline with a real number to
    impute. Empty only when `matches_df` carries no goals either -- nothing
    left to fall back on, and the xG columns then stay NaN rather than a number
    being invented.

    Module-level rather than a `FixtureFeatureContext` method because
    `build_training_frame` needs the identical series to impute the same cells
    the serving path imputes; a second derivation of "the league's xG rate"
    would be one more pair of halves that can disagree.

    Point-in-time caveat, and why the training path does not call this: every
    row here is a function of the whole `matches_df`, so the value is only
    valid as-of the end of that frame. Serving is fine -- a live fixture's date
    is after everything in the context, so "the whole frame" is genuinely the
    past. Training is not: one rate written into 3,040 historical rows puts
    2025-26 xG into a 2018 fixture. Use `_xg_league_avg_rates_by_date` there,
    which derives the same quantity per fixture date from strictly earlier data.
    """
    avg = _league_goals_per_match(matches_df)
    if xg_current is not None and not xg_current.empty:
        return xg_current.mean()
    if avg is None:
        return pd.Series(dtype=float)
    return pd.Series({col: avg for col in XG_STAT_COLS}, dtype=float)


def _xg_league_avg_rates_by_date(
    understat_df: pd.DataFrame, matches_df: pd.DataFrame, dates: pd.Series
) -> pd.DataFrame:
    """`_xg_league_avg_rates` computed as of each row's own match date, indexed
    by date — the no-lookahead form of the same quantity.

    Same definition (Understat's cross-team mean of each team's rolling form
    when available, else the actual-goals rate from `matches_df`), restricted to
    matches strictly before each date, because a feature row for fixture F may
    only contain information available before F kicks off. Serving needs no
    equivalent: its context holds no data later than the fixture it prices, so
    the whole-frame rate already is the as-of rate there.

    Implemented as one backward `merge_asof` from the distinct fixture dates
    onto Understat's per-team rolling form, rather than a filter-and-mean per
    date — the same idiom `xg_form.attach_xg_features` uses, and the reason it
    is affordable on 3,040 rows.

    Dates before Understat's first match (or with no `matches_df` goals to fall
    back on) get a NaN rate, which `resolve_missing_xg` declines to substitute.
    That leaves the cell missing rather than filling it with a rate derived from
    later matches, which is the same trade #43's `_league_goals_per_match`
    fallback already makes and is why those contract entries stay 0.0.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    out = pd.DataFrame(index=dates.index, columns=list(XG_STAT_COLS), dtype=float)

    if understat_df is None or understat_df.empty:
        # Same fallback as `_xg_league_avg_rates`, made point-in-time: the
        # actual-goals rate from matches strictly before each date, as an
        # expanding mean. Computed this way rather than from the whole frame so
        # a cold Understat does not reintroduce the look-ahead the Understat
        # branch above exists to avoid — the fallback is a rate too, and a rate
        # taken from the future is still the future.
        if matches_df is None or matches_df.empty:
            return out
        if "goals_home" not in matches_df.columns or "goals_away" not in matches_df.columns:
            return out
        # Per-date team-goals AND per-date FIXTURES, from the same rows, so the
        # two halves of the rate describe the same population. A date holds
        # however many fixtures were played on it — a real Premier League
        # matchday is 10 fixtures, not 1 — so counting dates as matches
        # inflated this rate by roughly the fixtures-per-date (measured on this
        # project's 8-season window: 3.23 fixtures per date on average, max 10,
        # which put the end-of-window rate at 4.61 where the true league rate is
        # 1.43 — a 3.2x error, and 1.43 is the value this now reproduces).
        # Caught by CodeRabbit on #51.
        #
        # A fixture counts if EITHER of its goal values is known, which is
        # exactly the set of rows contributing to the numerator: a fixture with
        # `goals_home` NaN and `goals_away` = 2 contributes 2 team-goals and is
        # 1 fixture, so it counts once. A date on which every fixture has both
        # goal values NaN therefore contributes 0 to both — see the reindex
        # below, which keeps such a date in the index as a zero row rather than
        # dropping it — so the numerator and denominator can never describe
        # different populations.
        long_goals = pd.DataFrame(
            {
                "date": pd.to_datetime(matches_df["date"]),
                "home": matches_df["goals_home"],
                "away": matches_df["goals_away"],
            }
        )
        long_goals = long_goals[long_goals[["home", "away"]].notna().any(axis=1)]
        if long_goals.empty:
            return out

        team_goals = pd.concat(
            [
                long_goals.rename(columns={"home": "goals"})[["date", "goals"]],
                long_goals.rename(columns={"away": "goals"})[["date", "goals"]],
            ],
            ignore_index=True,
        )
        per_date = pd.DataFrame(
            {
                "goals": team_goals.groupby("date")["goals"].sum(),
                "fixtures": long_goals.groupby("date").size(),
            }
        ).sort_index()
        if per_date.empty:
            return out

        # A matchday on which no fixture has a goal value still belongs in the
        # index, as a zero-contribution row: it must not dilute either half of
        # the rate. Dropping the row instead would be wrong in a way that is
        # easy to miss -- the *next* date's shifted rate would then be read off
        # the row before it, so a fixture played on an all-unplayed matchday
        # would lose the rate that matchday's predecessors established.
        all_match_dates = pd.DatetimeIndex(sorted(pd.to_datetime(matches_df["date"]).unique()))
        per_date = per_date.reindex(all_match_dates).fillna(0.0)
        if per_date.empty:
            return out

        # Cumulative team-goals / (2 x cumulative fixtures), evaluated STRICTLY
        # BEFORE each date — hence the shift on both columns. Without it the
        # first matchday's own 0-0 would seed the series and a 2020-08-01
        # fixture would be imputed 0.0, which is the very encoding this change
        # exists to remove, reintroduced through the fallback. The shifted
        # first row is NaN (0/0) because no prior match exists, which is the
        # honest answer and also avoids dividing by zero.
        cumulative = per_date.cumsum()
        # `shift(1)` then `fillna(0)`, NOT `shift(1)` alone: shift propagates a
        # NaN one row forward, which would wipe out the next date's numerator and
        # denominator. Filling the shifted gap with 0 gives the right answer on
        # both counts -- the first date's row becomes 0/0, which pandas
        # evaluates to NaN (the honest "no prior match" answer, and no
        # ZeroDivisionError) -- while every later date gets the true prior
        # cumulative. The `fillna(0.0)` on `per_date` above is likewise not
        # optional: pandas' `cumsum` propagates NaN rather than skipping it
        # (verified on 3.0.6), so a zero-contribution row left as NaN would
        # poison every later cumulative, not just its own.
        shifted = pd.DataFrame(
            {
                "goals": cumulative["goals"].shift(1).fillna(0.0),
                "fixtures": cumulative["fixtures"].shift(1).fillna(0.0),
            },
            index=per_date.index,
        )
        shifted["rate"] = shifted["goals"] / (2 * shifted["fixtures"])
        rates = shifted["rate"]
        for key in XG_STAT_COLS:
            out[key] = rates.reindex(pd.Series(dates).to_numpy()).to_numpy()
        return out

    long_df, base_cols = xg_form.build_rolling_xg(understat_df)
    teams = long_df["team"].drop_duplicates().to_numpy()
    unique_dates = pd.Series(dates).drop_duplicates().sort_values().to_numpy()
    # One backward `merge_asof` per team, then a cross-team mean per date.
    # `allow_exact_matches=False` is the no-lookahead guarantee, identical to
    # the one `xg_form.attach_xg_features` relies on. Per-team rather than a
    # single `by="team"` merge because pandas requires the left keys sorted
    # within each group, and `merge_asof` will not sort them for you.
    right = as_asof_key(long_df[["date", "team"] + list(base_cols)]).sort_values("date")
    per_team = {}
    for team, group in right.groupby("team", sort=False):
        left = as_asof_key(pd.DataFrame({"date": unique_dates}))
        per_team[team] = pd.merge_asof(
            left, group, on="date", direction="backward", allow_exact_matches=False
        )
    merged = pd.concat(per_team.values(), axis=0, keys=per_team.keys())
    per_date = merged.groupby("date", sort=True)[list(base_cols)].mean()

    # Back out to one row per INPUT, in input order. `unique_dates` was
    # de-duplicated and sorted above and `per_date` is indexed by those same
    # dates, so reindexing on the input's own (unsorted, repeated) dates is
    # what restores both the order and the row count -- positional indexing
    # here would silently misalign every duplicate date.
    return per_date.reindex(pd.Series(dates).to_numpy())[list(XG_STAT_COLS)].reset_index(drop=True)


def _add_targets(matches_df: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.copy()
    zeros = pd.Series(0, index=df.index)
    df["total_corners"] = df.get("hc", zeros).fillna(0) + df.get("ac", zeros).fillna(0)
    df["total_cards"] = (
        df.get("hy", zeros).fillna(0)
        + df.get("ay", zeros).fillna(0)
        + df.get("hr", zeros).fillna(0)
        + df.get("ar", zeros).fillna(0)
    )
    # Per-team (not match-total, unlike corners/cards above) -- shots is
    # naturally asymmetric between a match's two sides, unlike corners/cards
    # which are conventionally quoted as one combined match line.
    df["home_shots"] = df.get("hs", zeros).fillna(0)
    df["away_shots"] = df.get("as", zeros).fillna(0)
    df["home_shots_on_target"] = df.get("hst", zeros).fillna(0)
    df["away_shots_on_target"] = df.get("ast", zeros).fillna(0)
    return df


def build_training_frame(
    seasons: list[str] | None = None, matches_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Returns (df, feature_cols) — one row per historical match, with
    targets (goals_home/away, ftr, total_corners, total_cards) plus all
    model-ready feature columns (rolling form, ratings, h2h, rest days).
    `feature_cols` is the canonical list to pass to any model."""
    if matches_df is None:
        matches_df = football_data.load_training_data(seasons=seasons)
    matches_df = _add_targets(matches_df)

    long_df, base_feature_cols = rolling_form.build_rolling_form(matches_df)
    long_df, confidence = cold_start.apply_cold_start_fallback(long_df, base_feature_cols)
    long_df["confidence"] = confidence

    home_join = (
        long_df[long_df["was_home"]]
        .set_index(["date", "team"])[base_feature_cols + ["confidence"]]
        .rename(columns={c: f"home_{c}" for c in base_feature_cols})
        .rename(columns={"confidence": "confidence_home"})
        .reset_index()
        .rename(columns={"team": "team_home"})
    )
    away_join = (
        long_df[~long_df["was_home"]]
        .set_index(["date", "team"])[base_feature_cols + ["confidence"]]
        .rename(columns={c: f"away_{c}" for c in base_feature_cols})
        .rename(columns={"confidence": "confidence_away"})
        .reset_index()
        .rename(columns={"team": "team_away"})
    )

    # Left side keeps its rows (a match with an unreadable date still appears, with
    # missing features); right side drops them, because `NaT` matches `NaT` and
    # duplicate `(NaT, team)` keys would fan out and break one-row-per-match.
    df = as_date_key(matches_df).merge(
        drop_unmatchable(as_date_key(home_join)), on=["date", "team_home"], how="left"
    )
    df = df.merge(
        drop_unmatchable(as_date_key(away_join)), on=["date", "team_away"], how="left"
    )

    # EXP-2026-18 (docs/AI_CONTINUITY.md): off-season squad continuity —
    # keyed by (season, team), not row-position like the rolling-form/Elo
    # blocks below, so it's merged onto df directly rather than concatenated.
    # NaN for a team-season with no prior-season data (promotion) is left
    # as-is, same as h2h/rest-days elsewhere in this function.
    #
    # CORRECTION to what this comment used to claim here: it said "XGBoost
    # handles missing values natively", which is false and was load-bearing
    # to get wrong. It does not: `manifest.train_all` fits on
    # `train_df[feature_cols].fillna(0)` and `val_df[feature_cols].fillna(0)`
    # (see models/manifest.py), so these boosters have never seen a missing
    # value in any of these columns and there is no native missing-value
    # direction to route one by. What they HAVE seen is 0.0: on this project's
    # own data 46.88% of the rows the shipped `ml_scoreline` boosters are
    # fitted on carry a literal 0.0 in `*_squad_continuity` — a promoted team's
    # row — and every learned split on the column sits inside [0.585, 0.968],
    # with none below the nine-season observed minimum of 0.539. The booster
    # therefore learned exactly one thing from this column ("0.0 means no
    # prior-season squad data"), and its response is flat to five decimal
    # places across continuity = 0.0 .. 0.539.
    #
    # That is worth stating because the naive reading of this column is
    # seductive and wrong. 0.0 is -7.79 standard deviations below the
    # present-rows mean, and no club in nine seasons retained less than 53.9%
    # of its minutes, so 0.0 is a value this feature cannot really take — which
    # makes it look like the worst silent-wrong-number in the whole feature set.
    # It is the best-protected one. Substituting the league mean (0.8442) at
    # serving instead was measured against real outcomes on the shipped
    # boosters and made things *worse*: +0.00038 RPS on the affected rows for
    # the home side, +0.00008 for the away side, because reading a promoted
    # club as "average squad continuity" tells the model it is an established
    # one. The encoding, measured column by column, is
    # `models.ml_scoreline.MISSING_VALUE_ENCODING`; see that constant and
    # `models/manifest.py` for the training side of the same contract.
    continuity = squad_change.team_season_continuity_table(sorted(matches_df["season"].unique()))
    df = df.merge(
        continuity.rename(columns={"team": "team_home", "squad_continuity": "home_squad_continuity"}),
        on=["season", "team_home"],
        how="left",
    )
    df = df.merge(
        continuity.rename(columns={"team": "team_away", "squad_continuity": "away_squad_continuity"}),
        on=["season", "team_away"],
        how="left",
    )

    elo_feats = ratings.replay_elo(matches_df).reset_index(drop=True)
    pi_feats = ratings.replay_pi_ratings(matches_df).reset_index(drop=True)
    h2h_feats = head_to_head.build_h2h_features(matches_df).reset_index(drop=True)

    # Champions League/Europa League/Conference League/FA Cup/EFL Cup
    # fixture dates for whatever window each source's free tier actually
    # grants (see data/other_competitions.py) — corrects rest_days for a
    # team whose true previous match was a midweek cup/European fixture
    # invisible to matches_df alone, and feeds the two congestion features
    # below. An empty result (e.g. no FOOTBALL_DATA_KEY, ESPN unreachable)
    # degrades both to their PL-only baseline rather than failing.
    other_fixtures_df = other_competitions.get_team_fixture_calendar()
    rest_feats = rest_days.build_rest_days(matches_df, other_fixtures_df).reset_index(drop=True)
    congestion_feats = fixture_congestion.build_congestion_features(matches_df, other_fixtures_df).reset_index(
        drop=True
    )

    referee_feats = referee.build_referee_features(matches_df).reset_index(drop=True)

    understat_seasons = sorted({str(s)[:4] for s in matches_df["season"].unique()})
    xg_data = understat.load_xg_data(seasons=understat_seasons)
    xg_feats, xg_cols = xg_form.attach_xg_features(matches_df, xg_data)
    xg_feats = xg_feats.reset_index(drop=True)

    # A promoted team's first match of a season has no Understat history for
    # `merge_asof` to reach back to, so its xG columns arrive missing. On this
    # project's own window that is 20 of 3,040 rows (0.46-0.53% per column) and
    # every one of them is exactly that shape -- a new club's debut.
    #
    # They are resolved here, to the league-average rate, by the same
    # `xg_form.resolve_missing_xg` that `FixtureFeatureContext.build_row` calls
    # for the same gap on a live fixture. Before this, the serving half did it
    # and the fitting half did not: `manifest.train_all`'s
    # `train_df[feature_cols].fillna(0)` taught every booster that a missing xG
    # is 0.0, meaning the club created exactly zero expected goals and (via the
    # deltas) scored exactly to expectation, while serving supplied a rate. A
    # ~0.0003 RPS disagreement, small enough to survive indefinitely and to get
    # worse the moment either side moves.
    #
    # Three placement choices, all load-bearing:
    #
    # * Here rather than in `train_all`, because every consumer of this frame
    #   scores these rows — the walk-forward folds, the backtest, the in-season
    #   calibration — and filling only inside `train_all` would leave all of
    #   them scoring the row the way the old boosters were fitted and serving
    #   it the way it is served.
    # * Before `attach_xg_delta_features` below rather than after, because the
    #   deltas are then derived from the imputed rate — goals minus league
    #   expectation is a real over/under-performance reading, whereas goals
    #   minus 0.0 is the "exactly to expectation" fiction the imputation exists
    #   to remove.
    # * Per-row (`_xg_league_avg_rates_by_date`) rather than one rate for the
    #   whole frame. The serving side can use a single whole-window rate because
    #   its context holds nothing later than the fixture it prices. Training
    #   cannot: one rate written into every historical row would put 2025-26 xG
    #   into a 2018 fixture, which is look-ahead in a feature column. Measured
    #   on this window, the two differ by 0.167 goals on average (max 0.413) on
    #   exactly the 20 affected rows, so this is a real difference and not a
    #   rounding artefact. A row whose date predates Understat's coverage gets
    #   no rate at all and stays missing, which `train_all`'s fill then encodes
    #   as 0.0 — the same encoding serving uses for that same case.
    xg_league_avg = _xg_league_avg_rates_by_date(xg_data, matches_df, matches_df["date"])
    for side in ("home", "away"):
        for stat_key in XG_STAT_COLS:
            col = f"{side}_{stat_key}"
            if col not in xg_cols:
                continue
            rates = xg_league_avg[stat_key]
            xg_feats[col] = [
                xg_form.resolve_missing_xg(reading, rate)
                for reading, rate in zip(xg_feats[col].to_numpy(), rates)
            ]

    streak_feats, streak_cols = streaks.attach_streak_features(matches_df)
    streak_feats = streak_feats.reset_index(drop=True)

    stakes_feats, stakes_cols = table_context.attach_table_context_features(matches_df)
    stakes_feats = stakes_feats.reset_index(drop=True)

    shot_situation_data = understat_shots.load_shot_situation_data(seasons=understat_seasons)
    situation_feats, situation_cols = shot_situation.attach_shot_situation_features(matches_df, shot_situation_data)
    situation_feats = situation_feats.reset_index(drop=True)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            elo_feats,
            pi_feats,
            h2h_feats,
            rest_feats,
            congestion_feats,
            referee_feats,
            xg_feats,
            streak_feats,
            stakes_feats,
            situation_feats,
        ],
        axis=1,
    )
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["pi_diff"] = df["pi_home"] - df["pi_away"]

    xg_delta_feats, xg_delta_cols = xg_form.attach_xg_delta_features(df)
    df = pd.concat([df, xg_delta_feats], axis=1)

    feature_cols = (
        [f"home_{c}" for c in base_feature_cols]
        + [f"away_{c}" for c in base_feature_cols]
        + ["elo_home", "elo_away", "elo_diff", "pi_home", "pi_away", "pi_diff"]
        + ["h2h_home_goal_diff_avg", "h2h_home_win_rate"]
        + ["rest_days_home", "rest_days_away", "is_first_match_of_season_home", "is_first_match_of_season_away"]
        + [referee.FEATURE_COL]
        + xg_cols
        + xg_delta_cols
        + streak_cols
        + ["home_squad_continuity", "away_squad_continuity"]
        # NOTE: stakes_cols (table_context.py — points off top4/relegation,
        # games played) is deliberately *not* included here. Measured
        # directly: it made ml_scoreline's held-out RPS/Brier measurably
        # worse (0.2074->0.2087 RPS, 0.6157->0.6190 Brier) and also
        # regressed Cards MAE (1.6173->1.6343), with zero presence in
        # ml_scoreline's top-15 SHAP features either. Unlike fouls (which
        # earned a place in the cards feature set specifically), this
        # doesn't clearly help any of the three current markets, so — same
        # "keep only if it earns it" discipline — it stays computed (still
        # in `df`, just not in `feature_cols`) but unused for now. Revisit
        # if a market where genuine table-position stakes plausibly matter
        # more directly (e.g. a future BTTS/goals-total submodel) gets built.
        #
        # situation_cols (shot_situation.py — rolling set-piece xG share)
        # is excluded the same way, for the same reason: measured directly,
        # it made ml_scoreline's held-out RPS/Brier measurably worse
        # (0.2074->0.2087 RPS, 0.6157->0.6190 Brier, despite genuinely
        # showing up in ml_scoreline's own top-15 SHAP — it's used, just not
        # helpfully) and regressed Cards MAE too (1.6173->1.6444, no SHAP
        # presence there). Corners MAE ticked down slightly (2.68->2.657)
        # but with zero SHAP presence in corners either, so that's most
        # likely noise from the extra columns changing tree structure, not
        # a real effect — not enough to justify keeping it anywhere. Stays
        # computed (in `df`) but unused; the one-time Understat shot-level
        # fetch this required is fully cached either way, so revisiting
        # this costs nothing if a future market wants it.
        #
        # congestion_cols (fixture_congestion.py — games_last_14_days_*,
        # european_fixture_last_4_days_*) is excluded the same way (see
        # EXP-2026-17 in docs/AI_CONTINUITY.md). Measured directly via
        # walk-forward: adding them worsened mean held-out RPS
        # (0.199931->0.200029) and Brier (0.581041->0.581281) versus the
        # already-shipped feature set. The likely cause is coverage, not the
        # signal being fake: `other_fixtures_df` (data/other_competitions.py)
        # only reaches Champions League for ~2 of this model's 8 training
        # seasons (football-data.org's free tier depth limit) and Europa
        # League/Conference League/FA Cup fixtures for the *current* season
        # hadn't even been played yet at measurement time — so both columns
        # are a constant zero for the large majority of training rows, which
        # is exactly the asymmetric-coverage pattern EXP-2026-04's
        # shot-situation features already showed adds noise rather than
        # signal. Stays computed (in `df`) but unused; revisit once
        # other_competitions.py's coverage is deeper (a full season of
        # EL/UECL/FA Cup/EFL Cup data, or more historical CL seasons if the
        # API tier ever allows it).
    )

    return df, feature_cols


def _current_h2h(matches_df: pd.DataFrame, home: str, away: str, window: int = head_to_head.H2H_WINDOW) -> dict:
    pair = head_to_head._pair_key(home, away)
    df = matches_df.copy()
    df["pair"] = [head_to_head._pair_key(h, a) for h, a in zip(df["team_home"], df["team_away"])]
    past = df[df["pair"] == pair].sort_values("date").tail(window)
    if past.empty:
        return {"h2h_home_goal_diff_avg": None, "h2h_home_win_rate": None}

    diffs = [
        (gd if h == home else -gd)
        for h, gd in zip(past["team_home"], past["goals_home"] - past["goals_away"])
    ]
    return {
        "h2h_home_goal_diff_avg": sum(diffs) / len(diffs),
        "h2h_home_win_rate": sum(1 for d in diffs if d > 0) / len(diffs),
    }


def _current_rest_days(
    matches_df: pd.DataFrame, team: str, as_of_date, other_fixtures_df: pd.DataFrame | None = None
) -> dict:
    team_matches = matches_df[(matches_df["team_home"] == team) | (matches_df["team_away"] == team)]
    if team_matches.empty:
        return {"rest_days": None, "is_first_match_of_season": True}
    last_date = team_matches["date"].max()
    same_season = (
        team_matches[team_matches["date"] == last_date]["season"].iloc[0] == matches_df["season"].iloc[-1]
    )
    # commence_time from a live fixtures source (e.g. the Odds API) is
    # tz-aware (UTC); matches_df's own `date` column is always tz-naive —
    # same mismatch tracking/store.py's `_naive` and data/fixtures.py's
    # `_future_only` already normalize elsewhere.
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)

    # A Champions League/cup fixture strictly before this one (but still
    # after the last PL match) is often the team's *true* most recent
    # match — see data/other_competitions.py / features/rest_days.py.
    if other_fixtures_df is not None and not other_fixtures_df.empty:
        other_dates = pd.to_datetime(other_fixtures_df.loc[other_fixtures_df["team"] == team, "date"])
        prior_other_dates = other_dates[other_dates < as_of]
        if not prior_other_dates.empty:
            last_date = max(last_date, prior_other_dates.max())

    return {
        "rest_days": (as_of - last_date).days,
        "is_first_match_of_season": not same_season,
    }


def _current_congestion(
    matches_df: pd.DataFrame, team: str, as_of_date, other_fixtures_df: pd.DataFrame | None = None
) -> dict:
    """Live-serving counterpart to `fixture_congestion.build_congestion_features`
    (which computes the same signals historically, in bulk, via shift/no-
    lookahead) — one team's games-in-last-14-days and whether it played a
    European match in the last 4 days, as of `as_of_date`."""
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)

    team_matches = matches_df[(matches_df["team_home"] == team) | (matches_df["team_away"] == team)]
    dates = list(team_matches["date"])
    european_dates: list = []
    if other_fixtures_df is not None and not other_fixtures_df.empty:
        team_other = other_fixtures_df[other_fixtures_df["team"] == team]
        other_dates = pd.to_datetime(team_other["date"])
        dates += list(other_dates)
        is_european = team_other["competition"].isin(fixture_congestion.EUROPEAN_COMPETITIONS)
        european_dates = list(other_dates[is_european.to_numpy()])

    games_window_start = as_of - pd.Timedelta(days=fixture_congestion.GAMES_WINDOW_DAYS)
    european_window_start = as_of - pd.Timedelta(days=fixture_congestion.EUROPEAN_WINDOW_DAYS)
    return {
        "games_last_14_days": sum(1 for d in dates if games_window_start <= d < as_of),
        "european_fixture_last_4_days": int(any(european_window_start <= d < as_of for d in european_dates)),
    }


class FixtureFeatureContext:
    """Precomputes everything needed to build a live feature row for *any*
    (home, away) pair once per `matches_df` — the expensive part (Elo/Pi
    replay, every team's current rolling form, xG form) happens once here,
    so each individual fixture lookup afterward is cheap. Powers both
    `build_features_for_fixtures` (a batch of fixtures at once) and
    `models.ml_scoreline`'s live-serving wrapper (one pair at a time, on
    demand, from routes.py's already-cached matches_df) — one implementation
    of "build a live feature row," not two."""

    def __init__(self, matches_df: pd.DataFrame):
        self.matches_df = matches_df.sort_values("date").reset_index(drop=True)

        self.form = rolling_form.latest_form(self.matches_df)
        self.base_feature_cols = list(self.form.columns)
        self.league_avg = self.form.mean()
        self.games_played = self.matches_df.melt(value_vars=["team_home", "team_away"])["value"].value_counts()

        self.elo = ratings.fit_elo(self.matches_df)
        self.pi = ratings.fit_pi_ratings(self.matches_df)

        # Includes any already-scheduled future Champions League/cup
        # fixtures too (not just played ones) — a live prediction needs to
        # know about a European match yet to happen just as much as one
        # that already did. See data/other_competitions.py.
        self.other_fixtures_df = other_competitions.get_team_fixture_calendar()

        # Referee is never known this far ahead of an upcoming fixture (see
        # features/referee.py) — every live prediction uses the league average.
        self.referee_fallback = referee.league_average_card_rate(self.matches_df)

        understat_seasons = sorted({str(s)[:4] for s in self.matches_df["season"].unique()})
        xg_data = understat.load_xg_data(seasons=understat_seasons)
        self.xg_current = xg_form.latest_xg_form(xg_data)
        self.xg_stat_cols = {f"{stat}_last_{w}": w for stat in ("xg_for", "xg_against") for w in xg_form.WINDOWS}
        # The league-average rate `build_row`'s cold-start blend shrinks toward.
        # Understat's own cross-team mean when it has data; when it doesn't,
        # `_league_goals_per_match` supplies the same quantity from actual goals
        # in `matches_df`, so the blend always has a real target instead of
        # degrading every xG column to NaN (which the serving line would then
        # turn into a literal 0.0 — see that loop's comment). Same function
        # `build_training_frame` uses, deliberately: the two ends impute the
        # same cells with the same rate.
        self.xg_league_avg = _xg_league_avg_rates(self.xg_current, self.matches_df)

        self.current_streaks = streaks.latest_streaks(self.matches_df)

        # compute_standings sums whatever rows it's given with no season
        # filtering of its own (see its docstring) — matches_df here can
        # span 8+ seasons, so this must be scoped to the current season
        # only, same as routes.py::get_projected_table's own call.
        current_season = football_data.season_str(football_data.CURRENT_SEASON_START_YEAR)
        season_matches = self.matches_df[self.matches_df["season"] == current_season]
        standings = compute_standings(season_matches) if not season_matches.empty else pd.DataFrame()
        self.current_stakes = table_context.live_stakes(standings)

        # EXP-2026-18: off-season squad continuity for the current season
        # only (a team's rating for this doesn't change match-to-match).
        # Degrades to an empty series (NaN for every team, same as a
        # promoted team's own no-prior-season case) rather than raising —
        # unlike the training path above, this runs on every live request,
        # and vaastav's archive not yet having a GW1 file for a season that
        # just started must not take the whole app down with it.
        try:
            continuity_table = squad_change.team_season_continuity_table([current_season])
            self.squad_continuity = continuity_table.set_index("team")["squad_continuity"]
        except RuntimeError:
            self.squad_continuity = pd.Series(dtype=float)

        # Current season only, not the full 8-season window (unlike xG form
        # above) — deliberately, and confirmed necessary live: shot-level
        # data is one Understat request *per match* (2,500-3,000 requests
        # for the full historical window, per understat_shots.py's own
        # docstring), which is a one-time cost worth paying on a machine
        # whose disk cache persists forever, but is repeated in full on
        # every restart wherever the disk is ephemeral (e.g. a Docker
        # deployment) — confirmed live to make first-request latency there
        # completely impractical. `hub_analytics.py`'s own Team Hub display
        # already scopes this to the current season only for the same
        # reason; the `set_piece_xg_share_last_5/10` cold-start blend
        # (below) already degrades gracefully toward the league average
        # early in a season when there aren't enough current-season matches
        # yet, same as every other rolling feature here.
        shot_situation_data = understat_shots.load_shot_situation_data(
            seasons=[str(football_data.CURRENT_SEASON_START_YEAR)]
        )
        self.shot_situation_current = shot_situation.latest_shot_situation_form(shot_situation_data)
        self.shot_situation_league_avg = (
            self.shot_situation_current.mean() if not self.shot_situation_current.empty else pd.Series(dtype=float)
        )
        self.shot_situation_stat_cols = {f"set_piece_xg_share_last_{w}": w for w in shot_situation.WINDOWS}

    def build_row(self, home: str, away: str, commence_time=None) -> dict:
        row = {"team_home": home, "team_away": away, "commence_time": commence_time}

        for team, prefix in [(home, "home_"), (away, "away_")]:
            n_games = int(self.games_played.get(team, 0))
            team_form = self.form.loc[team] if team in self.form.index else pd.Series(dtype=float)
            # cold-start blend: weight real form by games_played / window, same idea as
            # cold_start.apply_cold_start_fallback but for a single "current" row.
            for col in self.base_feature_cols:
                w = cold_start._window_of(col) or 1
                weight = min(n_games / w, 1.0)
                avg = self.league_avg.get(col)
                current_val = team_form.get(col)
                if pd.isna(current_val):
                    current_val = avg
                blended = weight * current_val + (1 - weight) * avg if avg is not None and not pd.isna(avg) else current_val
                row[f"{prefix}{col}"] = blended
            row[f"confidence_{prefix.rstrip('_')}"] = "current" if n_games >= max(rolling_form.LAG_WINDOWS) else (
                "blended" if n_games > 0 else "none"
            )

        row["elo_home"] = self.elo.get_team_rating(home)
        row["elo_away"] = self.elo.get_team_rating(away)
        row["elo_diff"] = row["elo_home"] - row["elo_away"]
        row["pi_home"] = self.pi.get_team_rating(home)
        row["pi_away"] = self.pi.get_team_rating(away)
        row["pi_diff"] = row["pi_home"] - row["pi_away"]

        row.update(_current_h2h(self.matches_df, home, away))

        fixture_date = commence_time or pd.Timestamp.now()
        home_rest = _current_rest_days(self.matches_df, home, fixture_date, self.other_fixtures_df)
        away_rest = _current_rest_days(self.matches_df, away, fixture_date, self.other_fixtures_df)
        row["rest_days_home"] = home_rest["rest_days"]
        row["rest_days_away"] = away_rest["rest_days"]
        row["is_first_match_of_season_home"] = home_rest["is_first_match_of_season"]
        row["is_first_match_of_season_away"] = away_rest["is_first_match_of_season"]

        home_congestion = _current_congestion(self.matches_df, home, fixture_date, self.other_fixtures_df)
        away_congestion = _current_congestion(self.matches_df, away, fixture_date, self.other_fixtures_df)
        row["games_last_14_days_home"] = home_congestion["games_last_14_days"]
        row["games_last_14_days_away"] = away_congestion["games_last_14_days"]
        row["european_fixture_last_4_days_home"] = home_congestion["european_fixture_last_4_days"]
        row["european_fixture_last_4_days_away"] = away_congestion["european_fixture_last_4_days"]

        row[referee.FEATURE_COL] = self.referee_fallback

        for team, prefix in [(home, "home"), (away, "away")]:
            n_games = int(self.games_played.get(team, 0))
            team_xg = self.xg_current.loc[team] if team in self.xg_current.index else pd.Series(dtype=float)
            for stat_col, w in self.xg_stat_cols.items():
                weight = min(n_games / w, 1.0)
                avg = self.xg_league_avg.get(stat_col)
                # No Understat xG for this team (`team_xg` is an empty Series
                # when `xg_current` has no row for it, so `.get()` returns
                # None rather than NaN).
                #
                # Degrading straight to NaN here was the original defect, and
                # it was worse than it looked: the NaN did not stay NaN.
                # `ml_scoreline.MLScorelineModel.predict` ends in `.fillna(0)`
                # (see `_row_to_matrix`), so every one of these columns reached
                # the booster as a literal 0.0 -- asserting the team created
                # exactly zero expected goals, and (via the delta loop below)
                # scored exactly to expectation. 0.0 also sits 2.9-3.5 SD below
                # these columns' real means, far outside anything the fitted
                # boosters had support for.
                #
                # The substitute is the league-average expected-goals rate,
                # which `__init__` always populates: Understat's own
                # cross-team mean when available, else the actual-goals rate in
                # `matches_df` (`_league_goals_per_match`). It is a real,
                # in-distribution number that says the honest thing -- "this
                # team's xG rate is not measurable right now; the league average
                # is the best available estimate" -- and it leaves the delta
                # loop below doing real work, since goals minus the league rate
                # is a genuine over/under-performance reading rather than the
                # "exactly to expectation" fiction 0.0 asserted.
                #
                # `xg_form.resolve_missing_xg` is where that decision lives, and
                # `build_training_frame` calls the same function for the same
                # gap in the rows the boosters are fitted on. It used to encode
                # a missing xG as 0.0 there (via `manifest.train_all`'s
                # `fillna(0)`), which made train and serve disagree on what "no
                # xG" means; the two halves now have one implementation.
                #
                # Chosen by measurement, not taste: simulating a cold Understat
                # on each of the four available seasons and scoring against
                # real outcomes (mean RPS, lower better):
                #
                #     league-avg xG, deltas recomputed  0.18855  <- this
                #     league-avg xG, deltas zeroed       0.19035
                #     xG=0.0 (before this fix)            0.19059
                #     xG=NaN, routed natively             0.19152
                #     real xG (warm reference)            0.18412
                #
                # Re-deriving the deltas is what earns the 0.002 RPS: zeroing
                # them instead lands at 0.19035, barely better than the 0.0 it
                # replaces, because per-team over/under-performance is exactly
                # what those columns carry. And routing NaN through XGBoost's
                # native missing-value handling is the WORST option tested.
                current_val = xg_form.resolve_missing_xg(team_xg.get(stat_col), avg)
                blended = weight * current_val + (1 - weight) * avg if avg is not None and not pd.isna(avg) else current_val
                row[f"{prefix}_{stat_col}"] = blended

        for side in ("home", "away"):
            for w in xg_form.WINDOWS:
                row[f"{side}_xg_delta_for_last_{w}"] = row[f"{side}_last_{w}_goals_for"] - row[f"{side}_xg_for_last_{w}"]
                row[f"{side}_xg_delta_against_last_{w}"] = (
                    row[f"{side}_last_{w}_goals_against"] - row[f"{side}_xg_against_last_{w}"]
                )

        row["home_current_streak"] = int(self.current_streaks.get(home, 0))
        row["away_current_streak"] = int(self.current_streaks.get(away, 0))

        row["home_squad_continuity"] = self.squad_continuity.get(home)
        row["away_squad_continuity"] = self.squad_continuity.get(away)

        for team, prefix in [(home, "home"), (away, "away")]:
            stakes = self.current_stakes.loc[team] if team in self.current_stakes.index else None
            row[f"{prefix}_points_off_top4"] = float(stakes["points_off_top4"]) if stakes is not None else 0.0
            row[f"{prefix}_points_off_relegation"] = (
                float(stakes["points_off_relegation"]) if stakes is not None else 0.0
            )
            row[f"{prefix}_games_played_this_season"] = float(stakes["played"]) if stakes is not None else 0.0

        for team, prefix in [(home, "home"), (away, "away")]:
            n_games = int(self.games_played.get(team, 0))
            team_situation = (
                self.shot_situation_current.loc[team]
                if team in self.shot_situation_current.index
                else pd.Series(dtype=float)
            )
            for stat_col, w in self.shot_situation_stat_cols.items():
                weight = min(n_games / w, 1.0)
                avg = self.shot_situation_league_avg.get(stat_col)
                current_val = team_situation.get(stat_col)
                if pd.isna(current_val):
                    current_val = avg
                blended = weight * current_val + (1 - weight) * avg if avg is not None and not pd.isna(avg) else current_val
                row[f"{prefix}_{stat_col}"] = blended

        return row


def build_features_for_fixtures(
    fixtures_df: pd.DataFrame,
    matches_df: pd.DataFrame | None = None,
    seasons: list[str] | None = None,
    context: FixtureFeatureContext | None = None,
) -> pd.DataFrame:
    """Feature rows for upcoming (not-yet-played) fixtures, using each team's
    *current* state (latest rolling form / ratings / h2h / rest days) rather
    than the shift(1) historical features `build_training_frame` produces.
    `fixtures_df` needs `team_home`, `team_away`, `commence_time` columns
    (see `data.fixtures.get_upcoming_fixtures`).

    Pass an already-built `context` (e.g. `manifest.load_models()`'s own
    `"context"`) whenever one already exists — constructing a
    `FixtureFeatureContext` replays Elo/Pi ratings and rebuilds every
    team's rolling/xG/shot-situation form from scratch, expensive enough
    that rebuilding it per-request (rather than reusing routes.py's
    5-minute-cached one) was confirmed live to both slow every request and
    leak enough memory to OOM a resource-constrained deployment. Omit it
    only for one-off/offline use (research scripts, tests) where no shared
    context exists yet."""
    if context is None:
        if matches_df is None:
            matches_df = football_data.load_training_data(seasons=seasons)
            current = football_data.fetch_current_season_partial()
            if current is not None and not current.empty:
                matches_df = pd.concat([matches_df, current], ignore_index=True)
        context = FixtureFeatureContext(matches_df)

    rows = [
        context.build_row(fixture["team_home"], fixture["team_away"], fixture.get("commence_time"))
        for _, fixture in fixtures_df.iterrows()
    ]
    return pd.DataFrame(rows)
