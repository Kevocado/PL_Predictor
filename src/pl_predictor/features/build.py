"""build.py — the single feature-construction entry point.

Every consumer (training, backtesting, the Streamlit app, and upcoming-
fixture prediction) calls `build_training_frame()` / `build_features_for`
rather than reimplementing feature logic — this is what keeps notebooks from
going stale the way FPL_Optimizer/exploration.ipynb did (it duplicated logic
inline instead of importing the shared module).
"""

from __future__ import annotations

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
from .date_keys import as_date_key, drop_unmatchable

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
        # turn into a literal 0.0 — see that loop's comment).
        self.xg_league_avg = self._xg_league_avg_series(self.xg_current)

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

    def _xg_league_avg_series(self, xg_current: pd.DataFrame) -> pd.Series:
        """Per-`xg_stat_cols` league-average expected-goals rate, keyed the way
        `build_row`'s blend looks it up.

        Understat's own cross-team mean when it has data; otherwise the
        actual-goals rate from `matches_df` (`_league_goals_per_match`), so a
        cold Understat still leaves the blend with a real target. Empty only
        when `matches_df` carries no goals either -- nothing left to fall back
        on, and the blend degrades to NaN rather than inventing a number."""
        avg = _league_goals_per_match(self.matches_df)
        if xg_current is not None and not xg_current.empty:
            return xg_current.mean()
        if avg is None:
            return pd.Series(dtype=float)
        return pd.Series({col: avg for col in self.xg_stat_cols}, dtype=float)

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
                current_val = team_xg.get(stat_col)
                if current_val is None or pd.isna(current_val):
                    # No Understat xG for this team (`team_xg` is an empty
                    # Series when `xg_current` has no row for it, so `.get()`
                    # returns None rather than NaN).
                    #
                    # Degrading straight to NaN here was the original defect,
                    # and it was worse than it looked: the NaN did not stay
                    # NaN. `ml_scoreline.MLScorelineModel.predict` ends in
                    # `.fillna(0)` (see `_row_to_matrix`), so every one of
                    # these columns reached the booster as a literal 0.0 --
                    # asserting the team created exactly zero expected goals,
                    # and (via the delta loop below) scored exactly to
                    # expectation. 0.0 also sits 2.9-3.5 SD below these
                    # columns' real means, far outside anything the fitted
                    # boosters had support for.
                    #
                    # The substitute is the league-average expected-goals rate,
                    # which `__init__` now always populates: Understat's own
                    # cross-team mean when available, else the actual-goals
                    # rate in `matches_df` (`_league_goals_per_match`). It is
                    # a real, in-distribution number that says the honest
                    # thing -- "this team's xG rate is not measurable right now;
                    # the league average is the best available estimate" -- and
                    # it leaves the delta loop below doing real work, since
                    # goals minus the league rate is a genuine
                    # over/under-performance reading rather than the "exactly
                    # to expectation" fiction 0.0 asserted.
                    #
                    # Chosen by measurement, not taste: simulating a cold
                    # Understat on each of the four available seasons and
                    # scoring against real outcomes (mean RPS, lower better):
                    #
                    #     league-avg xG, deltas recomputed  0.18855  <- this
                    #     league-avg xG, deltas zeroed       0.19035
                    #     xG=0.0 (before this fix)            0.19059
                    #     xG=NaN, routed natively             0.19152
                    #     real xG (warm reference)            0.18412
                    #
                    # Re-deriving the deltas is what earns the 0.002 RPS:
                    # zeroing them instead lands at 0.19035, barely better
                    # than the 0.0 it replaces, because per-team
                    # over/under-performance is exactly what those columns
                    # carry. And routing NaN through XGBoost's native
                    # missing-value handling is the WORST option tested --
                    # these boosters are fitted on `fillna(0)`-imputed frames
                    # (see `manifest.train_all`) and have never seen a missing
                    # value in these columns, so there is no learned default
                    # direction to route it by.
                    #
                    # Not fixed here: TRAINING still encodes a cold-missing xG
                    # as 0.0, via `manifest.train_all`'s
                    # `train_df[feature_cols].fillna(0)`, so train and serve now
                    # disagree on what "no xG" means (0.0 vs the league rate).
                    # Measured, that disagreement costs almost nothing: all
                    # train/serve pairings scored within ~0.0003 RPS of each
                    # other in a retrain-per-fold check, and the shipped
                    # boosters see this path only when Understat is cold. The
                    # serving side is kept because it is the side whose number
                    # gets reported. Aligning training means editing
                    # `manifest.py` and retraining, which is a separate change.
                    current_val = avg
                if current_val is None or pd.isna(current_val):
                    # `matches_df` has no goals either: genuinely nothing to
                    # say. NaN is honest here and is all that is left; it is
                    # still not what the model sees (see `_row_to_matrix`).
                    current_val = float("nan")
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
