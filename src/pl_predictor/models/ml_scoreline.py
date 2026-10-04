"""ml_scoreline.py — feeds engineered features (rolling form, Elo/Pi,
referee, xG) into XGBoost expected-goals regressors, then prices markets via
penaltyblog's `create_dixon_coles_grid` — evaluated on held-out data, and
(when it wins) usable as a genuine `models/manifest.py::chosen_model`
alongside Dixon-Coles/Bivariate-Poisson.

`predict_fixture(model, home, away)` — used everywhere in the app
(routes.py, player_goals.py, projected_table.py) — assumes a model that's
self-sufficient given just two team names. That's true for Dixon-Coles/
Bivariate-Poisson (team-level parameters fit directly from goals), but an ML
model needs a live *feature vector* for that specific fixture. `MLScorelineModel`
below closes that gap: it holds a `features.build.FixtureFeatureContext`
(the same one `build_features_for_fixtures` uses, built once from the
current `matches_df`) and builds each fixture's row on demand inside
`.predict()` — so it satisfies the exact same duck-typed interface
(`.teams`, `.predict(home, away, max_goals)`) as the penaltyblog models, and
plugs into every existing call site with no changes there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import penaltyblog as pb
import xgboost as xgb
from sklearn.metrics import log_loss

MIN_LAMBDA = 0.05  # create_dixon_coles_grid requires strictly positive lambdas

#: Every feature column that can legitimately arrive missing at serving, and
#: the value the fitted boosters were trained to see in that slot.
#:
#: This is a property of the *fitted model*, not of the feature, which is why it
#: lives here rather than in `features/`: it is what `manifest.train_all`'s
#: `train_df[feature_cols].fillna(0)` taught the boosters, and it is therefore
#: the only value serving may substitute without creating a train/serve
#: disagreement. Every entry is 0.0 because that is what training fills; the
#: dict exists so the set of columns that rely on it is *named*, and so a
#: feature added to `feature_cols` that arrives NaN does not silently inherit
#: an encoding nobody chose.
#:
#: Measured on the shipped `models/ml_scoreline_{home,away}.json` fitted over
#: its own 2,700-row training window (`n_train` in `models/manifest.json`);
#: "0.0 share" is the share of those rows the booster sees a literal 0.0 in,
#: which equals the training NaN share unless noted.
#:
#:   feature                      NaN%    0.0 share   present mean / sd / min / max    z(0)
#:   squad_continuity (x2)        46.88%    46.88%     0.8442 / 0.1084 / 0.539 / 1.000  -7.79
#:   h2h_home_win_rate            12.70%    36.64%     0.3720 / 0.3198 / 0.000 / 1.000  -1.16
#:   h2h_home_goal_diff_avg       12.70%    21.35%    -0.0646 / 1.3092 / -6.00 / 5.000  +0.05
#:   rest_days_home                0.46%     0.46%    10.6989 / 33.2257 / 2.00 / 813     -0.32
#:   rest_days_away                0.53%     0.53%     9.4904 / 16.7784 / 2.00 / 454     -0.57
#:   xg_{for,against}_last_{5,10} (x8)
#:                                 0.46-0.53%  same      ~1.48 / ~0.47 / ~0.29 / ~3.9  -2.8..-3.5
#:   xg_delta_*_last_{5,10} (x8)
#:                                 0.46-0.53%  same     ~-0.06 / ~0.39 / -2.7 / 2.2  +0.12..0.21
#:
#: Why these stay 0.0 rather than being imputed, in one line each — the full
#: reasoning, with the RPS measurements behind it, is in
#: `features/build.py`'s module docstring and in this PR's description:
#:
#: * `squad_continuity` is the only one where 0.0 is *semantically* impossible
#:   (no club in nine seasons retained under 53.9% of its minutes), so it looks
#:   like the worst offender at -7.79 SD. It is the best-protected column in the
#:   set, though, and "fixing" it is a measured regression: 46.88% of the rows
#:   the boosters were fitted on carry exactly that 0.0, every learned split on
#:   it sits inside [0.585, 0.968] (none below the observed minimum), and its
#:   response is *flat* to five decimal places across continuity = 0.0 .. 0.539.
#:   The booster learned one thing from this column -- "0.0 means no prior-season
#:   data" -- and reading 0.844 as a promoted team instead costs +0.00038 RPS on
#:   the affected rows because it makes a promoted club look like an established
#:   one. See also `_apply_missing_value_encoding` below, and the squad-continuity
#:   merge in `features/build.py::build_training_frame`, which carries the same
#:   measurement from the feature side.
#: * `h2h_*`: 0.0 is a *reachable* value of both (no goal difference; never won
#:   in five meetings), sits +0.05 and -1.16 SD from the mean, and is inside the
#:   observed range for both. Substituting the present-rows mean moves the
#:   shipped boosters by 0.00000 and -0.00001 RPS respectively.
#: * `rest_days_*`: 0.0 means "played the same day", which no PL team can --
#:   the observed minimum is 2 days -- but it is the encoding for "no prior match
#:   in the window", which is exactly the condition that produces the NaN, and
#:   `is_first_match_of_season_*` sits beside it carrying the same fact. Every
#:   learned split is >= 4.0, so 0.0 lands in the leftmost region the booster
#:   has for it. Substituting the league mean measured worse (+0.00526 RPS on
#:   the affected rows).
#: * the 16 xG columns are already handled upstream, on BOTH sides:
#:   `xg_form.resolve_missing_xg` substitutes the league-average rate, called
#:   from `build.build_training_frame` when fitting and from
#:   `build.FixtureFeatureContext.build_row` when serving. They only reach this
#:   dict when that imputation had nothing to work with -- a `matches_df`
#:   carrying no goals at all, where the rate is unavailable and the frame stays
#:   NaN. That is what these 0.0 encode, and it is why they are still 0.0 and
#:   not the league rate: with no league rate in `matches_df` there is no rate
#:   to substitute at serving either, so the two ends still meet at 0.0.
#:
#:   One residual difference worth naming, because it is a real property of
#:   "one function" rather than one fixed number: the *rate* is a statistic of
#:   whatever `matches_df` the caller holds, and training and serving do not
#:   hold the same one. Serving's context includes the in-progress season, so
#:   its league average differs from the fitted window's by a small amount
#:   (measured on this project's data: 0.0023-0.0464 goals across the four
#:   rates). So the two ends run identical code on slightly different inputs,
#:   which is the honest and correct state for a statistic -- a frozen constant
#:   would go stale as the league changes. This is categorically different from
#:   the skew this replaced, where the two ends ran *different code* and one of
#:   them invented a number (0.0) rather than measuring one.
MISSING_VALUE_ENCODING: dict[str, float] = {
    # Promotion / no prior-season squad data. See the note above: this is a
    # learned encoding, not an unrepresentable value.
    "home_squad_continuity": 0.0,
    "away_squad_continuity": 0.0,
    # Pair has never met in the loaded window; 0.0 is a real reading of both.
    "h2h_home_goal_diff_avg": 0.0,
    "h2h_home_win_rate": 0.0,
    # Team has no match in the loaded window at all (see `build._current_rest_days`).
    "rest_days_home": 0.0,
    "rest_days_away": 0.0,
}
MISSING_VALUE_ENCODING.update(
    {f"{side}_xg_{stat}_last_{w}": 0.0 for side in ("home", "away") for stat in ("for", "against") for w in (5, 10)}
)
MISSING_VALUE_ENCODING.update(
    {
        f"{side}_xg_delta_{stat}_last_{w}": 0.0
        for side in ("home", "away")
        for stat in ("for", "against")
        for w in (5, 10)
    }
)

# `DEFAULT_HYPERPARAMS` is the single source of truth both `_regressor()`
# and `evaluate/tune_hyperparams.py` (the Optuna search space) start from.
#
# Tuned via `evaluate/tune_hyperparams.py` (50-trial Optuna/TPE search,
# objective = mean RPS across `evaluate/walk_forward.py`'s 5 rolling
# validation folds, not the single fixed holdout — chosen specifically to
# avoid overfitting the hyperparameters to one season's quirks). Measured
# directly: walk-forward mean RPS 0.20128->0.19955 (-0.86%), AND — checked
# separately, since a walk-forward win alone wouldn't rule out overfitting
# those specific folds — the single fixed chronological-holdout RPS/Brier
# also improved (0.20736->0.20690 RPS, 0.61572->0.61403 Brier), so this is
# a corroborated win, not a walk-forward-specific artifact. The dominant
# change is much stronger L2 regularization (reg_lambda 1.0->18.1) — this
# was expected: this session found repeatedly (fouls in the shared set,
# table-context, shot-situation) that adding columns to this feature set
# hurt via overfitting on ~2,280 rows; substantially more L2 shrinkage is
# a direct, corroborated fix for exactly that, not a new hypothesis.
#
# Previous defaults (kept here for reference / a quick revert if a future
# retrain regresses unexpectedly): n_estimators=300, learning_rate=0.05,
# max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0,
# reg_lambda=1.0 (XGBoost's own internal default when unset — confirmed via
# a fitted booster's saved config; the sklearn wrapper reports "None" for
# unset, which does NOT mean zero, a mistake this project's own earlier
# research made and caught here via the same before/after discipline used
# throughout).
DEFAULT_HYPERPARAMS: dict = {
    "n_estimators": 250,
    "learning_rate": 0.0232,
    "max_depth": 4,
    "subsample": 0.5705,
    "colsample_bytree": 0.9011,
    "reg_alpha": 0.002,
    "reg_lambda": 18.1027,
}


def _regressor(hyperparams: dict | None = None) -> xgb.XGBRegressor:
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    return xgb.XGBRegressor(objective="count:poisson", random_state=42, **params)


def train_goal_regressors(
    X_train,
    goals_home_train,
    goals_away_train,
    dates: pd.Series | None = None,
    xi: float = 0.0018,
    hyperparams: dict | None = None,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """`dates` (each row's match date, same length/order as `X_train`) is
    optional but should always be passed in production: without it every
    match counts equally regardless of age, unlike Dixon-Coles/Bivariate-
    Poisson which already downweight older matches via the same
    `dixon_coles_weights(xi=0.0018)` used here — omitting it would let a
    growing pile of old, equally-weighted matches dilute recent signal
    (playing styles, squad economics) as the training window widens.

    `hyperparams` overrides any subset of `DEFAULT_HYPERPARAMS` (e.g. from
    an Optuna trial) — omit for production defaults."""
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if len(sample_weight) != len(X_train):
            raise ValueError("sample_weight must have one value per training row")
    if dates is not None:
        date_weights = np.asarray(pb.models.dixon_coles_weights(dates, xi=xi), dtype=np.float64)
        sample_weight = date_weights if sample_weight is None else sample_weight * date_weights
    home_model = _regressor(hyperparams)
    home_model.fit(X_train, goals_home_train, sample_weight=sample_weight)
    away_model = _regressor(hyperparams)
    away_model.fit(X_train, goals_away_train, sample_weight=sample_weight)
    return home_model, away_model


def predict_grid(home_model: xgb.XGBRegressor, away_model: xgb.XGBRegressor, x_row: pd.DataFrame, max_goals: int = 10):
    lam_home = max(float(home_model.predict(x_row)[0]), MIN_LAMBDA)
    lam_away = max(float(away_model.predict(x_row)[0]), MIN_LAMBDA)
    return pb.models.create_dixon_coles_grid(lam_home, lam_away, rho=0.0, max_goals=max_goals)


def predict_grids_batch(home_model: xgb.XGBRegressor, away_model: xgb.XGBRegressor, X: pd.DataFrame, max_goals: int = 10) -> list:
    """Same result as calling `predict_grid` once per row, but one XGBoost
    `.predict()` call for the whole batch instead of one call per row.
    XGBoost has fixed per-call overhead (booster/DMatrix setup) that's
    negligible for one prediction but dominates when looped — measured at
    ~5s for 380 rows looped vs a small fraction of a second batched. Always
    prefer this over a Python loop of `predict_grid` calls whenever more
    than a handful of fixtures need scoring at once (holdout evaluation,
    live calibration, backtesting)."""
    lam_home = np.maximum(home_model.predict(X), MIN_LAMBDA)
    lam_away = np.maximum(away_model.predict(X), MIN_LAMBDA)
    return [
        pb.models.create_dixon_coles_grid(float(h), float(a), rho=0.0, max_goals=max_goals)
        for h, a in zip(lam_home, lam_away)
    ]


def evaluate_on_holdout(home_model, away_model, X_val: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    result_code = {"H": 0, "D": 1, "A": 2}
    grids = predict_grids_batch(home_model, away_model, X_val)
    probs = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
    outcomes = val_df["ftr"].map(result_code).to_numpy()
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == outcomes).astype(float)
    buckets = np.clip((confidence * 10).astype(int), 0, 9)
    ece = sum(abs(correct[buckets == bucket].mean() - confidence[buckets == bucket].mean()) * (buckets == bucket).mean() for bucket in range(10) if (buckets == bucket).any())
    return {
        "rps": float(pb.metrics.rps_average(probs, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
        "log_loss": float(log_loss(outcomes, probs, labels=[0, 1, 2])),
        "ece": float(ece),
        "coverage": float(np.isfinite(probs).all(axis=1).mean()),
    }


def _apply_missing_value_encoding(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve every missing cell in a serving frame to a value the fitted
    boosters were fitted to see in that column, and say so loudly for the ones
    nobody chose.

    Two populations, deliberately handled differently:

    * **Columns in `MISSING_VALUE_ENCODING`.** Their NaN is a real, expected
      state with a measured reason (see that constant's docstring). They are
      filled with their documented value, which is 0.0 for every one of them --
      identical to what `manifest.train_all` filled during fitting, which is
      the whole point: substituting anything else is a train/serve
      disagreement, and for `squad_continuity` a measured regression.

    * **Every other column.** These are supposed to arrive populated. A NaN
      here is new behaviour nobody tested -- a newly added feature, or a
      degraded loader (a `matches_df` with no shot/corner columns leaves the
      whole `*_last_{3,5,10}_shots*` family missing) -- and the old behaviour
      was to absorb it into the same `fillna(0)` as the contract columns,
      silently, with no record that a number had been invented. That is the
      failure mode this function exists to stop.

      It warns and fills with 0.0 rather than raising. Raising would be louder
      and is tempting, but it converts "a slightly wrong number in an
      otherwise-working degraded mode" into a 500 for the whole /facts
      endpoint -- and a caller passing a `matches_df` without `hs`/`as`/`hc`/`ac`
      is enough to trigger it, so the guard would fire on legitimate input.
      A named warning is the strongest signal that does not have a worse
      failure mode than the bug it replaces; `tests/test_missing_value_encoding.py`
      pins it, and says plainly what would be needed to close it properly
      (recording each column's fit-time NaN rate in the manifest, which needs
      the model artefacts and a retrain).

    No cell that is already populated is touched, so a fully-populated row
    reaches the booster byte-identically -- asserted with `assert_frame_equal`
    in that test module, against a transcription of the pre-change expression.
    """
    contract = {c: v for c, v in MISSING_VALUE_ENCODING.items() if c in frame.columns}
    if contract:
        frame[list(contract)] = frame[list(contract)].fillna(contract)

    still_missing = frame.columns[frame.isna().any()].tolist()
    if still_missing:
        warnings.warn(
            "features/build.py emitted NaN for %d feature column(s) that are not "
            "in ml_scoreline.MISSING_VALUE_ENCODING, so they are being filled "
            "with 0.0 -- a number no training run ever chose for them, and not "
            "the one a fitted booster was fitted to see: %s. If this is a new "
            "feature, add it to MISSING_VALUE_ENCODING with the encoding "
            "manifest.train_all will use (currently 0.0 for everything, via "
            "`train_df[feature_cols].fillna(0)`) and retrain, or drop it from "
            "`feature_cols`."
            % (len(still_missing), ", ".join(map(repr, still_missing))),
            UserWarning,
            stacklevel=2,
        )
        frame = frame.fillna(0.0)
    return frame


def _row_to_matrix(row, feature_cols: list[str]) -> pd.DataFrame:
    """Shape one feature row into the numeric matrix the fitted boosters
    require. Shared by both single-row serving entry points below so they
    cannot drift apart on what "missing" means.

    Why a fill is here at all, now that it is easy to misread as "unknown means
    zero":

    * **dtype, not semantics.** A single-row frame built from a dict leaves an
      all-`None` column (e.g. `h2h_*` for a pair with no prior meetings) as
      `object` dtype even after `fillna`. XGBoost's `inplace_predict` rejects
      object dtypes outright, so the final `.astype(float)` is load-bearing
      regardless of the fill.
    * **train/serve symmetry.** These boosters are fitted on
      `manifest.train_all`'s `train_df[feature_cols].fillna(0)` (see
      `models/manifest.py`), so 0.0 is the value they were fitted to see in a
      missing slot. `_apply_missing_value_encoding` is what applies it, named
      per column in `MISSING_VALUE_ENCODING` and measured column by column.

    What this does *not* claim is that 0.0 is a sensible stand-in for any given
    feature. For most of the 22 it genuinely is not -- `squad_continuity` can
    never be 0.0 for an established club -- and that was measured rather than
    assumed: see `MISSING_VALUE_ENCODING`. The fix that is right lives
    upstream, in `features/xg_form.py::resolve_missing_xg`, which supplies a real
    league-average xG rate on the serving path (`build_row`) *and* on the
    fitting path (`build_training_frame`), so those 16 columns arrive already
    populated at both ends and this fill is not what decides what a missing
    measurement means. `predict_many_from_rows` routes through the same helper
    for the same reason: it used to be a second, independent copy of this fill,
    which is exactly the kind of drift that let #43 fix one side of the xG
    encoding and not the other.

    Measured effect of that upstream fix, simulating a cold Understat on each of
    the four available seasons (mean RPS against real outcomes, lower better):
    0.19059 before -> 0.18855 after, against 0.18412 with real xG available.
    """
    frame = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
    return _apply_missing_value_encoding(frame).astype(float)


class MLScorelineModel:
    """Live-serving wrapper: satisfies the same `.teams` + `.predict(home,
    away, max_goals)` interface as penaltyblog's goal models, so it's a
    drop-in `models/manifest.py::chosen_model` option. `context` is a
    `features.build.FixtureFeatureContext` built from the current
    `matches_df` — expensive parts (Elo/Pi replay, every team's rolling
    form) already done once at construction, so `.predict()` itself is
    cheap."""

    def __init__(self, home_model, away_model, feature_cols: list[str], teams: list[str], context):
        self.home_model = home_model
        self.away_model = away_model
        self.feature_cols = feature_cols
        self.teams = teams
        self.context = context

    def predict(self, home: str, away: str, max_goals: int = 10, **_kwargs):
        row = self.context.build_row(home, away)
        return predict_grid(self.home_model, self.away_model, _row_to_matrix(row, self.feature_cols), max_goals=max_goals)

    def predict_from_row(self, row, max_goals: int = 10):
        """Score a fixture using its own precomputed, point-in-time feature
        columns (e.g. a row of `build_training_frame`'s output) instead of
        `context`'s "current state as of today" lookup. This is the only
        leakage-safe way to evaluate a fixture that's already in the
        historical record — `.predict(home, away)` would otherwise pull in
        each team's rolling form/Elo/xG as of *now*, which for a past match
        includes results that happened after it (and, for anything in the
        held-out season, effectively the season's own outcome). Genuine
        upcoming fixtures have no such point-in-time features to fall back
        on, so they correctly keep using `.predict(home, away)` instead."""
        return predict_grid(
            self.home_model, self.away_model, _row_to_matrix(row, self.feature_cols), max_goals=max_goals
        )

    def predict_many_from_rows(self, df: pd.DataFrame, max_goals: int = 10) -> list:
        """Batched `predict_from_row` — see `predict_grids_batch`'s
        docstring for why this matters. Used wherever a whole historical
        frame (calibration, backtest) needs scoring at once instead of
        fixture-by-fixture."""
        # Same `_apply_missing_value_encoding` the single-row path uses. It
        # used to be a bare `fillna(0)` of its own — a second, independent
        # copy of the one decision that has to match training, which is how
        # #43's xG fix landed on one side of a serving/serving boundary it
        # should never have been able to straddle.
        X = _apply_missing_value_encoding(df[self.feature_cols].copy()).astype(float)
        return predict_grids_batch(self.home_model, self.away_model, X, max_goals=max_goals)
