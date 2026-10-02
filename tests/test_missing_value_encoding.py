"""The missing-value encoding contract: which feature columns may arrive
missing at serving, what value the fitted boosters were fitted to see in that
slot, and what happens when one that may not arrives missing anyway.

Background, because the shape of this module is not obvious from its name.

`models/ml_scoreline.py` used to end its serving path in a bare
`.fillna(0)`, and `features/build.py`'s squad-continuity merge carried a
comment claiming XGBoost handles missing values natively. That claim was
false — `models/manifest.py::train_all` fits on
`train_df[feature_cols].fillna(0)`, so the boosters have never seen a missing
value in any column — and it was load-bearing to get wrong, because it invites
"fixing" any column whose 0.0 looks unrepresentative.

`squad_continuity` is the one that looks worst and is in fact the best
protected. Measured over the shipped boosters' own 2,700-row fitted window:

* `0.0` sits 7.79 standard deviations below the present-rows mean, and no club
  in nine seasons retained less than 53.9% of its minutes, so it is a value
  the feature cannot really take.
* But 46.88% of the rows the boosters were fitted on carry exactly that 0.0,
  and every learned split on the column is inside [0.585, 0.968] — none below
  the observed minimum. The booster learned one thing from this column ("0.0
  means no prior-season squad data") and is *flat* to five decimals across
  continuity = 0.0 .. 0.539.
* Substituting the league mean at serving instead measured **worse** against
  real outcomes on the shipped boosters: +0.00038 RPS on the affected rows
  (home side), +0.00008 (away side). It reads a promoted club as established.

So these tests pin the *encoding*, not a substitute value. The "impute
instead" version of this suite is deliberately not written: it would encode a
measured regression as a requirement.

Hermetic: fully synthetic league, every network-touching loader stubbed, no
`data/cache/` dependency, no model artefacts. The equivalence tests fit their
own tiny boosters on the same synthetic frame *with the same `.fillna(0)`
training encoding* that `train_all` uses, which is the whole point — the
encoding under test is the one the booster is fitted to.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from pl_predictor.data import other_competitions, understat as understat_module, understat_shots
from pl_predictor.features import build as build_module
from pl_predictor.features import squad_change, xg_form
from pl_predictor.features.build import FixtureFeatureContext, build_training_frame
from pl_predictor.models import ml_scoreline

XG_COLS = [
    f"{side}_xg_{stat}_last_{w}"
    for side in ("home", "away")
    for stat in ("for", "against")
    for w in xg_form.WINDOWS
]
XG_DELTA_COLS = [
    f"{side}_xg_delta_{stat}_last_{w}"
    for side in ("home", "away")
    for stat in ("for", "against")
    for w in xg_form.WINDOWS
]
#: Every column the audit found able to arrive NaN out of `build_row`.
EXPECTED_CONTRACT = {
    "home_squad_continuity",
    "away_squad_continuity",
    "h2h_home_goal_diff_avg",
    "h2h_home_win_rate",
    "rest_days_home",
    "rest_days_away",
    *XG_COLS,
    *XG_DELTA_COLS,
}

TEAMS = [f"T{i}" for i in range(6)]
#: The raw per-team stat columns `features/rolling_form.py` reads (`hs`, `as`,
#: ...). Omitting them is what reproduces a partially-populated season.
SHOT_COLS = ["hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar", "hf", "af"]


def _synthetic_matches(n_teams: int = 6, n_seasons: int = 3, with_shots: bool = True) -> pd.DataFrame:
    """A small, fully synthetic league. No cache and no network."""
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for season_idx in range(n_seasons):
        season = f"{2020 + season_idx}-{2021 + season_idx}"
        date = pd.Timestamp(f"{2020 + season_idx}-08-01")
        for rnd in range(n_teams // 2 * 4):
            home = teams[(rnd * 2 + season_idx) % n_teams]
            away = teams[(rnd * 2 + 1 + season_idx) % n_teams]
            gh, ga = (rnd + season_idx) % 4, (rnd * 2 + season_idx) % 3
            row = {
                "date": date,
                "season": season,
                "team_home": home,
                "team_away": away,
                "goals_home": gh,
                "goals_away": ga,
                "ftr": "H" if gh > ga else ("A" if gh < ga else "D"),
            }
            if with_shots:
                row.update(
                    hs=(rnd * 3) % 17, **{"as": (rnd * 5) % 15}, hst=(rnd * 2) % 7,
                    ast=(rnd * 3) % 6, hc=(rnd * 4) % 9, ac=(rnd * 6) % 8,
                    hy=(rnd * 2) % 4, ay=(rnd * 5) % 3, hr=(rnd % 2),
                    ar=((rnd + 1) % 2), hf=(rnd * 3) % 13, af=((rnd + 2) * 3) % 11,
                )
            rows.append(row)
            date = date + pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def _continuity_stub(matches: pd.DataFrame, *, serving_seasons_warm: bool):
    """`team_season_continuity_table`, stubbed to reproduce promotion.

    Returns real per-team continuity for every season the synthetic league
    contains — so `build_training_frame` gets a column with genuine spread and
    the booster has something to learn — while returning **nothing** for a
    season the league does not contain. `FixtureFeatureContext` asks for the
    current season alone, so every serving lookup misses, which is exactly the
    real situation for a promoted team (and for a cold vaastav archive).

    `serving_seasons_warm=False` makes *every* lookup miss, including the ones
    `build_training_frame` makes; `True` answers them, so training sees a
    column with real values and real NaNs while serving sees none.
    """
    known = sorted(set(matches["season"].unique()))
    rows = []
    for season in known:
        for i, team in enumerate(TEAMS):
            if i >= len(TEAMS) - 2 and season != known[-1]:
                continue  # promoted: no prior-season data for this pairing
            rows.append(
                {"season": season, "team": team, "squad_continuity": 0.55 + 0.45 * ((i + len(season)) % 6) / 5.0}
            )

    def _table(seasons):
        if not serving_seasons_warm:
            return pd.DataFrame(columns=["season", "team", "squad_continuity"])
        return pd.DataFrame(rows).query("season in @seasons")

    return _table


def _stub_loaders(monkeypatch, matches, *, xg: bool = False, serving_continuity: bool = False, calendar: bool = True):
    """Stub every loader that could reach the network.

    `xg=False` reproduces a cold Understat — the degradation PR #43 addressed
    and `build_row` now handles upstream with a league-average rate.
    `serving_continuity=False` reproduces a team with no squad-continuity row.
    """
    empty = other_competitions._EMPTY.copy() if calendar else pd.DataFrame()
    monkeypatch.setattr(other_competitions, "get_team_fixture_calendar", lambda: empty.copy())
    if xg:
        def _warm(seasons):
            return pd.DataFrame({"date": [], "xg_home": [], "xg_away": []})
        monkeypatch.setattr(understat_module, "load_xg_data", _warm)
    else:
        monkeypatch.setattr(understat_module, "load_xg_data", lambda **_: pd.DataFrame())
    monkeypatch.setattr(understat_shots, "load_shot_situation_data", lambda **_: pd.DataFrame())
    monkeypatch.setattr(
        squad_change,
        "team_season_continuity_table",
        _continuity_stub(matches, serving_seasons_warm=serving_continuity),
    )


@pytest.fixture
def synthetic(monkeypatch):
    matches = _synthetic_matches()
    _stub_loaders(monkeypatch, matches)
    return matches


def _tiny_models(matches):
    """Two small boosters fitted the way `manifest.train_all` fits the real
    ones: on `train_df[feature_cols].fillna(0)`. No artefacts needed, and the
    encoding they are fitted to see is therefore exactly the one under test.
    """
    df, cols = build_training_frame(matches_df=matches)
    X = df[cols].fillna(0).astype(float)
    params = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.2, "random_state": 0}
    home = xgb.XGBRegressor(objective="count:poisson", **params)
    away = xgb.XGBRegressor(objective="count:poisson", **params)
    home.fit(X, df["goals_home"])
    away.fit(X, df["goals_away"])
    return home, away, cols


def _served(row, cols, home_m, away_m):
    return ml_scoreline.predict_grid(home_m, away_m, ml_scoreline._row_to_matrix(row, list(cols)))


# --------------------------------------------------------------------------
# The contract itself
# --------------------------------------------------------------------------


def test_contract_names_exactly_the_columns_that_can_arrive_missing(monkeypatch, synthetic):
    """The dict is not a guess; it is the measured NaN surface of `build_row`.

    Every pair below is walked and the raw `build_row` dict is checked *before*
    any fill — that pre-fill frame is the entire defect surface. A feature added
    to `feature_cols` without a decision shows up here as a failure rather than
    as a silently fabricated 0.0.
    """
    _stub_loaders(monkeypatch, synthetic, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)

    seen: set[str] = set()
    for home, away in [("T0", "T1"), ("T0", "T2"), ("T2", "T3"), ("T4", "T5"), ("Unknown", "T1"), ("T1", "Unknown")]:
        row = ctx.build_row(home, away, commence_time=pd.Timestamp("2024-08-01"))
        seen.update(
            c for c, v in row.items() if v is None or (isinstance(v, float) and pd.isna(v))
        )
    # Only columns in `feature_cols` reach the booster -- `_row_to_matrix`
    # reindexes to them, so anything else `build_row` happens to emit
    # (the deliberately-unused `set_piece_xg_share_*`, `confidence_*`,
    # `team_home`) cannot produce a fabricated 0.0 and is not this test's
    # subject.
    _, feature_cols = build_training_frame(matches_df=synthetic)
    seen &= set(feature_cols)

    # Nothing may arrive NaN that the contract does not already cover: that is
    # the actual defect (a fabricated 0.0 nobody chose), and it is the half of
    # the assertion that is cheap to get wrong.
    not_covered = seen - set(ml_scoreline.MISSING_VALUE_ENCODING)
    assert not not_covered, (
        f"these columns arrive NaN and are not in MISSING_VALUE_ENCODING, so "
        f"they are filled with a 0.0 no training run chose: {sorted(not_covered)}"
    )

    # What this fixture reaches is exactly the six columns PR #43 did *not*
    # touch. The other 16 are the xG family, which `build_row` now fills
    # upstream with the league-average rate, so they no longer leak — they are
    # in the contract as the standing record that they are encoded rather than
    # imputed, and because `_league_goals_per_match` returning None (a
    # `matches_df` with no goals at all) is the one remaining way they could.
    # That path turns out to raise in `FixtureFeatureContext.__init__` rather
    # than reach serving, so it is covered by
    # `test_missing_goals_raises_rather_than_serving_fabricated_xg` instead of
    # here. Asserting the exact set, not a count, is what stops this test
    # quietly narrowing to whatever the fixture happens to produce.
    assert seen == {
        "home_squad_continuity", "away_squad_continuity",
        "h2h_home_goal_diff_avg", "h2h_home_win_rate",
        "rest_days_home", "rest_days_away",
    }, f"the NaN surface of build_row changed: {sorted(seen)}"

    # The remaining 16 are reachable only with a fully-populated league and no
    # goals at all, so they are walked explicitly to keep the contract and the
    # surface pinned together.
    unreached = set(ml_scoreline.MISSING_VALUE_ENCODING) - seen
    assert unreached == set(XG_COLS) | set(XG_DELTA_COLS), (
        f"expected only the 16 xG columns to be unreachable here; got {sorted(unreached)}"
    )
    for col in unreached:
        assert ml_scoreline.MISSING_VALUE_ENCODING[col] == 0.0
        seen.add(col)

    assert seen == EXPECTED_CONTRACT, (
        "MISSING_VALUE_ENCODING and the columns that can arrive NaN out of "
        "build_row have diverged.\n"
        f"  arrived NaN but not in contract: {sorted(seen - set(ml_scoreline.MISSING_VALUE_ENCODING))}\n"
        f"  in contract but never arrived NaN: {sorted(set(ml_scoreline.MISSING_VALUE_ENCODING) - seen)}\n"
        "A column in the contract that never goes missing is harmless; a column "
        "that goes missing and is not in the contract gets a fabricated 0.0 that "
        "no training run chose."
    )


def test_contract_columns_are_all_real_feature_columns(synthetic):
    """Every name in the contract must actually exist in `feature_cols`.

    A typo'd name would make the contract look complete while silently covering
    nothing — the same failure as an empty dict.
    """
    _, cols = build_training_frame(matches_df=synthetic)
    unknown = sorted(set(ml_scoreline.MISSING_VALUE_ENCODING) - set(cols))
    assert not unknown, f"contract names columns that are not features: {unknown}"


def test_contract_has_no_columns_outside_the_features_module_contract():
    """The contract is a closed set: 22 columns, all of them NaN-capable.

    Not a tautology guard, though. The failure it exists to catch is the
    contract growing to "cover" a column that cannot go missing, which would
    make the dict look thorough while quietly documenting nothing.
    """
    assert len(ml_scoreline.MISSING_VALUE_ENCODING) == 22
    assert set(ml_scoreline.MISSING_VALUE_ENCODING) == EXPECTED_CONTRACT


def test_every_contract_value_matches_what_training_fills():
    """The contract and `train_all`'s `.fillna(0)` must not drift.

    This is the train/serve symmetry the whole constant exists to protect, and
    it is checkable without a retrain: `train_all` fills every feature column
    with 0.0, so if the contract ever names a value other than 0.0 for a
    column that can go missing, the two halves of the contract have quietly
    stopped agreeing. A future non-zero encoding would need this test changed
    *and* `train_all` changed, together.
    """
    non_zero = {c: v for c, v in ml_scoreline.MISSING_VALUE_ENCODING.items() if v != 0.0}
    assert not non_zero, (
        f"contract values {non_zero} disagree with train_all's fillna(0); "
        "training still fills these columns with 0.0, so serving them with "
        "anything else is a silent train/serve mismatch"
    )


# --------------------------------------------------------------------------
# The two properties the audit actually established
# --------------------------------------------------------------------------


def test_cold_squad_continuity_serves_the_encoding_the_boosters_were_fitted_on(monkeypatch, synthetic):
    """A cold `squad_continuity` must reach the booster as 0.0 — the value
    46.88% of the fitted rows carry — and the booster must respond to it as it
    responds to a *trained* 0.0 row.

    Written as an equivalence against the model's own fitted encoding rather
    than as "is not 0.0", because the audit's finding is that 0.0 here is not a
    bug to be removed: it is the only value this booster reads as "no
    prior-season squad data". The test fails if someone substitutes a league
    mean, and it fails if someone routes NaN through XGBoost's native missing
    handling, which these boosters have never seen either.
    """
    # `serving_continuity=True` still leaves every *serving* lookup cold: the
    # stub only answers for seasons the synthetic league contains, and
    # `FixtureFeatureContext` asks for the current season alone. That is the
    # real promoted-team shape — training has genuine values and genuine NaNs,
    # serving has none.
    _stub_loaders(monkeypatch, synthetic, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    home_m, away_m, cols = _tiny_models(synthetic)

    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    assert row["home_squad_continuity"] is None, (
        "precondition: with the serving continuity lookup cold, build_row must "
        "emit a genuine missing value here -- if it stopped, this test is no "
        "longer covering the case it exists for"
    )
    # And the training frame really does carry both populated values and
    # missing ones, so the booster has a genuine 0.0 region to have learned.
    train_df, _ = build_training_frame(matches_df=synthetic)
    assert train_df["home_squad_continuity"].notna().any(), (
        "precondition: training must contain real squad_continuity values, or "
        "the booster cannot have learned anything about this column"
    )
    assert train_df["home_squad_continuity"].isna().any(), (
        "precondition: training must contain missing squad_continuity, or there "
        "is no trained 0.0 region to be equivalent to"
    )

    served = ml_scoreline._row_to_matrix(row, list(cols))
    assert served.loc[0, "home_squad_continuity"] == 0.0
    assert np.isfinite(served.loc[0, "home_squad_continuity"])

    # The equivalence that matters: a served 0.0 must score identically to the
    # same row with 0.0 written in explicitly, i.e. nothing downstream treats
    # this missing value differently from a trained one.
    explicit = _served({**row, "home_squad_continuity": 0.0}, cols, home_m, away_m)
    from_missing = _served(row, cols, home_m, away_m)
    assert explicit.home_win == pytest.approx(from_missing.home_win, abs=0)
    assert explicit.away_win == pytest.approx(from_missing.away_win, abs=0)

    # And it must differ from the "league mean" substitute the audit measured as
    # a regression, or this test has stopped being able to tell the two apart.
    imputed = _served({**row, "home_squad_continuity": 0.8442}, cols, home_m, away_m)
    assert imputed.home_win != pytest.approx(from_missing.home_win, abs=1e-9), (
        "serving 0.8442 for a missing squad_continuity gave the same prediction "
        "as serving 0.0, so this booster cannot distinguish them and the "
        "encoding argument in MISSING_VALUE_ENCODING no longer applies -- "
        "re-measure it on the shipped artefacts rather than assuming"
    )


def test_a_genuine_zero_in_a_count_feature_is_still_served_as_zero(monkeypatch, synthetic):
    """`fillna` must not be able to tell a real 0 from a missing one, because
    for a count feature they are the same number and both are correct.

    `home_last_3_goals_for` = 0.0 means "this team scored no goals in its last
    three" — a real, frequent, in-support reading. Unlike `squad_continuity`,
    0 is inside this feature's range and is what training fills for it too. The
    test exists to stop a future "impute the mean everywhere" change from
    quietly rewriting genuine zeros into league averages, which is the blanket
    fix this change rejects on measurement.
    """
    _stub_loaders(monkeypatch, synthetic, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    home_m, away_m, cols = _tiny_models(synthetic)

    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    league_mean = float(synthetic["goals_home"].mean())
    assert league_mean != 0.0, "precondition: the league mean must differ from 0.0 for this to bite"

    row["home_last_3_goals_for"] = 0.0  # a real zero, not a missing one
    served = ml_scoreline._row_to_matrix(row, list(cols))
    assert served.loc[0, "home_last_3_goals_for"] == 0.0

    # Same guarantee on the batched path, which used to be an independent copy
    # of this fill.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a populated row must not warn at all
        batched = ml_scoreline._apply_missing_value_encoding(
            pd.DataFrame([row])[list(cols)]
        )
    assert batched.loc[0, "home_last_3_goals_for"] == 0.0

    # And end to end: a genuinely goalless recent record must not score
    # identically to a league-average one. Direction is deliberately not
    # asserted -- on a six-team synthetic league the learned relationship is
    # not the real one, and only the non-identity matters here: if a blanket
    # "impute the mean" ever reached this column the two would collapse onto
    # each other and this would fail.
    goals_scored = float(home_m.predict(served)[0])
    goals_averaged = float(
        home_m.predict(
            ml_scoreline._row_to_matrix({**row, "home_last_3_goals_for": league_mean}, list(cols))
        )[0]
    )
    assert goals_scored != goals_averaged, (
        "a genuine 0 goals scored and a league-average record score identically, "
        "so the fill is rewriting real zeros into league averages and this test "
        "has stopped being able to see it"
    )


# --------------------------------------------------------------------------
# Warm predictions must not move
# --------------------------------------------------------------------------


def test_warm_row_is_bit_identical_to_the_pre_change_expression(monkeypatch, synthetic):
    """Every populated cell survives untouched.

    Compared against a straight transcription of the old
    `reindex(...).fillna(0).astype(float)` rather than a stored number, so this
    stays a real check if the fixtures change. `assert_frame_equal` defaults to
    `check_exact=True`, spelled out because "approximately the same" would not
    have caught a fill that nudged a value.
    """
    _stub_loaders(monkeypatch, synthetic, xg=True, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    home_m, away_m, cols = _tiny_models(synthetic)

    for home, away in [("T0", "T1"), ("T2", "T3"), ("T4", "T5")]:
        row = ctx.build_row(home, away, commence_time=pd.Timestamp("2024-08-01"))
        old = pd.DataFrame([row]).reindex(columns=list(cols), fill_value=0).fillna(0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a warm row must not warn at all
            new = ml_scoreline._row_to_matrix(row, list(cols))
            batched = ml_scoreline._apply_missing_value_encoding(
                pd.DataFrame([row])[list(cols)]
            ).astype(float)
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        pd.testing.assert_frame_equal(batched, old, check_exact=True)
        np.testing.assert_array_equal(home_m.predict(new), home_m.predict(old))
        np.testing.assert_array_equal(away_m.predict(new), away_m.predict(old))


def test_predict_many_from_rows_is_bit_identical_to_the_pre_change_expression(monkeypatch, synthetic):
    """The batched entry point used to be a second, independent `fillna(0)`.

    Routing it through the shared helper must not move a single cell, including
    for rows where a contract column *is* missing — the batch path feeds whole
    precomputed historical frames, where `squad_continuity` is missing for
    exactly the promoted teams that make up 46.88% of them.
    """
    _stub_loaders(monkeypatch, synthetic, xg=True, serving_continuity=True)
    df, cols = build_training_frame(matches_df=synthetic)

    frame = df[list(cols)].copy()
    missing_before = int(frame.isna().sum().sum())
    assert missing_before > 0, (
        "precondition: the historical frame must contain missing cells for this "
        "test to mean anything"
    )
    assert any(
        frame[c].isna().any() for c in ml_scoreline.MISSING_VALUE_ENCODING
    ), "precondition: a contract column must actually be missing in the frame"

    old = frame.fillna(0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing here is outside the contract
        new = ml_scoreline._apply_missing_value_encoding(frame.copy()).astype(float)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_serving_entry_points_share_one_implementation(monkeypatch, synthetic):
    """The drift that caused this bug class, pinned.

    `_row_to_matrix` and `predict_many_from_rows` used to each carry their own
    `fillna(0)`. Two copies of the one decision that has to match training is
    how the xG encoding ended up fixed on one side of a boundary it should
    never have been able to straddle. Checked on behaviour, not on source text.
    """
    _stub_loaders(monkeypatch, synthetic, xg=True, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    home_m, away_m, cols = _tiny_models(synthetic)
    model = ml_scoreline.MLScorelineModel(
        home_model=home_m, away_model=away_m, feature_cols=list(cols), teams=[], context=ctx
    )

    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    single_matrix = ml_scoreline._row_to_matrix(row, list(cols))
    batch_matrix = ml_scoreline._apply_missing_value_encoding(
        pd.DataFrame([{**row}])[list(cols)]
    ).astype(float)

    # The matrices themselves must be identical, cell for cell -- that is the
    # claim ("one implementation of what missing means"), and it is exact.
    pd.testing.assert_frame_equal(batch_matrix, single_matrix, check_exact=True)

    # And therefore the predictions. Compared at float32 tolerance rather than
    # exactly: `predict_grid` takes `float(...)` off a float32 array while
    # `predict_grids_batch` off a float64 one, so the two paths differ in the
    # 8th significant figure for reasons that predate this change and have
    # nothing to do with the missing-value encoding. Tight enough that only a
    # genuinely different fill would trip it.
    single = ml_scoreline.predict_grid(home_m, away_m, single_matrix)
    batched = model.predict_many_from_rows(pd.DataFrame([{**row}])[list(cols)])[0]
    for attr in ("home_win", "draw", "away_win"):
        a, b = getattr(single, attr), getattr(batched, attr)
        assert abs(a - b) < 1e-6, f"{attr} differs between entry points: {a!r} vs {b!r}"


# --------------------------------------------------------------------------
# The path that used to be silent
# --------------------------------------------------------------------------


def test_a_missing_column_outside_the_contract_is_named_rather_than_absorbed(monkeypatch, synthetic):
    """A column nobody encoded must produce a warning that says which one.

    This is the failure mode being closed. Before, any NaN reaching the serving
    frame was absorbed into the same `fillna(0)` as the contract columns, so a
    newly added feature that arrived missing became a fabricated 0.0 with no
    signal at all. It is a warning rather than a raise on purpose: a
    `matches_df` without `hs`/`as`/`hc`/`ac` legitimately leaves the whole
    `*_last_{3,5,10}_shots*` family missing, and raising there would turn a
    degraded-but-working mode into a 500.
    """
    _stub_loaders(monkeypatch, synthetic, xg=True, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    home_m, away_m, cols = _tiny_models(synthetic)

    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    # Two real feature columns that the contract deliberately does not cover --
    # on a fully-populated league neither can go missing, which is exactly why
    # they are not in it. NaN-ing one is the "new feature that arrives missing"
    # case this guard exists for.
    for outside_column in ("home_last_10_shots_on_target_for", "home_last_5_corners_against"):
        assert outside_column in cols, (
            f"{outside_column} is not a feature on this checkout; pick another "
            "column that is, or this test is not exercising the serving path"
        )
        assert outside_column not in ml_scoreline.MISSING_VALUE_ENCODING
        row[outside_column] = float("nan")

    with pytest.warns(UserWarning) as caught:
        served = ml_scoreline._row_to_matrix(row, list(cols))

    messages = " ".join(str(w.message) for w in caught)
    for outside_column in ("home_last_10_shots_on_target_for", "home_last_5_corners_against"):
        assert outside_column in messages, (
            f"the warning must name every offending column; {outside_column} is "
            f"missing from: {messages}"
        )
    assert "MISSING_VALUE_ENCODING" in messages
    # Still served, not raised on -- the point of the design.
    assert np.isfinite(served.to_numpy()).all()
    assert served.loc[0, "home_last_10_shots_on_target_for"] == 0.0


def test_degraded_matches_df_without_shot_columns_names_the_shot_family(monkeypatch):
    """The real degraded case, so the warning's message is exercised on the
    population it was written for rather than on an invented column name.

    `matches_df` with no `hs`/`as`/`hc`/`ac` is not hypothetical — it is what a
    partially-populated season looks like, and it is exactly the case where a
    blanket `fillna(0)` invents dozens of shot and corners numbers silently.
    """
    matches = _synthetic_matches(with_shots=False)
    _stub_loaders(monkeypatch, matches, xg=True, serving_continuity=True)
    ctx = FixtureFeatureContext(matches)
    _, _, cols = _tiny_models(matches)

    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    leaked = {c for c, v in row.items() if v is None or (isinstance(v, float) and pd.isna(v))}
    # Only columns in `feature_cols` reach the booster (`reindex` drops the
    # rest), so only those can produce the fabricated 0.0 being guarded
    # against. The deliberately-unused `set_piece_xg_share_*` columns are
    # computed but not in `feature_cols`, so they are excluded here rather
    # than counted as a gap.
    outside = (leaked - set(ml_scoreline.MISSING_VALUE_ENCODING)) & set(cols)
    assert outside, (
        "precondition: a matches_df with no shot/corner columns must leave "
        f"features missing, or this test is covering nothing (leaked: {sorted(leaked)})"
    )
    assert any("shots" in c for c in outside), (
        f"expected the shots family in the leaked set; got {sorted(outside)[:10]}"
    )

    with pytest.warns(UserWarning) as caught:
        served = ml_scoreline._row_to_matrix(row, list(cols))
    messages = " ".join(str(w.message) for w in caught)
    for col in sorted(outside)[:5]:
        assert col in messages
    assert np.isfinite(served.to_numpy()).all()


def test_no_warning_when_nothing_outside_the_contract_is_missing(monkeypatch, synthetic):
    """Negative case: the warning must not become noise.

    If it fires on a fully-populated row it is telling operators about a
    problem that does not exist, and it will be filtered out of logs within a
    week — taking the real signal with it.
    """
    _stub_loaders(monkeypatch, synthetic, xg=True, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    _, _, cols = _tiny_models(synthetic)

    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ml_scoreline._row_to_matrix(row, list(cols))


def test_missing_goals_raises_rather_than_serving_fabricated_xg(monkeypatch, synthetic):
    """The one path that could still reach the 16 xG columns fails loudly.

    `build_row`'s xG loop falls back to `_league_goals_per_match` when Understat
    is cold, and that returns `None` when `matches_df` carries no goals at all.
    The audit's answer to "can a failure still degrade silently?" is: not for
    xG — `FixtureFeatureContext.__init__` raises `KeyError` on `goals_home`
    before a row is ever built, so no fabricated 0.0 is ever served.

    Pinned because the alternative is invisible: if that raise were ever
    "fixed" by defaulting the rate to something, these 16 columns would start
    leaking again and this test would be the only thing that noticed.
    """
    _stub_loaders(monkeypatch, synthetic, serving_continuity=True)
    assert build_module._league_goals_per_match(synthetic.drop(columns=["goals_home", "goals_away"])) is None
    with pytest.raises(KeyError):
        FixtureFeatureContext(synthetic.drop(columns=["goals_home", "goals_away"]))


def test_both_entry_points_actually_call_the_shared_helper(monkeypatch, synthetic):
    """Structural, and behavioural where it can be.

    Restoring `predict_many_from_rows`' own private `fillna(0)` is a *silent*
    change: for every input the suite has, it produces the identical matrix, so
    no output-comparing test can catch it. That is exactly how the xG encoding
    came to be fixed on one side of a boundary it should never have straddled,
    so the sharing is pinned by counting calls through the helper instead — the
    one thing that differs between the shared implementation and two copies of
    it.
    """
    _stub_loaders(monkeypatch, synthetic, xg=True, serving_continuity=True)
    ctx = FixtureFeatureContext(synthetic)
    home_m, away_m, cols = _tiny_models(synthetic)
    model = ml_scoreline.MLScorelineModel(
        home_model=home_m, away_model=away_m, feature_cols=list(cols), teams=[], context=ctx
    )
    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))

    real = ml_scoreline._apply_missing_value_encoding
    calls: list = []

    def _counting(frame):
        calls.append(list(frame.columns))
        return real(frame)

    monkeypatch.setattr(ml_scoreline, "_apply_missing_value_encoding", _counting)

    ml_scoreline._row_to_matrix(row, list(cols))
    assert len(calls) == 1, f"_row_to_matrix bypassed the shared helper (calls={len(calls)})"
    assert calls[0] == list(cols)

    model.predict_many_from_rows(pd.DataFrame([{**row}])[list(cols)])
    assert len(calls) == 2, (
        "predict_many_from_rows bypassed the shared helper and carries its own "
        f"missing-value decision again (calls={len(calls)}); that duplication is "
        "how one half of the xG encoding got fixed without the other half"
    )


def test_train_all_still_fills_feature_columns_with_zero(monkeypatch, synthetic):
    """The other half of the contract, on the training side.

    `MISSING_VALUE_ENCODING` is only correct while `models/manifest.py` fills
    training with 0.0. `test_every_contract_value_matches_what_training_fills`
    pins the dict's *side*; this pins the trainer's, because the two can
    drift independently and nothing else in the suite would notice.

    Checked against the source rather than by re-running a retrain, which is
    out of scope here (it needs the model artefacts and a retrain command, and
    #43 already measured the resulting train/serve mismatch at ~0.0003 RPS).
    A transcription of the expression would not do: it would keep passing if
    `train_all` itself changed, which is the only thing worth checking. Every
    fill of a `*_feature_cols` selection must use 0.
    """
    from pl_predictor.models import manifest as manifest_module

    with open(manifest_module.__file__, encoding="utf-8") as handle:
        source = handle.read()
    fills = re.findall(r"\[(?:ml_)?feature_cols\]\.fillna\(([^)]*)\)", source)
    assert fills, (
        "no `fillna` found on a feature_cols selection in manifest.py; the "
        "training encoding moved and this test needs to know how"
    )
    wrong = [f for f in fills if f != "0"]
    assert not wrong, (
        f"manifest.py fills training feature columns with {wrong}, not 0; "
        "MISSING_VALUE_ENCODING and every fitted booster were built against 0.0, "
        "so serving now disagrees with training until that is retrained"
    )


def test_build_row_no_longer_claims_xgboost_handles_missing_natively(monkeypatch, synthetic):
    """The comment that was actively misleading is asserted gone.

    `features/build.py`'s squad-continuity merge claimed XGBoost handles missing
    values natively. It does not — `train_all` fits on `fillna(0)` — and it is
    the sentence that makes a -7.79 SD encoding look like a bug rather than a
    contract. A prose fix can rot silently, so the claim is pinned here: the
    broader rule this project already uses is that a comment is only
    trustworthy if a test fails when it stops being true.
    """
    with open(build_module.__file__, encoding="utf-8") as handle:
        text = handle.read()
    offending = [
        line.strip()
        for line in text.splitlines()
        if "missing values natively" in line and not line.strip().startswith("# CORRECTION")
        and "was load-bearing" not in line and "claim" not in line.lower()
    ]
    assert not offending, (
        "features/build.py still claims XGBoost handles missing values natively; "
        f"manifest.train_all fits on fillna(0), so it never sees one. Lines: {offending}"
    )
