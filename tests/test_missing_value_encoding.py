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

import json
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
    # Capacity sized to actually learn the columns these tests reason about.
    # It was `n_estimators=80, max_depth=3`, and on a 36-row synthetic league
    # that is few enough trees that the booster spent its splits elsewhere: when
    # the training frame's xG columns changed (PR #50's imputation), the tiny
    # model dropped from 6 splits on `squad_continuity` to 1, and a single split
    # at 0.55 cannot separate 0.0 from 0.8442 — so
    # `test_cold_squad_continuity_serves_the_encoding_the_boosters_were_fitted_on`
    # failed on a fixture artifact rather than on a real defect. The production
    # boosters use 250 trees at depth 4 and split that column 21 times, so this
    # brings the stand-in closer to what it stands in for rather than papering
    # over the assertion.
    params = {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.1, "random_state": 0}
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

    Every value being 0.0 is consistent with the xG columns now being imputed
    upstream (`xg_form.resolve_missing_xg`, called by `build_training_frame` on
    the fitting path and `build_row` on the serving path): that dict's 0.0 is
    only reached for an xG cell the imputation could not resolve, which is the
    `matches_df`-with-no-goals case where both ends also have no league rate and
    so still meet at 0.0.
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
    out of scope here (it needs the model artefacts and a retrain command).
    A transcription of the expression would not do: it would keep passing if
    `train_all` itself changed, which is the only thing worth checking. Every
    fill of a `*_feature_cols` selection must use 0.

    Note this is about what `train_all`'s own fill does, not about what the
    xG columns receive — those are resolved upstream by
    `xg_form.resolve_missing_xg`, shared with the serving path, and this fill
    never sees them. See `test_the_training_frame_and_the_serving_row_impute_a_missing_xg_the_same_way`
    for the pairing that is now the interesting one.
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


# --------------------------------------------------------------------------
# One implementation of "what a missing xG is", on both sides
# --------------------------------------------------------------------------


def test_the_training_frame_and_the_serving_row_impute_a_missing_xg_the_same_way(
    monkeypatch, synthetic
):
    """The skew #43 documented and left open, closed and pinned.

    PR #43 made `build_row` substitute the league-average expected-goals rate
    for a missing xG, and deliberately recorded that training still encoded the
    same gap as `0.0` via `manifest.train_all`'s `train_df[feature_cols]
    .fillna(0)` -- so the fitted boosters and the serving path disagreed about
    what a missing xG meant. Measured at ~0.0003 RPS, which is precisely why it
    would have survived: too small to notice, still a real disagreement.

    Both ends now call `xg_form.resolve_missing_xg`, and this asserts the
    *values* agree rather than that the shared function exists. The point of
    the fix is the single implementation, but a shared name is not the claim --
    two callers of one function that disagrees with itself would pass any
    call-counting check. Here the training frame is built cold and the serving
    row is built cold on the same synthetic league, and every xG cell the
    training frame had to resolve must equal what serving substitutes.
    """
    _stub_loaders(monkeypatch, synthetic, xg=False)  # cold Understat both ends
    ctx = FixtureFeatureContext(synthetic)
    train_df, cols = build_training_frame(matches_df=synthetic)

    league_rate = float(
        pd.concat([synthetic["goals_home"], synthetic["goals_away"]], ignore_index=True).mean()
    )

    # Which cells the imputation is responsible for. Determined by rebuilding
    # the frame with the shared function neutered, because after the fix the
    # real frame has no NaN left to count -- that is the whole point of it. If
    # this returns nothing, the fixture no longer produces a missing xG and
    # every assertion below would pass vacuously.
    real_resolver = xg_form.resolve_missing_xg
    monkeypatch.setattr(
        xg_form,
        "resolve_missing_xg",
        lambda reading, league_rate: (
            float("nan") if reading is None or pd.isna(reading) else float(reading)
        ),
    )
    try:
        unresolved_df, _ = build_training_frame(matches_df=synthetic)
    finally:
        monkeypatch.setattr(xg_form, "resolve_missing_xg", real_resolver)

    unresolved = {
        c: int(unresolved_df[c].isna().sum()) for c in XG_COLS if c in unresolved_df.columns
    }
    assert unresolved, "precondition: no xG columns in the training frame at all"
    assert sum(unresolved.values()) > 0, (
        "precondition: the fixture produces no missing xG for the training frame "
        "to resolve, so the assertions below would pass without comparing "
        "anything"
    )

    # Serving, cold: every xG column is the league rate.
    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01"))
    for col in XG_COLS:
        assert row[col] == pytest.approx(league_rate, abs=1e-9), (
            f"serving gave {row[col]!r} for a missing {col}, expected the league "
            f"rate {league_rate!r}"
        )

    # Training, cold: every gap resolved to a rate. Not the same number serving
    # used, and deliberately so — serving prices one fixture whose whole context
    # is in the past, so its whole-window rate is already as-of; a training row
    # from 2020 must not be handed a rate built from 2025 data. So the pairing
    # asserted here is on the *function* and on the shape, and the per-date
    # values are checked against `rates` below.
    rates = build_module._xg_league_avg_rates_by_date(
        pd.DataFrame(), synthetic, synthetic["date"]
    )
    assert rates.notna().to_numpy().any(), (
        "precondition: the cold-Understat as-of rate series is all NaN, so the "
        "assertions below would compare nothing"
    )
    first_matchday = synthetic["date"] == synthetic["date"].min()
    for col in XG_COLS:
        assert col in train_df.columns, f"precondition: {col} missing from the training frame"
        assert train_df[col].notna().any(), (
            f"{col} resolved no cells at all; the imputation is not running"
        )
        # And no cell was invented for the very first matchday, where no prior
        # match exists and therefore no as-of rate either.
        assert train_df.loc[first_matchday, col].isna().all(), (
            f"{col} was imputed on the first matchday, where no prior match "
            "exists: filling it would mean inventing a rate from nothing"
        )

    # The per-date values themselves, on every column: the gaps take their own
    # row's as-of rate, and the rows that already had a reading are untouched.
    for col in XG_COLS:
        if col not in train_df.columns:
            continue
        side, stat_key = col.split("_", 1)
        assert stat_key in build_module.XG_STAT_COLS, (
            f"{col} does not decompose into an XG_STAT_COLS key; the test's key "
            "derivation would silently miss the column"
        )
        was_present = unresolved_df[col].notna()
        assert (~was_present).any(), (
            f"{col} has no gaps to impute; this test has stopped covering the case"
        )
        expected = rates[stat_key].to_numpy(dtype=float)
        got = train_df[col].to_numpy(dtype=float)
        no_rate = np.isnan(expected)
        # Present readings survive byte-identical: an imputation that rewrote every
        # cell would satisfy every other assertion here, because both ends would
        # agree on a wrong number.
        np.testing.assert_array_equal(
            got[was_present.to_numpy()], unresolved_df[col].to_numpy(dtype=float)[was_present.to_numpy()]
        )
        # Gaps with a rate take it.
        gap = ~was_present.to_numpy() & ~no_rate
        assert got[gap] == pytest.approx(expected[gap], abs=1e-9), (
            f"{col}'s gaps took something other than their own date's league rate"
        )
        # Gaps with no rate stay missing, rather than becoming 0.0 here — the
        # contract fill downstream does that, and only then.
        gap_no_rate = ~was_present.to_numpy() & no_rate
        assert np.isnan(got[gap_no_rate]).all(), (
            f"{col} was filled on rows with no as-of rate available"
        )

    # And the deltas, which is where the two encodings differ most visibly:
    # against a league rate they are goals-minus-expectation (per-team, varying),
    # against 0.0 they were goals-scored, asserting the team hit expectation
    # exactly. Checked on the rows that were imputed, which is where the two
    # differ; on rows with real xG both encodings agree trivially.
    for side in ("home", "away"):
        for stat in ("for", "against"):
            for w in xg_form.WINDOWS:
                delta = f"{side}_xg_delta_{stat}_last_{w}"
                xg_col = f"{side}_xg_{stat}_last_{w}"
                goals = f"{side}_last_{w}_goals_{stat}"
                if delta not in train_df.columns:
                    continue
                gap = (~unresolved_df[xg_col].notna()).to_numpy() & ~np.isnan(
                    rates[f"xg_{stat}_last_{w}"].to_numpy(dtype=float)
                )
                assert gap.any(), (
                    f"precondition: no row in {delta} was both missing xG and "
                    "imputable, so the assertion below is vacuous"
                )
                assert not train_df.loc[gap, delta].isna().any(), (
                    f"{delta} is NaN on imputed rows: goals are known even with "
                    "no xG, so goals-minus-expectation is computable"
                )
                expected = (
                    train_df.loc[gap, goals].to_numpy(dtype=float)
                    - rates.loc[gap, f"xg_{stat}_last_{w}"].to_numpy(dtype=float)
                )
                assert train_df.loc[gap, delta].to_numpy(dtype=float) == pytest.approx(
                    expected, abs=1e-9
                ), (
                    f"{delta} is not goals-minus-league-rate on imputed rows; a "
                    "cold xG encoded as 0.0 makes this column plain goals-scored, "
                    "which asserts the team hit its expectation exactly"
                )


def test_the_training_frame_really_calls_the_shared_xg_imputation(monkeypatch, synthetic):
    """Structural: the training path must go through `resolve_missing_xg`.

    Restoring `build_training_frame`'s bare NaN pass-through is a *silent*
    change for every input this fixture has -- it only differs on rows whose xG
    is missing, and no output-comparing assertion in the suite would notice a
    return to 0.0 there. That is exactly how the xG encoding came to be fixed on
    one side of a boundary it should never have been able to straddle, so the
    sharing is pinned by counting calls through the one function, matching
    `test_both_entry_points_actually_call_the_shared_helper` on the serving side.
    """
    _stub_loaders(monkeypatch, synthetic, xg=False)
    real = xg_form.resolve_missing_xg
    calls: list = []

    def _counting(reading, league_rate):
        calls.append(reading)
        return real(reading, league_rate)

    monkeypatch.setattr(xg_form, "resolve_missing_xg", _counting)
    build_training_frame(matches_df=synthetic)

    assert calls, (
        "build_training_frame no longer calls xg_form.resolve_missing_xg, so the "
        "fitting path carries its own idea of what a missing xG is again -- that "
        "duplication is how one half of the xG encoding got fixed without the "
        "other half"
    )
    assert any(reading is None or pd.isna(reading) for reading in calls), (
        "resolve_missing_xg was called only with present readings, so it was "
        "never asked the question it exists to answer; the fixture no longer "
        "produces a missing xG and this test has stopped covering the case"
    )


def test_a_partly_warm_training_frame_imputes_only_the_missing_cells(monkeypatch, synthetic):
    """The real shape: some rows have xG, some do not, in the same frame.

    Fully-cold coverage is what the tests above use, because it is easy to
    arrange, but it cannot catch an imputation that resolves *every* cell --
    with nothing present, replacing everything with the rate is
    indistinguishable from replacing only the gaps. The production shape is the
    opposite: Understat covers the recent seasons and the gaps are the promoted
    teams' debut matches at the start of each one, so a frame carries both.

    So Understat is stubbed to cover the last synthetic season only. The
    assertion is that every cell Understat answered is byte-identical to the
    reading it produced, and only the gaps take the league rate.
    """
    known = sorted(synthetic["season"].unique())
    covered = known[-1]

    def _partly_warm(seasons):
        # Shape it like `data/understat.py::fetch_season`'s output, which is
        # what both `attach_xg_features` (`_team_perspective` reads the team
        # names) and `latest_xg_form` consume.
        part = synthetic[synthetic["season"] == covered]
        return pd.DataFrame(
            {
                "date": part["date"].to_numpy(),
                "team_home": part["team_home"].to_numpy(),
                "team_away": part["team_away"].to_numpy(),
                "xg_home": 1.1 + 0.05 * (part["goals_home"] % 4).to_numpy(),
                "xg_away": 1.3 + 0.05 * (part["goals_away"] % 4).to_numpy(),
                "goals_home": part["goals_home"].to_numpy(),
                "goals_away": part["goals_away"].to_numpy(),
            }
        )

    _stub_loaders(monkeypatch, synthetic, xg=False, serving_continuity=True)
    u = _partly_warm(None)
    monkeypatch.setattr(understat_module, "load_xg_data", lambda **_: u)

    fixed_df, cols = build_training_frame(matches_df=synthetic)

    # The reference: what the pre-fix frame carried, from the same stub, so the
    # two differ only by the imputation. Built with the shared function
    # neutered, which is also the only way to see which cells were missing --
    # after the fix there are none.
    real = xg_form.resolve_missing_xg
    monkeypatch.setattr(
        xg_form,
        "resolve_missing_xg",
        lambda reading, league_rate: (
            float("nan") if reading is None or pd.isna(reading) else float(reading)
        ),
    )
    try:
        raw_df, _ = build_training_frame(matches_df=synthetic)
    finally:
        monkeypatch.setattr(xg_form, "resolve_missing_xg", real)

    # Precondition: the frame really does carry both populated and missing xG,
    # or the comparison below distinguishes nothing.
    for col in XG_COLS:
        assert col in raw_df.columns
    populated = int(raw_df["home_xg_for_last_5"].notna().sum())
    assert 0 < populated < len(raw_df), (
        f"expected a partly-populated column, got {populated}/{len(raw_df)} "
        "populated; the fixture no longer reproduces the real shape"
    )
    # Post-fix the only cells still missing are those with no as-of rate at all
    # — rows predating the stub's Understat coverage. That is the honest answer
    # (`resolve_missing_xg` declines to invent a number) and it is why the xG
    # entries in MISSING_VALUE_ENCODING stay 0.0 rather than becoming the rate.
    still_missing = raw_df["home_xg_for_last_5"].isna() & fixed_df["home_xg_for_last_5"].isna()
    assert int(fixed_df["home_xg_for_last_5"].notna().sum()) > int(
        raw_df["home_xg_for_last_5"].notna().sum()
    ), "the imputation resolved no cells at all"
    assert (
        synthetic["date"].to_numpy()[still_missing.to_numpy()] < synthetic["date"].min() + pd.Timedelta(days=1)
    ).all() or synthetic["date"].to_numpy()[still_missing.to_numpy()].min() < u["date"].min(), (
        "a row still missing after imputation is dated at or after the stub's "
        "first Understat match, so it should have had a rate to take"
    )

    # The rate each row's own date implies — the point-in-time form, since that is
    # what `build_training_frame` uses. Derived from the same stub rather than
    # hardcoded, so it stays correct if the fixture changes.
    rates = build_module._xg_league_avg_rates_by_date(u, synthetic, synthetic["date"])
    assert rates.notna().to_numpy().any(), (
        "precondition: the as-of rate series is entirely NaN, so the assertions "
        "below would compare nothing"
    )

    for col in XG_COLS:
        was_present = raw_df[col].notna()
        assert was_present.any() and (~was_present).any(), (
            f"{col} is entirely one or the other on this fixture; the test needs "
            "both populated and missing cells in the same column"
        )
        # Every reading Understat actually produced survives untouched.
        np.testing.assert_array_equal(
            fixed_df.loc[was_present, col].to_numpy(),
            raw_df.loc[was_present, col].to_numpy(),
        )
        # Only the gaps take the league rate. The rate series is keyed
        # `{stat}_last_{w}` (see `build.XG_STAT_COLS`), and the column is
        # `{side}_{that key}`, so the split is on the first underscore only.
        side, stat_key = col.split("_", 1)
        assert side in ("home", "away") and stat_key in build_module.XG_STAT_COLS, (
            f"{col} does not decompose into a side plus an XG_STAT_COLS key; the "
            "test's key derivation would silently miss the column"
        )
        # Row-wise: each row's own date's rate, not one window-wide number.
        expected = rates.loc[~was_present.to_numpy(), stat_key].to_numpy(dtype=float)
        got = fixed_df.loc[~was_present, col].to_numpy(dtype=float)
        # Row-wise: each row's own date's rate. Rows with no as-of rate (predating
        # the stub's Understat coverage) must stay missing — that is what
        # `resolve_missing_xg` does when there is nothing to substitute, and it
        # is why those cells remain in MISSING_VALUE_ENCODING rather than
        # becoming the rate. Compared element-wise so a mixed column (some rows
        # imputed, some not) is checked properly instead of by an `any`/`all`
        # that would pass on the majority case.
        expected = rates.loc[~was_present.to_numpy(), stat_key].to_numpy(dtype=float)
        got = fixed_df.loc[~was_present, col].to_numpy(dtype=float)
        assert got.shape == expected.shape
        no_rate = np.isnan(expected)
        assert np.array_equal(np.isnan(got), no_rate), (
            f"{col}: the set of rows still missing after imputation does not match "
            f"the set with no as-of rate. got missing at {np.flatnonzero(np.isnan(got))[:5]}, "
            f"expected missing at {np.flatnonzero(no_rate)[:5]}"
        )
        resolved = ~no_rate
        if resolved.any():
            assert got[resolved] == pytest.approx(expected[resolved], abs=1e-9), (
                f"{col}'s gaps took something other than their own date's league rate"
            )


def test_the_goals_fallback_counts_fixtures_not_match_dates():
    """The denominator is fixtures, not dates. CodeRabbit's Major on #51.

    A real Premier League matchday is 10 fixtures, not 1, and the bug was that
    `matches_so_far` counted *dates* while the numerator carried the team-goals
    of every fixture on that date — so the rate came out inflated by roughly the
    fixtures-per-date. Measured on this project's real 8-season window: the
    rate ended at 4.61 where the true league rate is 0.78, a 6x error.

    Written as an explicit arithmetic identity on a synthetic league where
    several fixtures share one date, because that is the case the bug hides in:
    with exactly one fixture per date, `len(dates)` and the fixture count
    coincide and the wrong denominator is accidentally right. The `_synthetic_matches`
    fixture used elsewhere in this module spaces its matches 7 days apart, so it
    could never have caught this.

    The identity asserted is the definition of the quantity:
    cumulative team-goals / (2 x cumulative fixtures), strictly before the date.
    """
    dates = pd.to_datetime(
        ["2024-08-10"] * 4 + ["2024-08-17"] * 4 + ["2024-08-24"] * 4 + ["2024-08-31"] * 4
    )
    matches = pd.DataFrame(
        {
            "date": dates,
            "season": "2024-2025",
            # Four fixtures per date, deliberately asymmetric, and the first
            # matchday is 3+0, 1+2, 0+1, 0+0 -- seven team-goals -- so the
            # correct rate is 7/8 while a date-count denominator gives 7/2.
            "goals_home": [3, 1, 0, 0, 2, 0, 1, 0, 0, 0, 2, 0, 1, 0, 2, 0],
            "goals_away": [0, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 0, 1],
        }
    )
    rates = build_module._xg_league_avg_rates_by_date(
        pd.DataFrame(), matches, matches["date"]
    )["xg_for_last_5"].to_numpy(dtype=float)

    # Precondition: this league must actually put several fixtures on one date,
    # or the test is asserting the buggy denominator's own arithmetic.
    fixtures_per_date = matches.groupby("date").size()
    assert fixtures_per_date.max() > 1, (
        "precondition: the fixture league must have shared matchdates, else the "
        "date-count and fixture-count denominators coincide and this test is vacuous"
    )

    # The rate is per input ROW, so every fixture on a date carries that date's
    # rate. Index by date rather than by position, which is what made the first
    # draft of this test read NaN at index 1 (still the first matchday).
    all_dates = sorted(pd.to_datetime(matches["date"]).unique())
    first_row_of = [int(np.flatnonzero(pd.to_datetime(matches["date"]) == d)[0]) for d in all_dates]
    got = rates[first_row_of]

    # The first date has no prior match at all: no rate, and no ZeroDivisionError.
    assert np.isnan(got[0]), (
        f"the first matchday got rate {got[0]!r}; with zero prior fixtures the "
        "honest answer is no rate, and 0/0 must not raise or produce 0.0"
    )

    # Explicit identity, evaluated by hand for each date from the fixtures
    # strictly before it.
    known = matches.assign(date=pd.to_datetime(matches["date"]))
    expected = []
    for date in all_dates:
        prior = known[known["date"] < date]
        team_goals = float(prior["goals_home"].sum() + prior["goals_away"].sum())
        n_fixtures = len(prior)
        expected.append(team_goals / (2 * n_fixtures) if n_fixtures else np.nan)

    # The wrong denominator, computed the same way, for the contrast assertion
    # below: date-count instead of fixture-count.
    wrong = []
    for date in all_dates:
        n_prior_dates = sum(1 for d in all_dates if d < date)
        prior = known[known["date"] < date]
        team_goals = float(prior["goals_home"].sum() + prior["goals_away"].sum())
        wrong.append(team_goals / (2 * n_prior_dates) if n_prior_dates else np.nan)

    assert got[1:] == pytest.approx(np.array(expected[1:]), abs=1e-12), (
        "the imputed rate is not cumulative team-goals / (2 x cumulative FIXTURES): "
        f"got {got[1:]}, expected {np.array(expected[1:])}"
    )
    # And it must actually differ from the date-count version, or this test
    # cannot tell the two denominators apart.
    assert not np.allclose(got[1:], np.array(wrong[1:]), atol=1e-12), (
        "the fixture-count and date-count denominators produced identical rates "
        "on this payload, so the assertion above cannot distinguish them"
    )

    # Magnitude on a real multi-matchday date: after the first matchday of 4
    # fixtures scoring 3+0, 1+2, 0+1, 0+0 (7 team-goals over 4 fixtures), the
    # rate must be 7/8, NOT 7/2.
    assert got[1] == pytest.approx(7 / 8, abs=1e-12), (
        f"rate after one 4-fixture matchday is {got[1]}, expected 7 team-goals / "
        f"(2 x 4 fixtures) = {7/8}. A rate of 7/2 would mean the denominator "
        "counted the one matchdate as a single match"
    )
    # And the difference from the date-count denominator must be visible, not a
    # floating-point hair: 7/8 = 0.875 versus 7/2 = 3.5.
    assert abs(got[1] - 7 / 2) > 1.0, (
        "the date-count denominator produced a rate within 1.0 of the correct "
        f"one ({got[1]}); these denominators must be plainly distinguishable"
    )


def test_the_goals_fallback_drops_dates_whose_fixtures_all_lack_goals():
    """A matchday whose fixtures are all unplayed contributes to neither half.

    If such a date reached the numerator it would add nothing, but if it
    reached the *denominator* the rate would be diluted by matches that never
    happened — and if it reached only one of the two, the two would describe
    different populations. All-NaN dates are dropped from both, which is what
    this asserts.
    """
    dates = pd.to_datetime(["2024-08-10"] * 2 + ["2024-08-17"] * 2 + ["2024-08-24"] * 2)
    matches = pd.DataFrame(
        {
            "date": dates,
            "season": "2024-2025",
            # The middle matchday is entirely unplayed.
            "goals_home": [1, 2, np.nan, np.nan, 0, 1],
            "goals_away": [0, 1, np.nan, np.nan, 2, 0],
        }
    )
    rates = build_module._xg_league_avg_rates_by_date(
        pd.DataFrame(), matches, matches["date"]
    )["xg_for_last_5"].to_numpy(dtype=float)

    all_dates = sorted(pd.to_datetime(matches["date"]).unique())
    first_row_of = [int(np.flatnonzero(pd.to_datetime(matches["date"]) == d)[0]) for d in all_dates]
    by_date = rates[first_row_of]

    assert np.isnan(by_date[0]), "precondition: the first date must have no rate"
    # 2024-08-17: the only prior fixtures are 2024-08-10's two, carrying 4
    # team-goals -- 4 / (2 x 2) = 1.0. Note the all-NaN matchday sits ON this
    # date, so it is excluded by the "strictly before" rule regardless.
    assert by_date[1] == pytest.approx(1.0, abs=1e-12)
    # 2024-08-24: the prior fixtures are still only 2024-08-10's two (the
    # all-NaN matchday contributed no fixture) and 2024-08-24's own two are
    # excluded as the future. Still 4 / (2 x 2) = 1.0.
    #
    # This is the assertion that matters: had the unplayed matchday reached the
    # denominator it would read 4 / (2 x 3) = 0.667, and had it been dropped
    # from the index before the shift, 2024-08-24 would have taken its shifted
    # rate from the wrong row and come out with no rate at all.
    assert by_date[2] == pytest.approx(1.0, abs=1e-12), (
        f"rate on 2024-08-24 is {by_date[2]}; expected 4 team-goals over the 2 "
        "played fixtures that precede it. 0.667 would mean the unplayed "
        "matchday was counted in the denominator; NaN would mean it was "
        "dropped from the index and shifted onto the wrong row"
    )
    assert abs(by_date[2] - 4 / 6) > 0.1, (
        "the unplayed matchday leaked into the denominator; the two candidate "
        "rates are 1.0 and 0.667 and must be plainly distinguishable"
    )


def test_the_goals_fallback_rate_does_not_read_the_fixture_s_own_goals(monkeypatch, synthetic):
    """The cold-Understat branch, checked for the same look-ahead.

    `_xg_league_avg_rates_by_date`'s Understat branch is a backward
    `merge_asof`, so its no-lookahead property comes from the argument
    `allow_exact_matches=False`. The fallback branch derives an expanding mean
    by hand instead, which means the guarantee is only as good as that
    arithmetic — and CodeRabbit's Major on #50 was pointed at exactly the kind
    of place where it is quietly one `shift()` short. A match's own goals
    entering its own row's expected-goals rate is the sharpest possible form of
    that leak: the row's target is a function of those goals.

    Perturb one match's goals and require that no earlier row, and not that row
    itself, moves. Later rows moving is correct and expected — the expanding
    mean is *supposed* to absorb matches once they are in the past.
    """
    _stub_loaders(monkeypatch, synthetic, xg=False, serving_continuity=True)
    monkeypatch.setattr(understat_module, "load_xg_data", lambda **_: pd.DataFrame())

    dates = synthetic["date"]
    baseline = build_module._xg_league_avg_rates_by_date(pd.DataFrame(), synthetic, dates)

    target = len(synthetic) // 2
    target_date = dates.iloc[target]
    assert (dates < target_date).any() and (dates > target_date).any(), (
        "precondition: need rows on both sides of the perturbed match"
    )

    for column in ("goals_home", "goals_away"):
        perturbed = synthetic.copy()
        perturbed.iloc[target, perturbed.columns.get_loc(column)] = (
            perturbed.iloc[target, perturbed.columns.get_loc(column)] + 7
        )
        after = build_module._xg_league_avg_rates_by_date(pd.DataFrame(), perturbed, dates)

        moved = ~np.all(
            np.isclose(baseline.to_numpy(dtype=float), after.to_numpy(dtype=float), equal_nan=True),
            axis=1,
        )
        before = (dates < target_date).to_numpy()
        at = (dates == target_date).to_numpy()
        assert not moved[before].any(), (
            f"perturbing {column} at {target_date.date()} moved "
            f"{int(moved[before].sum())} EARLIER rows' rates, so the fallback "
            "reads the future"
        )
        assert not moved[at].any(), (
            f"perturbing {column} moved the perturbed match's OWN imputed rate; "
            "a row's expected-goals rate must not contain that row's own goals"
        )


def test_the_training_frame_imputes_each_row_as_of_that_row_s_own_date(monkeypatch, synthetic):
    """No-lookahead on the *rate*, which is the subtler half of the fix.

    The first version of this change derived one league rate from the whole
    window and wrote it into every historical row — correct on the serving side,
    where a context holds nothing later than the fixture it prices, but
    look-ahead on the fitting side, where it puts 2025-26 xG into a 2018
    fixture. CodeRabbit caught it on #50.

    Perturbing *future* Understat xG must leave earlier rows' rates untouched.
    That is the direct statement of the guarantee, and it is the only form of it
    that a future refactor cannot quietly break: any change that starts reading
    the whole frame instead of the strictly-earlier one moves these numbers.
    """
    covered = sorted(synthetic["season"].unique())[-1]
    part = synthetic[synthetic["season"] == covered]

    def _warm(seasons):
        return pd.DataFrame(
            {
                "date": part["date"].to_numpy(),
                "team_home": part["team_home"].to_numpy(),
                "team_away": part["team_away"].to_numpy(),
                "xg_home": 1.1 + 0.05 * (part["goals_home"] % 4).to_numpy(),
                "xg_away": 1.3 + 0.05 * (part["goals_away"] % 4).to_numpy(),
                "goals_home": part["goals_home"].to_numpy(),
                "goals_away": part["goals_away"].to_numpy(),
            }
        )

    def _rates_with_future_scrambled():
        """Build the frame with Understat's *later* xG replaced by nonsense."""
        scrambled = _warm(None)
        late = scrambled["date"] > pd.Timestamp("2021-06-01")
        assert late.any(), (
            "precondition: the stub must have rows after the cut, or scrambling "
            "the future changes nothing and this test passes vacuously"
        )
        scrambled = scrambled.copy()
        scrambled.loc[late, "xg_home"] = 99.0
        scrambled.loc[late, "xg_away"] = 99.0
        monkeypatch.setattr(understat_module, "load_xg_data", lambda **_: scrambled)
        return build_training_frame(matches_df=synthetic)[0]

    _stub_loaders(monkeypatch, synthetic, xg=False, serving_continuity=True)
    monkeypatch.setattr(understat_module, "load_xg_data", _warm)
    baseline_df, _ = build_training_frame(matches_df=synthetic)

    scrambled_df = _rates_with_future_scrambled()

    cut = pd.Timestamp("2021-06-01")
    early = synthetic["date"] < cut
    assert early.any() and (~early).any(), (
        "precondition: need rows on both sides of the cut for this to mean anything"
    )

    for col in XG_COLS:
        np.testing.assert_allclose(
            scrambled_df.loc[early, col].to_numpy(dtype=float),
            baseline_df.loc[early, col].to_numpy(dtype=float),
            equal_nan=True,
            err_msg=(
                f"{col} changed for rows dated before {cut.date()} when only LATER "
                "Understat xG was scrambled, so the imputed rate reads the future"
            ),
        )


def test_a_warm_xg_is_not_touched_by_the_shared_imputation():
    """The imputation must be a no-op on a reading that is already there.

    Every other property of this change is about the missing case; this one
    guards the overwhelmingly common one. A regression that resolved *every*
    xG cell to the league rate instead of only the missing ones would satisfy
    every equivalence test above -- both ends would still agree, because both
    would be wrong -- while quietly discarding all real signal.
    """
    assert xg_form.resolve_missing_xg(1.83, 1.5) == 1.83
    assert xg_form.resolve_missing_xg(0.0, 1.5) == 0.0, "a real zero must survive"
    # And a present-but-falsy reading is not mistaken for a missing one.
    assert xg_form.resolve_missing_xg(0.0, None) == 0.0


def test_an_unavailable_league_rate_leaves_a_missing_xg_missing():
    """No league rate means nothing to substitute, and inventing one is the bug.

    This is the `matches_df`-with-no-goals case. The xG cell stays NaN, and
    `ml_scoreline.MISSING_VALUE_ENCODING` is then the only thing deciding what
    the booster sees -- which is why its 0.0 entries are still correct, and why
    the training and serving halves of the contract still meet at the same
    value on this path.
    """
    assert pd.isna(xg_form.resolve_missing_xg(None, None))
    assert pd.isna(xg_form.resolve_missing_xg(float("nan"), float("nan")))
    # A present reading is unaffected by the absence of any fallback.
    assert xg_form.resolve_missing_xg(2.0, None) == 2.0


def test_the_committed_artefacts_were_fitted_by_this_pipeline():
    """If the refit is skipped, the committed boosters are the OLD ones — and
    nothing else in the suite would say so.

    This is the failure mode with no guard at all. `train_all` writes six
    artefacts plus the manifest, in that order and non-atomically, so a run that
    raises midway leaves boosters from one code version beside a manifest from
    another; a refit that is skipped entirely leaves the previous pair in place
    with no marker at all. Either way the model keeps serving, the holdout RPS
    in `manifest.json` still looks current, and nothing compares the shipped
    boosters against the code that is meant to have produced them. `trained_at`
    cannot catch it — it records when the run happened, not what it ran.

    So `train_all` records `feature_pipeline_version` and this asserts the
    committed `models/manifest.json` agrees with the constant in the code on
    this checkout. Bump the constant in the same commit as any change to the
    fitted feature values; this then fails until the artefacts are refitted.

    Reads the committed file rather than calling `train_all`: the question is
    about what is in the repository, which a refit in-process could not answer.
    """
    from pl_predictor.models import manifest as manifest_module

    if not manifest_module.MANIFEST_PATH.exists():
        pytest.skip("no committed models/manifest.json to check")

    committed = json.loads(manifest_module.MANIFEST_PATH.read_text())
    assert committed.get("feature_pipeline_version") == manifest_module.FEATURE_PIPELINE_VERSION, (
        f"models/manifest.json was written by a feature pipeline at "
        f"{committed.get('feature_pipeline_version')!r}, but this checkout's code "
        f"is at {manifest_module.FEATURE_PIPELINE_VERSION!r}. The committed "
        "boosters were fitted on values this serving path no longer produces. "
        "Re-run manifest.train_all() and commit models/, or revert the change."
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
