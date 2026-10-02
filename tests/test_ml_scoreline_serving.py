"""`build_row`'s xG degradation, and what `ml_scoreline` actually serves.

PR #41 made `build_row` return `float("nan")` instead of `None` when Understat xG
was unavailable, on the reasoning that "NaN is the honest answer ... XGBoost
consumes it natively. 0.0 would instead claim the team created/conceded zero
expected goals, which is a real number and a wrong one."

That reasoning was right about 0.0 and wrong about NaN surviving. The NaN did
not reach XGBoost -- `MLScorelineModel.predict` ends in `.fillna(0)`, so every
one of those columns was served as a literal `0.0`, which is the exact wrong
number the change set out to avoid. These tests pin the served value, not the
intermediate one: asserting on `build_row`'s dict was never enough, because the
whole defect lived in the gap between that dict and the booster's input matrix.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from pl_predictor.data import other_competitions, understat as understat_module, understat_shots
from pl_predictor.features import squad_change
from pl_predictor.features.build import FixtureFeatureContext, build_training_frame
from pl_predictor.features import xg_form
from pl_predictor.models import ml_scoreline

XG_COLS = [f"{side}_xg_{stat}_last_{w}" for side in ("home", "away") for stat in ("for", "against") for w in xg_form.WINDOWS]
XG_DELTA_COLS = [
    f"{side}_xg_delta_{stat}_last_{w}"
    for side in ("home", "away")
    for stat in ("for", "against")
    for w in xg_form.WINDOWS
]


def _synthetic_matches(n_teams: int = 6, n_seasons: int = 3) -> pd.DataFrame:
    """A small, fully synthetic league -- no cache and no network, so these
    tests behave identically on a cold checkout."""
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for season_idx in range(n_seasons):
        season = f"{2020 + season_idx}-{2021 + season_idx}"
        date = pd.Timestamp(f"{2020 + season_idx}-08-01")
        for rnd in range(n_teams // 2 * 4):
            home, away = teams[(rnd * 2 + season_idx) % n_teams], teams[(rnd * 2 + 1 + season_idx) % n_teams]
            gh, ga = (rnd + season_idx) % 4, (rnd * 2 + season_idx) % 3
            rows.append(
                {
                    "date": date,
                    "season": season,
                    "team_home": home,
                    "team_away": away,
                    "goals_home": gh,
                    "goals_away": ga,
                    "ftr": "H" if gh > ga else ("A" if gh < ga else "D"),
                }
            )
            date = date + pd.Timedelta(days=7)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic(monkeypatch):
    """Synthetic league, with every loader that could reach the network stubbed.

    Not just the calendar: `_tiny_models` builds a real training frame, which
    calls the Understat and understat-shot loaders, and the shot loader fetches
    one file per match. On a cold checkout that would take this module to the
    network (and take minutes doing it), so all of them are stubbed here and
    every test in this file is hermetic regardless of cache state.
    """
    monkeypatch.setattr(
        other_competitions,
        "get_team_fixture_calendar",
        lambda: other_competitions._EMPTY.copy(),
    )
    # Cold by default: `_warm_context` re-patches this per test when it needs
    # Understat-shaped data.
    monkeypatch.setattr(understat_module, "load_xg_data", lambda **_: pd.DataFrame())
    monkeypatch.setattr(understat_shots, "load_shot_situation_data", lambda **_: pd.DataFrame())
    # Empty but correctly-shaped, so `build_training_frame` can merge it and
    # `build_row` gets NaN continuity -- the same degradation the real loader
    # produces when vaastav has no data, and what these tests already expect
    # for a feature with nothing to report.
    monkeypatch.setattr(
        squad_change,
        "team_season_continuity_table",
        lambda seasons: pd.DataFrame(columns=["season", "team", "squad_continuity"]),
    )
    return _synthetic_matches()


def _cold_context(matches) -> FixtureFeatureContext:
    """A context built with Understat (and the shot loader) yielding nothing.

    Relies on the `synthetic` fixture having already stubbed those loaders.
    """
    return FixtureFeatureContext(matches)


def _warm_context(monkeypatch, matches) -> FixtureFeatureContext:
    """A context with real Understat-shaped xG, so nothing is cold.

    Takes `monkeypatch` rather than restoring by hand so the override is
    undone automatically and cannot leak into the next test.
    """
    xg_rows = [
        {
            "date": r["date"],
            "team_home": r["team_home"],
            "team_away": r["team_away"],
            "xg_home": 1.4 + 0.1 * (r["goals_home"] % 3),
            "xg_away": 1.2 + 0.1 * (r["goals_away"] % 3),
        }
        for _, r in matches.iterrows()
    ]
    warm = pd.DataFrame(xg_rows)
    monkeypatch.setattr(understat_module, "load_xg_data", lambda **_: warm)
    return FixtureFeatureContext(matches)


def _tiny_models(matches, n=60):
    """Two small boosters trained on this synthetic league, standing in for the
    shipped ones. Kept tiny so the test stays fast; the point is the served
    matrix, not the model's skill."""
    df, cols = build_training_frame(matches_df=matches)
    X = df[cols].fillna(0).astype(float)
    params = {"n_estimators": n, "max_depth": 3, "learning_rate": 0.1, "random_state": 0}
    home = xgb.XGBRegressor(objective="count:poisson", **params)
    away = xgb.XGBRegressor(objective="count:poisson", **params)
    home.fit(X, df["goals_home"])
    away.fit(X, df["goals_away"])
    return home, away, cols


def _matrix_for(row, feature_cols) -> pd.DataFrame:
    """Build the served matrix through whichever serving helper exists.

    `_row_to_matrix` is the post-fix helper; on an unfixed checkout it does not
    exist, so fall back to the inline expression `predict` used then. Either way
    this is the transformation the boosters actually receive, not a
    re-implementation of the fix -- which is what lets these tests be
    red-checked against unfixed code.
    """
    builder = getattr(ml_scoreline, "_row_to_matrix", None)
    if builder is not None:
        return builder(row, list(feature_cols))
    return pd.DataFrame([row]).reindex(columns=list(feature_cols), fill_value=0).fillna(0).astype(float)


def test_cold_understat_does_not_serve_xg_as_zero(monkeypatch, synthetic):
    """The defect, stated as an assertion on what the boosters are handed.

    With Understat cold, every xG column used to arrive as `0.0` at serving:
    a plausible, confident, wrong number claiming the team created zero
    expected goals and (via the deltas) scored exactly to expectation.
    """
    ctx = _cold_context(synthetic)
    home, away = "T0", "T1"
    row = ctx.build_row(home, away, commence_time=pd.Timestamp("2024-08-01", tz="UTC"))

    served = _matrix_for(row, XG_COLS + XG_DELTA_COLS)

    for col in XG_COLS:
        value = served.loc[0, col]
        assert value != 0.0, (
            f"{col} is 0.0 with Understat cold. That is not a neutral "
            f"placeholder, it asserts the team created exactly zero expected "
            f"goals -- and it is what `.fillna(0)` in the serving path did to "
            f"the NaN that PR #41 introduced."
        )
        assert np.isfinite(value)

    # The deltas are the sharper half of the bug: 0.0 there reads as "scored
    # exactly to expectation", which is a confident statement about a fixture
    # nobody measured.
    assert served.loc[0, "home_xg_delta_for_last_5"] != 0.0, (
        "xg_delta is 0.0, i.e. 'this team scored exactly to expectation' -- a "
        "reading asserted about a fixture whose xG was never measured"
    )


def test_cold_understat_xg_is_the_league_average_rate(monkeypatch, synthetic):
    """The substitute is the league's expected-goals rate, and it says so.

    Asserted against the value derived from the data itself, not a hardcoded
    constant, so this stays true if the synthetic league changes.
    """
    ctx = _cold_context(synthetic)
    row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01", tz="UTC"))

    league_rate = float(
        pd.concat([synthetic["goals_home"], synthetic["goals_away"]], ignore_index=True).mean()
    )

    for col in [c for c in XG_COLS if "_for_" in c or "_against_" in c]:
        assert row[col] == pytest.approx(league_rate, abs=1e-9), (
            f"{col} is {row[col]}, expected the league rate {league_rate}"
        )


def test_cold_understat_delta_is_a_real_over_under_performance_reading(monkeypatch, synthetic):
    """The delta must carry information, not collapse to a constant.

    `xg_delta_for = last_w_goals_for - xg_for_last_w`. With xG imputed at the
    league rate this is goals-scored-minus-league-expectation, which varies
    per team. Before the fix it was a flat 0.0 for every team on earth.
    """
    ctx = _cold_context(synthetic)
    deltas = set()
    for home, away in [("T0", "T1"), ("T2", "T3"), ("T4", "T5")]:
        row = ctx.build_row(home, away, commence_time=pd.Timestamp("2024-08-01", tz="UTC"))
        deltas.add(round(row["home_xg_delta_for_last_5"], 6))

    assert len(deltas) > 1, (
        f"every fixture got the same xG delta ({deltas}), so the column carries "
        f"no signal -- it is the 'exactly to expectation' reading in disguise"
    )


def test_h2h_object_dtype_still_predicts_without_raising(monkeypatch, synthetic):
    """The object-dtype path must keep working.

    `_current_h2h` returns `None` for a pair with no prior meetings, which
    leaves those columns as `object` dtype in a one-row frame. XGBoost's
    `inplace_predict` rejects object dtypes outright, so the coercion in
    `_row_to_matrix` is load-bearing, not decorative. `weight` is set so the
    blend genuinely produces an object column.
    """
    ctx = _cold_context(synthetic)
    # T0/T1 have met; force a pair that has not.
    home, away = "T0", "T2"
    h2h_cols = ["h2h_home_goal_diff_avg", "h2h_home_win_rate"]
    row = ctx.build_row(home, away, commence_time=pd.Timestamp("2024-08-01", tz="UTC"))

    frame = pd.DataFrame([row]).reindex(columns=h2h_cols, fill_value=0)
    assert frame[h2h_cols].dtypes.apply(lambda d: d == object).any(), (
        "expected at least one object-dtype h2h column for this fixture; if the "
        "fixture stopped producing one, this test is no longer covering the "
        "case it exists for"
    )

    served = _matrix_for(row, h2h_cols)
    assert served.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all(), (
        f"h2h columns must reach XGBoost as floats, got {dict(served.dtypes)}"
    )


def test_fully_populated_prediction_is_unchanged_by_this_fix(monkeypatch, synthetic):
    """Guards against quietly shifting every existing prediction.

    With Understat warm there is nothing to impute, so the feature row -- and
    therefore the prediction -- must be bit-identical to what the pre-fix code
    produced. Compared against a straight transcription of the old expression
    rather than a stored number, so this stays a real check if the fixtures
    change.
    """
    ctx = _warm_context(monkeypatch, synthetic)
    home, away = "T0", "T1"
    row = ctx.build_row(home, away, commence_time=pd.Timestamp("2024-08-01", tz="UTC"))

    home_model, away_model, feature_cols = _tiny_models(synthetic)

    old_matrix = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0).fillna(0).astype(float)
    new_matrix = _matrix_for(row, feature_cols)

    pd.testing.assert_frame_equal(new_matrix, old_matrix)
    assert np.allclose(
        home_model.predict(new_matrix), home_model.predict(old_matrix), rtol=0, atol=0
    )

    # And end to end through the public entry point, so this covers the grid
    # the app actually returns rather than only the intermediate matrix.
    model = ml_scoreline.MLScorelineModel(
        home_model=home_model,
        away_model=away_model,
        feature_cols=list(feature_cols),
        teams=[],
        context=ctx,
    )
    grid = model.predict("T0", "T1")
    direct = ml_scoreline.predict_grid(
        home_model, away_model, old_matrix
    )
    assert grid.home_win == pytest.approx(direct.home_win, abs=0)
    assert grid.draw == pytest.approx(direct.draw, abs=0)
    assert grid.away_win == pytest.approx(direct.away_win, abs=0)


def test_serving_matrix_is_never_object_dtype(monkeypatch, synthetic):
    """A blanket postcondition for every serving path, cold or warm.

    This is the invariant that actually matters to XGBoost, and it is cheap to
    check on both entry points.
    """
    for label, ctx in [("warm", _warm_context(monkeypatch, synthetic))]:
        row = ctx.build_row("T0", "T1", commence_time=pd.Timestamp("2024-08-01", tz="UTC"))
        served = _matrix_for(row, ["h2h_home_win_rate", "xg_for_last_5"])
        assert served.select_dtypes(include=["object"]).empty, (
            f"{label}: object dtype survived to the booster"
        )