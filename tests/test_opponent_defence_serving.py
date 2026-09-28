"""Serving-path tests for the opponent-defensive-strength features.

EXP-2026-25 promoted `opponent_defence_last{3,5,10}` into the fitted G+A model.
That makes `predict_goal_contribution` responsible for supplying three values
that `blended_current_form` cannot know, because they describe *the fixture being
priced* rather than the player's own history.

The hazard these tests exist for is specific: the feature is worth ~0.0008 log
loss, and a silently-missing feature scores as a constant offset against a
fitted coefficient, which is the same class of defect as NFL `0628c6d`.
"""

import numpy as np
import pandas as pd
import pytest

from pl_predictor.models import player_goals

FEATURES = ("opponent_defence_last3", "opponent_defence_last5", "opponent_defence_last10")


def _two_column(values, index) -> np.ndarray:
    """sklearn-shaped output.

    A numpy array, not a DataFrame: `predict_proba` is indexed with `[:, 1]`,
    which is positional on an array but *label*-based on a DataFrame in pandas
    2.x. A DataFrame stub therefore fails with `InvalidIndexError` rather than
    testing anything.
    """
    return np.tile(np.array([[1.0 - values, values]]), (len(index), 1))


def _model(columns=None) -> dict:
    """A stub contribution model that records the row it was asked to score."""

    class _Recorder:
        def __init__(self):
            self.seen = None

        def predict_proba(self, matrix):
            self.seen = matrix
            return _two_column(0.5, matrix.index)

    class _Calibrator:
        def predict_proba(self, values):
            return _two_column(0.5, range(len(values)))

    recorder = _Recorder()
    names = list(columns or FEATURES)
    return {
        "features": names,
        "columns": names + ["position"],
        "model": recorder,
        "calibrator": _Calibrator(),
        "_recorder": recorder,
    }


def test_the_opponent_terms_are_actually_fitted():
    """A model fitted without them would serve 0.0 for every one of them and
    never know it. This is the assertion that the promotion actually landed."""
    from pl_predictor.evaluate.goal_contribution_research import OPPONENT_FEATURES

    assert list(player_goals._OPPONENT_DEFENCE_FEATURES) == list(OPPONENT_FEATURES), (
        "the serving constant has drifted from the fitted feature list")


def test_a_supplied_rating_reaches_the_model():
    model = _model()
    player_goals.predict_goal_contribution(
        {}, {}, "MID", model, opponent_defence={"opponent_defence_last3": 1.75})
    row = model["_recorder"].seen
    for feature in FEATURES:
        assert feature in row.columns, f"{feature} missing from the scored row"


def test_an_unresolvable_opponent_serves_the_league_prior_not_a_crash():
    """A caller that cannot name the opponent passes `None`. That must score as
    0.0 -- the league-average prior the model was calibrated against -- and not
    raise, because a missing feature must not take down player predictions."""
    model = _model()
    probability = player_goals.predict_goal_contribution({}, {}, "MID", model, opponent_defence=None)
    assert probability == 0.5  # stubbed model/calibrator
    row = model["_recorder"].seen
    assert (row[list(FEATURES)] == 0.0).all().all(), "must default to the league prior, not NaN"


def test_a_partially_resolved_opponent_defaults_the_rest_to_the_prior():
    model = _model()
    player_goals.predict_goal_contribution(
        {}, {}, "MID", model, opponent_defence={"opponent_defence_last5": -0.5})
    row = model["_recorder"].seen
    assert row["opponent_defence_last5"].iloc[0] == pytest.approx(-0.5)
    assert row["opponent_defence_last3"].iloc[0] == 0.0


def test_a_stale_opponent_term_in_rates_does_not_win():
    """Mirrors `test_was_home_argument_overrides_a_conflicting_rates_value`.
    `opponent_defence_last*` describes the fixture being priced, so an explicit
    rating must beat anything reachable through the feature dicts. Those dicts
    are built by `blended_current_form` from the player's own historical rows
    and know nothing about who is being faced -- letting them win would be a
    train/serve skew of the kind this module exists to prevent."""
    model = _model()
    rates = {"opponent_defence_last3": 99.0}
    player_goals.predict_goal_contribution(
        rates, {}, "MID", model, opponent_defence={"opponent_defence_last3": 1.0})
    assert model["_recorder"].seen["opponent_defence_last3"].iloc[0] == pytest.approx(1.0)


def test_a_fitted_model_is_scored_with_every_column_it_declares():
    """The end-to-end contract. If the fitted column list gains a name the
    serving row cannot produce, `reindex(..., fill_value=0.0)` silently scores
    it as a constant and nobody finds out until a prediction looks slightly
    wrong. A real fitted model is used so the list is the production list."""
    model = player_goals.fit_goal_contribution_model()
    if not model:
        pytest.skip("no fitted contribution model available")
    declared = set(model["columns"])
    assert set(FEATURES) <= declared, "fitted model lost the opponent terms"

    probability = player_goals.predict_goal_contribution(
        {}, {}, "MID", model, is_home=True,
        opponent_defence={"opponent_defence_last3": 0.5, "opponent_defence_last5": 0.2},
    )
    assert probability is not None
    assert 0.0 <= probability <= 1.0


def test_the_route_helper_rates_the_defence_each_side_faces():
    """`_rank_fixture_players` calls this once and hands each club the rating of
    the defence *it* is about to meet. Getting the two the wrong way round would
    hand every home player the away side's own defensive record."""
    from pl_predictor.api import routes

    # Season labels are long form ("2026-2027") because `season_str` is the single
    # producer: `fetch_season` does `df["season"] = season` from
    # `default_completed_seasons()`, and `fetch_current_season_partial` uses the
    # same helper. An earlier note here claimed `_get_matches_df` mixes the short
    # "2024-25" form with the long one; that was wrong, and it would have invited a
    # name-normalisation fix for a problem that does not exist.
    matches_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-08", "2026-08-15", "2026-08-22"]),
        "season": ["2026-2027"] * 3,
        "team_home": ["Arsenal", "Arsenal", "Chelsea"],
        "team_away": ["Leaky", "Leaky", "Leaky"],
        "goals_home": [0, 0, 5],
        "goals_away": [4, 4, 0],
    })

    original = routes._get_matches_df
    routes._get_matches_df = lambda: matches_df
    try:
        rated = routes._opponent_defence_for_fixture("Arsenal", "Leaky", pd.Timestamp("2026-08-29"))
    finally:
        routes._get_matches_df = original

    # Leaky's prior three (conceded - scored): 0-4, 0-4, 5-0 -> -4, -4, 5 -> mean -1.
    assert rated["Arsenal"]["opponent_defence_last3"] == pytest.approx(-1.0), (
        "Arsenal faces Leaky, whose last three reads -4, -4, +5")
    # Arsenal's prior two in 2026-27 (it is not in the 08-22 fixture, which is
    # Chelsea v Leaky): 4 and 4.
    assert rated["Leaky"]["opponent_defence_last3"] == pytest.approx(4.0), (
        "Leaky faces Arsenal, whose prior reads 4, 4")


def test_the_route_helper_survives_a_fixture_it_cannot_resolve():
    """No kickoff, no history, or a malformed frame all mean the league prior,
    never a 500 on the fixture's player list."""
    from pl_predictor.api import routes

    assert routes._opponent_defence_for_fixture("Arsenal", "Chelsea", None) == {}

    original = routes._get_matches_df
    routes._get_matches_df = lambda: pd.DataFrame()
    try:
        assert routes._opponent_defence_for_fixture("Arsenal", "Chelsea", pd.Timestamp("2026-08-29")) == {}
    finally:
        routes._get_matches_df = original


def test_a_gameweek_one_fixture_is_unrated_on_the_serving_side_too():
    """Grouping by season is what makes this true. The fitting path cannot rate a
    club before it has played in that season, so the serving path must not hand a
    gameweek-1 fixture a rating smuggled in from the previous season -- the model
    was calibrated expecting 0.0 there."""
    from pl_predictor.api import routes

    # Only 2025-26 has been played. Arsenal leaked 4 and then 4 last season, so
    # grouping by club alone would hand a 2026-27 gameweek-1 fixture a rating of
    # -4 that the fitted model has never seen an example of.
    matches_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-08-09", "2025-08-16"]),
        "season": ["2025-2026", "2025-2026"],
        "team_home": ["Arsenal", "Arsenal"],
        "team_away": ["Leaky", "Leaky"],
        "goals_home": [0, 0],
        "goals_away": [4, 4],
    })

    original = routes._get_matches_df
    routes._get_matches_df = lambda: matches_df
    try:
        opening = routes._opponent_defence_for_fixture("Chelsea", "Arsenal", pd.Timestamp("2026-08-15"))
    finally:
        routes._get_matches_df = original

    assert "opponent_defence_last3" not in opening["Chelsea"], (
        "Arsenal has not played in 2026-27, so a gameweek-1 fixture must be unrated")
