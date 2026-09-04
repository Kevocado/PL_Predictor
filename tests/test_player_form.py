"""Leakage and sanity checks for the player-prediction feature layer."""

import pandas as pd
import pytest

from pl_predictor.data.fpl_history import load_player_gw_history
from pl_predictor.features.player_form import blended_current_form, build_historical_player_form, position_rate_priors


@pytest.fixture(scope="module")
def player_gws():
    return load_player_gw_history(seasons=["2023-24", "2024-25"])


def test_rolling_player_form_first_appearance_is_nan(player_gws):
    played, feature_cols = build_historical_player_form(player_gws)
    first_rows = played.sort_values("kickoff_time").groupby("element", sort=False).head(1)
    for col in feature_cols:
        assert first_rows[col].isna().all(), f"{col} should be NaN for a player's first appearance"


def test_rolling_player_form_never_uses_same_row_stat(player_gws):
    """A player's rolling goals-per-90 rate, right after their first
    appearance, must be computed purely from that first appearance (not
    include the second row's own goals via off-by-one)."""
    played, _ = build_historical_player_form(player_gws)
    played = played.sort_values(["element", "kickoff_time"])
    element = played["element"].iloc[0]
    rows = played[played["element"] == element].reset_index(drop=True)
    if len(rows) < 2:
        pytest.skip("sampled player has fewer than 2 appearances")
    expected = rows.loc[0, "goals_scored"] / rows.loc[0, "minutes"] * 90
    assert rows.loc[1, "goals_per90_last5"] == pytest.approx(expected)


def test_position_rate_priors_cover_outfield_positions(player_gws):
    priors = position_rate_priors(player_gws)
    for position in ("DEF", "MID", "FWD"):
        assert position in priors
        assert priors[position]["goals_per90"] >= 0
        assert priors[position]["avg_minutes"] > 0
    # forwards should score more per-90 than defenders, on average
    assert priors["FWD"]["goals_per90"] > priors["DEF"]["goals_per90"]


def test_blended_current_form_handles_player_with_no_current_season_history():
    """A player with zero rows this season (e.g. a summer signing who
    hasn't debuted yet) hits `fetch_player_summary`'s empty-history path —
    `pd.DataFrame([])`, which has no columns at all, not just no rows.
    `rank_team_players` loops every squad member with no per-player
    try/except, so one such player used to take down predictions for the
    player's *entire team* with `KeyError: 'minutes'` (confirmed live on
    the public deployment: every fixture's Likely scorers & assists came
    back empty)."""
    empty_history = pd.DataFrame([])
    rates, confidence = blended_current_form(empty_history, prior_season=None, position="FWD", position_priors={})
    assert confidence == "none"
    assert rates["avg_minutes"] == 60.0
