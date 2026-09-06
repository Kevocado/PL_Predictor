"""predict_player's reliability-adjusted goals/assists estimate — see
evaluate/player_stat_reliability.py for the study that found `threat`/
`creativity` add real predictive signal beyond the plain rolling rate,
while `influence` doesn't. Tests the blending logic itself (no network
calls — fitting real coefficients from FPL history is covered by just
calling `fit_reliability_coefficients()` directly against live data
elsewhere, this only checks predict_player's math given known inputs)."""

import math

import pandas as pd

from pl_predictor.models import player_goals


def test_falls_back_to_plain_rate_without_coefficients():
    rates = {"goals_per90": 0.5, "assists_per90": 0.2, "avg_minutes": 90, "threat": 40.0, "creativity": 30.0}
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0, reliability_coeffs=None)

    # No coefficients passed -> plain rate * scale (strength=1.5/1.5=1.0 given LEAGUE_AVERAGE_TEAM_GOALS defaults)
    expected_goals_lam = 0.5 * (1.5 / player_goals.LEAGUE_AVERAGE_TEAM_GOALS) * 1.0 * 1.0
    expected_assists_lam = 0.2 * (1.5 / player_goals.LEAGUE_AVERAGE_TEAM_GOALS) * 1.0 * 1.0
    assert pred["expected_goals"] == expected_goals_lam
    assert pred["expected_assists"] == expected_assists_lam
    assert pred["anytime_goal_prob"] == 1 - math.exp(-expected_goals_lam)
    assert pred["anytime_goal_contribution_prob"] == 1 - math.exp(-(expected_goals_lam + expected_assists_lam))


def test_falls_back_to_plain_rate_when_extra_stat_missing():
    """A brand-new player with no threat/creativity history yet (e.g. debut
    via position-average priors only) should still get a sane estimate,
    not crash or silently zero out."""
    rates = {"goals_per90": 0.3, "assists_per90": 0.1, "avg_minutes": 60}  # no "threat"/"creativity" keys
    coeffs = {"goals": {"intercept": 0.01, "coef_rate": 0.5, "coef_extra": 0.01}}
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0, reliability_coeffs=coeffs)

    scale = (1.5 / player_goals.LEAGUE_AVERAGE_TEAM_GOALS) * (60 / 90) * 1.0
    assert pred["expected_goals"] == 0.3 * scale  # plain rate, coeffs present but "threat" missing from rates


def test_uses_reliability_adjusted_estimate_when_available():
    rates = {"goals_per90": 0.4, "assists_per90": 0.15, "avg_minutes": 90, "threat": 50.0, "creativity": 20.0}
    coeffs = {
        "goals": {"intercept": 0.011, "coef_rate": 0.065, "coef_extra": 0.008},
        "assists": {"intercept": 0.024, "coef_rate": 0.019, "coef_extra": 0.006},
    }
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0, reliability_coeffs=coeffs)

    goals_estimate = 0.011 + 0.065 * 0.4 + 0.008 * 50.0
    assists_estimate = 0.024 + 0.019 * 0.15 + 0.006 * 20.0
    scale = (1.5 / player_goals.LEAGUE_AVERAGE_TEAM_GOALS) * 1.0 * 1.0
    assert pred["expected_goals"] == goals_estimate * scale
    assert pred["expected_assists"] == assists_estimate * scale


def test_reliability_estimate_never_goes_negative():
    """A very low rate/threat combination could push the linear estimate
    below zero — must clip, not produce a negative expected-goals count."""
    rates = {"goals_per90": 0.0, "assists_per90": 0.0, "avg_minutes": 90, "threat": 0.0, "creativity": 0.0}
    coeffs = {"goals": {"intercept": -0.5, "coef_rate": 0.065, "coef_extra": 0.008}}
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0, reliability_coeffs=coeffs)
    assert pred["expected_goals"] >= 0.0
    assert 0.0 <= pred["anytime_goal_prob"] < 1.0


def test_position_model_does_not_replace_prior_rate_without_current_form():
    class ZeroModel:
        def predict(self, _):
            return [0.0]

    rates = {"goals_per90": 0.5, "assists_per90": 0.2, "avg_minutes": 90}
    models = {("FWD", "goals"): ZeroModel(), ("FWD", "assists"): ZeroModel()}
    pred = player_goals.predict_player(rates, team_goal_expectation=1.5, availability=1.0, position="FWD", position_rate_models=models)
    strength = 1.5 / player_goals.LEAGUE_AVERAGE_TEAM_GOALS
    assert pred["expected_goals"] == 0.5 * strength
    assert pred["expected_assists"] == 0.2 * strength


def test_confirmed_name_matching_accepts_longer_fpl_legal_names():
    element = {"first_name": "Robert", "second_name": "Lynch Sánchez", "web_name": "Sánchez"}
    assert player_goals._element_matches_confirmed_name(element, {player_goals._normalise_name("Robert Sánchez")})


def test_fit_reliability_coefficients_against_real_history():
    """Real end-to-end check against the actual FPL history archive (this
    project already caches it — no new network dependency introduced by
    this test)."""
    coeffs = player_goals.fit_reliability_coefficients()
    assert "goals" in coeffs
    assert "assists" in coeffs
    for key in ("goals", "assists"):
        assert set(coeffs[key]) == {"intercept", "coef_rate", "coef_extra"}
        # both coefficients found positive on real data (see the reliability
        # study's own printed results) — a sign-flip would mean something
        # broke, not just a marginal-value finding.
        assert coeffs[key]["coef_rate"] > 0
        assert coeffs[key]["coef_extra"] > 0


def _bootstrap_with_one_striker():
    return {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "elements": [{
            "id": 10, "team": 1, "element_type": 4, "web_name": "Striker",
            "status": "a", "first_name": "Test", "second_name": "Striker",
        }],
    }


def test_rank_team_players_merges_live_shots_into_rates(monkeypatch):
    """rank_team_players's own per-request fetch_player_summary() call
    (live current-season history) has no shots/shots_on_target columns at
    all -- FPL's API doesn't provide them. player_shots_by_element is the
    live-serving counterpart to _load_history_with_shots's training-frame
    merge: same crosswalk-and-date join, scoped to one player's live
    history instead of the bulk historical frame. Without this wired in,
    predict_player's rates.get("shots_per90", 0.0) silently returns 0.0 for
    every player, indistinguishable from "genuinely no shots" (confirmed
    live: this was happening for every player in every fixture)."""
    history = pd.DataFrame([
        {"GW": 1, "minutes": 90, "goals_scored": 1, "assists": 0, "kickoff_time": "2025-08-16T14:00:00Z"},
        {"GW": 2, "minutes": 90, "goals_scored": 0, "assists": 1, "kickoff_time": "2025-08-23T14:00:00Z"},
        {"GW": 3, "minutes": 90, "goals_scored": 0, "assists": 0, "kickoff_time": "2025-08-30T14:00:00Z"},
    ])
    monkeypatch.setattr(player_goals.fpl_api, "fetch_player_summary", lambda *a, **k: (history, None))
    monkeypatch.setattr(player_goals, "predict_lineup", lambda *a, **k: {"predicted_starter": True, "expected_minutes": 90.0})

    shots_df = pd.DataFrame([
        {"date": pd.Timestamp("2025-08-16"), "shots": 4, "shots_on_target": 2},
        {"date": pd.Timestamp("2025-08-23"), "shots": 2, "shots_on_target": 1},
    ])

    results = player_goals.rank_team_players(
        "Arsenal", team_goal_expectation=1.8, bootstrap=_bootstrap_with_one_striker(), current_event=4,
        position_priors={}, is_home=True, player_shots_by_element={10: shots_df},
    )
    assert results[0]["expected_shots"] > 0
    assert results[0]["expected_shots_on_target"] > 0


def test_rank_team_players_without_shots_data_still_works(monkeypatch):
    """No player_shots_by_element entry (crosswalk miss, or the dict simply
    isn't passed) must degrade to today's existing behavior, not crash."""
    history = pd.DataFrame([
        {"GW": 1, "minutes": 90, "goals_scored": 1, "assists": 0, "kickoff_time": "2025-08-16T14:00:00Z"},
    ])
    monkeypatch.setattr(player_goals.fpl_api, "fetch_player_summary", lambda *a, **k: (history, None))
    monkeypatch.setattr(player_goals, "predict_lineup", lambda *a, **k: {"predicted_starter": True, "expected_minutes": 90.0})

    results = player_goals.rank_team_players(
        "Arsenal", team_goal_expectation=1.8, bootstrap=_bootstrap_with_one_striker(), current_event=2,
        position_priors={}, is_home=True,
    )
    assert results[0]["expected_shots"] == 0.0
    assert results[0]["expected_shots_on_target"] == 0.0


def test_player_prediction_schema_carries_shots_fields():
    from pl_predictor.api.schemas import PlayerPrediction

    assert "expected_shots" in PlayerPrediction.model_fields
    assert "expected_shots_on_target" in PlayerPrediction.model_fields
    assert "anytime_shot_on_target_prob" in PlayerPrediction.model_fields
    # All three optional -- a crosswalk-miss player must validate with None.
    PlayerPrediction(
        player_id=1, name="Test", position="FWD", anytime_goal_prob=0.1,
        anytime_assist_prob=0.05, anytime_goal_contribution_prob=0.14,
        status="a", news="", confidence="current", predicted_starter=True,
        confirmed_starter=False, expected_minutes=90.0, is_penalty_taker=False,
        is_set_piece_taker=False,
    )
