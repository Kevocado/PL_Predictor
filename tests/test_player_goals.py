"""predict_player's reliability-adjusted goals/assists estimate — see
evaluate/player_stat_reliability.py for the study that found `threat`/
`creativity` add real predictive signal beyond the plain rolling rate,
while `influence` doesn't. Tests the blending logic itself (no network
calls — fitting real coefficients from FPL history is covered by just
calling `fit_reliability_coefficients()` directly against live data
elsewhere, this only checks predict_player's math given known inputs)."""

import math

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
