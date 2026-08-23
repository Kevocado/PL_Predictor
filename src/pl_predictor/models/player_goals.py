"""player_goals.py — anytime goalscorer / assist probability per player.

    λ_player = player's expected-goals-per-appearance estimate
               × (this fixture's team expected goals / league-average team goals)
               × (expected minutes this match / 90, from recent appearances)
               × live availability multiplier

Ties player predictions to the already-fitted scoreline model's
fixture-specific team strength (`models.scoreline.predict_fixture`'s
`home_goal_expectation`/`away_goal_expectation`) rather than an independent
player regressor, so a player's chance rises/falls with how much the match
model expects their team to score in this specific fixture. Availability
reuses the exact live-status business rule already proven in
`FPL_Optimizer/scout.py`: zero out injured/suspended/unavailable players,
scale doubtful ones by their `chance_of_playing_next_round`.

The "expected-goals-per-appearance estimate" itself: `evaluate/
player_stat_reliability.py` tested whether FPL's ICT Index (and the wider
stat surface `features/player_form.py` computes) actually predicts a
player's future output beyond the plain rolling goals/assists rate this
formula used to rely on alone. Result — not assumed, measured on a real
held-out season: `threat` adds real incremental signal for goals (R² on a
goals~[rate, threat] regression beats goals~rate alone), `creativity` does
the same for assists; the ICT Index's `influence` sub-component does not
(near-zero/negative gain — redundant with the existing rate, not reliable
on its own). So `predict_player` now blends in `threat`/`creativity` via a
small linear regression fitted on real historical data (`fit_reliability_coefficients`),
not the raw rate alone — `influence` and the other more marginal
candidates (bps, bonus, xG/xA on their own) are left out, matching "keep
only what earns it" the same way match-model features were tested this
session. `reliability_coeffs` is optional specifically so this stays
backward compatible (omit it to fall back to the plain rate, e.g. in a unit
test with no fitted coefficients handy)."""

from __future__ import annotations

import math

from ..data import fpl_api, fpl_history
from ..data.team_names import to_canonical
from ..features import player_form
from . import scoreline

# Reuse the same reference constant scoreline.py already uses for an
# average-strength team, so "how much stronger/weaker is this fixture's
# team than average" has one consistent definition across the project.
LEAGUE_AVERAGE_TEAM_GOALS = scoreline.FALLBACK_GOAL_EXPECTANCY

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# (target rate key, reliability-tested stat key) -> which of player_form's
# blended_current_form output keys the reliability-adjusted estimate reads.
_RELIABILITY_SPEC = {
    "goals": ("goals_per90", "threat"),
    "assists": ("assists_per90", "creativity"),
}


def fit_reliability_coefficients(seasons: list[str] | None = None) -> dict:
    """Fits the two small linear regressions `evaluate/
    player_stat_reliability.py` validated (goals ~ [goals_per90_last10,
    threat_last10], assists ~ [assists_per90_last10, creativity_last10]) on
    *all* available historical seasons (not held out — this is for
    production use, unlike the reliability study's own train/test split).
    Cheap (a couple of features, tens of thousands of rows, sub-second) —
    meant to be refit on a lightweight periodic cadence (see
    `api/routes.py`'s cache for this), not persisted to disk like the match
    model's much more expensive XGBoost fits."""
    from sklearn.linear_model import LinearRegression

    seasons = seasons or fpl_history.default_completed_seasons()
    df = fpl_history.load_player_gw_history(seasons=seasons)
    played, _ = player_form.build_historical_player_form(df)

    coeffs = {}
    for target_key, (rate_stat, extra_stat) in _RELIABILITY_SPEC.items():
        rate_col, extra_col = f"{rate_stat}_last10", f"{extra_stat}_last10"
        target_col = "goals_scored" if target_key == "goals" else "assists"
        d = played.dropna(subset=[rate_col, extra_col, target_col])
        if len(d) < 100:
            continue
        model = LinearRegression().fit(d[[rate_col, extra_col]].to_numpy(), d[target_col].to_numpy())
        coeffs[target_key] = {
            "intercept": float(model.intercept_),
            "coef_rate": float(model.coef_[0]),
            "coef_extra": float(model.coef_[1]),
        }
    return coeffs


def anytime_probability(lam: float) -> float:
    return 1 - math.exp(-lam)


def _expected_per_appearance(rates: dict, target_key: str, reliability_coeffs: dict | None) -> float:
    """The reliability-adjusted estimate for one target (goals or assists)
    if fitted coefficients are available and the extra stat is present in
    `rates`; falls back to the plain per-90 rate otherwise (e.g. no
    coefficients passed, or a brand-new player with no `threat`/`creativity`
    history yet)."""
    rate_stat, extra_stat = _RELIABILITY_SPEC[target_key]
    plain_rate = rates.get(rate_stat, 0.0) or 0.0

    coeffs = (reliability_coeffs or {}).get(target_key)
    extra_val = rates.get(extra_stat)
    if coeffs is None or extra_val is None:
        return plain_rate

    estimate = coeffs["intercept"] + coeffs["coef_rate"] * plain_rate + coeffs["coef_extra"] * extra_val
    return max(estimate, 0.0)


def predict_player(
    rates: dict, team_goal_expectation: float, availability: float, reliability_coeffs: dict | None = None
) -> dict:
    """`rates` is `features.player_form.blended_current_form`'s output.
    `reliability_coeffs` (from `fit_reliability_coefficients`) is optional —
    pass it to use the reliability-adjusted goals/assists estimate;
    omit for the plain per-90 rate (e.g. in tests)."""
    strength_multiplier = team_goal_expectation / LEAGUE_AVERAGE_TEAM_GOALS
    minutes_fraction = min(rates["avg_minutes"] / 90, 1.0)
    scale = strength_multiplier * minutes_fraction * availability

    goals_estimate = _expected_per_appearance(rates, "goals", reliability_coeffs)
    assists_estimate = _expected_per_appearance(rates, "assists", reliability_coeffs)

    lam_goals = goals_estimate * scale
    lam_assists = assists_estimate * scale

    return {
        "expected_goals": lam_goals,
        "expected_assists": lam_assists,
        "anytime_goal_prob": anytime_probability(lam_goals),
        "anytime_assist_prob": anytime_probability(lam_assists),
    }


def rank_team_players(
    team: str,
    team_goal_expectation: float,
    bootstrap: dict,
    current_event: int | None,
    position_priors: dict,
    limit: int = 8,
    reliability_coeffs: dict | None = None,
) -> list[dict]:
    """Ranked (by anytime-goal probability) list of a team's players for one
    fixture, given that fixture's team expected goals from the scoreline
    model. `reliability_coeffs` (from `fit_reliability_coefficients`) is
    optional — pass it for the reliability-adjusted goals/assists estimate."""
    teams_by_id = {t["id"]: t["name"] for t in bootstrap["teams"]}
    team_id = next(
        (tid for tid, name in teams_by_id.items() if to_canonical(name, source="fpl") == team),
        None,
    )
    if team_id is None:
        return []

    results = []
    for el in bootstrap["elements"]:
        if el["team"] != team_id:
            continue
        position = POSITION_MAP.get(el["element_type"], "Unknown")

        history, prior_season = fpl_api.fetch_player_summary(el["id"], current_event)
        rates, confidence = player_form.blended_current_form(history, prior_season, position, position_priors)
        availability = fpl_api.availability_multiplier(el["status"], el.get("chance_of_playing_next_round"))
        pred = predict_player(rates, team_goal_expectation, availability, reliability_coeffs)

        results.append(
            {
                "player_id": el["id"],
                "name": el["web_name"],
                "position": position,
                "status": el["status"],
                "news": el.get("news", ""),
                "availability": availability,
                "confidence": confidence,
                **pred,
            }
        )

    results.sort(key=lambda r: -r["anytime_goal_prob"])
    return results[:limit]
