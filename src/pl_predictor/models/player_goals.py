"""player_goals.py — anytime goalscorer / assist probability per player.

    λ_player = player's rolling goals-per-90 rate
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
"""

from __future__ import annotations

import math

from ..data import fpl_api
from ..data.team_names import to_canonical
from ..features import player_form
from . import scoreline

# Reuse the same reference constant scoreline.py already uses for an
# average-strength team, so "how much stronger/weaker is this fixture's
# team than average" has one consistent definition across the project.
LEAGUE_AVERAGE_TEAM_GOALS = scoreline.FALLBACK_GOAL_EXPECTANCY

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def anytime_probability(lam: float) -> float:
    return 1 - math.exp(-lam)


def predict_player(rates: dict, team_goal_expectation: float, availability: float) -> dict:
    """`rates` is `features.player_form.blended_current_form`'s output
    (goals_per90, assists_per90, avg_minutes)."""
    strength_multiplier = team_goal_expectation / LEAGUE_AVERAGE_TEAM_GOALS
    minutes_fraction = min(rates["avg_minutes"] / 90, 1.0)
    scale = strength_multiplier * minutes_fraction * availability

    lam_goals = rates["goals_per90"] * scale
    lam_assists = rates["assists_per90"] * scale

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
) -> list[dict]:
    """Ranked (by anytime-goal probability) list of a team's players for one
    fixture, given that fixture's team expected goals from the scoreline
    model."""
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
        pred = predict_player(rates, team_goal_expectation, availability)

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
