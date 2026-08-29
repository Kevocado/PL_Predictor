"""FPL player projections and constraint-safe recommendation helpers.

The scorer intentionally consumes only information available before the
deadline: the official FPL bootstrap/fixture feeds and the match model's
pre-kickoff scoreline forecast.  It is a transparent baseline while the
historical hybrid experiment accumulates enough labelled player-gameweek
data to be promoted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_POINTS = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CLEAN_SHEET_POINTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _availability(element: dict) -> float:
    status = element.get("status", "a")
    if status in {"i", "s", "u"}:
        return 0.0
    if status == "d":
        chance = element.get("chance_of_playing_next_round")
        return float(chance) / 100 if chance is not None else 0.5
    return 1.0


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _expected_minutes(element: dict, availability: float) -> float:
    starts = _float(element.get("starts"))
    minutes = _float(element.get("minutes"))
    games = max(_float(element.get("appearances")), starts, minutes / 60.0, 1.0)
    start_rate = min(1.0, starts / games)
    # A non-starter can still return points as a substitute.  This deliberately
    # avoids treating a hot but low-minutes player as a 90-minute certainty.
    base = 25.0 + 65.0 * start_rate
    return round(base * availability, 1)


def _fixture_projection(element: dict, fixture: dict, is_home: bool, team_goal_expectancy: float, opponent_goal_expectancy: float) -> tuple[float, dict]:
    position = POSITION_NAMES.get(int(element.get("element_type", 0)), "MID")
    availability = _availability(element)
    minutes = _expected_minutes(element, availability)
    appearance = 2.0 * min(minutes / 60.0, 1.0)
    season_minutes = max(_float(element.get("minutes")), 1.0)
    xg90 = 90 * _float(element.get("expected_goals")) / season_minutes
    xa90 = 90 * _float(element.get("expected_assists")) / season_minutes
    # Conservative individual shares; total team xG still comes from the
    # independently trained match model, not bookmaker prices.
    goal_share = min(0.42, max(0.025, xg90 / 1.7))
    assist_share = min(0.34, max(0.02, xa90 / 1.35))
    goal_points = GOAL_POINTS[position] * team_goal_expectancy * goal_share * (minutes / 90.0)
    assist_points = 3.0 * team_goal_expectancy * assist_share * (minutes / 90.0)
    clean_prob = float(np.exp(-max(opponent_goal_expectancy, 0.0)))
    clean_points = CLEAN_SHEET_POINTS[position] * clean_prob * min(minutes / 60.0, 1.0)
    bps_form = min(1.2, _float(element.get("form")) / 8.0)
    total = appearance + goal_points + assist_points + clean_points + bps_form
    opponent = fixture["team_a"] if is_home else fixture["team_h"]
    difficulty = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
    return total, {
        "opponent_id": int(opponent),
        "was_home": is_home,
        "difficulty": int(difficulty or 3),
        "expected_goals": round(team_goal_expectancy, 2),
        "clean_sheet_probability": round(clean_prob, 3),
        "expected_minutes": minutes,
    }


def build_projections(
    bootstrap: dict,
    fixtures: list[dict],
    event_id: int,
    scoreline_for_fixture: Callable[[str, str], dict] | None = None,
) -> dict:
    """Return current gameweek projections, including every double fixture.

    ``scoreline_for_fixture`` is injected by the API layer so this module is
    testable without loading any match model or network source.
    """
    teams = {int(team["id"]): team["name"] for team in bootstrap.get("teams", [])}
    event_fixtures = [item for item in fixtures if item.get("event") == event_id and not item.get("finished")]
    by_team: dict[int, list[dict]] = defaultdict(list)
    for fixture in event_fixtures:
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        if scoreline_for_fixture:
            prediction = scoreline_for_fixture(teams.get(home, str(home)), teams.get(away, str(away)))
            home_xg = float(prediction.get("home_goal_expectation", 1.35))
            away_xg = float(prediction.get("away_goal_expectation", 1.15))
            model_source = prediction.get("model_source", "independent_scoreline")
        else:
            home_xg, away_xg, model_source = 1.35, 1.15, "fpl_baseline"
        by_team[home].append({"fixture": fixture, "is_home": True, "team_xg": home_xg, "opponent_xg": away_xg, "model_source": model_source})
        by_team[away].append({"fixture": fixture, "is_home": False, "team_xg": away_xg, "opponent_xg": home_xg, "model_source": model_source})

    players = []
    for element in bootstrap.get("elements", []):
        team_id = int(element.get("team", 0))
        position = POSITION_NAMES.get(int(element.get("element_type", 0)), "MID")
        fixture_rows = by_team.get(team_id, [])
        fixture_details = []
        total = 0.0
        expected_minutes = 0.0
        for row in fixture_rows:
            points, detail = _fixture_projection(element, row["fixture"], row["is_home"], row["team_xg"], row["opponent_xg"])
            detail["opponent"] = teams.get(detail.pop("opponent_id"), "Unknown")
            detail["model_source"] = row["model_source"]
            fixture_details.append(detail)
            total += points
            expected_minutes += detail["expected_minutes"]
        availability = _availability(element)
        form = _float(element.get("form"))
        drivers = [
            f"{len(fixture_details)} fixture{'s' if len(fixture_details) != 1 else ''}",
            f"{expected_minutes:.0f} expected minutes",
            f"form {form:.1f}",
        ]
        if fixture_details:
            drivers.append(f"{sum(item['expected_goals'] for item in fixture_details):.2f} expected team goals")
        players.append(
            {
                "player_id": int(element["id"]),
                "name": f"{element.get('first_name', '')} {element.get('second_name', '')}".strip() or element.get("web_name", "Unknown"),
                "web_name": element.get("web_name", ""),
                "team": teams.get(team_id, "Unknown"),
                "team_id": team_id,
                "position": position,
                "price": round(_float(element.get("now_cost")) / 10.0, 1),
                "status": element.get("status", "a"),
                "news": element.get("news", ""),
                "availability": availability,
                "expected_minutes": round(expected_minutes, 1),
                "projected_points": round(total, 2),
                "fixture_count": len(fixture_details),
                "fixtures": fixture_details,
                "drivers": drivers,
            }
        )
    players.sort(key=lambda item: item["projected_points"], reverse=True)
    return {
        "gameweek": event_id,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "model_source": "hybrid_match_context_baseline",
        "data_freshness": "live official FPL API + independent pre-match scoreline model",
        "players": players,
    }


def _eligible(players: list[dict]) -> list[dict]:
    return [p for p in players if p["fixture_count"] and p["availability"] > 0 and p["expected_minutes"] >= 20]


def _valid_team(players: list[dict], budget: float | None = None) -> bool:
    counts = defaultdict(int)
    clubs = defaultdict(int)
    for player in players:
        counts[player["position"]] += 1
        clubs[player["team_id"]] += 1
    if any(clubs[club] > 3 for club in clubs):
        return False
    if budget is not None and sum(p["price"] for p in players) > budget + 1e-9:
        return False
    return True


def optimal_xi(players: list[dict], formation: str | None = None) -> dict:
    """Find the highest projected legal starting XI from a player pool.

    ``formation`` may be e.g. ``"3-4-3"``; otherwise the optimiser chooses
    the best legal formation itself.
    """
    candidates = _eligible(players)
    if not candidates:
        return {"starting_xi": [], "captain": None, "vice_captain": None, "bench": [], "projected_points": 0.0}
    rows, lower, upper = [np.ones(len(candidates))], [11.0], [11.0]
    limits = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    if formation is not None:
        try:
            defenders, midfielders, forwards = (int(value) for value in formation.split("-"))
        except ValueError as exc:
            raise ValueError("Formation must look like 3-4-3.") from exc
        if not (3 <= defenders <= 5 and 2 <= midfielders <= 5 and 1 <= forwards <= 3 and defenders + midfielders + forwards == 10):
            raise ValueError("Formation must be a legal FPL outfield formation.")
        limits = {"GK": (1, 1), "DEF": (defenders, defenders), "MID": (midfielders, midfielders), "FWD": (forwards, forwards)}
    for position, (lo, hi) in limits.items():
        rows.append(np.array([1.0 if p["position"] == position else 0.0 for p in candidates]))
        lower.append(float(lo)); upper.append(float(hi))
    for team_id in {p["team_id"] for p in candidates}:
        rows.append(np.array([1.0 if p["team_id"] == team_id else 0.0 for p in candidates]))
        lower.append(0.0); upper.append(3.0)
    solution = milp(
        c=-np.array([p["projected_points"] for p in candidates]),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        options={"time_limit": 2.0},
    )
    best = [p for p, chosen in zip(candidates, solution.x if solution.success else []) if chosen > 0.5]
    if len(best) != 11:
        return {"starting_xi": [], "captain": None, "vice_captain": None, "bench": [], "projected_points": 0.0}
    ordered = sorted(best, key=lambda p: (p["position"], -p["projected_points"]))
    captain, vice = sorted(best, key=lambda p: p["projected_points"], reverse=True)[:2]
    bench = sorted([p for p in candidates if p["player_id"] not in {x["player_id"] for x in best}], key=lambda p: p["projected_points"], reverse=True)[:4]
    return {"starting_xi": ordered, "captain": captain, "vice_captain": vice, "bench": bench, "projected_points": round(sum(p["projected_points"] for p in best), 2)}


def build_squad(players: list[dict], budget: float = 100.0) -> dict:
    """Highest-projected legal 15-player £100m squad."""
    candidates = _eligible(players)
    if not candidates:
        raise ValueError("Not enough available FPL players to build a legal squad")
    rows = [np.ones(len(candidates)), np.array([p["price"] for p in candidates])]
    lower, upper = [15.0, 0.0], [15.0, budget]
    for position, required in SQUAD_SHAPE.items():
        rows.append(np.array([1.0 if p["position"] == position else 0.0 for p in candidates]))
        lower.append(float(required)); upper.append(float(required))
    for team_id in {p["team_id"] for p in candidates}:
        rows.append(np.array([1.0 if p["team_id"] == team_id else 0.0 for p in candidates]))
        lower.append(0.0); upper.append(3.0)
    solution = milp(
        c=-np.array([p["projected_points"] for p in candidates]),
        integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)), options={"time_limit": 3.0},
    )
    selected = [p for p, chosen in zip(candidates, solution.x if solution.success else []) if chosen > 0.5]
    if len(selected) != 15:
        raise ValueError("Unable to satisfy the £100m squad budget with available players")
    spent = sum(p["price"] for p in selected)
    xi = optimal_xi(selected)
    return {"squad": sorted(selected, key=lambda p: (p["position"], -p["projected_points"])), "budget": budget, "spent": round(spent, 1), "remaining": round(budget - spent, 1), **xi}


def transfer_recommendations(players: list[dict], current_ids: list[int], bank: float = 0.0, free_transfers: int = 1) -> dict:
    """Rank legal like-for-like transfers from a current 15-player squad."""
    pool = _eligible(players)
    # An injured, suspended, or blank-GW player is precisely someone a user
    # may need to sell, so current-squad validation intentionally uses the
    # complete live pool rather than the incoming-player eligibility filter.
    current = [p for p in players if p["player_id"] in set(current_ids)]
    if len(current) != 15:
        raise ValueError("The supplied squad does not match 15 currently available FPL players")
    clubs = defaultdict(int)
    for p in current:
        clubs[p["team_id"]] += 1
    ideas = []
    for outgoing in current:
        for incoming in pool:
            if incoming["position"] != outgoing["position"] or incoming["player_id"] in set(current_ids):
                continue
            cost = max(0.0, incoming["price"] - outgoing["price"] - bank)
            if clubs[incoming["team_id"]] + (0 if incoming["team_id"] == outgoing["team_id"] else 1) > 3:
                continue
            gain = incoming["projected_points"] - outgoing["projected_points"]
            if gain > 0:
                ideas.append({"out": outgoing, "in": incoming, "cost": round(cost, 1), "projected_gain": round(gain, 2), "net_gain": round(gain - (4 if free_transfers < 1 else 0), 2)})
    ideas.sort(key=lambda x: x["net_gain"], reverse=True)
    return {"free_transfers": max(0, int(free_transfers)), "bank": round(bank, 1), "recommendations": ideas[:10]}
