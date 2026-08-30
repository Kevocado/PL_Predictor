"""Display-only summaries for the Data Hub's team and player tabs."""

from __future__ import annotations

import pandas as pd

from ..data import fpl_api, understat, understat_shots
from ..data.team_names import to_canonical
from ..features import streaks
from ..models import player_ratings


def _number(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def _form_summary(group: pd.DataFrame) -> tuple[float | None, str]:
    """Current form and direction, using only this season's completed rows."""
    ordered = group.sort_values("date")
    if ordered.empty:
        return None, "new"
    form = ordered.tail(5)["points"]
    current = float(form.mean())
    if len(ordered) < 2:
        return _number(current), "new"
    window = min(3, len(ordered) - 1)
    recent = ordered.tail(window)["points"].mean()
    previous = ordered.iloc[-(2 * window):-window]["points"].mean()
    if pd.isna(previous):
        return _number(current), "new"
    delta = float(recent - previous)
    return _number(current), "up" if delta > 0.15 else "down" if delta < -0.15 else "steady"


def _team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    zeros = pd.Series(0, index=matches.index)
    home = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["team_home"],
            "opponent": matches["team_away"],
            "venue": "Home",
            "goals_for": matches["goals_home"],
            "goals_against": matches["goals_away"],
            "shots": matches.get("hs"),
            "shots_on_target": matches.get("hst"),
            "corners": matches.get("hc"),
            "fouls": matches.get("hf"),
            "cards": matches.get("hy", zeros).fillna(0) + matches.get("hr", zeros).fillna(0),
        }
    )
    away = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["team_away"],
            "opponent": matches["team_home"],
            "venue": "Away",
            "goals_for": matches["goals_away"],
            "goals_against": matches["goals_home"],
            "shots": matches.get("as"),
            "shots_on_target": matches.get("ast"),
            "corners": matches.get("ac"),
            "fouls": matches.get("af"),
            "cards": matches.get("ay", zeros).fillna(0) + matches.get("ar", zeros).fillna(0),
        }
    )
    rows = pd.concat([home, away], ignore_index=True)
    rows["result"] = rows.apply(
        lambda row: "W" if row.goals_for > row.goals_against else ("D" if row.goals_for == row.goals_against else "L"), axis=1
    )
    rows["points"] = rows["result"].map({"W": 3, "D": 1, "L": 0})
    return rows.sort_values(["team", "date"]).reset_index(drop=True)


def _understat_team_rows(season_start: int) -> pd.DataFrame:
    try:
        source = understat.load_xg_data(seasons=[str(season_start)])
    except RuntimeError:
        return pd.DataFrame(columns=["date", "team", "opponent", "venue", "xg_for", "xg_against"])
    if source.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "venue", "xg_for", "xg_against"])
    home = pd.DataFrame(
        {
            "date": source["date"],
            "team": source["team_home"],
            "opponent": source["team_away"],
            "venue": "Home",
            "xg_for": source["xg_home"],
            "xg_against": source["xg_away"],
        }
    )
    away = pd.DataFrame(
        {
            "date": source["date"],
            "team": source["team_away"],
            "opponent": source["team_home"],
            "venue": "Away",
            "xg_for": source["xg_away"],
            "xg_against": source["xg_home"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _set_piece_rows(season_start: int) -> pd.DataFrame:
    try:
        source = understat_shots.load_shot_situation_data(seasons=[str(season_start)])
    except RuntimeError:
        return pd.DataFrame(columns=["date", "team", "opponent", "venue", "set_piece_xg_share"])
    if source.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "venue", "set_piece_xg_share"])
    home = pd.DataFrame(
        {
            "date": source["date"],
            "team": source["team_home"],
            "opponent": source["team_away"],
            "venue": "Home",
            "set_piece_xg_share": source["home_set_piece_xg_share"],
        }
    )
    away = pd.DataFrame(
        {
            "date": source["date"],
            "team": source["team_away"],
            "opponent": source["team_home"],
            "venue": "Away",
            "set_piece_xg_share": source["away_set_piece_xg_share"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _team_fpl_totals(bootstrap: dict | None) -> dict[str, dict[str, float | int]]:
    """Assists and expected-assists per team, summed straight off the FPL
    bootstrap's per-player totals — the same source `build_player_hub`
    (below) already reads `assists`/`expected_assists` from, just grouped
    by team instead of listed per player. `season_matches` (used for every
    other team_hub stat) has no assist data at all, so this is the only
    source for it."""
    totals: dict[str, dict[str, float | int]] = {}
    if not bootstrap:
        return totals
    team_names = {team["id"]: to_canonical(team["name"], source="fpl") for team in bootstrap.get("teams", [])}
    for element in bootstrap.get("elements", []):
        team = team_names.get(element["team"])
        if team is None:
            continue
        entry = totals.setdefault(team, {"assists": 0, "xa": 0.0, "has_xa": False})
        entry["assists"] += int(element.get("assists", 0))
        xa = element.get("expected_assists")
        if xa is not None and not pd.isna(xa):
            entry["xa"] += float(xa)
            entry["has_xa"] = True
    return totals


def build_team_hub(matches: pd.DataFrame, season: str, bootstrap: dict | None = None) -> dict:
    """Return current-season form, underlying performance, and team styles."""
    season_matches = matches[matches["season"] == season].copy()
    if season_matches.empty:
        return {"season": season, "teams": []}

    season_start = int(season[:4])
    rows = _team_match_rows(season_matches)
    keys = ["date", "team", "opponent", "venue"]
    rows = rows.merge(_understat_team_rows(season_start), on=keys, how="left")
    rows = rows.merge(_set_piece_rows(season_start), on=keys, how="left")
    current_streaks = streaks.latest_streaks(season_matches)
    fpl_totals = _team_fpl_totals(bootstrap)
    teams = []
    for team, group in rows.groupby("team", sort=True):
        played = len(group)
        recent = group.sort_values("date", ascending=False).head(10)
        xg_for = group["xg_for"].sum(min_count=1)
        xg_against = group["xg_against"].sum(min_count=1)
        form_points_per_match, form_trend = _form_summary(group)
        fpl_team_totals = fpl_totals.get(team, {})
        teams.append(
            {
                "team": team,
                "played": played,
                "points": int(group["points"].sum()),
                "wins": int((group["result"] == "W").sum()),
                "draws": int((group["result"] == "D").sum()),
                "losses": int((group["result"] == "L").sum()),
                "goals_for": int(group["goals_for"].sum()),
                "goals_against": int(group["goals_against"].sum()),
                "assists": int(fpl_team_totals.get("assists", 0)),
                "xa": _number(fpl_team_totals["xa"]) if fpl_team_totals.get("has_xa") else None,
                "points_per_match": _number(group["points"].mean()),
                "form_points_per_match": form_points_per_match,
                "form_trend": form_trend,
                "goals_for_per_match": _number(group["goals_for"].mean()),
                "goals_against_per_match": _number(group["goals_against"].mean()),
                "shots_per_match": _number(group["shots"].mean()),
                "shots_on_target_per_match": _number(group["shots_on_target"].mean()),
                "corners_per_match": _number(group["corners"].mean()),
                "fouls_per_match": _number(group["fouls"].mean()),
                "cards_per_match": _number(group["cards"].mean()),
                "xg_for": _number(xg_for),
                "xg_against": _number(xg_against),
                "goals_minus_xg": _number(group["goals_for"].sum() - xg_for) if not pd.isna(xg_for) else None,
                "goals_conceded_minus_xg": _number(group["goals_against"].sum() - xg_against) if not pd.isna(xg_against) else None,
                "set_piece_xg_share": _number(group["set_piece_xg_share"].mean()),
                "streak": int(current_streaks.get(team, 0)),
                "recent_matches": [
                    {
                        "date": row.date.date().isoformat(),
                        "opponent": row.opponent,
                        "venue": row.venue,
                        "result": row.result,
                        "score": f"{int(row.goals_for)}-{int(row.goals_against)}",
                        "xg_for": _number(row.xg_for),
                        "xg_against": _number(row.xg_against),
                    }
                    for row in recent.itertuples()
                ],
            }
        )
    return {"season": season, "teams": teams}


def build_player_hub(bootstrap: dict) -> dict:
    """Return current FPL form plus fixed-scale role-aware ratings."""
    team_names = {team["id"]: to_canonical(team["name"], source="fpl") for team in bootstrap.get("teams", [])}
    positions = {position["id"]: position["singular_name_short"] for position in bootstrap.get("element_types", [])}
    ratings = player_ratings.rate_bootstrap_elements(bootstrap.get("elements", []), positions)
    players = []
    for element in bootstrap.get("elements", []):
        team = team_names.get(element["team"], "Unknown")
        status = element.get("status", "a")
        players.append(
            {
                "id": int(element["id"]),
                "name": element["web_name"],
                "team": team,
                "position": positions.get(element.get("element_type"), "Unknown"),
                "status": status,
                "chance_of_playing": element.get("chance_of_playing_next_round"),
                "minutes": int(element.get("minutes", 0)),
                "starts": int(element.get("starts", 0)),
                "goals": int(element.get("goals_scored", 0)),
                "assists": int(element.get("assists", 0)),
                "xg": _number(element.get("expected_goals")),
                "xa": _number(element.get("expected_assists")),
                "xgi": _number(element.get("expected_goal_involvements")),
                "threat": _number(element.get("threat")),
                "creativity": _number(element.get("creativity")),
                "ict": _number(element.get("ict_index")),
                "bps": int(element.get("bps", 0)),
                "bonus": int(element.get("bonus", 0)),
                "news": element.get("news", ""),
                **ratings.get(int(element["id"]), {}),
            }
        )
    players.sort(key=lambda player: (player.get("overall_rating", 0), player["minutes"]), reverse=True)
    leaderboards = {
        position: sorted(
            (player for player in players if player["position"] == position),
            key=lambda player: player.get("overall_rating", 0),
            reverse=True,
        )[:5]
        for position in ("GK", "DEF", "MID", "FWD")
    }
    return {
        "players": players,
        "leaderboards": leaderboards,
        "rating_model_source": "role_aware_evidence_baseline",
        "data_freshness": "cached official FPL bootstrap",
    }
