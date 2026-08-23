"""football_data_org.py — live current-season Premier League fixtures,
results, and standings via football-data.org's free tier.

Unlike API-Football's free tier (which explicitly walls off the current
season entirely — verified directly: any request with `season=2026` returns
"Free plans do not have access to this season, try from 2022 to 2024"),
football-data.org's Matches and Standings endpoints for a competition
default to the *current* season automatically — no season param needed to
unlock it, so there's nothing to gate. Confirmed live:
`/competitions/2021/matches` (no season param) returns the full current
380-fixture Premier League season in one call, already reflecting the
matches played so far (goals, status, referee).

What this source does *not* have on the free tier: match-level stats beyond
the final score — no shots/corners/cards, so it can't replace
football-data.co.uk as a *training-feature* source. It's used here purely
as a fast, single-call, always-current source for the live fixture
list/schedule, known results (for reconciling predictions quickly), and
precomputed standings — football-data.co.uk remains the source for
everything training-feature-related (rolling shots/corners/cards, referee
tendencies).
"""

from __future__ import annotations

import pandas as pd
import requests

from ..config import FOOTBALL_DATA_KEY, FOOTBALL_DATA_ORG_BASE_URL, FOOTBALL_DATA_ORG_COMPETITION_ID
from .team_names import to_canonical


class FootballDataOrgKeyMissing(Exception):
    pass


def _headers() -> dict:
    if not FOOTBALL_DATA_KEY:
        raise FootballDataOrgKeyMissing("FOOTBALL_DATA_KEY is not set")
    return {"X-Auth-Token": FOOTBALL_DATA_KEY}


def fetch_matches(competition_id: int = FOOTBALL_DATA_ORG_COMPETITION_ID) -> pd.DataFrame:
    """The full current season's fixtures, one row per match, played or
    not. Columns: event_id, team_home, team_away, commence_time, status,
    finished, goals_home, goals_away, ftr (all four None until finished),
    matchday (football-data.org's own gameweek number — football-data.co.uk
    has no equivalent column, so this is the only source for it)."""
    resp = requests.get(
        f"{FOOTBALL_DATA_ORG_BASE_URL}/competitions/{competition_id}/matches",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])

    rows = []
    for m in matches:
        finished = m["status"] == "FINISHED"
        score = (m.get("score") or {}).get("fullTime") or {}
        goals_home = score.get("home") if finished else None
        goals_away = score.get("away") if finished else None
        ftr = None
        if finished and goals_home is not None and goals_away is not None:
            ftr = "H" if goals_home > goals_away else "A" if goals_away > goals_home else "D"
        rows.append(
            {
                "event_id": m["id"],
                "team_home": to_canonical(m["homeTeam"]["name"], source="football_data_org"),
                "team_away": to_canonical(m["awayTeam"]["name"], source="football_data_org"),
                "commence_time": m["utcDate"],
                "status": m["status"],
                "finished": finished,
                "goals_home": goals_home,
                "goals_away": goals_away,
                "ftr": ftr,
                "matchday": m.get("matchday"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
    return df


def fetch_standings(competition_id: int = FOOTBALL_DATA_ORG_COMPETITION_ID) -> pd.DataFrame:
    """Live current-season league table, already computed by
    football-data.org — no need to reconstruct it from individual match
    results the way `models/projected_table.py::compute_standings` does
    for the football-data.co.uk path. Columns: team, position, played,
    points, goal_diff, wins, draws, losses, goals_for, goals_against."""
    resp = requests.get(
        f"{FOOTBALL_DATA_ORG_BASE_URL}/competitions/{competition_id}/standings",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    table = next((s["table"] for s in data.get("standings", []) if s.get("type") == "TOTAL"), [])

    rows = [
        {
            "team": to_canonical(row["team"]["name"], source="football_data_org"),
            "position": row["position"],
            "played": row["playedGames"],
            "points": row["points"],
            "goal_diff": row["goalDifference"],
            "wins": row["won"],
            "draws": row["draw"],
            "losses": row["lost"],
            "goals_for": row["goalsFor"],
            "goals_against": row["goalsAgainst"],
        }
        for row in table
    ]
    return pd.DataFrame(rows)
