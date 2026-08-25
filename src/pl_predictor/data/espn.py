"""Confirmed Premier League starting lineups from ESPN's public match feed.

The feed publishes a roster only after the team sheets are available. It is
used solely to constrain an already-local player-probability model; failure
to fetch it is deliberately indistinguishable from "lineups not released".
"""

from __future__ import annotations

from datetime import datetime
import unicodedata

import requests

from ..config import ESPN_SCOREBOARD_URL


def _normalise(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed.casefold() if char.isalnum())


def fetch_confirmed_lineups(home: str, away: str, kickoff: datetime | None) -> dict[str, list[str]]:
    """Return confirmed starter names keyed by the supplied team names.

    An empty mapping means ESPN has not published the lineups yet or is
    temporarily unavailable. No caller needs to special-case either state.
    """
    if kickoff is None:
        return {}
    try:
        scoreboard = requests.get(
            f"{ESPN_SCOREBOARD_URL}/scoreboard", params={"dates": kickoff.strftime("%Y%m%d")}, timeout=15
        )
        scoreboard.raise_for_status()
        event = next(
            (
                candidate
                for candidate in scoreboard.json().get("events", [])
                if {_normalise(competitor["team"]["displayName"]) for competitor in candidate["competitions"][0]["competitors"]}
                == {_normalise(home), _normalise(away)}
            ),
            None,
        )
        if event is None:
            return {}

        response = requests.get(f"{ESPN_SCOREBOARD_URL}/summary", params={"event": event["id"]}, timeout=15)
        response.raise_for_status()
        return _confirmed_starters(response.json(), home, away)
    except (KeyError, requests.RequestException, ValueError):
        return {}


def _confirmed_starters(payload: dict, home: str, away: str) -> dict[str, list[str]]:
    by_team = {_normalise(home): home, _normalise(away): away}
    lineups: dict[str, list[str]] = {}
    for roster in payload.get("rosters", []):
        team = by_team.get(_normalise(roster.get("team", {}).get("displayName", "")))
        starters = [
            row.get("athlete", {}).get("displayName", "")
            for row in roster.get("roster", [])
            if row.get("starter") and row.get("athlete", {}).get("displayName")
        ]
        if team and len(starters) == 11:
            lineups[team] = starters
    return lineups
