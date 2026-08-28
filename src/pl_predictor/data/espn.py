"""Confirmed Premier League starting lineups from ESPN's public match feed.

The feed publishes a roster only after the team sheets are available. It is
used solely to constrain an already-local player-probability model; failure
to fetch it is deliberately indistinguishable from "lineups not released".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
import unicodedata

import requests

from ..config import ESPN_SCOREBOARD_URL

# Team sheets are only ever published shortly before kickoff (typically
# ~1h), never for fixtures further out or long finished — outside this
# window the call can only return {} anyway, so skipping it entirely avoids
# a slow/unreliable network round-trip for every non-imminent fixture.
# Confirmed the hard way: an uncached, unconditional call here was the
# dominant cost of building `public_snapshot.py` across a full season.
_LINEUP_WINDOW = timedelta(hours=30)

# Scoreboard listings are shared across every fixture kicking off the same
# day (a full gameweek is usually 3-4 distinct dates) — cache per date
# rather than re-fetching identical data once per fixture.
_scoreboard_cache: dict[str, tuple[float, dict]] = {}
_SCOREBOARD_CACHE_TTL_SECONDS = 300


def _normalise(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed.casefold() if char.isalnum())


def _within_lineup_window(kickoff: datetime) -> bool:
    now = datetime.now(timezone.utc)
    reference = kickoff if kickoff.tzinfo is not None else kickoff.replace(tzinfo=timezone.utc)
    return -_LINEUP_WINDOW <= (reference - now) <= _LINEUP_WINDOW


def _fetch_scoreboard(date_key: str) -> dict:
    cached = _scoreboard_cache.get(date_key)
    if cached is not None and (time.time() - cached[0]) < _SCOREBOARD_CACHE_TTL_SECONDS:
        return cached[1]
    scoreboard = requests.get(f"{ESPN_SCOREBOARD_URL}/scoreboard", params={"dates": date_key}, timeout=8)
    scoreboard.raise_for_status()
    payload = scoreboard.json()
    _scoreboard_cache[date_key] = (time.time(), payload)
    return payload


def fetch_confirmed_lineups(home: str, away: str, kickoff: datetime | None) -> dict[str, list[str]]:
    """Return confirmed starter names keyed by the supplied team names.

    An empty mapping means ESPN has not published the lineups yet, the
    fixture isn't imminent enough for lineups to exist, or the feed is
    temporarily unavailable. No caller needs to special-case any of these.
    """
    if kickoff is None or not _within_lineup_window(kickoff):
        return {}
    try:
        payload = _fetch_scoreboard(kickoff.strftime("%Y%m%d"))
        event = next(
            (
                candidate
                for candidate in payload.get("events", [])
                if {_normalise(competitor["team"]["displayName"]) for competitor in candidate["competitions"][0]["competitors"]}
                == {_normalise(home), _normalise(away)}
            ),
            None,
        )
        if event is None:
            return {}

        response = requests.get(f"{ESPN_SCOREBOARD_URL}/summary", params={"event": event["id"]}, timeout=8)
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
