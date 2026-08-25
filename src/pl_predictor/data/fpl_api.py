"""fpl_api.py — live FPL API client: current player pool, live availability
status, and per-player current-season history.

Ported from FPL_Optimizer/fpl_data.py — same endpoints, same
domain-keyed-plus-TTL cache pattern for per-player history (cache is keyed
by the current gameweek so it auto-invalidates when a new gameweek starts,
plus a few-hours TTL within a gameweek).
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from ..config import FPL_API_BASE_URL, FPL_EVENT_CACHE_DIR, FPL_PLAYER_CACHE_DIR
from .team_names import to_canonical

HISTORY_CACHE_TTL_SECONDS = 6 * 3600
FPL_REQUEST_TIMEOUT_SECONDS = 10
FPL_FAILURE_BACKOFF_SECONDS = 60
_failed_fetches: dict[str, float] = {}
_fixtures_memory_cache: tuple[float, list[dict]] | None = None

# a=available, i=injured, d=doubtful, s=suspended, u=unavailable
UNAVAILABLE_STATUSES = {"i", "s", "u"}

NUMERIC_STRING_COLS = [
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


def fetch_bootstrap() -> dict:
    resp = requests.get(f"{FPL_API_BASE_URL}/bootstrap-static/", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures() -> list[dict]:
    """Return FPL fixtures, retaining the last successful response offline."""
    global _fixtures_memory_cache
    cache_path = FPL_EVENT_CACHE_DIR / "fixtures.json"
    cache_key = str(cache_path)
    now = time.time()
    if _fixtures_memory_cache is not None and now - _fixtures_memory_cache[0] < 60:
        return _fixtures_memory_cache[1]
    if now - _failed_fetches.get(cache_key, 0) < FPL_FAILURE_BACKOFF_SECONDS:
        if cache_path.exists():
            fixtures = json.loads(cache_path.read_text())
            _fixtures_memory_cache = (now, fixtures)
            return fixtures
        raise requests.ConnectionError("FPL fixture feed is temporarily unavailable")
    try:
        response = requests.get(f"{FPL_API_BASE_URL}/fixtures/", timeout=FPL_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        fixtures = response.json()
        cache_path.write_text(json.dumps(fixtures))
        _fixtures_memory_cache = (now, fixtures)
        _failed_fetches.pop(cache_key, None)
        return fixtures
    except requests.RequestException:
        _failed_fetches[cache_key] = now
        if cache_path.exists():
            try:
                fixtures = json.loads(cache_path.read_text())
                _fixtures_memory_cache = (now, fixtures)
                return fixtures
            except json.JSONDecodeError:
                pass
        raise


def fetch_event_live(event_id: int) -> dict:
    """Confirmed per-player outcomes, retaining final gameweek data offline."""
    cache_path = FPL_EVENT_CACHE_DIR / f"event_{int(event_id)}.json"
    cache_key = str(cache_path)
    now = time.time()
    if now - _failed_fetches.get(cache_key, 0) < FPL_FAILURE_BACKOFF_SECONDS:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        raise requests.ConnectionError("FPL event feed is temporarily unavailable")
    try:
        response = requests.get(f"{FPL_API_BASE_URL}/event/{event_id}/live/", timeout=FPL_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload))
        _failed_fetches.pop(cache_key, None)
        return payload
    except requests.RequestException:
        _failed_fetches[cache_key] = now
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                pass
        raise


def _cached_fixture_id(bootstrap: dict, home_id: int, away_id: int, target_date: pd.Timestamp) -> int | None:
    """Find a finished fixture from already-cached FPL player histories."""
    for element in bootstrap.get("elements", []):
        team_id = int(element.get("team", -1))
        if team_id not in {home_id, away_id}:
            continue
        cache_path = FPL_PLAYER_CACHE_DIR / f"{int(element['id'])}.json"
        try:
            history = json.loads(cache_path.read_text()).get("history", []) if cache_path.exists() else []
        except json.JSONDecodeError:
            continue
        expected_opponent = away_id if team_id == home_id else home_id
        expected_home = team_id == home_id
        for row in history:
            kickoff_time = row.get("kickoff_time")
            if not kickoff_time or int(row.get("opponent_team", -1)) != expected_opponent:
                continue
            fixture_time = pd.Timestamp(kickoff_time)
            fixture_time = fixture_time.tz_localize(None) if fixture_time.tzinfo is not None else fixture_time
            if bool(row.get("was_home")) == expected_home and abs((fixture_time - target_date).total_seconds()) < 3 * 86400:
                return int(row["fixture"])
    return None


def _cached_outcomes_matching_score(
    target_date: pd.Timestamp, expected_score: tuple[int, int] | None
) -> dict[int, dict]:
    if expected_score is None:
        return {}
    candidates: dict[int, list[tuple[int, dict]]] = {}
    for cache_path in FPL_PLAYER_CACHE_DIR.glob("*.json"):
        try:
            history = json.loads(cache_path.read_text()).get("history", [])
        except json.JSONDecodeError:
            continue
        for row in history:
            kickoff_time = row.get("kickoff_time")
            if not kickoff_time or row.get("fixture") is None:
                continue
            fixture_time = pd.Timestamp(kickoff_time)
            fixture_time = fixture_time.tz_localize(None) if fixture_time.tzinfo is not None else fixture_time
            if abs((fixture_time - target_date).total_seconds()) < 3 * 86400:
                candidates.setdefault(int(row["fixture"]), []).append((int(cache_path.stem), row))
    matched = []
    for fixture_id, rows in candidates.items():
        home_goals = sum(int(row.get("goals_scored", 0) or 0) for _, row in rows if row.get("was_home"))
        away_goals = sum(int(row.get("goals_scored", 0) or 0) for _, row in rows if not row.get("was_home"))
        if (home_goals, away_goals) == expected_score:
            matched.append((fixture_id, rows))
    if len(matched) != 1:
        return {}
    return {
        player_id: {
            "goals": int(row.get("goals_scored", 0) or 0),
            "assists": int(row.get("assists", 0) or 0),
            "started": bool(int(row.get("starts", 0) or 0)) or int(row.get("minutes", 0) or 0) >= 60,
            "was_home": bool(row.get("was_home")),
        }
        for player_id, row in matched[0][1]
    }


def fixture_player_outcomes(
    home: str,
    away: str,
    kickoff,
    bootstrap: dict | None = None,
    expected_score: tuple[int, int] | None = None,
) -> dict[int, dict]:
    """Resolve official FPL goals/assists for one finished fixture.

    ``event/{gw}/live`` contains a per-fixture explanation block, so this is
    correct for double gameweeks too. Missing source data deliberately returns
    an empty mapping rather than making a completed-fixture response fail.
    """
    bootstrap = bootstrap or fetch_bootstrap()
    teams = {to_canonical(team["name"], source="fpl"): int(team["id"]) for team in bootstrap.get("teams", [])}
    home_id, away_id = teams.get(home), teams.get(away)
    player_teams = {
        int(element["id"]): int(element["team"])
        for element in bootstrap.get("elements", [])
        if element.get("id") is not None and element.get("team") is not None
    }
    target_date = pd.Timestamp(kickoff).tz_localize(None) if pd.Timestamp(kickoff).tzinfo is not None else pd.Timestamp(kickoff)
    if home_id is None or away_id is None:
        return _cached_outcomes_matching_score(target_date, expected_score)
    try:
        fixtures = fetch_fixtures()
    except requests.RequestException:
        fixtures = []
    fixture = next(
        (
            item for item in fixtures
            if item.get("team_h") == home_id and item.get("team_a") == away_id
            and item.get("kickoff_time")
            and abs((pd.Timestamp(item["kickoff_time"]).tz_localize(None) - target_date).total_seconds()) < 3 * 86400
        ),
        None,
    )
    # FPL often publishes team scores and player event explanations before
    # it flips the fixture's `finished` flag. A completed score from our
    # result feed is sufficient to reconcile here, so only reject fixtures
    # that have neither final score field yet.
    if (
        fixture is not None
        and not fixture.get("finished")
        and (fixture.get("team_h_score") is None or fixture.get("team_a_score") is None)
    ):
        return {}
    fixture_id = int(fixture["id"]) if fixture is not None else _cached_fixture_id(bootstrap, home_id, away_id, target_date)
    if fixture_id is None:
        return _cached_outcomes_matching_score(target_date, expected_score)
    outcomes: dict[int, dict] = {}
    event_elements = []
    if fixture is not None and fixture.get("event") is not None:
        try:
            event_elements = fetch_event_live(int(fixture["event"])).get("elements", [])
        except requests.RequestException:
            event_elements = []
    for element in event_elements:
        goals = assists = starts = minutes = 0
        explanations = [item for item in element.get("explain", []) if int(item.get("fixture", -1)) == fixture_id]
        for explanation in explanations:
            stats = {item.get("identifier"): item.get("value", 0) for item in explanation.get("stats", [])}
            goals += int(stats.get("goals_scored", 0) or 0)
            assists += int(stats.get("assists", 0) or 0)
            starts += int(stats.get("starts", 0) or 0)
            minutes += int(stats.get("minutes", 0) or 0)
        if explanations:
            # Older FPL event payloads occasionally omit `starts`; a long
            # appearance is the safest available fallback for a labelled
            # reconstruction, while live snapshots still require ESPN's XI.
            player_id = int(element["id"])
            player_team = player_teams.get(player_id)
            outcomes[player_id] = {
                "goals": goals,
                "assists": assists,
                "started": starts > 0 or minutes >= 60,
                "was_home": player_team == home_id if player_team is not None else None,
            }
    # FPL's player-summary cache carries the same final fixture rows. It is a
    # resilient fallback for historical reconciliation when event/live has
    # expired, is temporarily unavailable, or a prior request already cached
    # the player detail but not the event response.
    for element in bootstrap.get("elements", []):
        if int(element.get("team", -1)) not in {home_id, away_id}:
            continue
        player_id = int(element["id"])
        if player_id in outcomes:
            continue
        cache_path = FPL_PLAYER_CACHE_DIR / f"{player_id}.json"
        try:
            history = json.loads(cache_path.read_text()).get("history", []) if cache_path.exists() else []
        except json.JSONDecodeError:
            history = []
        row = next((item for item in history if int(item.get("fixture", -1)) == fixture_id), None)
        if row is None:
            continue
        outcomes[player_id] = {
            "goals": int(row.get("goals_scored", 0) or 0),
            "assists": int(row.get("assists", 0) or 0),
            "started": bool(int(row.get("starts", 0) or 0)) or int(row.get("minutes", 0) or 0) >= 60,
            "was_home": int(element["team"]) == home_id,
        }
    return outcomes or _cached_outcomes_matching_score(target_date, expected_score)


def get_current_event(bootstrap: dict) -> int | None:
    events = bootstrap.get("events", [])
    for e in events:
        if e.get("is_current") or e.get("is_next"):
            return e["id"]
    unfinished = [e["id"] for e in events if not e.get("finished")]
    return min(unfinished) if unfinished else None


def player_status_table(bootstrap: dict) -> pd.DataFrame:
    """One row per current player: id, name, team, status,
    chance_of_playing_next_round, news — the live-availability signal used
    to gate player-goal predictions."""
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    rows = [
        {
            "element": el["id"],
            "web_name": el["web_name"],
            "name": f"{el['first_name']} {el['second_name']}",
            "team": teams.get(el["team"], "Unknown"),
            "status": el["status"],
            "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            "news": el.get("news", ""),
        }
        for el in bootstrap["elements"]
    ]
    return pd.DataFrame(rows)


def availability_multiplier(status: str, chance_of_playing_next_round) -> float:
    """1.0 = fully expected to play, 0.0 = ruled out. Mirrors the exact
    business rule already used in FPL_Optimizer/scout.py."""
    if status in UNAVAILABLE_STATUSES:
        return 0.0
    if status == "d":
        chance = chance_of_playing_next_round
        return (chance / 100.0) if chance is not None and not pd.isna(chance) else 0.5
    return 1.0


def fetch_player_summary(player_id: int, current_event: int | None = None) -> tuple[pd.DataFrame, dict | None]:
    """Returns (history_df, prior_season_row) for one player: this season's
    played-gameweek rows so far, and last season's season-total stats (or
    None if the player has no FPL history at all)."""
    cache_path = FPL_PLAYER_CACHE_DIR / f"{player_id}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            # A concurrent request for the same player (routes.py's
            # rank_team_players fans out one fetch per squad member, and
            # FastAPI runs sync endpoints in a thread pool — two overlapping
            # requests for the same team can race on the same cache file)
            # can interleave two writes into one corrupt file — confirmed
            # directly: several cache files found with two JSON documents
            # concatenated back to back. Treated as a cache miss rather than
            # crashing the request; the atomic write below (temp file +
            # os.replace) prevents new corruption going forward.
            cached = None
        if cached is not None:
            fresh_gw = cached.get("current_event") == current_event
            fresh_time = (time.time() - cached.get("fetched_at", 0)) < HISTORY_CACHE_TTL_SECONDS
            if fresh_gw and fresh_time:
                return _normalize_history(pd.DataFrame(cached["history"])), cached.get("prior_season")

    resp = requests.get(f"{FPL_API_BASE_URL}/element-summary/{player_id}/", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    history = data.get("history", [])
    history_past = data.get("history_past", [])
    prior_season = history_past[-1] if history_past else None

    payload = json.dumps(
        {"current_event": current_event, "fetched_at": time.time(), "history": history, "prior_season": prior_season}
    )
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(payload)
    os.replace(tmp_path, cache_path)  # atomic on POSIX — no interleaved-write corruption possible
    return _normalize_history(pd.DataFrame(history)), prior_season


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={"round": "GW"})
    for col in NUMERIC_STRING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
