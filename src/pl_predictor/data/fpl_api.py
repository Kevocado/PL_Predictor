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

from ..config import FPL_API_BASE_URL, FPL_PLAYER_CACHE_DIR

HISTORY_CACHE_TTL_SECONDS = 6 * 3600

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
