"""sportsbook_api.py — live EPL odds via RapidAPI's Sportsbook API
(https://rapidapi.com/sportsbook-api-sportsbook-api-default/api/sportsbook-api2),
the odds/value-bets pipeline's primary live-odds source since 2026-09.
`odds_api.py` (The Odds API) exhausts its monthly credit quota too often to
serve as the day-to-day source; that module is left in place, unused by
default, in case it's ever worth reverting to or supplementing with.

One event = one request for real prices — confirmed live: no batch/bulk
odds endpoint exists despite trying comma-separated `eventKeys`, repeated
`eventKeys=` params, and an `eventKeys[]` array (all either 400 or silently
keep only the last key). `/v0/competitions/{key}/events` lists fixtures
with market *keys* attached but never outcomes/prices — `includeOdds=true`
and `withOdds=true` are silently ignored. A full EPL gameweek costs ~10
requests to price every fixture, against a confirmed 150/day cap (the
`x-ratelimit-requests-limit` response header) — both event-list and
per-event responses are cached to disk per `SPORTSBOOK_CACHE_TTL_SECONDS`
(6h), so the periodic background refresh (~40 requests/day for a
10-fixture gameweek) stays well inside that.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import (
    SPORTSBOOK_API_BASE_URL,
    SPORTSBOOK_API_HOST,
    SPORTSBOOK_API_KEY,
    SPORTSBOOK_CACHE_DIR,
    SPORTSBOOK_CACHE_TTL_SECONDS,
    SPORTSBOOK_EPL_COMPETITION_KEY,
)
from .team_names import to_canonical

# This project's own market vocabulary (same names odds_api.py/value_bets.py
# already use) -- only full-match (REGULATION_TIME) markets are mapped;
# HALF_1 and other segments are ignored.
_H2H = "h2h"
_TOTALS = "totals"
_BTTS = "btts"

# The only totals line this project prices elsewhere (model + The Odds
# API) -- a fixture can carry other lines (3.0, 3.5, ...) per book,
# discarded so de-vigging always compares the same line across books.
_TOTALS_LINE = 2.5


class SportsbookAPIKeyMissing(RuntimeError):
    """Raised when SPORTSBOOK_API_KEY isn't set. Subscribe (free tier) to
    the Sportsbook API on RapidAPI and add `SPORTSBOOK_API_KEY=...` to your
    .env file (see .env.example)."""


def _require_api_key() -> str:
    if not SPORTSBOOK_API_KEY:
        raise SportsbookAPIKeyMissing(
            "SPORTSBOOK_API_KEY is not set. Subscribe to the Sportsbook API on "
            "RapidAPI and add `SPORTSBOOK_API_KEY=...` to your .env file (see .env.example)."
        )
    return SPORTSBOOK_API_KEY


def _headers() -> dict:
    return {"X-RapidAPI-Key": _require_api_key(), "X-RapidAPI-Host": SPORTSBOOK_API_HOST}


def _cache_path(name: str) -> Path:
    return SPORTSBOOK_CACHE_DIR / f"{name}.json"


def _is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def fetch_epl_events_raw(force_refresh: bool = False) -> list[dict]:
    """Upcoming EPL fixtures with team names/keys and market *keys* attached
    (no prices yet — see fetch_event_odds_raw for that). One request;
    cached to disk."""
    cache_path = _cache_path("epl_events")
    if not force_refresh and _is_fresh(cache_path, SPORTSBOOK_CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    resp = requests.get(
        f"{SPORTSBOOK_API_BASE_URL}/competitions/{SPORTSBOOK_EPL_COMPETITION_KEY}/events",
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json().get("events", [])
    cache_path.write_text(json.dumps(events))
    return events


def fetch_event_odds_raw(event_key: str, force_refresh: bool = False) -> dict | None:
    """Real prices for one event. One request per call; cached to disk per
    event so a partial refresh (some fixtures already fetched this cycle)
    doesn't re-spend budget on fixtures that haven't gone stale yet."""
    cache_path = _cache_path(f"event_{event_key}")
    if not force_refresh and _is_fresh(cache_path, SPORTSBOOK_CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    resp = requests.get(
        f"{SPORTSBOOK_API_BASE_URL}/events",
        headers=_headers(),
        params={"eventKeys": event_key},
        timeout=15,
    )
    resp.raise_for_status()
    # The API nests each requested event's own data as a one-element list
    # (confirmed live: {"events": [[{...}]]}) rather than flattening it.
    groups = resp.json().get("events", [])
    event = groups[0][0] if groups and groups[0] else None
    if event is not None:
        cache_path.write_text(json.dumps(event))
    return event


def match_project_event_ids(fixtures_df: pd.DataFrame, events: list[dict]) -> dict[str, str]:
    """Maps this project's own fixture `event_id` -> Sportsbook API event
    `key`, matched by canonicalized (team_home, team_away) and commence_time
    within a day — the same fuzzy-match tolerance
    `tracking/store.py::reconcile_fixture_market_predictions` already uses
    for a different source pairing. Returns only fixtures matched on both
    sides; an unmatched project fixture (e.g. a match this provider hasn't
    posted odds for yet) is simply absent, not an error."""
    if fixtures_df.empty or not events:
        return {}

    by_teams: dict[tuple[str, str], list[dict]] = {}
    for event in events:
        participants = {p["key"]: p["name"] for p in event.get("participants", [])}
        home_key = event.get("homeParticipantKey")
        if home_key not in participants or len(participants) != 2:
            continue
        home = to_canonical(participants[home_key], source="sportsbook_api")
        away_name = next(name for key, name in participants.items() if key != home_key)
        away = to_canonical(away_name, source="sportsbook_api")
        by_teams.setdefault((home, away), []).append(event)

    mapping: dict[str, str] = {}
    for _, fixture in fixtures_df.iterrows():
        candidates = by_teams.get((fixture["team_home"], fixture["team_away"]))
        if not candidates:
            continue
        fixture_time = pd.to_datetime(fixture["commence_time"], utc=True)
        best = min(
            candidates,
            key=lambda event: abs((pd.to_datetime(event["startTime"], utc=True) - fixture_time).total_seconds()),
        )
        if abs((pd.to_datetime(best["startTime"], utc=True) - fixture_time).total_seconds()) <= 24 * 3600:
            mapping[fixture["event_id"]] = best["key"]

    return mapping


def _quotes_for_markets(event: dict, market_type: str, segment: str, modifier: float | None = None) -> dict[str, list[dict]]:
    """Merges outcomes across every market entry matching (market_type,
    segment) — a fixture can carry duplicate/stale market-key entries with
    no books at all (confirmed live) alongside the one real, priced entry.
    Returns {outcome_type: [{"bookmaker": ..., "price": ...}, ...]}."""
    by_type: dict[str, list[dict]] = {}
    for market in event.get("markets", []):
        if market.get("type") != market_type or market.get("segment") != segment:
            continue
        for book, rows in market.get("outcomes", {}).items():
            for row in rows:
                if modifier is not None and row.get("modifier") != modifier:
                    continue
                by_type.setdefault(row["type"], []).append({"bookmaker": book, "price": row["payout"]})
    return by_type


def event_to_rows(
    event: dict,
    project_event_id: str,
    team_home: str,
    team_away: str,
    commence_time,
    fetched_at: pd.Timestamp,
) -> list[dict]:
    """Normalizes one Sportsbook API event's markets into this project's
    long-format odds row shape (event_id, market, outcome_name, bookmaker,
    price, point) — the same shape `odds_api.events_to_frame()` produces,
    so `value_bets.py`'s `_best_quote`/`_devig_*` need no changes to
    consume either source. `project_event_id` is *this project's own*
    fixture event_id (from `match_project_event_ids`), never the Sportsbook
    API's own event key — keeps event_id stable for anything that persists
    it (tracking_store, the value-bet ledger)."""
    rows: list[dict] = []

    home_key = event.get("homeParticipantKey")

    def add(market: str, outcome_name: str, quotes: list[dict], point: float | None = None) -> None:
        for quote in quotes:
            rows.append(
                {
                    "event_id": project_event_id,
                    "commence_time": commence_time,
                    "team_home": team_home,
                    "team_away": team_away,
                    "bookmaker": quote["bookmaker"],
                    "market": market,
                    "outcome_name": outcome_name,
                    "price": quote["price"],
                    "point": point,
                    "odds_fetched_at": fetched_at,
                }
            )

    # MONEYLINE_3WAY: each book emits one "WIN" row per participant (side
    # identified by participantKey, not a HOME/AWAY label) plus one "DRAW"
    # row with no participant. Some books (e.g. Kalshi, a prediction
    # market) also emit "WIN_NO"/"DRAW_NO" (the lay/no side) mixed into the
    # same array -- those aren't a back price on this outcome and are
    # skipped rather than mismapped onto home/away/draw.
    h2h_by_type = _quotes_for_markets(event, "MONEYLINE_3WAY", "REGULATION_TIME")
    home_quotes, away_quotes = [], []
    for market in event.get("markets", []):
        if market.get("type") != "MONEYLINE_3WAY" or market.get("segment") != "REGULATION_TIME":
            continue
        for book, rows_ in market.get("outcomes", {}).items():
            for row in rows_:
                if row.get("type") != "WIN" or row.get("participantKey") is None:
                    continue
                quote = {"bookmaker": book, "price": row["payout"]}
                (home_quotes if row["participantKey"] == home_key else away_quotes).append(quote)
    add(_H2H, team_home, home_quotes)
    add(_H2H, team_away, away_quotes)
    add(_H2H, "Draw", h2h_by_type.get("DRAW", []))

    totals_by_type = _quotes_for_markets(event, "POINT_TOTAL", "REGULATION_TIME", modifier=_TOTALS_LINE)
    add(_TOTALS, "Over", totals_by_type.get("OVER", []), point=_TOTALS_LINE)
    add(_TOTALS, "Under", totals_by_type.get("UNDER", []), point=_TOTALS_LINE)

    btts_by_type = _quotes_for_markets(event, "BOTH_TEAMS_TO_SCORE", "REGULATION_TIME")
    add(_BTTS, "Yes", btts_by_type.get("YES", []))
    add(_BTTS, "No", btts_by_type.get("NO", []))

    return rows


def fetch_epl_odds(fixtures_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Live odds for every fixture in `fixtures_df` that this provider has
    posted a line for, in the same long-format shape
    `odds_api.events_to_frame()` returns. Rides on top of whichever
    fixture list the caller already has (`data/fixtures.py`, unchanged) —
    this module is odds-only, not a fixture source."""
    events = fetch_epl_events_raw(force_refresh=force_refresh)
    event_id_map = match_project_event_ids(fixtures_df, events)
    fetched_at = pd.Timestamp.now(tz="UTC")

    rows: list[dict] = []
    for _, fixture in fixtures_df.iterrows():
        event_key = event_id_map.get(fixture["event_id"])
        if event_key is None:
            continue
        event = fetch_event_odds_raw(event_key, force_refresh=force_refresh)
        if event is None:
            continue
        rows.extend(
            event_to_rows(
                event,
                project_event_id=fixture["event_id"],
                team_home=fixture["team_home"],
                team_away=fixture["team_away"],
                commence_time=fixture["commence_time"],
                fetched_at=fetched_at,
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
    return df
