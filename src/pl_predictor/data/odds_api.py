"""odds_api.py — live EPL odds via The Odds API (https://the-odds-api.com/).

Free tier is 500 credits/month (cost = markets x regions per call), so calls
are cached to disk per-gameweek and reused rather than re-fetched on every
notebook/app run. Missing API key raises `OddsAPIKeyMissing` rather than
crashing callers — see `data/fixtures.py` for the graceful fallback.

Note: the bulk `/sports/{sport}/odds` endpoint used here only serves "core"
markets (h2h, spreads, totals). BTTS, corners, and cards are "additional
markets" that The Odds API only exposes per-event via
`/sports/{sport}/events/{event_id}/odds` (confirmed by hitting the bulk
endpoint directly — requesting `btts` in bulk returns a 422 "Markets not
supported by this endpoint"). Not implemented here — those stay model-only
predictions, same as corners/cards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_API_SPORT_KEY, ODDS_CACHE_DIR
from .team_names import to_canonical

DEFAULT_MARKETS = ["h2h", "totals"]
DEFAULT_REGIONS = "uk"
CACHE_TTL_SECONDS = 12 * 3600


class OddsAPIKeyMissing(RuntimeError):
    """Raised when ODDS_API_KEY isn't set. Sign up free (no card) at
    https://the-odds-api.com/ and add it to .env (see .env.example)."""


def _require_api_key() -> str:
    if not ODDS_API_KEY:
        raise OddsAPIKeyMissing(
            "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com/ "
            "and add `ODDS_API_KEY=...` to your .env file (see .env.example)."
        )
    return ODDS_API_KEY


def _cache_path(gameweek_key: str) -> Path:
    return ODDS_CACHE_DIR / f"gw_{gameweek_key}.json"


def _is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def fetch_epl_odds_raw(
    markets: list[str] | None = None,
    regions: str = DEFAULT_REGIONS,
    gameweek_key: str = "current",
    force_refresh: bool = False,
) -> list[dict]:
    """Return the raw event list from The Odds API, cached per `gameweek_key`.
    On a cache hit, no network call (and no credits) are used."""
    markets = markets or DEFAULT_MARKETS
    cache_path = _cache_path(gameweek_key)

    if not force_refresh and _is_fresh(cache_path, CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    api_key = _require_api_key()
    resp = requests.get(
        f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT_KEY}/odds",
        params={
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "apiKey": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"[odds_api] credits used={used} remaining={remaining}")

    events = resp.json()
    cache_path.write_text(json.dumps(events))
    return events


def events_to_frame(events: list[dict], fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Flatten raw event JSON into one row per (event, bookmaker, market,
    outcome) — long format, easiest to aggregate/de-vig downstream."""
    rows = []
    for event in events:
        home = to_canonical(event["home_team"], source="odds_api")
        away = to_canonical(event["away_team"], source="odds_api")
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    # h2h outcome names are literally the team names (raw,
                    # not yet canonicalized) for a home/away win outcome;
                    # canonicalize so lookups by our short team names match.
                    # "Draw"/"Over"/"Under"/"Yes"/"No" pass through unchanged
                    # since they have no alias entry.
                    outcome_name = to_canonical(outcome["name"], source="odds_api")
                    rows.append(
                        {
                            "event_id": event["id"],
                            "commence_time": event["commence_time"],
                            "team_home": home,
                            "team_away": away,
                            "bookmaker": bookmaker["key"],
                            "market": market["key"],
                            "outcome_name": outcome_name,
                            "price": outcome["price"],
                            "point": outcome.get("point"),
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df["odds_fetched_at"] = fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC")
    return df


def fetch_epl_odds(
    markets: list[str] | None = None,
    regions: str = DEFAULT_REGIONS,
    gameweek_key: str = "current",
    force_refresh: bool = False,
) -> pd.DataFrame:
    events = fetch_epl_odds_raw(
        markets=markets, regions=regions, gameweek_key=gameweek_key, force_refresh=force_refresh
    )
    cache_path = _cache_path(gameweek_key)
    fetched_at = pd.Timestamp(cache_path.stat().st_mtime, unit="s", tz="UTC") if cache_path.exists() else pd.Timestamp.now(tz="UTC")
    return events_to_frame(events, fetched_at=fetched_at)
