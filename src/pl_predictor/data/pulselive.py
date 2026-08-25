"""pulselive.py — current-season fixtures, results, and match-level stats
from the Premier League's own site backend (footballapi.pulselive.com).

Confirmed live, no authentication required. Used as a fast supplement to
`football_data.py` for the IN-PROGRESS season only — football-data.co.uk
remains the deep 8-season historical source (already cached, no reason to
touch it), but its publishing cadence for the current season is
unpredictable and can lag for a long time (confirmed directly: zero rows
for the 2026-2027 season despite a full gameweek already played).
pulselive.com's own `/stats/match/{id}` endpoint has the same shots/
shots-on-target/corners/fouls/cards detail football-data.co.uk provides,
confirmed available well within 24 hours of full time — this is what
closes that freshness gap.

Grey-area note: this is an undocumented endpoint, not an officially
licensed API like football-data.org (whose free tier explicitly permits
personal use). Used here for private, non-redistributed personal use only,
per explicit user sign-off — never surface this as "official Premier
League data" to anyone else, never redistribute what it returns.

Column mapping onto football-data.co.uk's own convention (so the rest of
the feature pipeline — rolling_form.py's BASE_STATS, referee.py — needs no
changes to consume this):
    hs/as   <- total_scoring_att          (total shots)
    hst/ast <- ontarget_scoring_att       (shots on target)
    hc/ac   <- corner_taken               (corners)
    hf/af   <- fk_foul_lost               (fouls committed BY that team —
                                            "lost the foul contest" = conceded
                                            the free kick = committed it)
    hy/ay   <- total_yel_card
    hr/ar   <- total_red_card             (present but null when zero)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import PULSELIVE_BASE_URL, PULSELIVE_CACHE_DIR, PULSELIVE_COMPETITION_ID
from .football_data import CURRENT_SEASON_START_YEAR, season_str
from .team_names import to_canonical

_STAT_MAP = {
    "total_scoring_att": "s",
    "ontarget_scoring_att": "st",
    "corner_taken": "c",
    "fk_foul_lost": "f",
    "total_yel_card": "y",
    "total_red_card": "r",
    "possession_percentage": "possession",
}


def _fetch_with_retry(fn, attempts: int = 3, backoff: float = 2.0):
    last_err = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry on any transient fetch failure
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"pulselive fetch failed after {attempts} attempts") from last_err


def current_comp_season_id() -> int:
    """The most recent competition-season id pulselive knows about for the
    Premier League — sorted desc by the API itself, so `content[0]` is
    always "whatever season is current," no year-number bookkeeping needed
    on this side (unlike football_data.py's CURRENT_SEASON_START_YEAR,
    which has to be bumped by hand each August)."""

    def _get():
        resp = requests.get(
            f"{PULSELIVE_BASE_URL}/competitions/{PULSELIVE_COMPETITION_ID}/compseasons",
            params={"page": 0, "size": 1, "sort": "desc"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    data = _fetch_with_retry(_get)
    return int(data["content"][0]["id"])


def fetch_fixtures(comp_season_id: int | None = None) -> pd.DataFrame:
    """Every fixture in one season (finished and upcoming), paginated.
    Always fetched fresh (no file cache) — this is meant to be current, and
    the request cost is only ~4 pages. Columns: pulselive_id, date, season,
    team_home, team_away, team_home_id, team_away_id, goals_home,
    goals_away, finished, gameweek."""
    comp_season_id = comp_season_id or current_comp_season_id()

    rows = []
    page = 0
    while True:

        def _get(page=page):
            resp = requests.get(
                f"{PULSELIVE_BASE_URL}/fixtures",
                params={"compSeasons": comp_season_id, "page": page, "pageSize": 100, "sort": "asc"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

        data = _fetch_with_retry(_get)
        for f in data["content"]:
            teams = f["teams"]
            home, away = teams[0], teams[1]
            finished = f["status"] == "C"
            rows.append(
                {
                    "pulselive_id": f["id"],
                    "date": f["kickoff"]["millis"],
                    "team_home": to_canonical(home["team"]["name"], source="pulselive"),
                    "team_away": to_canonical(away["team"]["name"], source="pulselive"),
                    "team_home_id": home["team"]["club"]["id"],
                    "team_away_id": away["team"]["club"]["id"],
                    "goals_home": home["score"] if finished else None,
                    "goals_away": away["score"] if finished else None,
                    "finished": finished,
                    "gameweek": f["gameweek"]["gameweek"],
                }
            )
        if page + 1 >= data["pageInfo"]["numPages"]:
            break
        page += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], unit="ms")
    return df


def _match_cache_path(pulselive_id: int, kind: str) -> Path:
    return PULSELIVE_CACHE_DIR / f"{kind}_{pulselive_id}.json"


def fetch_match_stats(pulselive_id: int, force_refresh: bool = False) -> dict:
    """Cache-or-fetch one match's team-level stats, already reduced to the
    football-data.co.uk-style short keys (`_STAT_MAP`). Returns
    `{team_id_str: {"s": ..., "st": ..., "c": ..., "f": ..., "y": ..., "r": ...}}`.
    Only ever called for *finished* matches — a finished match's stats are
    immutable, so this caches forever once fetched, same as every other
    per-match cache in this codebase."""
    cache_path = _match_cache_path(pulselive_id, "stats")
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())

    def _get():
        resp = requests.get(f"{PULSELIVE_BASE_URL}/stats/match/{pulselive_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    data = _fetch_with_retry(_get)
    result = {}
    for team_id, block in data.get("data", {}).items():
        stat_values = {m["name"]: m.get("value") for m in block.get("M", [])}
        values = {short: stat_values.get(long) for long, short in _STAT_MAP.items()}
        values["possession"] = values["possession"] if values["possession"] is not None else stat_values.get("possession")
        result[team_id] = values

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def fetch_match_referee(pulselive_id: int, force_refresh: bool = False) -> str | None:
    """Cache-or-fetch one match's main referee's display name, from the
    fixture-detail endpoint's `matchOfficials` list. Returns None if no
    "MAIN" official is listed (shouldn't normally happen for a finished
    match, but never assumed)."""
    cache_path = _match_cache_path(pulselive_id, "referee")
    if cache_path.exists() and not force_refresh:
        cached = json.loads(cache_path.read_text())
        return cached["referee"]

    def _get():
        resp = requests.get(f"{PULSELIVE_BASE_URL}/fixtures/{pulselive_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    data = _fetch_with_retry(_get)
    referee = None
    for official in data.get("matchOfficials", []):
        if official.get("role") == "MAIN":
            referee = official.get("name", {}).get("display")
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"referee": referee}))
    return referee


def fetch_current_season_matches(comp_season_id: int | None = None) -> pd.DataFrame:
    """Finished matches of the current season, in the same column shape as
    `football_data.py::fetch_season` (so it's a drop-in supplement for
    `fetch_current_season_partial`): date, season, team_home, team_away,
    goals_home, goals_away, ftr, hs, as, hst, ast, hf, af, hc, ac, hy, ay,
    hr, ar, referee. Fetches (and caches) per-match stats/referee only for
    matches not already cached — cheap and gets cheaper every call as more
    of the season becomes cached."""
    fixtures = fetch_fixtures(comp_season_id)
    if fixtures.empty:
        return fixtures

    finished = fixtures[fixtures["finished"]].copy()
    if finished.empty:
        return pd.DataFrame(
            columns=[
                "date", "team_home", "team_away", "goals_home", "goals_away", "ftr",
                "hs", "as", "hst", "ast", "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar", "referee", "season",
            ]
        )

    season_label = None
    rows = []
    for _, m in finished.iterrows():
        stats = fetch_match_stats(int(m["pulselive_id"]))
        referee = fetch_match_referee(int(m["pulselive_id"]))
        home_stats = stats.get(str(int(m["team_home_id"])), {})
        away_stats = stats.get(str(int(m["team_away_id"])), {})
        gh, ga = m["goals_home"], m["goals_away"]
        ftr = "H" if gh > ga else "A" if ga > gh else "D"
        rows.append(
            {
                "date": m["date"],
                "team_home": m["team_home"],
                "team_away": m["team_away"],
                "goals_home": gh,
                "goals_away": ga,
                "ftr": ftr,
                "hs": home_stats.get("s"),
                "as": away_stats.get("s"),
                "hst": home_stats.get("st"),
                "ast": away_stats.get("st"),
                "hf": home_stats.get("f"),
                "af": away_stats.get("f"),
                "hc": home_stats.get("c"),
                "ac": away_stats.get("c"),
                "hy": home_stats.get("y"),
                "ay": away_stats.get("y"),
                "hr": home_stats.get("r"),
                "ar": away_stats.get("r"),
                "hp": home_stats.get("possession"),
                "ap": away_stats.get("possession"),
                "referee": referee,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = season_str(CURRENT_SEASON_START_YEAR)
    # A count stat absent from pulselive's per-match `M` list appears to
    # mean "zero occurred" rather than "unknown" (confirmed pattern:
    # total_red_card is consistently absent, not present-as-0, for a team
    # with no red card) — applies the same way to corners/cards/shots.
    for col in ("hs", "as", "hst", "ast", "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar"):
        df[col] = df[col].fillna(0)
    return df.sort_values("date").reset_index(drop=True)
