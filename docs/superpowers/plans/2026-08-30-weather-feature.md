# Weather Feature (EXP-2026-22) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether kickoff-day weather (wind, rain, temperature) at the
home team's stadium, sourced from Open-Meteo's free Historical Forecast
archive, improves `ml_scoreline`'s walk-forward RPS/Brier/log loss —
following this project's standard "compute as a candidate feature via
`extra_feature_frame`, evaluate, promote or reject and document either way"
protocol, the same shape as EXP-2026-18 (squad continuity).

**Architecture:** A static stadium-coordinates lookup
(`data/stadiums.py`) feeds a cached Open-Meteo client (`data/open_meteo.py`,
same fetch/cache/retry pattern as `data/clubelo.py`) which a new feature
module (`features/weather.py`) turns into one row of kickoff-day weather per
historical fixture. A new evaluation script
(`evaluate/weather_prior.py`, mirroring `evaluate/squad_change_prior.py`'s
CLI shape) wires this into `evaluate.walk_forward.prepare_folds`'s existing
`extra_feature_frame` mechanism to run the standard 5-fold walk-forward
comparison, printing per-season data coverage before any metric is trusted
(Open-Meteo's forecast archive only reaches back to ~2022, so 1-2 of the
5 folds may have partial coverage — this project's own EXP-2026-04/17
history shows an asymmetric-coverage feature can look like noise, so
coverage must be reported, not assumed away).

**Tech Stack:** Python, `pandas`, `requests` (already project dependencies —
no new packages needed).

**Spec:** No separate spec file — the architecture above was agreed directly
in chat (see the conversation's design discussion immediately preceding this
plan) and approved via `AskUserQuestion`, the same lightweight path
EXP-2026-18's own plan took.

## Global Constraints

- No-lookahead: only weather data that would have been knowable at or before
  a fixture's actual kickoff date may be used. Open-Meteo's *Historical
  Forecast* archive (`https://historical-forecast-api.open-meteo.com/v1/forecast`)
  is the correct source for this — it replays what was actually forecast at
  the time, unlike the ERA5 reanalysis archive, which is hindsight (see
  `docs/AI_CONTINUITY.md`'s "Free data research" table, Open-Meteo row).
  Never call the reanalysis/ERA5 archive endpoint for this feature.
- Attribution: Open-Meteo's non-commercial free tier is CC-BY 4.0 — requires
  crediting the source. Add an attribution line to `README.md`'s existing
  data-sources section (wherever `data/clubelo.py`/`data/understat.py`'s
  sources are already credited, if such a section exists — otherwise add
  one) as part of Task 4.
- Rate limit: 10,000 calls/day, 5,000/hour, 600/minute on the free tier
  (confirmed via `https://open-meteo.com/en/terms`). This plan's fetch
  pattern is one call per (team, season) — about 100 calls total for a full
  5-season backfill of ~20 clubs — nowhere near the limit; never fetch
  per-fixture (would be ~1,900 calls and provides no benefit since Open-Meteo
  returns a full date-range's daily data in one call).
- Never merge this feature into `features/build.py`'s production
  `feature_cols` unless it clears the promotion gate in
  `docs/AI_CONTINUITY.md` (average walk-forward AND fixed most-recent-season
  holdout both improve, no material fold regression).
- Follow existing caching conventions exactly: new cache directories are
  declared in `src/pl_predictor/config.py` alongside the existing
  `CLUBELO_CACHE_DIR`/`ODDS_CACHE_DIR` pattern, and added to the module-level
  list of directories that get created on import.

---

### Task 1: Stadium coordinates lookup

**Files:**
- Create: `src/pl_predictor/data/stadiums.py`
- Test: `tests/test_stadiums.py`

**Interfaces:**
- Produces: `STADIUM_COORDS: dict[str, tuple[float, float]]` (canonical team
  name → (latitude, longitude)); `stadium_location(team: str, season: str) ->
  tuple[float, float] | None` — later tasks call this, never `STADIUM_COORDS`
  directly, so a team's stadium can vary by season (Everton moved from
  Goodison Park to Everton Stadium/Bramley-Moore Dock for the 2025-26 season)
  without every caller needing to know that.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stadiums.py
from pl_predictor.data import stadiums


def test_stadium_location_returns_known_team_coords():
    lat, lon = stadiums.stadium_location("Arsenal", "2024-2025")
    assert lat == 51.5549
    assert lon == -0.1084


def test_stadium_location_returns_none_for_unknown_team():
    assert stadiums.stadium_location("Not A Real Club", "2024-2025") is None


def test_everton_stadium_changes_for_2025_26_onward():
    """Everton moved from Goodison Park to Everton Stadium (Bramley-Moore
    Dock) for the 2025-26 season — a fixture's weather must use whichever
    ground was actually in use that season, not a single fixed location."""
    old_lat, old_lon = stadiums.stadium_location("Everton", "2024-2025")
    new_lat, new_lon = stadiums.stadium_location("Everton", "2025-2026")
    assert (old_lat, old_lon) != (new_lat, new_lon)
    assert old_lat == 53.4388  # Goodison Park
    assert new_lat == 53.4494  # Everton Stadium
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stadiums.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pl_predictor.data.stadiums'`

- [ ] **Step 3: Write the implementation**

```python
# src/pl_predictor/data/stadiums.py
"""stadiums.py — static stadium latitude/longitude for every club that has
appeared in the Premier League during this project's standard training
window (see `data.football_data.default_completed_seasons`), keyed by this
project's canonical team names (`data.team_names.to_canonical`).

Used only to look up where a fixture was actually played, for
`features/weather.py`'s Open-Meteo lookup — no other module needs a club's
physical location. Coordinates are each ground's own listed location
(public, stable facts); verify against a current source before trusting a
club not already listed here (a newly promoted or newly-stadium-moved club).

A team can have more than one entry across seasons (see Everton below) —
always call `stadium_location(team, season)`, never index
`STADIUM_COORDS` directly.
"""

from __future__ import annotations

# (latitude, longitude) for each club's *current* ground, keyed by this
# project's canonical short name (see data/team_names.py::CANONICAL_TEAMS).
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Arsenal": (51.5549, -0.1084),
    "Aston Villa": (52.5092, -1.8848),
    "Bournemouth": (50.7352, -1.8380),
    "Brentford": (51.4907, -0.2887),
    "Brighton": (50.8617, -0.0837),
    "Burnley": (53.7890, -2.2302),
    "Cardiff": (51.4727, -3.2033),
    "Chelsea": (51.4816, -0.1910),
    "Coventry": (52.4483, -1.4954),
    "Crystal Palace": (51.3983, -0.0855),
    "Everton": (53.4494, -2.9622),  # Everton Stadium (Bramley-Moore Dock), 2025-26 onward
    "Fulham": (51.4750, -0.2217),
    "Huddersfield": (53.6541, -1.7681),
    "Hull": (53.7461, -0.3663),
    "Ipswich": (52.0553, 1.1451),
    "Leeds": (53.7778, -1.5722),
    "Leicester": (52.6204, -1.1422),
    "Liverpool": (53.4308, -2.9608),
    "Luton": (51.8844, -0.4326),
    "Man City": (53.4831, -2.2004),
    "Man United": (53.4631, -2.2913),
    "Middlesbrough": (54.5783, -1.2169),
    "Newcastle": (54.9756, -1.6217),
    "Norwich": (52.6223, 1.3095),
    "Nott'm Forest": (52.9400, -1.1327),
    "QPR": (51.5094, -0.2323),
    "Sheffield United": (53.3701, -1.4708),
    "Southampton": (50.9058, -1.3911),
    "Stoke": (52.9884, -2.1754),
    "Sunderland": (54.9144, -1.3883),
    "Swansea": (51.6428, -3.9351),
    "Tottenham": (51.6043, -0.0664),
    "Watford": (51.6499, -0.4013),
    "West Brom": (52.5092, -1.9639),
    "West Ham": (51.5386, -0.0166),
    "Wolves": (52.5902, -2.1301),
}

# Grounds that changed within the standard training window. Each entry:
# team -> list of (first_season_at_this_ground, (lat, lon)), oldest first.
# `stadium_location` uses the last entry whose season is <= the requested one.
_GROUND_CHANGES: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "Everton": [
        ("2015-2016", (53.4388, -2.9663)),  # Goodison Park
        ("2025-2026", (53.4494, -2.9622)),  # Everton Stadium
    ],
}


def stadium_location(team: str, season: str) -> tuple[float, float] | None:
    """`team`'s home-ground (latitude, longitude) as of `season`
    (football-data's `"YYYY-YYYY"` format, e.g. `"2024-2025"`). Returns
    `None` for a team with no known coordinates rather than raising —
    callers (features/weather.py) must degrade to NaN weather for that
    fixture, same as every other optional feature in this project."""
    changes = _GROUND_CHANGES.get(team)
    if changes is not None:
        applicable = [coords for start_season, coords in changes if start_season <= season]
        if applicable:
            return applicable[-1]
        return changes[0][1]
    return STADIUM_COORDS.get(team)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stadiums.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/data/stadiums.py tests/test_stadiums.py
git commit -m "feat: add stadium coordinates lookup for weather feature"
```

---

### Task 2: Open-Meteo client (fetch + cache)

**Files:**
- Create: `src/pl_predictor/data/open_meteo.py`
- Modify: `src/pl_predictor/config.py` (add `OPEN_METEO_BASE_URL`,
  `OPEN_METEO_CACHE_DIR`, and append `OPEN_METEO_CACHE_DIR` to the
  module-level list of cache directories created on import — follow the
  exact pattern already used for `CLUBELO_BASE_URL`/`CLUBELO_CACHE_DIR`)
- Test: `tests/test_open_meteo.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `fetch_weather_range(latitude: float, longitude: float,
  start_date: str, end_date: str, force_refresh: bool = False) ->
  pd.DataFrame` with columns `date`, `precip_mm`, `wind_kph`, `temp_max_c`,
  `temp_min_c` — one row per day in `[start_date, end_date]`. Later tasks
  (features/weather.py) call this per (stadium, season).

- [ ] **Step 1: Add config entries**

In `src/pl_predictor/config.py`, immediately after the existing
`CLUBELO_BASE_URL = "http://api.clubelo.com"` line, add:

```python
OPEN_METEO_BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
```

And immediately after `CLUBELO_CACHE_DIR = CACHE_DIR / "clubelo"`, add:

```python
OPEN_METEO_CACHE_DIR = CACHE_DIR / "open_meteo"
```

Then add `OPEN_METEO_CACHE_DIR` to the module-level list/tuple of cache
directories that already includes `CLUBELO_CACHE_DIR` (the block that
creates every cache dir on import — find it by searching for
`CLUBELO_CACHE_DIR,` in the file and add `OPEN_METEO_CACHE_DIR,` on its own
line right after it, matching the existing style exactly).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_open_meteo.py
import pandas as pd

from pl_predictor.data import open_meteo

_FAKE_RESPONSE = {
    "daily": {
        "time": ["2024-08-17", "2024-08-18", "2024-08-19"],
        "precipitation_sum": [0.0, 4.2, None],
        "wind_speed_10m_max": [12.3, 30.1, 8.0],
        "temperature_2m_max": [21.5, 18.0, 22.1],
        "temperature_2m_min": [12.0, 11.5, 13.0],
    }
}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_weather_range_parses_daily_arrays_into_one_row_per_date(monkeypatch, tmp_path):
    monkeypatch.setattr(open_meteo, "OPEN_METEO_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        open_meteo.requests, "get", lambda url, params, timeout: _FakeResponse(_FAKE_RESPONSE)
    )

    df = open_meteo.fetch_weather_range(51.5549, -0.1084, "2024-08-17", "2024-08-19")

    assert list(df["date"]) == ["2024-08-17", "2024-08-18", "2024-08-19"]
    assert df.iloc[0]["precip_mm"] == 0.0
    assert df.iloc[1]["wind_kph"] == 30.1
    assert pd.isna(df.iloc[2]["precip_mm"])  # a missing daily value must stay NaN, not crash


def test_fetch_weather_range_caches_and_does_not_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(open_meteo, "OPEN_METEO_CACHE_DIR", tmp_path)
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return _FakeResponse(_FAKE_RESPONSE)

    monkeypatch.setattr(open_meteo.requests, "get", fake_get)

    open_meteo.fetch_weather_range(51.5549, -0.1084, "2024-08-17", "2024-08-19")
    open_meteo.fetch_weather_range(51.5549, -0.1084, "2024-08-17", "2024-08-19")

    assert len(calls) == 1


def test_fetch_weather_range_never_raises_on_request_failure(monkeypatch, tmp_path):
    """Weather is a best-effort candidate feature, same discipline as
    other_competitions.py's fixture calendar — a network failure must
    degrade to an empty frame, not take a training run down with it."""
    monkeypatch.setattr(open_meteo, "OPEN_METEO_CACHE_DIR", tmp_path)

    def fake_get(url, params, timeout):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(open_meteo.requests, "get", fake_get)

    df = open_meteo.fetch_weather_range(51.5549, -0.1084, "2024-08-17", "2024-08-19")

    assert df.empty
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_open_meteo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pl_predictor.data.open_meteo'`

- [ ] **Step 4: Write the implementation**

```python
# src/pl_predictor/data/open_meteo.py
"""open_meteo.py — kickoff-day weather via Open-Meteo's Historical Forecast
API (https://open-meteo.com/en/docs/historical-forecast-api).

Deliberately uses the *forecast archive* endpoint
(`historical-forecast-api.open-meteo.com`), not the ERA5 reanalysis
archive — the forecast archive replays what was actually forecast at the
time, which is the correct no-lookahead source; reanalysis is hindsight
(see docs/AI_CONTINUITY.md's "Free data research" table, Open-Meteo row).

Coverage starts around 2022 for most source models — a fixture before that
gets an empty result, not an error; `features/weather.py` and
`evaluate/weather_prior.py` are responsible for reporting that coverage gap
rather than silently averaging over it.

Free-tier non-commercial terms (https://open-meteo.com/en/terms): under
10,000 calls/day, attribution required (CC-BY 4.0) — this project's own
attribution lives in README.md alongside its other data-source credits.
Cached to disk per (lat, lon, start_date, end_date) forever, same pattern
as `clubelo.py`'s per-date cache (a past date's weather never changes once
fetched).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from ..config import OPEN_METEO_BASE_URL, OPEN_METEO_CACHE_DIR

_DAILY_VARS = "precipitation_sum,wind_speed_10m_max,temperature_2m_max,temperature_2m_min"


def _cache_path(latitude: float, longitude: float, start_date: str, end_date: str) -> Path:
    return OPEN_METEO_CACHE_DIR / f"{latitude:.4f}_{longitude:.4f}_{start_date}_{end_date}.json"


def fetch_weather_range(
    latitude: float, longitude: float, start_date: str, end_date: str, force_refresh: bool = False
) -> pd.DataFrame:
    """Daily weather at (`latitude`, `longitude`) for every date in
    `[start_date, end_date]` (both `"YYYY-MM-DD"`). Columns: `date`,
    `precip_mm`, `wind_kph`, `temp_max_c`, `temp_min_c`. Returns an empty
    DataFrame (never raises) on any request failure or when Open-Meteo has
    no coverage for the requested range — this is a best-effort candidate
    feature, matching `data/other_competitions.py`'s degrade-to-empty
    discipline."""
    cache_path = _cache_path(latitude, longitude, start_date, end_date)
    if cache_path.exists() and not force_refresh:
        payload = json.loads(cache_path.read_text())
    else:
        try:
            resp = requests.get(
                OPEN_METEO_BASE_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": _DAILY_VARS,
                    "timezone": "UTC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:  # noqa: BLE001 - best-effort candidate feature, never take a caller down
            return pd.DataFrame(columns=["date", "precip_mm", "wind_kph", "temp_max_c", "temp_min_c"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))

    daily = payload.get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame(columns=["date", "precip_mm", "wind_kph", "temp_max_c", "temp_min_c"])

    return pd.DataFrame(
        {
            "date": daily["time"],
            "precip_mm": daily.get("precipitation_sum", [None] * len(daily["time"])),
            "wind_kph": daily.get("wind_speed_10m_max", [None] * len(daily["time"])),
            "temp_max_c": daily.get("temperature_2m_max", [None] * len(daily["time"])),
            "temp_min_c": daily.get("temperature_2m_min", [None] * len(daily["time"])),
        }
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_open_meteo.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pl_predictor/config.py src/pl_predictor/data/open_meteo.py tests/test_open_meteo.py
git commit -m "feat: add Open-Meteo historical-forecast client with disk caching"
```

---

### Task 3: Fixture weather feature

**Files:**
- Create: `src/pl_predictor/features/weather.py`
- Test: `tests/test_weather.py`

**Interfaces:**
- Consumes: `stadiums.stadium_location(team, season)` (Task 1),
  `open_meteo.fetch_weather_range(lat, lon, start_date, end_date)` (Task 2).
- Produces: `fixture_weather_features(matches_df: pd.DataFrame) ->
  pd.DataFrame` with columns `kickoff_date`, `team_home`, `team_away`,
  `precip_mm`, `wind_kph`, `temp_max_c`, `temp_min_c` — one row per fixture
  in `matches_df`, shaped to merge directly onto `evaluate.walk_forward.
  prepare_folds`'s default `extra_feature_frame` merge keys
  (`kickoff_date`, `team_home`, `team_away`). `matches_df` needs `season`,
  `date`, `team_home`, `team_away` columns (same shape
  `evaluate/squad_change_prior.py` already assumes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weather.py
import pandas as pd

from pl_predictor.features import weather


def _matches_df():
    return pd.DataFrame(
        [
            {"season": "2024-2025", "date": pd.Timestamp("2024-08-17"), "team_home": "Arsenal", "team_away": "Chelsea"},
            {"season": "2024-2025", "date": pd.Timestamp("2024-08-18"), "team_home": "Chelsea", "team_away": "Arsenal"},
        ]
    )


def test_fixture_weather_features_joins_home_teams_stadium_weather_by_date(monkeypatch):
    def fake_fetch(latitude, longitude, start_date, end_date, force_refresh=False):
        return pd.DataFrame(
            {
                "date": ["2024-08-17", "2024-08-18"],
                "precip_mm": [0.0, 5.0],
                "wind_kph": [10.0, 25.0],
                "temp_max_c": [20.0, 19.0],
                "temp_min_c": [12.0, 11.0],
            }
        )

    monkeypatch.setattr(weather.open_meteo, "fetch_weather_range", fake_fetch)

    result = weather.fixture_weather_features(_matches_df())

    arsenal_home = result[(result["team_home"] == "Arsenal") & (result["team_away"] == "Chelsea")].iloc[0]
    assert arsenal_home["precip_mm"] == 0.0
    assert arsenal_home["wind_kph"] == 10.0

    chelsea_home = result[(result["team_home"] == "Chelsea") & (result["team_away"] == "Arsenal")].iloc[0]
    assert chelsea_home["precip_mm"] == 5.0  # a different stadium/date -> different weather


def test_fixture_weather_features_degrades_to_nan_for_unknown_team(monkeypatch):
    monkeypatch.setattr(
        weather.open_meteo, "fetch_weather_range", lambda *a, **k: pd.DataFrame(columns=["date", "precip_mm", "wind_kph", "temp_max_c", "temp_min_c"])
    )
    matches = pd.DataFrame(
        [{"season": "2024-2025", "date": pd.Timestamp("2024-08-17"), "team_home": "Not A Real Club", "team_away": "Chelsea"}]
    )

    result = weather.fixture_weather_features(matches)

    assert pd.isna(result.iloc[0]["precip_mm"])


def test_fixture_weather_features_output_matches_walk_forward_merge_keys(monkeypatch):
    monkeypatch.setattr(
        weather.open_meteo, "fetch_weather_range", lambda *a, **k: pd.DataFrame(columns=["date", "precip_mm", "wind_kph", "temp_max_c", "temp_min_c"])
    )
    result = weather.fixture_weather_features(_matches_df())
    assert {"kickoff_date", "team_home", "team_away", "precip_mm", "wind_kph", "temp_max_c", "temp_min_c"} == set(result.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_weather.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pl_predictor.features.weather'`

- [ ] **Step 3: Write the implementation**

```python
# src/pl_predictor/features/weather.py
"""weather.py — kickoff-day weather (rain, wind, temperature) at the home
team's stadium, as a candidate no-lookahead feature (see docs/
AI_CONTINUITY.md EXP-2026-22 and data/open_meteo.py's own docstring for the
no-lookahead reasoning). Batches one Open-Meteo call per (team, season) —
not per fixture — since the API returns a full date range in one call and
this project's fixtures for a team/season all fall within that team's
season-length home-fixture window.

Weather affects both teams in a fixture equally (it is a property of the
match, not either team specifically), so this is a single set of
match-level columns, not home_/away_-prefixed like most other features in
`features/build.py`.
"""

from __future__ import annotations

import pandas as pd

from ..data import open_meteo, stadiums

_WEATHER_COLS = ["precip_mm", "wind_kph", "temp_max_c", "temp_min_c"]


def fixture_weather_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture in `matches_df` (needs `season`, `date`,
    `team_home`, `team_away`) with kickoff-day weather at the home team's
    stadium. NaN for a team with no known stadium (`data/stadiums.py`) or a
    date outside Open-Meteo's forecast-archive coverage — never raises."""
    df = matches_df.copy()
    df["kickoff_date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["_date_str"] = df["kickoff_date"].dt.strftime("%Y-%m-%d")

    weather_frames = []
    for (season, team), group in df.groupby(["season", "team_home"]):
        location = stadiums.stadium_location(team, season)
        if location is None:
            continue
        latitude, longitude = location
        start_date, end_date = group["_date_str"].min(), group["_date_str"].max()
        season_weather = open_meteo.fetch_weather_range(latitude, longitude, start_date, end_date)
        if season_weather.empty:
            continue
        season_weather = season_weather.rename(columns={"date": "_date_str"})
        matched = group[["kickoff_date", "team_home", "team_away", "_date_str"]].merge(
            season_weather, on="_date_str", how="left"
        )
        weather_frames.append(matched)

    if not weather_frames:
        result = df[["kickoff_date", "team_home", "team_away"]].copy()
        for col in _WEATHER_COLS:
            result[col] = pd.NA
        return result

    matched_all = pd.concat(weather_frames, ignore_index=True)
    result = df[["kickoff_date", "team_home", "team_away"]].merge(
        matched_all[["kickoff_date", "team_home", "team_away"] + _WEATHER_COLS],
        on=["kickoff_date", "team_home", "team_away"],
        how="left",
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_weather.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/features/weather.py tests/test_weather.py
git commit -m "feat: add fixture-level kickoff-day weather feature"
```

---

### Task 4: Walk-forward evaluation script + attribution

**Files:**
- Create: `src/pl_predictor/evaluate/weather_prior.py`
- Modify: `README.md` (add an Open-Meteo attribution line next to the
  project's other data-source credits — find where `data/clubelo.py` or
  `data/understat.py`'s sources are already mentioned, if present; if no
  such section exists, add one short "Data sources" bullet list)

**Interfaces:**
- Consumes: `walk_forward.prepare_folds` (existing), `weather.
  fixture_weather_features` (Task 3).
- Produces: a `run(seasons=None, min_train_seasons=3) -> dict` result and a
  `__main__` block printing per-season coverage and metrics, mirroring
  `evaluate/squad_change_prior.py`'s shape exactly.

- [ ] **Step 1: Write the evaluation script**

```python
# src/pl_predictor/evaluate/weather_prior.py
"""weather_prior.py — walk-forward evaluation of the kickoff-day weather
candidate feature (see docs/AI_CONTINUITY.md EXP-2026-22 and
`features/weather.py`'s own docstring).

Open-Meteo's forecast archive only reaches back to ~2022 (confirmed via
https://open-meteo.com/en/docs/historical-forecast-api), while this
project's standard walk-forward window starts at 2021-22 — so one or more
of the earliest folds may have partial or zero weather coverage. Per this
project's own EXP-2026-04/17 lesson (an asymmetric-coverage feature can
look like pure noise via SHAP-invisible tree-structure changes, not a real
effect), per-season coverage is printed before any metric is trusted, and
the comparison is run twice: once across the full standard window, and once
restricted to seasons with usable coverage.
"""

from __future__ import annotations

import pandas as pd

from ..data import football_data
from ..evaluate import walk_forward
from ..features import weather
from ..models import ml_scoreline

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}
MIN_COVERAGE = 0.90  # a season needs at least this fraction of fixtures with real (non-NaN) weather to count as "usable"


def _extra_feature_frame(seasons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    matches_df = football_data.load_training_data(seasons=seasons)
    extra = weather.fixture_weather_features(matches_df)
    return extra, ["precip_mm", "wind_kph", "temp_max_c", "temp_min_c"]


def _season_coverage(matches_df: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.copy()
    df["kickoff_date"] = pd.to_datetime(df["date"]).dt.normalize()
    merged = df.merge(extra, on=["kickoff_date", "team_home", "team_away"], how="left")
    coverage = merged.groupby("season").apply(lambda g: g["precip_mm"].notna().mean())
    return coverage.rename("coverage").reset_index()


def _rps_brier(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {"rps": float("nan"), "brier": float("nan"), "n": 0}
    import penaltyblog as pb

    probs = preds[["home_win", "draw", "away_win"]].to_numpy()
    outcomes = preds["ftr"].map(_RESULT_CODE).to_numpy()
    return {
        "rps": float(pb.metrics.rps_average(probs, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
        "n": int(len(preds)),
    }


def _fold_predictions(fold: dict) -> pd.DataFrame:
    home_model, away_model = ml_scoreline.train_goal_regressors(
        fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
    )
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, fold["X_val"])
    val_df = fold["val_df"].reset_index(drop=True)
    return pd.DataFrame(
        {
            "season": val_df["season"].to_numpy(),
            "ftr": val_df["ftr"].to_numpy(),
            "home_win": [g.home_win for g in grids],
            "draw": [g.draw for g in grids],
            "away_win": [g.away_win for g in grids],
        }
    )


def run(seasons: list[str] | None = None, min_train_seasons: int = 3) -> dict:
    seasons = seasons or football_data.default_completed_seasons(n=8)
    extra_frame, extra_cols = _extra_feature_frame(seasons)
    matches_df = football_data.load_training_data(seasons=seasons)
    coverage = _season_coverage(matches_df, extra_frame)

    baseline_folds = walk_forward.prepare_folds(seasons, min_train_seasons)
    candidate_folds = walk_forward.prepare_folds(
        seasons, min_train_seasons, extra_feature_frame=extra_frame, extra_feature_cols=extra_cols
    )

    baseline_preds = pd.concat([_fold_predictions(f) for f in baseline_folds], ignore_index=True)
    candidate_preds = pd.concat([_fold_predictions(f) for f in candidate_folds], ignore_index=True)

    usable_seasons = set(coverage[coverage["coverage"] >= MIN_COVERAGE]["season"])
    restricted_baseline = baseline_preds[baseline_preds["season"].isin(usable_seasons)]
    restricted_candidate = candidate_preds[candidate_preds["season"].isin(usable_seasons)]

    return {
        "coverage_by_season": coverage,
        "usable_seasons": sorted(usable_seasons),
        "overall_baseline": _rps_brier(baseline_preds),
        "overall_candidate": _rps_brier(candidate_preds),
        "restricted_baseline": _rps_brier(restricted_baseline),
        "restricted_candidate": _rps_brier(restricted_candidate),
        "per_season_baseline": baseline_preds.groupby("season").apply(lambda g: pd.Series(_rps_brier(g))),
        "per_season_candidate": candidate_preds.groupby("season").apply(lambda g: pd.Series(_rps_brier(g))),
    }


if __name__ == "__main__":
    results = run()

    print("=== Weather data coverage by season (fraction of fixtures with a real, non-NaN reading) ===")
    print(results["coverage_by_season"].to_string(index=False))
    print(f"\nSeasons with >= {MIN_COVERAGE:.0%} coverage: {results['usable_seasons']}")

    print("\n=== Overall (every fixture, full 8-season window) ===")
    print("baseline: ", results["overall_baseline"])
    print("candidate:", results["overall_candidate"])

    print("\n=== Coverage-restricted (usable-coverage seasons only) ===")
    print("baseline: ", results["restricted_baseline"])
    print("candidate:", results["restricted_candidate"])

    print("\n=== Per-season, overall (most-recent-season corroboration check) ===")
    print("baseline:")
    print(results["per_season_baseline"])
    print("candidate:")
    print(results["per_season_candidate"])
```

- [ ] **Step 2: Add attribution**

In `README.md`, find the section listing this project's external data
sources (search for `understat` or `clubelo` in the README to locate it).
Add a bullet: `- Weather: [Open-Meteo](https://open-meteo.com/) (CC-BY 4.0)`.
If no such section exists, add a short `## Data sources` section near the
bottom of the README listing this line alongside the project's other major
external sources (football-data.co.uk, Understat, the official FPL API, The
Odds API) so attribution is genuinely visible, not just in a docstring.

- [ ] **Step 3: Commit**

```bash
git add src/pl_predictor/evaluate/weather_prior.py README.md
git commit -m "feat: add weather walk-forward evaluation script and attribution"
```

---

### Task 5: Run the evaluation and document the result

**Files:**
- Modify: `docs/AI_CONTINUITY.md`

- [ ] **Step 1: Run the evaluation**

Run: `python -m pl_predictor.evaluate.weather_prior`

This performs a real backfill (~100 Open-Meteo calls, cached forever
afterward) and prints coverage, overall, coverage-restricted, and
per-season RPS/Brier for baseline vs. candidate.

- [ ] **Step 2: Apply the promotion gate**

Per `docs/AI_CONTINUITY.md`'s "Non-negotiable research protocol" (item 5):
promote only if the candidate improves the average AND the fixed
most-recent-season (2025-26) holdout, without a material fold regression.
Check both the full-window and coverage-restricted results — if the two
disagree, the coverage-restricted result is the one to trust (it excludes
seasons where the "improvement" could just be filling early-season NaNs
with the league's typical XGBoost missing-value handling rather than real
signal).

- [ ] **Step 3: Write the EXP-2026-22 entry**

Add a new entry to `docs/AI_CONTINUITY.md`'s "Completed experiment log"
(insert it before the "## Change checklist for future agents" heading, in
the same style as the existing EXP-2026-18/20/21 entries), following the
project's own experiment log template (see the `### EXP-YYYY-NN` template
a few sections above the log). Include: the question, why this isn't a
re-run of anything already rejected, the data/coverage caveat and its
actual measured coverage-by-season numbers, the full results table (overall
and coverage-restricted, baseline vs candidate, per-season), and an
explicit **Decision:** line — promoted or rejected, with the reasoning, per
the gate in Step 2. Write this up regardless of outcome; this project never
deletes a negative result.

- [ ] **Step 4: If promoted, wire into production (skip this step if rejected)**

Only if the gate in Step 2 passed: add `weather.fixture_weather_features`'s
merge into `features/build.py::build_training_frame` and
`FixtureFeatureContext`/`build_row` (follow the exact pattern
`squad_change`'s merge already uses in that file — merge by
`kickoff_date`/`team_home`/`team_away` for the training path, and add a
live equivalent to `FixtureFeatureContext.build_row` that looks up the
current season's weather forecast for `commence_time`), add the new
columns to `feature_cols`, retrain via the project's existing retrain
entrypoint, and re-run the fixed holdout check to confirm the production
retrain's numbers match what the walk-forward predicted (same verification
EXP-2026-18 did). This step intentionally has no pre-written code — it
depends entirely on Step 2's actual result, which isn't known yet.

- [ ] **Step 5: Commit**

```bash
git add docs/AI_CONTINUITY.md
git commit -m "docs: record EXP-2026-22 weather feature result"
```

(If Step 4 also produced changes, include `src/pl_predictor/features/build.py`,
`models/manifest.json`, and any other touched files in that same commit or a
follow-up one, matching however this project's other promoted-feature
commits are structured — see EXP-2026-18's commit history for reference.)
