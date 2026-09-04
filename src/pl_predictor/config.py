"""config.py — paths, env loading, and shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

# Public, read-only deployment — unset/false everywhere else, which
# reproduces this app's original private-network-only behavior exactly
# (api/main.py's own CORS comment). No login gate: the public deployment
# has no admin surface reachable at all (routes.py::_admin_only 404s the
# write endpoints regardless), so there's nothing a password would protect.
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"

# Precomputed data the public deployment serves instead of running this
# project's live-serving pipeline itself (see public_snapshot.py) —
# confirmed live that doing the real computation on a free-tier host's
# memory budget doesn't work. Generated locally (`python -m
# pl_predictor.public_snapshot`) and committed/pushed like any other
# tracked file; the local copy on disk is only the cold-start fallback for
# a freshly-built container — see PUBLIC_SNAPSHOT_REFRESH_URL below for how
# a running process actually stays current.
PUBLIC_SNAPSHOT_PATH = DATA_DIR / "public_snapshot.json"

# A running public-deployment process polls this raw-file URL (see
# api/routes.py::refresh_public_snapshot_from_remote) instead of relying on
# a full redeploy to pick up each new snapshot — the GitHub Actions refresh
# (.github/workflows/refresh-public-snapshot.yml) pushes a new
# data/public_snapshot.json several times a day, and rebuilding the whole
# Docker image (npm ci + pip install) for a pure data change was needless
# churn on top of this host's own idle-sleep cycle. Render's auto-deploy
# still fires on an actual code push; only THIS one data path is meant to
# be excluded from that (Dashboard -> service -> Settings -> Build Filters
# -> Ignored Paths -> `data/public_snapshot.json`, set once, by hand — no
# Render API for it at the time this was written).
PUBLIC_SNAPSHOT_REFRESH_URL = os.getenv(
    "PUBLIC_SNAPSHOT_REFRESH_URL",
    "https://raw.githubusercontent.com/Kevocado/PL_Predictor/main/data/public_snapshot.json",
)
PUBLIC_SNAPSHOT_POLL_SECONDS = int(os.getenv("PUBLIC_SNAPSHOT_POLL_SECONDS", "300"))

FOOTBALL_DATA_CACHE_DIR = CACHE_DIR / "football_data"
ODDS_CACHE_DIR = CACHE_DIR / "odds"
FPL_HISTORY_CACHE_DIR = CACHE_DIR / "fpl_history"
FPL_PLAYER_CACHE_DIR = CACHE_DIR / "fpl_players"
FPL_EVENT_CACHE_DIR = CACHE_DIR / "fpl_events"
FPL_RESEARCH_CACHE_DIR = CACHE_DIR / "fpl_research"
ODDS_SNAPSHOT_DIR = CACHE_DIR / "odds_snapshots"
UNDERSTAT_CACHE_DIR = CACHE_DIR / "understat"
# One file per MATCH, not per season (unlike every other cache dir here) —
# see data/understat_shots.py's module docstring.
UNDERSTAT_SHOTS_CACHE_DIR = CACHE_DIR / "understat_shots"
PULSELIVE_CACHE_DIR = CACHE_DIR / "pulselive"
CLUBELO_CACHE_DIR = CACHE_DIR / "clubelo"
OTHER_COMPETITIONS_CACHE_DIR = CACHE_DIR / "other_competitions"

COMPETITION = "ENG Premier League"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_SPORT_KEY = "soccer_epl"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FPL_API_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_HISTORY_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# football-data.org: current-season fixtures/results/standings in one call
# each, no per-day workaround needed (unlike API-Football's free tier,
# which walls off the current season entirely) — see data/football_data_org.py.
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")
FOOTBALL_DATA_ORG_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_ORG_COMPETITION_ID = 2021  # Premier League

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"

# Same ESPN site API `data/espn.py` already uses for PL lineups, just other
# competitions' league slugs — used by `data/other_competitions.py` for
# fixture-congestion data (Europa League/Conference League/domestic cups
# aren't on football-data.org's free tier; see that module's docstring).
# Confirmed live (2026-08-28, see EXP-2026-17 in docs/AI_CONTINUITY.md) —
# `limit=1000` is required in the request or ESPN's default 100-event cap
# returns qualifying-round matches from other countries before ever
# reaching a Premier League club's own fixture.
ESPN_SOCCER_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_CUP_COMPETITIONS = {
    "Europa League": "uefa.europa",
    "Conference League": "uefa.europa.conf",
    "FA Cup": "eng.fa",
    "EFL Cup": "eng.league_cup",
}

# pulselive.com: the Premier League's own site backend — no key, no
# documented terms. Used only as a fast current-season supplement to
# football-data.co.uk (whose CSV publishing can lag indefinitely) — private,
# non-redistributed personal use only. See data/pulselive.py.
PULSELIVE_BASE_URL = "https://footballapi.pulselive.com/football"
PULSELIVE_COMPETITION_ID = 1  # Premier League

# clubelo.com: free, no key. Terms/licence for reuse have NOT been verified
# yet (the site was unreachable from this project's development environment
# while this was written — see below) — confirm them before any live/
# redistributed use, same discipline docs/AI_CONTINUITY.md's free-data
# research table applies to every other source. Cross-league ratings used
# only as a promoted-team cold-start prior — see data/clubelo.py and
# features/promoted_team_prior.py. As of this writing the API's own server
# has not been reachable from this project's development environment (TCP
# connects, no response) — code is written and unit-tested against a
# mocked response, but has not yet run against a real fetch; see
# docs/AI_CONTINUITY.md EXP-2026-05 for status.
CLUBELO_BASE_URL = "http://api.clubelo.com"

# Not in cache/: unlike everything else there, this can't be regenerated by
# re-fetching — it's the accumulated record of what the model predicted
# *before* each match kicked off, which only exists if it was captured at
# the time.
TRACKING_DB_PATH = DATA_DIR / "tracking.db"

for _d in (
    DATA_DIR,
    CACHE_DIR,
    MODELS_DIR,
    FOOTBALL_DATA_CACHE_DIR,
    ODDS_CACHE_DIR,
    FPL_HISTORY_CACHE_DIR,
    FPL_PLAYER_CACHE_DIR,
    FPL_EVENT_CACHE_DIR,
    FPL_RESEARCH_CACHE_DIR,
    ODDS_SNAPSHOT_DIR,
    UNDERSTAT_CACHE_DIR,
    UNDERSTAT_SHOTS_CACHE_DIR,
    PULSELIVE_CACHE_DIR,
    CLUBELO_CACHE_DIR,
    OTHER_COMPETITIONS_CACHE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
