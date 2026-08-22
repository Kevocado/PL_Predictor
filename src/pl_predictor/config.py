"""config.py — paths, env loading, and shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

FOOTBALL_DATA_CACHE_DIR = CACHE_DIR / "football_data"
ODDS_CACHE_DIR = CACHE_DIR / "odds"

COMPETITION = "ENG Premier League"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_SPORT_KEY = "soccer_epl"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FPL_API_BASE_URL = "https://fantasy.premierleague.com/api"

for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR, FOOTBALL_DATA_CACHE_DIR, ODDS_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
