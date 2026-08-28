"""team_names.py — canonical team-name mapping.

Canonical form = the short names used by penaltyblog's football-data.co.uk
scraper (e.g. "Man City", "Nott'm Forest", "Wolves") since that's the
training-data source every other source has to join against.

Every other data source (The Odds API's full names, the official FPL API's
own short names) is mapped onto this canonical form via `to_canonical`.
"""

from __future__ import annotations

# Alias -> canonical, one entry per source. Covers the current 20 PL clubs
# plus clubs relegated/promoted within roughly the last 8 seasons (the
# training-data window), so historical joins don't silently drop rows.
_ODDS_API_ALIASES: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Cardiff City": "Cardiff",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Middlesbrough": "Middlesbrough",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# The official FPL API only ever covers current-season PL clubs and already
# uses short names close to (but not always identical to) football-data's.
_FPL_API_ALIASES: dict[str, str] = {
    "Man Utd": "Man United",
    "Nott'm Forest": "Nott'm Forest",
    "Spurs": "Tottenham",
    "Wolves": "Wolves",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
}

# clubelo.com's exact "Club" field spelling has NOT been confirmed against a
# live response (the API was unreachable from this project's development
# environment while this was written — see config.py::CLUBELO_BASE_URL).
# This covers only the handful of cases known from general ClubElo usage to
# differ from the canonical short form; everything else relies on
# `to_canonical`'s case-insensitive fallback against `CANONICAL_TEAMS`,
# which should already match ClubElo's own generally-short naming for most
# clubs. Verify and extend this against a real response before trusting it
# for anything beyond a manual spot-check.
_CLUBELO_ALIASES: dict[str, str] = {
    "Nottingham": "Nott'm Forest",
    "Man City": "Man City",
    "Man United": "Man United",
    "Wolverhampton": "Wolves",
}

_SOURCE_ALIASES = {
    "odds_api": _ODDS_API_ALIASES,
    "fpl": _FPL_API_ALIASES,
    # Understat uses the same full-name convention as The Odds API
    # ("Manchester City", "Nottingham Forest", ...) — same alias table.
    "understat": _ODDS_API_ALIASES,
    # pulselive.com's club names are the same full-name convention too
    # ("Tottenham Hotspur", "Brentford", ...) — confirmed live.
    "pulselive": _ODDS_API_ALIASES,
    "clubelo": _CLUBELO_ALIASES,
    # ESPN's site API (data/espn.py, data/other_competitions.py) uses the
    # same full-name convention as The Odds API's own feed for English
    # clubs ("Manchester City", "Newcastle United", ...) — NOT yet confirmed
    # against a live response for the cup-competition endpoints specifically
    # (see config.py::ESPN_SOCCER_BASE_URL); extend this if a real fetch
    # turns up a different spelling.
    "espn": _ODDS_API_ALIASES,
}

CANONICAL_TEAMS: set[str] = set(_ODDS_API_ALIASES.values())


def to_canonical(name: str, source: str = "odds_api") -> str:
    """Map a team name from `source` to its canonical (football-data.co.uk)
    short form. Falls back to the input unchanged (case/whitespace-normalized
    against the known canonical set) if no alias is registered, rather than
    raising — an unmapped team should surface as a visible join failure
    downstream, not crash data collection outright."""
    name = name.strip()

    if source == "football_data":
        return name

    if source == "football_data_org":
        # football-data.org's names are the same full names as The Odds
        # API's, just with a "FC"/"AFC" club-suffix (or, for Bournemouth
        # specifically, an "AFC " prefix) that the Odds-API alias table
        # doesn't carry — strip it and reuse that table rather than
        # duplicating all 20+ entries.
        base = name
        if base.startswith("AFC "):
            base = base[4:]
        elif base.endswith(" FC"):
            base = base[:-3]
        elif base.endswith(" AFC"):
            base = base[:-4]
        if base in _ODDS_API_ALIASES:
            return _ODDS_API_ALIASES[base]
        name = base

    aliases = _SOURCE_ALIASES.get(source, {})
    if name in aliases:
        return aliases[name]

    for canonical in CANONICAL_TEAMS:
        if canonical.lower() == name.lower():
            return canonical

    return name
