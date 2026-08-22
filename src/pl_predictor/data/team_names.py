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

_SOURCE_ALIASES = {
    "odds_api": _ODDS_API_ALIASES,
    "fpl": _FPL_API_ALIASES,
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

    aliases = _SOURCE_ALIASES.get(source, {})
    if name in aliases:
        return aliases[name]

    for canonical in CANONICAL_TEAMS:
        if canonical.lower() == name.lower():
            return canonical

    return name
