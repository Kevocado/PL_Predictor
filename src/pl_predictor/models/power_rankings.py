"""power_rankings.py — read the fitted scoreline model's own understanding
of each team's attacking/defensive strength. Not a new model: just a parse
of `model.get_params()`, which is already computed the moment the model is
fit.

Note: penaltyblog's two goal-model classes spell the defensive parameter
differently — `DixonColesGoalModel` uses `defence_{team}` (British), while
`BivariatePoissonGoalModel` uses `defense_{team}` (American). Both prefixes
are handled here since `models/manifest.py` picks whichever model scores
better and either could be the one loaded at runtime.
"""

from __future__ import annotations

_DEFENSE_PREFIXES = ("defence_", "defense_")
_ESTABLISHED_GAMES_THRESHOLD = 10

# penaltyblog's DixonColesGoalModel fixes attack parameters to average
# exactly 1.0 across every fitted team (verified directly against a live
# fit) — a real identifiability constraint, not a coincidence, so 1.0 is a
# safe "league-average attack" placeholder for a team the model hasn't
# fitted at all yet. Defence has no such fixed constraint, so its neutral
# placeholder is computed from whatever teams *are* fitted instead of
# hardcoded.
_NEUTRAL_ATTACK = 1.0


def power_rankings(model, current_teams: set[str] | None = None, games_played: dict[str, int] | None = None) -> list[dict]:
    """Ranked (by net strength, attack - defence) list of teams. Lower
    defence/defense values mean a stronger defence in penaltyblog's
    parameterization (it's a suppression term), so net strength is
    attack + (-defence) = attack - defence either way.

    `current_teams` (pass this season's 20 teams — see
    `data/fixtures.py::list_current_teams`) restricts the *fitted* half of
    the output to just them: the model is fit on the last several completed
    seasons plus whatever's played so far this one, so without this filter
    the list includes every team relegated/promoted through that whole
    window — 30+ teams, most of them not actually in the league right now.

    A team in `current_teams` the model has *never* fitted at all (a fresh
    promotion with zero matches anywhere in the loaded window, even after
    the latest retrain) is still included — with a neutral, league-average
    placeholder rating and `confidence: "new"` — rather than silently
    dropped, so newly-promoted sides are visible from day one instead of
    only appearing once they've accumulated enough matches for the model to
    have fit them. `games_played` (pass `FixtureFeatureContext.games_played`
    or equivalent) also tags every *fitted* team with a confidence tier —
    this reflects live data (updates every 5 minutes as new results come
    in), so a newly-promoted team's confidence can improve immediately even
    before the next retrain has produced a real rating for them."""
    params = model.get_params()

    def _confidence(team: str) -> str:
        if games_played is None:
            return "established"
        n = games_played.get(team, 0)
        if n >= _ESTABLISHED_GAMES_THRESHOLD:
            return "established"
        return "limited" if n > 0 else "new"

    fitted_teams = sorted({k[len("attack_") :] for k in params if k.startswith("attack_")})
    if current_teams is not None:
        fitted_teams = [t for t in fitted_teams if t in current_teams]

    rankings = []
    for team in fitted_teams:
        attack = params[f"attack_{team}"]
        defence = next(
            (params[f"{prefix}{team}"] for prefix in _DEFENSE_PREFIXES if f"{prefix}{team}" in params),
            None,
        )
        if defence is None:
            continue
        rankings.append(
            {
                "team": team,
                "attack": float(attack),
                "defence": float(defence),
                "net_strength": float(attack - defence),
                "games_played": games_played.get(team) if games_played is not None else None,
                "confidence": _confidence(team),
                "fitted": True,
            }
        )

    if current_teams is not None:
        missing = current_teams - {r["team"] for r in rankings}
        if missing and rankings:
            neutral_defence = sum(r["defence"] for r in rankings) / len(rankings)
            for team in sorted(missing):
                # A team can be missing from the model (never in a *trained*
                # season — e.g. Sunderland, whose only recent top-flight
                # season is the held-out validation split) while still
                # having plenty of real games_played — confidence should
                # reflect that real data, `fitted: False` separately flags
                # that the rating itself is still a neutral placeholder
                # rather than conflating "no rating yet" with "no data yet."
                rankings.append(
                    {
                        "team": team,
                        "attack": _NEUTRAL_ATTACK,
                        "defence": neutral_defence,
                        "net_strength": float(_NEUTRAL_ATTACK - neutral_defence),
                        "games_played": games_played.get(team, 0) if games_played is not None else 0,
                        "confidence": _confidence(team),
                        "fitted": False,
                    }
                )

    rankings.sort(key=lambda r: -r["net_strength"])
    return rankings
