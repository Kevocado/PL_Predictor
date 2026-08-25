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

import math

from . import scoreline

_DEFENSE_PREFIXES = ("defence_", "defense_")
_ESTABLISHED_GAMES_THRESHOLD = 10
_SHRINKAGE_GAMES = 19

# penaltyblog's DixonColesGoalModel fixes attack parameters to average
# exactly 1.0 across every fitted team (verified directly against a live
# fit) — a real identifiability constraint, not a coincidence, so 1.0 is a
# safe "league-average attack" placeholder for a team the model hasn't
# fitted at all yet. Defence has no such fixed constraint, so its neutral
# placeholder is computed from whatever teams *are* fitted instead of
# hardcoded.
_NEUTRAL_ATTACK = 1.0
_FORM_HALF_SEASON_GAMES = 8
_FORM_MAX_ADJUSTMENT = 0.35
_MOMENTUM_MAX_ADJUSTMENT = 0.08


def _raw_parameters(model, current_teams: set[str] | None) -> list[dict]:
    params = model.get_params()
    fitted_teams = sorted({key[len("attack_") :] for key in params if key.startswith("attack_")})
    if current_teams is not None:
        fitted_teams = [team for team in fitted_teams if team in current_teams]
    rows = []
    for team in fitted_teams:
        defence = next((params[f"{prefix}{team}"] for prefix in _DEFENSE_PREFIXES if f"{prefix}{team}" in params), None)
        if defence is not None:
            rows.append({"team": team, "attack": float(params[f"attack_{team}"]), "defence": float(defence), "fitted": True})
    if current_teams is not None and rows:
        neutral_defence = sum(row["defence"] for row in rows) / len(rows)
        missing = current_teams - {row["team"] for row in rows}
        rows.extend({"team": team, "attack": _NEUTRAL_ATTACK, "defence": neutral_defence, "fitted": False} for team in sorted(missing))
    return rows


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
    def _confidence(team: str) -> str:
        if games_played is None:
            return "established"
        n = games_played.get(team, 0)
        if n >= _ESTABLISHED_GAMES_THRESHOLD:
            return "established"
        return "limited" if n > 0 else "new"

    raw_rows = [{**row, "games_played": games_played.get(row["team"]) if games_played is not None else None, "confidence": _confidence(row["team"])} for row in _raw_parameters(model, current_teams)]

    rankings = []
    if raw_rows:
        neutral_attack = sum(row["attack"] for row in raw_rows) / len(raw_rows)
        neutral_defence = sum(row["defence"] for row in raw_rows) / len(raw_rows)
        for row in raw_rows:
            # DC is fitted over a long historical window, but a promoted or
            # newly-returned club can effectively have only a handful of
            # high-weight current matches. Shrink its fitted parameters toward
            # the current-league mean until it has a half-season of evidence.
            # Current-season games are supplied by the route; historical
            # appearances must not make a new campaign look established.
            n_current = row["games_played"] or 0
            weight = n_current / (n_current + _SHRINKAGE_GAMES)
            attack = neutral_attack + weight * (row["attack"] - neutral_attack)
            defence = neutral_defence + weight * (row["defence"] - neutral_defence)
            rankings.append({**row, "attack": float(attack), "defence": float(defence), "net_strength": float(attack - defence)})

    rankings.sort(key=lambda r: -r["net_strength"])
    return rankings


def dominance_power_rankings(
    prior_model, season_matches, current_teams: set[str] | None = None, include_momentum: bool = False
) -> list[dict]:
    """Prior-anchored ranking with capped opponent-adjusted match evidence.

    The fitted pre-season goal model supplies the baseline. Each completed
    match then contributes a bounded residual versus its expected goal
    difference plus a shots-on-target dominance signal. This makes a 2-0 win
    over a strong club meaningful without allowing one result to overthrow the
    entire pre-season ranking.
    """
    teams = current_teams or (set(season_matches["team_home"]) | set(season_matches["team_away"]))
    prior_rows = _raw_parameters(prior_model, teams)
    if not prior_rows:
        return []
    prior = {row["team"]: row for row in prior_rows}
    evidence = {team: 0.0 for team in teams}
    games = {team: 0 for team in teams}
    streaks = {team: 0 for team in teams}
    season_rows = () if season_matches is None or season_matches.empty else season_matches.sort_values("date").itertuples()
    for row in season_rows:
        prediction = scoreline.predict_fixture(prior_model, row.team_home, row.team_away)
        expected_diff = prediction["home_goal_expectation"] - prediction["away_goal_expectation"]
        goal_residual = math.tanh(((row.goals_home - row.goals_away) - expected_diff) / 1.5)
        if getattr(row, "hst", None) is not None and getattr(row, "ast", None) is not None:
            shot_signal = math.tanh((float(row.hst) - float(row.ast)) / 4.0)
        else:
            shot_signal = 0.0
        dominance = 0.7 * goal_residual + 0.3 * shot_signal
        evidence[row.team_home] += dominance
        evidence[row.team_away] -= dominance
        games[row.team_home] += 1
        games[row.team_away] += 1
        if row.goals_home == row.goals_away:
            streaks[row.team_home] = 0
            streaks[row.team_away] = 0
        else:
            winner, loser = (row.team_home, row.team_away) if row.goals_home > row.goals_away else (row.team_away, row.team_home)
            streaks[winner] = streaks[winner] + 1 if streaks[winner] > 0 else 1
            streaks[loser] = streaks[loser] - 1 if streaks[loser] < 0 else -1

    rows = []
    for team, base in prior.items():
        n = games.get(team, 0)
        form = evidence[team] / n if n else 0.0
        trust = n / (n + _FORM_HALF_SEASON_GAMES)
        momentum = _MOMENTUM_MAX_ADJUSTMENT * trust * math.tanh(streaks[team] / 3.0) if include_momentum else 0.0
        adjustment = _FORM_MAX_ADJUSTMENT * trust * form + momentum
        rows.append(
            {
                "team": team,
                "attack": base["attack"],
                "defence": base["defence"],
                "net_strength": float(base["attack"] - base["defence"] + adjustment),
                "games_played": n,
                "confidence": "established" if n >= _ESTABLISHED_GAMES_THRESHOLD else ("limited" if n else "preseason"),
                "fitted": base["fitted"],
                "form_adjustment": float(adjustment),
                "momentum_adjustment": float(momentum),
            }
        )
    return sorted(rows, key=lambda row: -row["net_strength"])


def elo_pi_power_rankings(elo, pi, current_teams: set[str], games_played: dict[str, int] | None = None) -> list[dict]:
    """Rank current teams from history-seeded Elo and Pi ratings.

    Elo rewards results relative to opposition; Pi also responds to goal
    margin. Each scale is standardised across the current league's teams
    before averaging, so neither arbitrary numeric scale dominates.
    """
    teams = sorted(current_teams)
    if not teams:
        return []

    elo_values = {team: float(elo.get_team_rating(team)) for team in teams}
    pi_values = {team: float(pi.get_team_rating(team)) for team in teams}

    def standardise(values: dict[str, float]) -> dict[str, float]:
        mean = sum(values.values()) / len(values)
        variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
        deviation = math.sqrt(variance)
        if deviation == 0:
            return {team: 0.0 for team in values}
        return {team: (value - mean) / deviation for team, value in values.items()}

    elo_standardised = standardise(elo_values)
    pi_standardised = standardise(pi_values)
    rows = []
    for team in teams:
        games = games_played.get(team, 0) if games_played is not None else 0
        rows.append(
            {
                "team": team,
                "attack": float(elo_standardised[team]),
                "defence": float(-pi_standardised[team]),
                "net_strength": float((elo_standardised[team] + pi_standardised[team]) / 2),
                "elo": elo_values[team],
                "pi": pi_values[team],
                "ranking_method": "elo_pi",
                "games_played": games if games_played is not None else None,
                "confidence": "established" if games >= _ESTABLISHED_GAMES_THRESHOLD else ("limited" if games else "new"),
                "fitted": True,
            }
        )
    return sorted(rows, key=lambda row: -row["net_strength"])


def blended_form_power_rankings(
    prior_model, elo, pi, current_teams: set[str], form_weight: float, games_played: dict[str, int] | None = None
) -> list[dict]:
    """Blend a fixed pre-season prior with an online Elo/Pi form signal.

    The blend uses standardised ranking scores, rather than mixing raw
    Dixon-Coles, Elo, and Pi scales. ``form_weight`` is deliberately a fixed
    design choice supplied by the evaluator, never learned from the season
    being ranked.
    """
    if not 0.0 <= form_weight <= 1.0:
        raise ValueError("form_weight must be between 0 and 1")

    prior_rows = dominance_power_rankings(prior_model, season_matches=None, current_teams=current_teams)
    live_form_rows = elo_pi_power_rankings(elo, pi, current_teams=current_teams, games_played=games_played)
    if not prior_rows or not live_form_rows:
        return []

    def standardise(values: dict[str, float]) -> dict[str, float]:
        mean = sum(values.values()) / len(values)
        variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
        deviation = math.sqrt(variance)
        if deviation == 0:
            return {team: 0.0 for team in values}
        return {team: (value - mean) / deviation for team, value in values.items()}

    prior_by_team = {row["team"]: row for row in prior_rows}
    form_by_team = {row["team"]: row for row in live_form_rows}
    teams = sorted(set(prior_by_team) & set(form_by_team))
    prior_scores = standardise({team: prior_by_team[team]["net_strength"] for team in teams})
    form_scores = standardise({team: form_by_team[team]["net_strength"] for team in teams})
    rows = []
    for team in teams:
        prior_row = prior_by_team[team]
        form_row = form_by_team[team]
        rows.append(
            {
                **prior_row,
                "net_strength": float((1.0 - form_weight) * prior_scores[team] + form_weight * form_scores[team]),
                "elo": form_row["elo"],
                "pi": form_row["pi"],
                "form_weight": form_weight,
                "ranking_method": "preseason_elo_pi_blend",
                "games_played": form_row["games_played"],
                "confidence": "limited" if form_row["games_played"] else "preseason",
            }
        )
    return sorted(rows, key=lambda row: -row["net_strength"])
