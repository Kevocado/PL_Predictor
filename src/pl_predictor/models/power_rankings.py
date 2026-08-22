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


def power_rankings(model) -> list[dict]:
    """Ranked (by net strength, attack - defence) list of every team the
    model was fit on. Lower defence/defense values mean a stronger defence
    in penaltyblog's parameterization (it's a suppression term), so net
    strength is attack + (-defence) = attack - defence either way."""
    params = model.get_params()

    teams = sorted({k[len("attack_") :] for k in params if k.startswith("attack_")})
    rankings = []
    for team in teams:
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
            }
        )

    rankings.sort(key=lambda r: -r["net_strength"])
    return rankings
