"""Historical validation for display-only power-ranking candidates."""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from ..data import football_data
from ..features import ratings
from ..models import power_rankings, scoreline


def _future_points_per_game(matches: pd.DataFrame) -> dict[str, float]:
    points: dict[str, int] = {}
    games: dict[str, int] = {}
    for row in matches.itertuples():
        home_points, away_points = (3, 0) if row.goals_home > row.goals_away else ((0, 3) if row.goals_home < row.goals_away else (1, 1))
        for team, earned in ((row.team_home, home_points), (row.team_away, away_points)):
            points[team] = points.get(team, 0) + earned
            games[team] = games.get(team, 0) + 1
    return {team: points[team] / games[team] for team in games}


def _correlation(rankings: list[dict], future: pd.DataFrame) -> float | None:
    future_ppg = _future_points_per_game(future)
    values = {row["team"]: row["net_strength"] for row in rankings}
    teams = sorted(set(values) & set(future_ppg))
    if len(teams) < 5:
        return None
    return float(spearmanr([values[team] for team in teams], [future_ppg[team] for team in teams]).statistic)


def run_power_ranking_research(
    seasons: list[str] | None = None, min_history_seasons: int = 3, checkpoints: tuple[int, ...] = (10, 50, 100, 190)
) -> dict:
    """Compare display-ranking candidates against future PPG.

    A positive Spearman correlation means a stronger displayed ranking tracks
    stronger results in the remaining part of that historical season. Live
    form candidates have fixed weights and are evaluated only after replaying
    the match prefix available at that checkpoint.
    """
    seasons = seasons or football_data.default_completed_seasons()
    matches = football_data.load_training_data(seasons=seasons)
    rows = []
    for season_index in range(min_history_seasons, len(seasons)):
        target_season = seasons[season_index]
        target = matches[matches["season"] == target_season].sort_values("date").reset_index(drop=True)
        historical = matches[matches["season"].isin(seasons[:season_index])]
        if target.empty or historical.empty:
            continue
        prior = scoreline.fit_dixon_coles(historical)
        teams = set(target["team_home"]) | set(target["team_away"])
        for checkpoint in checkpoints:
            observed = target.iloc[:checkpoint]
            future = target.iloc[checkpoint:]
            if observed.empty or future.empty:
                continue
            prior_only = power_rankings.dominance_power_rankings(prior, observed.iloc[:0], current_teams=teams)
            dominance = power_rankings.dominance_power_rankings(prior, observed, current_teams=teams)
            dominance_momentum = power_rankings.dominance_power_rankings(
                prior, observed, current_teams=teams, include_momentum=True
            )
            history_to_checkpoint = pd.concat([historical, observed], ignore_index=True)
            elo = ratings.fit_elo(history_to_checkpoint)
            pi = ratings.fit_pi_ratings(history_to_checkpoint)
            elo_pi = power_rankings.elo_pi_power_rankings(elo, pi, current_teams=teams)
            live_form_candidates = {
                "elo_pi_25_form_spearman_future_ppg": power_rankings.blended_form_power_rankings(
                    prior, elo, pi, current_teams=teams, form_weight=0.25
                ),
                "elo_pi_50_form_spearman_future_ppg": power_rankings.blended_form_power_rankings(
                    prior, elo, pi, current_teams=teams, form_weight=0.50
                ),
                "elo_pi_75_form_spearman_future_ppg": power_rankings.blended_form_power_rankings(
                    prior, elo, pi, current_teams=teams, form_weight=0.75
                ),
            }
            rows.append(
                {
                    "season": target_season,
                    "checkpoint_matches": checkpoint,
                    "preseason_spearman_future_ppg": _correlation(prior_only, future),
                    "dominance_spearman_future_ppg": _correlation(dominance, future),
                    "dominance_momentum_spearman_future_ppg": _correlation(dominance_momentum, future),
                    "elo_pi_spearman_future_ppg": _correlation(elo_pi, future),
                    **{name: _correlation(rankings, future) for name, rankings in live_form_candidates.items()},
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return {"rows": [], "summary": {}, "status": "No eligible historical seasons."}
    report["delta"] = report["dominance_spearman_future_ppg"] - report["preseason_spearman_future_ppg"]
    report["momentum_delta"] = report["dominance_momentum_spearman_future_ppg"] - report["preseason_spearman_future_ppg"]
    report["elo_pi_delta"] = report["elo_pi_spearman_future_ppg"] - report["preseason_spearman_future_ppg"]
    for form_weight in (25, 50, 75):
        report[f"elo_pi_{form_weight}_form_delta"] = (
            report[f"elo_pi_{form_weight}_form_spearman_future_ppg"] - report["preseason_spearman_future_ppg"]
        )
    return {
        "rows": report.to_dict(orient="records"),
        "summary": {
            "preseason_spearman_future_ppg": float(report["preseason_spearman_future_ppg"].mean()),
            "dominance_spearman_future_ppg": float(report["dominance_spearman_future_ppg"].mean()),
            "delta": float(report["delta"].mean()),
            "dominance_momentum_spearman_future_ppg": float(report["dominance_momentum_spearman_future_ppg"].mean()),
            "momentum_delta": float(report["momentum_delta"].mean()),
            "elo_pi_spearman_future_ppg": float(report["elo_pi_spearman_future_ppg"].mean()),
            "elo_pi_delta": float(report["elo_pi_delta"].mean()),
            **{
                f"elo_pi_{form_weight}_form_spearman_future_ppg": float(
                    report[f"elo_pi_{form_weight}_form_spearman_future_ppg"].mean()
                )
                for form_weight in (25, 50, 75)
            },
            **{
                f"elo_pi_{form_weight}_form_delta": float(report[f"elo_pi_{form_weight}_form_delta"].mean())
                for form_weight in (25, 50, 75)
            },
            "n_comparisons": int(len(report)),
        },
        "status": "Display-ranking research only; future PPG is the held-out target.",
    }
