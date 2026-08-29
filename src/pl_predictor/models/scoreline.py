"""scoreline.py — the core goals/1X2/scoreline engine, built on
penaltyblog's DixonColesGoalModel (and BivariatePoissonGoalModel as a
comparison candidate).

One fitted model's `FootballProbabilityGrid` gives 1X2, exact scorelines,
BTTS, and over/under goals all at once — no separate model needed for those
markets.

Note: penaltyblog's Cython loss functions require *writable* float64 numpy
arrays, not pandas Series (a pandas Series' underlying buffer is read-only to
Cython memoryviews) — hence the explicit `.to_numpy().astype(np.float64)` /
`.astype(str)` conversions below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import penaltyblog as pb
from sklearn.metrics import brier_score_loss, log_loss

from ..config import MODELS_DIR
from ..features import rolling_form

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}

# Which `predict_fixture`/`predict_fixtures_batch` result fields belong to
# each market — the registry `market_overrides` (see both functions below)
# and `models/manifest.py`'s per-market promotion decision both key off
# this. Extend it if a future model earns an override for a market not
# listed yet (e.g. `"1x2"`/`"exact_scoreline"` currently never have one —
# `ml_scoreline` wins those outright per EXP-2026-14/15 — but the
# mechanism is general, not hardcoded to `"over_2_5"`).
MARKET_FIELDS = {
    "1x2": ["home_win", "draw", "away_win"],
    "over_2_5": ["over_2_5", "under_2_5"],
    "btts": ["btts_yes", "btts_no"],
    "exact_scoreline": ["top_scorelines", "grid"],
}


def _writable_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    goals_home = df["goals_home"].to_numpy().astype(np.float64)
    goals_away = df["goals_away"].to_numpy().astype(np.float64)
    team_home = df["team_home"].to_numpy().astype(str)
    team_away = df["team_away"].to_numpy().astype(str)
    return goals_home, goals_away, team_home, team_away


def fit_dixon_coles(df: pd.DataFrame, xi: float = 0.0018) -> pb.models.DixonColesGoalModel:
    goals_home, goals_away, team_home, team_away = _writable_inputs(df)
    weights = np.asarray(pb.models.dixon_coles_weights(df["date"], xi=xi), dtype=np.float64)
    model = pb.models.DixonColesGoalModel(goals_home, goals_away, team_home, team_away, weights=weights)
    model.fit()
    return model


def fit_bivariate_poisson(df: pd.DataFrame, xi: float = 0.0018) -> pb.models.BivariatePoissonGoalModel:
    goals_home, goals_away, team_home, team_away = _writable_inputs(df)
    weights = np.asarray(pb.models.dixon_coles_weights(df["date"], xi=xi), dtype=np.float64)
    model = pb.models.BivariatePoissonGoalModel(goals_home, goals_away, team_home, team_away, weights=weights)
    model.fit()
    return model


def _top_n_scorelines(grid: pb.models.FootballProbabilityGrid, n: int = 5, max_goals: int = 8) -> list[dict]:
    scores = [
        {"home": h, "away": a, "prob": grid.exact_score(h, a)}
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
    ]
    return sorted(scores, key=lambda s: s["prob"], reverse=True)[:n]


# League-average goals-per-team-per-match — the fallback for models that
# genuinely have no way to score an unseen team (Dixon-Coles/Bivariate-Poisson
# only have fitted attack/defence parameters for teams present at fit time).
# Feature-driven models (MLScorelineModel) don't need this: Elo/Pi both
# return a sane league-average default (Elo 1500, Pi 0.0) for a team they've
# never seen, and rolling-form/xG cold-start-blend toward the league average
# the same way they already do for any team with few recent matches — so
# they can score literally any team from day one, and that prediction
# improves as the season goes rather than staying flat all year. See
# `_data_confidence` for the caveat this still leaves: the *prediction* is no
# longer a blind fallback, but it's still low-confidence until real matches
# accumulate.
FALLBACK_GOAL_EXPECTANCY = 1.35


def is_known_team(model, team: str) -> bool:
    return team in set(model.teams)


def _data_confidence(model, home: str, away: str) -> str | None:
    """How much real, observed data this fixture's prediction is actually
    resting on, for feature-driven models only (`None` for Dixon-Coles/
    Bivariate-Poisson, which don't have a live per-fixture feature context to
    measure this from). 'new' means both Elo/Pi/rolling-form are at or near
    their league-average defaults for at least one side (e.g. a newly
    promoted team's first few matches); 'limited' means a partial cold-start
    blend; 'established' means a full rolling window of real matches. Purely
    informational — never changes what's predicted, just how much to trust
    it."""
    if not hasattr(model, "context"):
        return None
    games_played = model.context.games_played
    min_games = min(games_played.get(home, 0), games_played.get(away, 0))
    if min_games >= max(rolling_form.LAG_WINDOWS):
        return "established"
    if min_games > 0:
        return "limited"
    return "new"


def multiclass_top_label_ece(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Standard top-label calibration error: bins predictions by the
    model's own confidence in its argmax class, compares bin-average
    accuracy to bin-average confidence. Generalizes
    `evaluate/goal_contribution_research.py::_ece`'s binary version to the
    1X2 3-class case."""
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == outcomes).astype(float)
    bucket = np.clip((confidence * bins).astype(int), 0, bins - 1)
    total = len(outcomes)
    if total == 0:
        return float("nan")
    return float(
        sum(
            abs(correct[bucket == b].mean() - confidence[bucket == b].mean()) * (bucket == b).sum() / total
            for b in range(bins)
            if (bucket == b).any()
        )
    )


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI over per-fixture values (e.g. per-row
    RPS contributions)."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate_grids_multi_market(grids: list, val_df: pd.DataFrame) -> dict:
    """Every scoreline market's metrics from an already-predicted list of
    grids (no refitting per market) — RPS/Brier/log-loss/ECE for 1X2,
    exact-scoreline log-loss, and BTTS / O-U-2.5 log-loss+Brier as separate
    markets. Model-agnostic: `grids` can come from any model whose
    `.predict`/batch path yields the same `FootballProbabilityGrid`-shaped
    objects — Dixon-Coles, Bivariate-Poisson, `MLScorelineModel`, and
    `CovariatePoissonModel` all do. This is the one canonical place
    scoreline market metrics are computed — `models/manifest.py`'s
    per-market promotion decision and every research module that compares
    scoreline candidates (`evaluate/scoreline_dominance_arms.py`,
    `evaluate/model_selection_by_segment.py`) call this rather than
    re-deriving it."""
    outcomes = val_df["ftr"].map(_RESULT_CODE).to_numpy()
    probs_1x2 = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
    per_row_rps = np.array([pb.metrics.rps_average(p.reshape(1, -1), int(o)) for p, o in zip(probs_1x2, outcomes)])

    goals_total = (val_df["goals_home"] + val_df["goals_away"]).to_numpy()
    over_actual = (goals_total > 2.5).astype(int)
    over_probs = np.clip(np.array([g.total_goals("over", 2.5) for g in grids]), 1e-6, 1 - 1e-6)

    btts_actual = ((val_df["goals_home"] > 0) & (val_df["goals_away"] > 0)).astype(int).to_numpy()
    btts_probs = np.clip(np.array([g.btts_yes for g in grids]), 1e-6, 1 - 1e-6)

    exact_score_probs = np.array(
        [
            max(g.exact_score(int(h), int(a)), 1e-6)
            for g, h, a in zip(grids, val_df["goals_home"], val_df["goals_away"])
        ]
    )

    rps_ci_low, rps_ci_high = bootstrap_ci(per_row_rps)

    return {
        "rps": float(per_row_rps.mean()),
        "rps_ci_low": rps_ci_low,
        "rps_ci_high": rps_ci_high,
        "brier_1x2": float(pb.metrics.multiclass_brier_score(probs_1x2, outcomes)),
        "log_loss_1x2": float(log_loss(outcomes, probs_1x2, labels=[0, 1, 2])),
        "ece_1x2": multiclass_top_label_ece(probs_1x2, outcomes),
        "exact_scoreline_log_loss": float(-np.mean(np.log(exact_score_probs))),
        "over_2_5_log_loss": float(log_loss(over_actual, over_probs)),
        "over_2_5_brier": float(brier_score_loss(over_actual, over_probs)),
        "btts_log_loss": float(log_loss(btts_actual, btts_probs)),
        "btts_brier": float(brier_score_loss(btts_actual, btts_probs)),
    }


def predict_grids_for_fixed_param_model(model, val_df: pd.DataFrame, max_goals: int = 10) -> list:
    """Raw grid objects for every `val_df` fixture (needs `team_home`/
    `team_away` columns), for a model with fixed, already-fitted
    attack/defence parameters (Dixon-Coles, Bivariate-Poisson) — falls back
    to `FALLBACK_GOAL_EXPECTANCY` for a team unseen at fit time, exactly
    like `predict_fixture`. For `models/manifest.py`'s multi-market
    candidate comparison, which needs the grid object itself (via
    `evaluate_grids_multi_market`), not `predict_fixtures_batch`'s already-
    extracted dict shape. Feature-driven models (`ml_scoreline`,
    `covariate_poisson`) have their own batch-eval helper instead
    (`predict_grids_batch` in each), since they need each row's own
    point-in-time features for leakage-safe evaluation — this generic
    team-name-only helper has no way to supply that."""
    grids = []
    for h, a in zip(val_df["team_home"], val_df["team_away"]):
        if is_known_team(model, h) and is_known_team(model, a):
            grids.append(model.predict(h, a, max_goals=max_goals))
        else:
            grids.append(
                pb.models.create_dixon_coles_grid(FALLBACK_GOAL_EXPECTANCY, FALLBACK_GOAL_EXPECTANCY, rho=0.0, max_goals=max_goals)
            )
    return grids


def predict_fixture(
    model, home: str, away: str, max_goals: int = 10, feature_row=None, market_overrides: dict | None = None
) -> dict:
    """`feature_row` (a full row from `features.build.build_training_frame`'s
    output, e.g. a validation-set row) is only relevant for models that need
    point-in-time features to avoid leakage — currently `MLScorelineModel`
    via its `predict_from_row`. Pass it when scoring a fixture that already
    happened (calibration, backtest); omit it for a genuine upcoming fixture,
    where there's no historical row to draw from and `.predict(home, away)`
    correctly uses each team's current state instead. Dixon-Coles/Bivariate-
    Poisson ignore it either way — their fitted attack/defence params are
    static regardless of when `.predict` is called.

    A model exposing `.context` (currently `MLScorelineModel`) is
    feature-driven rather than needing fixed per-team learned parameters, so
    it's never sent through the flat fallback below, however new a team is —
    see `FALLBACK_GOAL_EXPECTANCY`'s docstring and `_data_confidence`.

    `market_overrides` (e.g. `{"over_2_5": covariate_poisson_model}`, keyed
    by `MARKET_FIELDS`) lets specific markets in the result come from a
    separately fitted model instead of `model`, while every other field
    stays from `model` untouched — this is how `models/manifest.py`'s
    per-market promotion decision (`scoreline.market_overrides` in
    `manifest.json`, decided from real per-fold evidence at retrain time,
    never hardcoded here — see EXP-2026-15/16) actually takes effect at
    serving time. The override model is scored through this same function,
    recursively, without its own `market_overrides` (never chained)."""
    uses_live_features = hasattr(model, "context")
    known_home, known_away = is_known_team(model, home), is_known_team(model, away)
    fallback = not uses_live_features and not (known_home and known_away)

    if fallback:
        grid = pb.models.create_dixon_coles_grid(
            FALLBACK_GOAL_EXPECTANCY, FALLBACK_GOAL_EXPECTANCY, rho=0.0, max_goals=max_goals
        )
    elif feature_row is not None and hasattr(model, "predict_from_row"):
        grid = model.predict_from_row(feature_row, max_goals=max_goals)
    else:
        grid = model.predict(home, away, max_goals=max_goals)

    result = {
        "home_win": grid.home_win,
        "draw": grid.draw,
        "away_win": grid.away_win,
        "btts_yes": grid.btts_yes,
        "btts_no": grid.btts_no,
        "over_2_5": grid.total_goals("over", 2.5),
        "under_2_5": grid.total_goals("under", 2.5),
        # P(team scores >=2), derived from the same grid's marginal goal
        # distribution — not a separate model, same reasoning as
        # predicted_total_goals/predicted_margin elsewhere in this app.
        "home_2plus_prob": float(grid.home_goal_distribution()[2:].sum()),
        "away_2plus_prob": float(grid.away_goal_distribution()[2:].sum()),
        "home_goal_expectation": grid.home_goal_expectation,
        "away_goal_expectation": grid.away_goal_expectation,
        "top_scorelines": _top_n_scorelines(grid),
        "grid": grid.grid,
        "fallback": fallback,
        "data_confidence": _data_confidence(model, home, away),
    }
    if market_overrides:
        result["market_model_overrides"] = []
        for market, override_model in market_overrides.items():
            override_result = predict_fixture(override_model, home, away, max_goals=max_goals, feature_row=feature_row)
            for field in MARKET_FIELDS[market]:
                result[field] = override_result[field]
            result["market_model_overrides"].append(market)
    return result


def predict_fixtures_batch(
    model, fixtures_df: pd.DataFrame, max_goals: int = 10, market_overrides: dict | None = None
) -> list[dict]:
    """Same per-fixture dict shape as `predict_fixture`, for many fixtures
    at once. For a feature-driven model (has `.context` + `predict_many_
    from_rows` — currently `MLScorelineModel`), builds every fixture's
    feature row via the context once, then does a single batched XGBoost
    call instead of looping one `.predict()` per fixture — the difference
    between a fraction of a second and several seconds once there's more
    than a handful of fixtures (a full remaining season for the projected
    table, or the main fixtures list once the Odds API is returning many).
    Dixon-Coles/Bivariate-Poisson have no such benefit (static per-team
    params, no XGBoost call) and just fall back to the per-fixture loop —
    same for `CovariatePoissonModel`, whose `PoissonRegressor` predict has
    none of XGBoost's per-call overhead so the loop is already fast enough.

    `market_overrides`: see `predict_fixture`'s docstring. Applied here by
    recursing into this same function per override model (so an override
    model that itself has a batched fast path — e.g. a future feature-driven
    override — still gets it) and merging field-by-field, rather than
    re-deriving the merge logic for the batch case."""
    if hasattr(model, "context") and hasattr(model, "predict_many_from_rows"):
        fixtures = list(fixtures_df.itertuples(index=False))
        rows = [
            model.context.build_row(f.team_home, f.team_away, getattr(f, "commence_time", None)) for f in fixtures
        ]
        X = pd.DataFrame(rows).reindex(columns=model.feature_cols, fill_value=0).fillna(0).astype(float)
        grids = model.predict_many_from_rows(X, max_goals=max_goals)
        results = [
            {
                "home_win": g.home_win,
                "draw": g.draw,
                "away_win": g.away_win,
                "btts_yes": g.btts_yes,
                "btts_no": g.btts_no,
                "over_2_5": g.total_goals("over", 2.5),
                "under_2_5": g.total_goals("under", 2.5),
                "home_2plus_prob": float(g.home_goal_distribution()[2:].sum()),
                "away_2plus_prob": float(g.away_goal_distribution()[2:].sum()),
                "home_goal_expectation": g.home_goal_expectation,
                "away_goal_expectation": g.away_goal_expectation,
                "top_scorelines": _top_n_scorelines(g),
                "grid": g.grid,
                "fallback": False,
                "data_confidence": _data_confidence(model, f.team_home, f.team_away),
            }
            for g, f in zip(grids, fixtures)
        ]
    else:
        results = [
            predict_fixture(model, row["team_home"], row["team_away"], max_goals=max_goals)
            for _, row in fixtures_df.iterrows()
        ]

    if market_overrides:
        for market, override_model in market_overrides.items():
            override_results = predict_fixtures_batch(override_model, fixtures_df, max_goals=max_goals)
            for result, override_result in zip(results, override_results):
                for field in MARKET_FIELDS[market]:
                    result[field] = override_result[field]
                result.setdefault("market_model_overrides", []).append(market)
    return results


def predict_many(model, fixtures_df: pd.DataFrame, max_goals: int = 10) -> pd.DataFrame:
    rows = []
    for _, row in fixtures_df.iterrows():
        pred = predict_fixture(model, row["team_home"], row["team_away"], max_goals=max_goals)
        rows.append(
            {
                "team_home": row["team_home"],
                "team_away": row["team_away"],
                "home_win": pred["home_win"],
                "draw": pred["draw"],
                "away_win": pred["away_win"],
                "btts_yes": pred["btts_yes"],
                "over_2_5": pred["over_2_5"],
                "top_scoreline": pred["top_scorelines"][0],
            }
        )
    return pd.DataFrame(rows)


def save(model, path: Path | None = None) -> Path:
    path = path or (MODELS_DIR / "dixon_coles.pkl")
    model.save(str(path))
    return path


def load(path: Path | None = None):
    path = path or (MODELS_DIR / "dixon_coles.pkl")
    return pb.models.DixonColesGoalModel.load(str(path))
