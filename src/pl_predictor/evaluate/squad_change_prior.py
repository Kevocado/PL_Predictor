"""squad_change_prior.py — walk-forward evaluation of Phase 1's squad-
continuity signal (see docs/AI_CONTINUITY.md EXP-2026-18 and
`features/squad_change.py`'s own docstring for the full motivation).

Question: does knowing how much of a team's squad carried over from last
season help predict its matches, specifically in the fixtures where the
model's own frozen expectation and a team's actual results diverge the
most — an ex-post proxy for "the model didn't see a real squad-strength
change coming," the exact shape of the Newcastle 2026-27 observation that
prompted this experiment?

Learned from EXP-2026-14's mistake (a "cold-start" segment that turned
out to be one newly-promoted team's first 10 matches, re-run every
season, n=10): the "surprise" segment here is defined across every
team-season in the training window, not a single case, and its
composition is printed before any metric is trusted.
"""

from __future__ import annotations

import pandas as pd
import penaltyblog as pb

from ..data import football_data
from ..evaluate import walk_forward
from ..features import squad_change
from ..models import ml_scoreline

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}
SURPRISE_GAMES = 8  # first N games of a season used to define the "surprise" segment
SURPRISE_QUANTILE = 0.85  # flag the top/bottom (1 - quantile) team-seasons by |actual - expected| PPG


def _extra_feature_frame(seasons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """(season, team_home, team_away, home_squad_continuity,
    away_squad_continuity) for every fixture in `seasons`, shaped for
    `evaluate.walk_forward.prepare_folds`'s `extra_feature_frame`
    mechanism. `features.squad_change` only computes the per-(season,
    team) signal — this is the fixture-shaping step that belongs to the
    experiment, not the feature module."""
    matches_df = football_data.load_training_data(seasons=seasons)
    continuity = squad_change.team_season_continuity_table(seasons)

    extra = matches_df[["season", "team_home", "team_away"]].drop_duplicates().copy()
    extra = extra.merge(
        continuity.rename(columns={"team": "team_home", "squad_continuity": "home_squad_continuity"}),
        on=["season", "team_home"],
        how="left",
    )
    extra = extra.merge(
        continuity.rename(columns={"team": "team_away", "squad_continuity": "away_squad_continuity"}),
        on=["season", "team_away"],
        how="left",
    )
    return extra, ["home_squad_continuity", "away_squad_continuity"]


def _fold_predictions(fold: dict) -> pd.DataFrame:
    """Per-fixture predicted probabilities for one fold's val_df.
    Reimplemented rather than calling `ml_scoreline.evaluate_on_holdout`
    because that function only returns fold-aggregated RPS/Brier — the
    segment analysis below needs each fixture's own probabilities."""
    home_model, away_model = ml_scoreline.train_goal_regressors(
        fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
    )
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, fold["X_val"])
    val_df = fold["val_df"].reset_index(drop=True)
    return pd.DataFrame(
        {
            "season": val_df["season"].to_numpy(),
            "date": pd.to_datetime(val_df["date"]).to_numpy(),
            "team_home": val_df["team_home"].to_numpy(),
            "team_away": val_df["team_away"].to_numpy(),
            "ftr": val_df["ftr"].to_numpy(),
            "home_win": [g.home_win for g in grids],
            "draw": [g.draw for g in grids],
            "away_win": [g.away_win for g in grids],
        }
    )


def _team_perspective(preds: pd.DataFrame) -> pd.DataFrame:
    """Long, one-row-per-team-per-fixture frame: actual points earned and
    the model's own expected points (3*P(win) + 1*P(draw)) for that
    fixture, from each team's own side."""
    home = pd.DataFrame(
        {
            "season": preds["season"],
            "date": preds["date"],
            "team": preds["team_home"],
            "points": preds["ftr"].map({"H": 3, "D": 1, "A": 0}),
            "expected_points": preds["home_win"] * 3 + preds["draw"],
        }
    )
    away = pd.DataFrame(
        {
            "season": preds["season"],
            "date": preds["date"],
            "team": preds["team_away"],
            "points": preds["ftr"].map({"H": 0, "D": 1, "A": 3}),
            "expected_points": preds["away_win"] * 3 + preds["draw"],
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "date"])


def surprise_team_seasons(
    preds: pd.DataFrame, n_games: int = SURPRISE_GAMES, quantile: float = SURPRISE_QUANTILE
) -> pd.DataFrame:
    """One row per (season, team) with `actual_ppg`, `expected_ppg`, and
    `deviation` over that team-season's first `n_games` fixtures —
    "expected" is the frozen baseline model's own expectation (not a
    fixed external number), so a large deviation means "this team's
    results diverged from what the model itself expected," in either
    direction. Flags the top/bottom `1 - quantile` team-seasons by
    `|deviation|` as `is_surprise`. Does not print anything — the caller
    must report composition before trusting a segment metric on it."""
    long_df = _team_perspective(preds).sort_values(["season", "team", "date"])
    game_number = long_df.groupby(["season", "team"]).cumcount()
    first_n = long_df[game_number < n_games]
    summary = first_n.groupby(["season", "team"], as_index=False).agg(
        n_games=("points", "size"), actual_ppg=("points", "mean"), expected_ppg=("expected_points", "mean")
    )
    summary = summary[summary["n_games"] == n_games]  # only fully-observed windows
    summary["deviation"] = summary["actual_ppg"] - summary["expected_ppg"]
    threshold = summary["deviation"].abs().quantile(quantile)
    summary["is_surprise"] = summary["deviation"].abs() >= threshold
    return summary.sort_values("deviation")


def _rps_brier(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {"rps": float("nan"), "brier": float("nan"), "n": 0}
    probs = preds[["home_win", "draw", "away_win"]].to_numpy()
    outcomes = preds["ftr"].map(_RESULT_CODE).to_numpy()
    return {
        "rps": float(pb.metrics.rps_average(probs, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
        "n": int(len(preds)),
    }


def _segment_mask(preds: pd.DataFrame, surprise_keys: set[tuple[str, str]]) -> pd.Series:
    home_flag = preds.apply(lambda r: (r["season"], r["team_home"]) in surprise_keys, axis=1)
    away_flag = preds.apply(lambda r: (r["season"], r["team_away"]) in surprise_keys, axis=1)
    return home_flag | away_flag


def run(seasons: list[str] | None = None, min_train_seasons: int = 3) -> dict:
    seasons = seasons or football_data.default_completed_seasons(n=8)
    extra_frame, extra_cols = _extra_feature_frame(seasons)

    baseline_folds = walk_forward.prepare_folds(seasons, min_train_seasons)
    candidate_folds = walk_forward.prepare_folds(
        seasons,
        min_train_seasons,
        extra_feature_frame=extra_frame,
        extra_feature_cols=extra_cols,
        extra_merge_keys=["season", "team_home", "team_away"],
    )

    baseline_preds = pd.concat([_fold_predictions(f) for f in baseline_folds], ignore_index=True)
    candidate_preds = pd.concat([_fold_predictions(f) for f in candidate_folds], ignore_index=True)

    # The baseline model defines "surprising" by design — the segment
    # must not be built using the candidate's own predictions, or a
    # circularity would let the candidate "predict" a segment defined by
    # its own errors.
    surprise = surprise_team_seasons(baseline_preds)
    surprise_keys = set(zip(surprise[surprise["is_surprise"]]["season"], surprise[surprise["is_surprise"]]["team"]))

    def _segment(preds: pd.DataFrame) -> pd.DataFrame:
        return preds[_segment_mask(preds, surprise_keys)]

    per_season_baseline = baseline_preds.groupby("season").apply(lambda g: pd.Series(_rps_brier(g)))
    per_season_candidate = candidate_preds.groupby("season").apply(lambda g: pd.Series(_rps_brier(g)))

    return {
        "surprise_composition": surprise[surprise["is_surprise"]],
        "n_team_seasons_considered": len(surprise),
        "overall_baseline": _rps_brier(baseline_preds),
        "overall_candidate": _rps_brier(candidate_preds),
        "segment_baseline": _rps_brier(_segment(baseline_preds)),
        "segment_candidate": _rps_brier(_segment(candidate_preds)),
        "per_season_baseline": per_season_baseline,
        "per_season_candidate": per_season_candidate,
    }


if __name__ == "__main__":
    results = run()

    print(f"=== Surprise team-seasons (top/bottom by |actual - expected| PPG, first {SURPRISE_GAMES} games) ===")
    print(f"{len(results['surprise_composition'])} of {results['n_team_seasons_considered']} team-seasons flagged.")
    print(results["surprise_composition"].to_string(index=False))

    print("\n=== Overall (every fixture) ===")
    print("baseline: ", results["overall_baseline"])
    print("candidate:", results["overall_candidate"])

    print("\n=== Segment (fixtures involving a surprise team-season) ===")
    print("baseline: ", results["segment_baseline"])
    print("candidate:", results["segment_candidate"])

    print("\n=== Per-season, overall (most-recent-season corroboration check) ===")
    print("baseline:")
    print(results["per_season_baseline"])
    print("candidate:")
    print(results["per_season_candidate"])
