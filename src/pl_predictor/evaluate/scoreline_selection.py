"""How much of the published scoreline metric is selection rather than skill?

`models/manifest.py` chooses the scoreline model with

    chosen = min(candidates, key=candidates.get)

against the same `val_df` whose metrics it then writes into `manifest.json`. So
the headline RPS is a *best-of-four on 380 fixtures*, not the performance of a
model picked in advance. The reported value and `market_metrics.ml_scoreline.rps`
agree to 14 significant figures precisely because they are the same number, which
is the tell: the "holdout" is a selection set, and a holdout that is also a
selection set is no longer held out.

This module measures the gap instead of arguing about it, and produces an
unbiased alternative via **nested selection**: on each fold, the candidate is
chosen using only *earlier* folds and then scored on the current one. That is the
same walk-forward discipline the rest of this project already uses for model
changes, applied to the act of choosing a model.

Three quantities, all reported per candidate and as an aggregate:

- ``walk_forward_mean`` — the mean RPS over folds. Stable, and what a headline
  number should be, but still mildly contaminated: the candidate is named after
  seeing all the folds, including the one it is reported on.
- ``nested_selection`` — choose on earlier folds, score on the current one.
  Uncontaminated, at the cost of one fewer usable fold.
- ``incumbent_min_on_fold`` — what the project publishes today.

and ``optimism``, the gap between the first and the second *on the folds both
cover*, which is the number that says whether any of this matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import covariate_poisson, ml_scoreline, scoreline
from ..models.manifest import SELECTION_WALK_FORWARD_PATH
from .walk_forward import prepare_folds

# Where the diagnostics are written. Imported from `manifest` rather than
# recomputed: the writer and the reader must agree on the path by construction,
# and a second hardcoded path is how a retrain ends up quietly reporting
# "not computed" while a perfectly good file sits elsewhere on disk.

CANDIDATES = ("dixon_coles", "bivariate_poisson", "ml_scoreline", "covariate_poisson")


def evaluate_candidates(folds: list[dict]) -> pd.DataFrame:
    """One row per (fold, candidate) with that candidate's 1X2 RPS and Brier.

    Every candidate sees the identical `train_df`/`val_df` split, so the
    comparison isolates the model rather than the fold.
    """
    rows: list[dict] = []
    for fold in folds:
        train_df, val_df = fold["train_df"], fold["val_df"]
        grids = {
            "dixon_coles": scoreline.predict_grids_for_fixed_param_model(
                scoreline.fit_dixon_coles(train_df), val_df),
            "bivariate_poisson": scoreline.predict_grids_for_fixed_param_model(
                scoreline.fit_bivariate_poisson(train_df), val_df),
            "ml_scoreline": ml_scoreline.predict_grids_batch(
                *ml_scoreline.train_goal_regressors(
                    fold["X_train"], train_df["goals_home"], train_df["goals_away"]),
                fold["X_val"]),
            "covariate_poisson": covariate_poisson.predict_grids_batch(
                covariate_poisson.fit(train_df), val_df),
        }
        for candidate, candidate_grids in grids.items():
            metrics = scoreline.evaluate_grids_multi_market(candidate_grids, val_df)
            rows.append({
                "val_season": fold["val_season"],
                "n_train": len(train_df),
                "n_val": len(val_df),
                "candidate": candidate,
                "rps": metrics["rps"],
                # `evaluate_grids_multi_market` qualifies these by market: the
                # unqualified `brier` belongs to no single market, so a reader
                # comparing these to a 1X2 Brier elsewhere is not comparing like
                # with like. The per-match bootstrap bounds come along free and
                # are what makes a selection-aware interval possible.
                "brier_1x2": metrics["brier_1x2"],
                "log_loss_1x2": metrics["log_loss_1x2"],
                "rps_ci_low": metrics["rps_ci_low"],
                "rps_ci_high": metrics["rps_ci_high"],
            })
    return pd.DataFrame(rows)


def per_fold_rps(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot to fold x candidate. Used by every selection rule below."""
    return metrics.pivot_table(index="val_season", columns="candidate", values="rps").sort_index()


def nested_selection_rps(per_fold: pd.DataFrame) -> tuple[float, list[str]]:
    """Score each fold using the candidate that earlier folds favoured.

    The first fold has no earlier evidence, so it is not scored — reporting it
    would mean selecting on the fold being scored, which is the thing being
    measured. Returns the mean and the per-fold choice, so the choices can be
    inspected rather than trusted.
    """
    scores: list[float] = []
    chosen: list[str] = []
    for position in range(1, len(per_fold)):
        earlier = per_fold.iloc[:position].mean(axis=0)
        candidate = earlier.idxmin()
        scores.append(float(per_fold.iloc[position][candidate]))
        chosen.append(f"{per_fold.index[position]}:{candidate}")
    if not scores:
        return float("nan"), []
    return float(np.mean(scores)), chosen


def summarise(metrics: pd.DataFrame) -> dict:
    """The three numbers, plus how much the selection flatters the published one.

    `optimism` is `nested_selection - incumbent_min_on_fold`, **measured on the same
    folds**. That last clause is the whole correctness of the comparison: nested
    selection cannot score the opening fold, so comparing its mean against an
    all-folds mean would charge the honest estimator for a fold the optimistic
    one got to keep. On the shared folds, a positive `optimism` means the
    published figure is better than a selection that could not see the fold it is
    reported on -- which is the bias, quantified. Zero means the candidates were
    unambiguous and selection bought nothing.
    """
    per_fold = per_fold_rps(metrics)
    means = per_fold.mean(axis=0).sort_values()
    nested, chosen = nested_selection_rps(per_fold)

    # The folds nested selection actually scored, and the incumbent on those same
    # folds. Comparing over a shared index is the only fair pairing.
    scored = per_fold.iloc[1:]
    incumbent_shared = float(scored.min(axis=1).mean()) if len(scored) else float("nan")
    optimism = nested - incumbent_shared if len(scored) else float("nan")

    return {
        "walk_forward_mean": means,
        "best_by_walk_forward_mean": str(means.index[0]),
        # Every fold, as the project publishes it today.
        "incumbent_min_on_fold": float(per_fold.min(axis=1).mean()),
        # The shared-fold pairing the optimism figure is built from.
        "incumbent_min_on_fold_shared": incumbent_shared,
        "nested_selection": nested,
        "nested_choices": chosen,
        "optimism": optimism,
        "n_folds": int(len(per_fold)),
        "n_scored_by_nested": int(len(scored)),
    }


def run(seasons: list[str] | None = None, min_train_seasons: int = 3) -> dict:
    """Prepare folds, evaluate all four candidates on each, and summarise."""
    folds = prepare_folds(seasons=seasons, min_train_seasons=min_train_seasons)
    metrics = evaluate_candidates(folds)
    return {"metrics": metrics, "per_fold": per_fold_rps(metrics), **summarise(metrics)}


def to_record(result: dict) -> dict:
    """The JSON-serialisable subset `models/manifest.py` embeds.

    Only summary figures: the manifest is read by the API on every request, and
    a per-fold per-candidate table would be several hundred rows of numbers
    nothing displays. The per-fold table stays available from `run()`.
    """
    return {
        "n_folds": result["n_folds"],
        "n_scored_by_nested": result["n_scored_by_nested"],
        "walk_forward_mean": {k: float(v) for k, v in result["walk_forward_mean"].items()},
        "best_by_walk_forward_mean": result["best_by_walk_forward_mean"],
        "incumbent_min_on_fold": result["incumbent_min_on_fold"],
        "incumbent_min_on_fold_shared": result["incumbent_min_on_fold_shared"],
        "nested_selection": result["nested_selection"],
        "optimism": result["optimism"],
        "nested_choices": list(result["nested_choices"]),
    }


def write_cache(result: dict, path: Path = SELECTION_WALK_FORWARD_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_record(result), indent=2, sort_keys=True))
    return path


def main(seasons: list[str] | None = None, min_train_seasons: int = 3) -> None:
    result = run(seasons=seasons, min_train_seasons=min_train_seasons)
    print(result["per_fold"].round(6).to_string())
    print()
    for candidate, mean in result["walk_forward_mean"].items():
        print(f"  walk-forward mean  {candidate:<22} {mean:.6f}")
    print(f"  -> best by walk-forward mean: {result['best_by_walk_forward_mean']}")
    print()
    print(f"  published today (min per fold, {result['n_folds']} folds) : {result['incumbent_min_on_fold']:.6f}")
    print(f"  on the {result['n_scored_by_nested']} folds nested selection scores: "
          f"{result['incumbent_min_on_fold_shared']:.6f}")
    print(f"  nested selection, uncontaminated                    : {result['nested_selection']:.6f}")
    print(f"  OPTIMISM in the published number                     : {result['optimism']:+.6f}")
    print(f"  nested choices: {', '.join(result['nested_choices'])}")
    print(f"\n  wrote {write_cache(result)}")


if __name__ == "__main__":
    main()
