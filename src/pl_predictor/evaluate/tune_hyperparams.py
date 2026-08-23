"""tune_hyperparams.py — Optuna hyperparameter search for ml_scoreline.

Not wired into `train_all`/the hourly auto-retrain loop — same reasoning as
`walk_forward.py`: this is a periodic/manual tool, run directly, not part
of the live retrain cycle. The objective is `walk_forward_validate`'s mean
RPS across 5 rolling folds, not the single fixed `chronological_split`
holdout — with only one 380-row validation season, running many trials
against a single season risks overfitting the *hyperparameters* to that
season's specific quirks the same way individual features could (and did,
repeatedly, this session). Averaging across folds makes that much harder.

Trial budget is deliberately modest (well under 100) for the same reason —
more trials chasing marginal single-run gains just increases the chance of
finding a config that happens to suit these particular folds rather than
one that's genuinely better.

The study is stored in SQLite (`PROJECT_ROOT/models/optuna_study.db`) so a
later call with `load_if_exists=True` can add more trials incrementally
instead of restarting cold — this is what "fine-tune as the season
progresses" should mean in practice.
"""

from __future__ import annotations

import optuna

from ..config import MODELS_DIR
from .walk_forward import evaluate_folds, prepare_folds

STUDY_DB_PATH = MODELS_DIR / "optuna_study.db"
STUDY_NAME = "ml_scoreline_hyperparams"


def _suggest_hyperparams(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
    }


def run_study(n_trials: int = 40, seed: int = 42) -> tuple[optuna.Study, float]:
    """Runs (or resumes) the tuning study. `prepare_folds()` (the expensive,
    hyperparameter-independent feature-engineering step) runs exactly once
    here and is reused across every trial via the objective closure — see
    `walk_forward.py::prepare_folds`'s docstring for why that matters.
    Returns `(study, baseline_rps)` — the baseline is production defaults'
    mean RPS on these same prepared folds, for a fair before/after
    comparison."""
    folds = prepare_folds()
    baseline_rps = float(evaluate_folds(folds)["rps"].mean())

    def objective(trial: optuna.Trial) -> float:
        hyperparams = _suggest_hyperparams(trial)
        return float(evaluate_folds(folds, hyperparams)["rps"].mean())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study, baseline_rps


if __name__ == "__main__":
    study, baseline = run_study()
    print(f"Baseline (production defaults) mean RPS: {baseline:.5f}")
    print(f"Best trial mean RPS: {study.best_value:.5f}")
    print(f"Best hyperparams: {study.best_params}")
    if study.best_value < baseline:
        print(f"Improvement: {baseline - study.best_value:.5f} ({(baseline - study.best_value) / baseline * 100:.2f}%)")
    else:
        print("No improvement over production defaults — keeping current config.")
