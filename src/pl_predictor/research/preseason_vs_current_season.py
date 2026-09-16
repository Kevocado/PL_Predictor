"""preseason_vs_current_season.py — does training on current-season match
results (as production's auto-retrain actually does) help or hurt 1X2
prediction, versus training only on prior, fully-completed seasons?

This replaces an earlier, broken attempt at the same question
(`research/seasonal_degradation.py`, in a separate worktree): that version
compared 44 tracking-db predictions from one partial season (4 gameweeks) —
its own hypothesis test came back n=3 paired gameweeks, nowhere near enough
to act on, and its findings doc was never actually filled in (its entry
point had a `NameError` bug that meant it had never successfully run before
this module replaced it).

This instead reuses `evaluate/walk_forward.py`'s existing multi-season
fold machinery — the project's own established way of testing something
across the 8 real completed seasons of ml_scoreline's actual training data,
not one partial season's live-tracked bets:

- **Pre-season-only arm**: exactly walk_forward.py's existing fold —
  train once on strictly earlier seasons, evaluate the whole validation
  season with that one frozen model.
- **Current-season-including arm**: simulates what the live app's hourly
  auto-retrain (`api/routes.py::maybe_auto_retrain`) actually does —
  retrain after every gameweek's results land, using prior seasons *plus*
  the validation season's own matches played so far, then predict the next
  gameweek. The very first gameweek of a season has no current-season data
  yet either way, so both arms agree there.

Same `ml_scoreline.train_goal_regressors` defaults as production (no
`dates` recency weighting — see `models/manifest.py::train_all`'s own
comment on why that's deliberately unweighted for this model), so this is
an apples-to-apples comparison against what's actually deployed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy import stats
from sklearn.metrics import log_loss

from ..evaluate.walk_forward import prepare_folds
from ..models import ml_scoreline

# The PL has 20 teams -> 10 matches per gameweek. There's no explicit
# gameweek column in football-data.co.uk's training frame (unlike
# football-data.org's `matchday`), so chronological batches of 10 stand in
# for "one gameweek's worth of new results" -- the same granularity
# production's auto-retrain effectively reacts at (it checks hourly, but
# only retrains once *any* new match exists, so in practice it re-trains
# roughly this often during a normal matchweek).
MATCHES_PER_GAMEWEEK = 10


def chunk_by_gameweek(val_df: pd.DataFrame, chunk_size: int = MATCHES_PER_GAMEWEEK) -> list[pd.DataFrame]:
    """Splits one season's held-out matches into chronological gameweek-
    sized batches. The last chunk may be smaller than `chunk_size` (a
    season's match count isn't always an exact multiple, e.g. after
    postponements are folded back in)."""
    ordered = val_df.sort_values("date").reset_index(drop=True)
    return [ordered.iloc[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]


def _probs_and_outcomes(home_model, away_model, X: pd.DataFrame, val_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    result_code = {"H": 0, "D": 1, "A": 2}
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, X)
    probs = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
    outcomes = val_df["ftr"].map(result_code).to_numpy()
    return probs, outcomes


def _metrics_from_probs(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "rps": float(pb.metrics.rps_average(probs, outcomes)),
        "brier": float(pb.metrics.multiclass_brier_score(probs, outcomes)),
        "log_loss": float(log_loss(outcomes, probs, labels=[0, 1, 2])),
        "n": int(len(outcomes)),
    }


def evaluate_preseason_only(fold: dict, ml_feature_cols: list[str]) -> dict:
    """One model, trained once on strictly-prior seasons, scored against
    the entire validation season -- identical to what
    `evaluate/walk_forward.py::evaluate_folds` already computes."""
    home_model, away_model = ml_scoreline.train_goal_regressors(
        fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
    )
    probs, outcomes = _probs_and_outcomes(home_model, away_model, fold["X_val"], fold["val_df"])
    return _metrics_from_probs(probs, outcomes)


def evaluate_current_season_including(fold: dict, ml_feature_cols: list[str]) -> dict:
    """Retrains after every gameweek-sized chunk of the validation season,
    each time on (prior seasons + validation season matches played so far),
    then scores the *next* chunk with that freshly retrained model --
    concatenating every chunk's predictions before computing one RPS/Brier
    for the whole season, not an average-of-averages across
    differently-sized chunks."""
    chunks = chunk_by_gameweek(fold["val_df"])
    cumulative_train_df = fold["train_df"]
    cumulative_X = fold["X_train"]

    all_probs, all_outcomes = [], []
    for chunk in chunks:
        home_model, away_model = ml_scoreline.train_goal_regressors(
            cumulative_X, cumulative_train_df["goals_home"], cumulative_train_df["goals_away"]
        )
        X_chunk = chunk[ml_feature_cols].fillna(0)
        probs, outcomes = _probs_and_outcomes(home_model, away_model, X_chunk, chunk)
        all_probs.append(probs)
        all_outcomes.append(outcomes)

        cumulative_train_df = pd.concat([cumulative_train_df, chunk])
        cumulative_X = pd.concat([cumulative_X, X_chunk])

    return _metrics_from_probs(np.concatenate(all_probs), np.concatenate(all_outcomes))


def run_study(seasons: list[str] | None = None, min_train_seasons: int = 3) -> dict:
    """One (pre-season-only, current-season-including) RPS/Brier pair per
    validation season, plus a paired t-test across seasons -- season-level
    pairs (n = number of validation seasons), not the gameweek-level pairs
    an earlier, broken version of this study used on a single partial
    season (n=3, not remotely enough to act on)."""
    folds = prepare_folds(seasons=seasons, min_train_seasons=min_train_seasons)
    ml_feature_cols = list(folds[0]["X_train"].columns) if folds else []

    rows = []
    for fold in folds:
        preseason = evaluate_preseason_only(fold, ml_feature_cols)
        current = evaluate_current_season_including(fold, ml_feature_cols)
        rows.append(
            {
                "val_season": fold["val_season"],
                "n_train": len(fold["train_df"]),
                "n_val": len(fold["val_df"]),
                "preseason_rps": preseason["rps"],
                "preseason_brier": preseason["brier"],
                "current_rps": current["rps"],
                "current_brier": current["brier"],
            }
        )
    per_season = pd.DataFrame(rows)

    if len(per_season) >= 2:
        rps_test = stats.ttest_rel(per_season["preseason_rps"], per_season["current_rps"])
        brier_test = stats.ttest_rel(per_season["preseason_brier"], per_season["current_brier"])
        hypothesis_test = {
            "n_seasons": len(per_season),
            "rps_t_statistic": float(rps_test.statistic),
            "rps_p_value": float(rps_test.pvalue),
            "brier_t_statistic": float(brier_test.statistic),
            "brier_p_value": float(brier_test.pvalue),
            "mean_preseason_rps": float(per_season["preseason_rps"].mean()),
            "mean_current_rps": float(per_season["current_rps"].mean()),
        }
    else:
        hypothesis_test = {"n_seasons": len(per_season), "note": "fewer than 2 validation seasons -- no test run"}

    return {"per_season": per_season, "hypothesis_test": hypothesis_test}


if __name__ == "__main__":
    result = run_study()
    print(result["per_season"].to_string(index=False))
    print()
    print(result["hypothesis_test"])
