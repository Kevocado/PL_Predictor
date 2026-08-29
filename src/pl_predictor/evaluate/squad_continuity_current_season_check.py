"""squad_continuity_current_season_check.py — a genuinely live, out-of-
sample check for EXP-2026-18's squad-continuity feature (docs/
AI_CONTINUITY.md): does keeping `home_squad_continuity`/
`away_squad_continuity` in `ml_scoreline`'s feature set actually predict
the *actual, in-progress* 2026-27 season's completed fixtures better than
dropping them, using only data available before the season started?

Same discipline as `current_season_check.py` (EXP-2026-11's live-check
precedent) — deliberately NOT a replacement for EXP-2026-18's own 5-fold
historical walk-forward result, which is the primary evidence behind the
promotion decision. A handful of completed current-season fixtures is far
below `MIN_FIXTURES_FOR_A_DECISION`; treat any result here as directionally
noted, not decisive on its own, and re-run as the season progresses.
"""

from __future__ import annotations

import pandas as pd

from ..data import football_data
from ..features.build import build_training_frame
from ..models import ml_scoreline
from .current_season_check import MIN_FIXTURES_FOR_A_DECISION
from .scoreline_dominance_arms import _fold_metrics

SQUAD_CONTINUITY_COLS = ["home_squad_continuity", "away_squad_continuity"]


def run(n_seasons: int = 8) -> pd.DataFrame:
    historical_seasons = football_data.default_completed_seasons(n=n_seasons)
    current_partial = football_data.fetch_current_season_partial()
    if current_partial is None or current_partial.empty:
        raise RuntimeError("No completed current-season fixtures yet — nothing to check.")

    matches_df = (
        pd.concat([football_data.load_training_data(seasons=historical_seasons), current_partial], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    df, feature_cols = build_training_frame(matches_df=matches_df)
    ml_feature_cols = [c for c in feature_cols if "fouls" not in c]

    current_season_label = football_data.season_str(football_data.CURRENT_SEASON_START_YEAR)
    train_df = df[df["season"].isin(historical_seasons)]
    val_df = df[df["season"] == current_season_label]
    if val_df.empty:
        raise RuntimeError("No completed current-season fixtures yet — nothing to check.")

    rows = []
    for arm_name, cols in [
        ("baseline_without_squad_continuity", [c for c in ml_feature_cols if c not in SQUAD_CONTINUITY_COLS]),
        ("candidate_with_squad_continuity", ml_feature_cols),
    ]:
        X_train, X_val = train_df[cols].fillna(0), val_df[cols].fillna(0)
        home_model, away_model = ml_scoreline.train_goal_regressors(
            X_train, train_df["goals_home"], train_df["goals_away"]
        )
        metrics = _fold_metrics(home_model, away_model, X_val, val_df)
        rows.append({"arm": arm_name, "n_fixtures": len(val_df), **metrics})

    result = pd.DataFrame(rows)
    result.attrs["has_enough_power"] = bool(result["n_fixtures"].iloc[0] >= MIN_FIXTURES_FOR_A_DECISION)
    return result


if __name__ == "__main__":
    result = run()
    print(result.to_string(index=False))
    n = int(result["n_fixtures"].iloc[0])
    if n < MIN_FIXTURES_FOR_A_DECISION:
        print(
            f"\n  ! Only {n} completed fixtures so far — below the "
            f"{MIN_FIXTURES_FOR_A_DECISION}-fixture rule-of-thumb floor. "
            "Directionally noted only; not a basis for a decision on its own."
        )
