"""current_season_check.py — a genuinely live, out-of-sample check: does a
candidate arm actually predict the *current, in-progress* season's
completed fixtures better than production, using only data available
before the season started?

This is deliberately NOT a replacement for the multi-season chronological/
walk-forward comparison in `scoreline_dominance_arms.py` — with a whole
Premier League season at 380 fixtures, and the current season starting
with a single-digit-to-low-double-digit gameweek count of completed
matches, the sample here is small enough that per-fixture noise can easily
swamp a real effect. `has_enough_power` below gives an explicit, printed
answer to "is this big enough to decide anything" rather than leaving that
judgment implicit — treat a `False` there as "directionally noted, not
decisive," and re-run this as more of the season completes; it only grows
more informative from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import football_data
from ..features.build import build_training_frame
from ..models import market_models, ml_scoreline
from .scoreline_dominance_arms import ARM_SPECS, _dominance_extra_frame, _fold_metrics

# Rule of thumb, not a statistical proof: below this many completed
# fixtures, a count-market MAE/RPS difference this data has historically
# shown between arms (order of a few hundredths to a few tenths) is not
# reliably distinguishable from noise. Exists so the tool says so itself
# rather than requiring the reader to notice a tiny n in a table.
MIN_FIXTURES_FOR_A_DECISION = 60


def _current_season_frame(historical_seasons: list[str], current_partial: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    matches_df = pd.concat([football_data.load_training_data(seasons=historical_seasons), current_partial], ignore_index=True)
    matches_df = matches_df.sort_values("date").reset_index(drop=True)
    df, feature_cols = build_training_frame(matches_df=matches_df)
    current_season_label = football_data.season_str(football_data.CURRENT_SEASON_START_YEAR)
    return df, feature_cols, current_season_label


def evaluate_scoreline_arms_on_current_season() -> pd.DataFrame:
    """1X2/RPS for each arm's model, trained on strictly pre-season data,
    scored against the current season's completed fixtures only."""
    current_partial = football_data.fetch_current_season_partial()
    if current_partial is None or current_partial.empty:
        raise RuntimeError("No completed current-season fixtures yet — nothing to check.")

    rows = []
    for arm_name, spec in ARM_SPECS.items():
        historical_seasons = football_data.default_completed_seasons(n=spec["n_seasons"])
        df, feature_cols, current_season_label = _current_season_frame(historical_seasons, current_partial)
        ml_feature_cols = [c for c in feature_cols if "fouls" not in c]

        if spec["with_dominance"]:
            extra_frame, extra_cols = _dominance_extra_frame(historical_seasons)
            df = df.copy()
            df["kickoff_date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.merge(extra_frame, on=["kickoff_date", "team_home", "team_away"], how="left")
            ml_feature_cols = ml_feature_cols + list(extra_cols)

        train_df = df[df["season"].isin(historical_seasons)]
        val_df = df[df["season"] == current_season_label]
        if val_df.empty:
            continue
        X_train, X_val = train_df[ml_feature_cols].fillna(0), val_df[ml_feature_cols].fillna(0)

        home_model, away_model = ml_scoreline.train_goal_regressors(X_train, train_df["goals_home"], train_df["goals_away"])
        metrics = _fold_metrics(home_model, away_model, X_val, val_df)
        rows.append({"arm": arm_name, "n_fixtures": len(val_df), **metrics})

    result = pd.DataFrame(rows)
    if not result.empty:
        result.attrs["has_enough_power"] = bool(result["n_fixtures"].iloc[0] >= MIN_FIXTURES_FOR_A_DECISION)
    return result


def evaluate_count_market_arms_on_current_season(target: str) -> pd.DataFrame:
    """Same idea for a count market (`"total_corners"`/`"total_cards"`)."""
    line = 9.5 if target == "total_corners" else 3.5
    current_partial = football_data.fetch_current_season_partial()
    if current_partial is None or current_partial.empty:
        raise RuntimeError("No completed current-season fixtures yet — nothing to check.")

    rows = []
    for arm_name, spec in ARM_SPECS.items():
        historical_seasons = football_data.default_completed_seasons(n=spec["n_seasons"])
        df, feature_cols, current_season_label = _current_season_frame(historical_seasons, current_partial)

        if spec["with_dominance"]:
            extra_frame, extra_cols = _dominance_extra_frame(historical_seasons)
            df = df.copy()
            df["kickoff_date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.merge(extra_frame, on=["kickoff_date", "team_home", "team_away"], how="left")
            feature_cols = feature_cols + list(extra_cols)

        train_df = df[df["season"].isin(historical_seasons)]
        val_df = df[df["season"] == current_season_label]
        if val_df.empty or target not in val_df.columns:
            continue
        X_train, X_val = train_df[feature_cols].fillna(0), val_df[feature_cols].fillna(0)

        model = market_models.train_lambda_regressor(X_train, train_df[target])
        preds = model.predict(X_val)
        actual = val_df[target].to_numpy()

        rows.append(
            {
                "arm": arm_name,
                "target": target,
                "n_fixtures": len(val_df),
                "mae": float(np.mean(np.abs(preds - actual))),
                "rmse": float(np.sqrt(np.mean((preds - actual) ** 2))),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result.attrs["has_enough_power"] = bool(result["n_fixtures"].iloc[0] >= MIN_FIXTURES_FOR_A_DECISION)
    return result


if __name__ == "__main__":
    for target in ("total_corners", "total_cards"):
        result = evaluate_count_market_arms_on_current_season(target)
        print(f"\n--- {target} (current season) ---")
        print(result.to_string(index=False))
        n = result["n_fixtures"].iloc[0] if not result.empty else 0
        if n < MIN_FIXTURES_FOR_A_DECISION:
            print(
                f"  ! Only {n} completed fixtures so far — below the "
                f"{MIN_FIXTURES_FOR_A_DECISION}-fixture rule-of-thumb floor. "
                "Directionally noted only; not a basis for a decision yet."
            )
