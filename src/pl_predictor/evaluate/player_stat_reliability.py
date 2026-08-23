"""player_stat_reliability.py — does FPL's ICT Index (and the rest of the
stat surface `features/player_form.py` now computes) actually predict a
player's *future* output, or is it noise/redundant with the simple rolling
goals/assists rate this project already uses?

Direct answer to the user's question ("is FPLs ICT influence and other
measures actually reliable?") rather than an assumption either way — same
"test before believing" discipline used for every match-model feature this
session.

No extra walk-forward loop needed here the way `evaluate/walk_forward.py`
needs one for the match model: `player_form.py::build_historical_player_form`
already computes every rolling feature with `shift(1)`, so a row's feature
values only ever reflect that player's *prior* appearances — the row's own
`goals_scored`/`assists` (this match's actual outcome) is therefore already
a legitimate future-prediction target, not something the features could see.
What this module adds is a train/test split *by season* (mirroring
`chronological_split`'s discipline) so a stat's apparent value isn't just
overfit noise from evaluating on the same data its correlation was read
off of, plus an *incremental* check (does adding this stat to the existing
goals/assists-rate baseline reduce error, not just correlate on its own —
a stat can correlate with output for boring reasons, like both being driven
by "this player is just good," without adding anything the existing
baseline doesn't already capture).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..data import fpl_history
from ..features import player_form

# stat -> (target column, baseline feature the stat is being tested
# *incrementally* against). Baseline is always the existing goals/assists
# rate this project already computes — the question is whether each
# candidate adds signal beyond that, not whether it correlates in isolation.
CANDIDATES = {
    "ict_index_last10": ("goal_involvement", "goals_per90_last10"),
    "influence_last10": ("goal_involvement", "goals_per90_last10"),
    "creativity_last10": ("assists", "assists_per90_last10"),
    "threat_last10": ("goals_scored", "goals_per90_last10"),
    "expected_goals_per90_last10": ("goals_scored", "goals_per90_last10"),
    "expected_assists_per90_last10": ("assists", "assists_per90_last10"),
    "expected_goal_involvements_per90_last10": ("goal_involvement", "goals_per90_last10"),
    "bps_last10": ("goal_involvement", "goals_per90_last10"),
    "bonus_last10": ("goal_involvement", "goals_per90_last10"),
    "defensive_contribution_last10": ("goal_involvement", "goals_per90_last10"),
}


def _prepare(seasons: list[str] | None = None) -> pd.DataFrame:
    df = fpl_history.load_player_gw_history(seasons=seasons)
    played, _ = player_form.build_historical_player_form(df)
    played["goal_involvement"] = played["goals_scored"] + played["assists"]
    return played


def evaluate_candidates(seasons: list[str] | None = None, test_season: str | None = None) -> pd.DataFrame:
    """One row per candidate stat: correlation with its target (on the held-
    out season), and the R² improvement from adding it to the existing
    goals/assists-rate baseline (also held-out). Positive `r2_gain` means
    the stat adds real signal beyond what this project already uses;
    ~zero/negative means it's redundant or noise for this purpose."""
    played = _prepare(seasons)
    test_season = test_season or sorted(played["season"].unique())[-1]
    train = played[played["season"] != test_season]
    test = played[played["season"] == test_season]

    rows = []
    for stat, (target, baseline_col) in CANDIDATES.items():
        cols_needed = [stat, baseline_col, target]
        tr = train.dropna(subset=cols_needed)
        te = test.dropna(subset=cols_needed)
        if len(tr) < 50 or len(te) < 20:
            rows.append({"stat": stat, "target": target, "n_train": len(tr), "n_test": len(te), "corr": None, "r2_baseline": None, "r2_with_stat": None, "r2_gain": None})
            continue

        corr = float(te[stat].corr(te[target]))

        X_train_base = tr[[baseline_col]].to_numpy()
        X_test_base = te[[baseline_col]].to_numpy()
        y_train = tr[target].to_numpy()
        y_test = te[target].to_numpy()

        base_model = LinearRegression().fit(X_train_base, y_train)
        r2_baseline = r2_score(y_test, base_model.predict(X_test_base))

        X_train_full = tr[[baseline_col, stat]].to_numpy()
        X_test_full = te[[baseline_col, stat]].to_numpy()
        full_model = LinearRegression().fit(X_train_full, y_train)
        r2_with_stat = r2_score(y_test, full_model.predict(X_test_full))

        rows.append(
            {
                "stat": stat,
                "target": target,
                "n_train": len(tr),
                "n_test": len(te),
                "corr": corr,
                "r2_baseline": r2_baseline,
                "r2_with_stat": r2_with_stat,
                "r2_gain": r2_with_stat - r2_baseline,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = evaluate_candidates()
    pd.set_option("display.width", 160)
    print(result.to_string(index=False))
