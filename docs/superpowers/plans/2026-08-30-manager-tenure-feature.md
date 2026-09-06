# Manager Tenure Feature (EXP-2026-23) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a leading indicator of managerial change — days
since the current manager's appointment, and whether the team enters a
fixture under a different manager than it finished the prior season with —
improves `ml_scoreline`'s walk-forward RPS/Brier/log loss. This is the
"manager-change flag" follow-on explicitly named but never started in
EXP-2026-18 (`docs/AI_CONTINUITY.md`), following the same
compute-as-candidate-via-`extra_feature_frame`, evaluate,
promote-or-reject-and-document protocol as EXP-2026-18/22.

**Architecture:** A curated, hand-researched lookup table of Premier League
managerial tenures (`data/manager_history.py`) — a finite, stable, publicly
documented dataset, not a live API — feeds a new feature module
(`features/manager_tenure.py`) that computes, for any (team, date), tenure
length and a same-manager-as-last-season flag. A new evaluation script
(`evaluate/manager_tenure_prior.py`, mirroring `evaluate/
squad_change_prior.py`'s shape) wires this into `evaluate.walk_forward.
prepare_folds`'s existing `extra_feature_frame` mechanism for the standard
5-fold walk-forward comparison.

**Tech Stack:** Python, `pandas` (already a project dependency — no new
packages, no new external HTTP calls at serving or training time; the
manager-history table is static data checked into the repo).

**Spec:** No separate spec file — the architecture above was agreed
directly in chat and approved via `AskUserQuestion`, the same lightweight
path EXP-2026-18/22's plans took.

## Global Constraints

- No-lookahead: a manager's appointment date is historical fact once it has
  happened, so this feature has no lookahead risk by construction — unlike
  weather or squad continuity, there is no "was this knowable yet" edge
  case to design around, only "is the table itself accurate."
- Data accuracy is the real risk here, not licensing or lookahead: Task 1
  requires actually researching and verifying real managerial dates from a
  stable public source before this table is trusted for evaluation. Do not
  invent or guess dates — an inaccurate row silently produces a wrong
  feature value with no test able to catch it (unit tests here can only
  check internal consistency, e.g. no overlapping tenures, not historical
  truth).
- Never merge this feature into `features/build.py`'s production
  `feature_cols` unless it clears the promotion gate in `docs/
  AI_CONTINUITY.md` (average walk-forward AND fixed most-recent-season
  holdout both improve, no material fold regression).
- Cover every team-season in the standard 8-season training window
  (`data.football_data.default_completed_seasons(n=8)`) — a team missing
  from the table must degrade to "no tenure signal" (NaN), never crash the
  feature builder, matching every other optional feature in this project
  (h2h, rest days, squad continuity).

---

### Task 1: Research and curate the manager-history table

**Files:**
- Create: `src/pl_predictor/data/manager_history.py`
- Test: `tests/test_manager_history.py`

**Interfaces:**
- Produces: `MANAGER_TENURES: list[dict]` — one dict per continuous
  managerial tenure, each with keys `team` (canonical short name, see
  `data/team_names.py::CANONICAL_TEAMS`), `manager` (str), `start_date`
  (`"YYYY-MM-DD"`, the date the manager took charge — first match in
  interim/caretaker charge counts as the start), and `end_date`
  (`"YYYY-MM-DD"` or `None` for a manager still in post at the time this
  table was last updated). `manager_asof(team: str, date) -> dict | None`
  — the tenure record covering `team` on `date`, or `None` if the table has
  no coverage for that team/date. Later tasks (features/manager_tenure.py)
  call `manager_asof`, never `MANAGER_TENURES` directly.

- [ ] **Step 1: Research the table**

Before writing any code, research actual Premier League managerial
tenures for every team that appears in `data.football_data.
default_completed_seasons(n=8)`'s window (currently 2018-19 through
2025-26 — confirm the exact list by running
`python -c "from pl_predictor.data import football_data; print(football_data.default_completed_seasons(n=8))"`)
against a stable public reference (e.g. Wikipedia's "List of Premier League
managers" and/or each club's own official managerial-history page — cross-
check at least two sources per team where dates are ambiguous, e.g. a
caretaker period). Build a plain list of dicts (`team`, `manager`,
`start_date`, `end_date`) covering every tenure change for every team in
that window. Treat a caretaker/interim spell as its own tenure entry if it
lasted more than 2 matches; fold a 1-2 match caretaker spell into the
surrounding permanent tenures' gap (record it as a short gap with no
entry — `manager_asof` returning `None` for a few days is an acceptable,
honestly-represented gap, not a bug to paper over with a guess).

This is a genuine research task, not a mechanical one — do not fabricate
dates. If a source is unclear or contradictory for a specific tenure,
prefer the club's own official site's history page as the tiebreaker, and
leave a one-line comment on that entry noting the ambiguity.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_manager_history.py
from pl_predictor.data import manager_history


def test_manager_asof_returns_the_tenure_covering_the_date():
    # Unai Emery has managed Aston Villa continuously since his November
    # 2022 appointment — a fixture well inside that tenure must resolve to it.
    record = manager_history.manager_asof("Aston Villa", "2024-01-15")
    assert record is not None
    assert record["manager"] == "Unai Emery"
    assert record["start_date"] <= "2024-01-15"
    assert record["end_date"] is None or record["end_date"] >= "2024-01-15"


def test_manager_asof_returns_none_for_a_date_with_no_coverage():
    assert manager_history.manager_asof("Not A Real Club", "2024-01-15") is None


def test_manager_tenures_do_not_overlap_for_the_same_team():
    """A data-integrity check on the curated table itself: two tenures for
    the same team must never claim the same date — this cannot verify
    historical *accuracy*, only that the table is internally consistent."""
    by_team: dict[str, list[dict]] = {}
    for tenure in manager_history.MANAGER_TENURES:
        by_team.setdefault(tenure["team"], []).append(tenure)

    for team, tenures in by_team.items():
        ordered = sorted(tenures, key=lambda t: t["start_date"])
        for earlier, later in zip(ordered, ordered[1:]):
            earlier_end = earlier["end_date"] or "9999-12-31"
            assert earlier_end <= later["start_date"], (
                f"{team}: {earlier['manager']} (ends {earlier['end_date']}) overlaps "
                f"{later['manager']} (starts {later['start_date']})"
            )


def test_every_team_in_the_standard_training_window_has_at_least_one_tenure():
    from pl_predictor.data import football_data

    seasons = football_data.default_completed_seasons(n=8)
    matches_df = football_data.load_training_data(seasons=seasons)
    teams_in_window = set(matches_df["team_home"]).union(matches_df["team_away"])

    covered_teams = {tenure["team"] for tenure in manager_history.MANAGER_TENURES}
    missing = teams_in_window - covered_teams
    assert not missing, f"No manager-history coverage for: {sorted(missing)}"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_manager_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pl_predictor.data.manager_history'`

- [ ] **Step 4: Write the implementation**

```python
# src/pl_predictor/data/manager_history.py
"""manager_history.py — a curated, hand-researched table of Premier League
managerial tenures, covering every team in this project's standard
8-season training window (see docs/AI_CONTINUITY.md EXP-2026-23 for the
motivation — the "manager-change flag" follow-on named but never started
in EXP-2026-18).

Unlike every other external-data module in this project, this is static
data checked into the repo, not a live fetch — Premier League managerial
history is a finite, stable, publicly documented record (Wikipedia's "List
of Premier League managers", club official sites), not something worth
building a scraper for. Accuracy depends entirely on the research done
when this table was built (see Task 1 of the plan this module came from) —
verify against a current source before trusting an entry for a team/date
not already covered here.

A caretaker/interim spell of 1-2 matches is deliberately omitted rather
than guessed at — `manager_asof` returning `None` for a few days around a
managerial change is an honest gap, not a bug.
"""

from __future__ import annotations

# One entry per continuous managerial tenure. `end_date=None` means still
# in post as of when this table was last updated (see the module docstring
# for the research/verification discipline before extending this table).
MANAGER_TENURES: list[dict] = [
    {"team": "Aston Villa", "manager": "Unai Emery", "start_date": "2022-11-01", "end_date": None},
    {"team": "Man United", "manager": "Erik ten Hag", "start_date": "2022-05-23", "end_date": "2024-10-28"},
    {"team": "Man United", "manager": "Ruben Amorim", "start_date": "2024-11-11", "end_date": None},
    {"team": "Tottenham", "manager": "Ange Postecoglou", "start_date": "2023-06-06", "end_date": "2025-06-06"},
    # NOTE: this table is a starting point, not a finished dataset — the
    # implementer completing Task 1 must research and add every remaining
    # team/tenure in the standard training window (run
    # `default_completed_seasons(n=8)` to get the exact team list) before
    # `test_every_team_in_the_standard_training_window_has_at_least_one_tenure`
    # will pass. The four entries above are a high-confidence starting
    # scaffold, not a claim that research is done.
]


def manager_asof(team: str, date: str) -> dict | None:
    """The tenure record covering `team` on `date` (`"YYYY-MM-DD"`), or
    `None` if the table has no coverage for that team/date (an unmapped
    team, or a gap between two tenures — see module docstring)."""
    for tenure in MANAGER_TENURES:
        if tenure["team"] != team:
            continue
        if tenure["start_date"] <= date and (tenure["end_date"] is None or date <= tenure["end_date"]):
            return tenure
    return None
```

- [ ] **Step 5: Complete the research and fill in the full table**

Using the research from Step 1, replace the scaffold `MANAGER_TENURES`
list above with the complete table covering every team in the standard
training window. Run the tests after each team you add to catch overlap
mistakes early rather than debugging the whole table at once.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_manager_history.py -v`
Expected: PASS (including
`test_every_team_in_the_standard_training_window_has_at_least_one_tenure`,
which will only pass once the table is genuinely complete)

- [ ] **Step 7: Commit**

```bash
git add src/pl_predictor/data/manager_history.py tests/test_manager_history.py
git commit -m "feat: add curated Premier League manager-tenure history table"
```

---

### Task 2: Manager tenure feature

**Files:**
- Create: `src/pl_predictor/features/manager_tenure.py`
- Test: `tests/test_manager_tenure_feature.py`

**Interfaces:**
- Consumes: `manager_history.manager_asof(team, date)` (Task 1).
- Produces: `fixture_manager_tenure_features(matches_df: pd.DataFrame) ->
  pd.DataFrame` with columns `kickoff_date`, `team_home`, `team_away`,
  `home_manager_tenure_days`, `away_manager_tenure_days`,
  `home_new_manager_this_season`, `away_new_manager_this_season` — shaped
  to merge directly onto `evaluate.walk_forward.prepare_folds`'s default
  `extra_feature_frame` merge keys (`kickoff_date`, `team_home`,
  `team_away`). `home_new_manager_this_season` is `True` when the manager
  in charge for this fixture started *after* the previous season's final
  matchday for that team (captures "different manager than we ended last
  season with", independent of exactly how long ago they were appointed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manager_tenure_feature.py
import pandas as pd

from pl_predictor.features import manager_tenure


def test_tenure_days_counts_from_the_managers_start_date(monkeypatch):
    monkeypatch.setattr(
        manager_tenure.manager_history,
        "manager_asof",
        lambda team, date: {"team": team, "manager": "Test Manager", "start_date": "2024-06-01", "end_date": None},
    )
    matches = pd.DataFrame(
        [{"season": "2024-2025", "date": pd.Timestamp("2024-08-17"), "team_home": "Arsenal", "team_away": "Chelsea"}]
    )

    result = manager_tenure.fixture_manager_tenure_features(matches)

    row = result.iloc[0]
    assert row["home_manager_tenure_days"] == (pd.Timestamp("2024-08-17") - pd.Timestamp("2024-06-01")).days
    assert row["away_manager_tenure_days"] == (pd.Timestamp("2024-08-17") - pd.Timestamp("2024-06-01")).days


def test_new_manager_this_season_flag_true_when_appointed_after_prior_seasons_last_match(monkeypatch):
    """A manager appointed in the 2024 close season (after 2023-24 ended)
    must flag as new for 2024-25's opening fixtures."""
    def fake_asof(team, date):
        return {"team": team, "manager": "New Boss", "start_date": "2024-07-01", "end_date": None}

    monkeypatch.setattr(manager_tenure.manager_history, "manager_asof", fake_asof)
    matches = pd.DataFrame(
        [{"season": "2024-2025", "date": pd.Timestamp("2024-08-17"), "team_home": "Arsenal", "team_away": "Chelsea"}]
    )

    result = manager_tenure.fixture_manager_tenure_features(matches)

    assert result.iloc[0]["home_new_manager_this_season"] == True  # noqa: E712
    assert result.iloc[0]["away_new_manager_this_season"] == True  # noqa: E712


def test_new_manager_this_season_flag_false_for_a_long_serving_manager(monkeypatch):
    def fake_asof(team, date):
        return {"team": team, "manager": "Long Server", "start_date": "2019-01-01", "end_date": None}

    monkeypatch.setattr(manager_tenure.manager_history, "manager_asof", fake_asof)
    matches = pd.DataFrame(
        [{"season": "2024-2025", "date": pd.Timestamp("2024-08-17"), "team_home": "Arsenal", "team_away": "Chelsea"}]
    )

    result = manager_tenure.fixture_manager_tenure_features(matches)

    assert result.iloc[0]["home_new_manager_this_season"] == False  # noqa: E712


def test_degrades_to_nan_for_a_team_with_no_manager_history_coverage(monkeypatch):
    monkeypatch.setattr(manager_tenure.manager_history, "manager_asof", lambda team, date: None)
    matches = pd.DataFrame(
        [{"season": "2024-2025", "date": pd.Timestamp("2024-08-17"), "team_home": "Arsenal", "team_away": "Chelsea"}]
    )

    result = manager_tenure.fixture_manager_tenure_features(matches)

    row = result.iloc[0]
    assert pd.isna(row["home_manager_tenure_days"])
    assert pd.isna(row["home_new_manager_this_season"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manager_tenure_feature.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pl_predictor.features.manager_tenure'`

- [ ] **Step 3: Write the implementation**

```python
# src/pl_predictor/features/manager_tenure.py
"""manager_tenure.py — a leading indicator of managerial change: how long
the current manager has been in charge, and whether the team enters this
fixture under a different manager than it finished the *prior* season
with (see docs/AI_CONTINUITY.md EXP-2026-23 and
`data/manager_history.py`'s own docstring for the curated-data source).

No-lookahead by construction: a manager's appointment date is historical
fact once it has happened, so every value here was genuinely knowable
before the fixture's own kickoff.
"""

from __future__ import annotations

import pandas as pd

from ..data import manager_history

_PRIOR_SEASON_END_MONTH_DAY = "06-01"  # a manager appointed on/after June 1 of the season's start year counts as "new this season" — close-season appointments happen June-August; an in-season sacking mid-2024-25 (e.g. October) is not "new this season" for the *following* season unless they're still there next August.


def _tenure_for(team: str, date_str: str) -> dict | None:
    return manager_history.manager_asof(team, date_str)


def _season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def fixture_manager_tenure_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture in `matches_df` (needs `season`, `date`,
    `team_home`, `team_away`) with each side's current manager's tenure
    length in days and whether that manager is new since the prior
    season's close. NaN for a team/date with no manager-history coverage —
    never raises."""
    df = matches_df.copy()
    df["kickoff_date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["_date_str"] = df["kickoff_date"].dt.strftime("%Y-%m-%d")

    rows = []
    for _, fixture in df.iterrows():
        row = {
            "kickoff_date": fixture["kickoff_date"],
            "team_home": fixture["team_home"],
            "team_away": fixture["team_away"],
        }
        close_season_cutoff = f"{_season_start_year(fixture['season'])}-{_PRIOR_SEASON_END_MONTH_DAY}"
        for team, prefix in [(fixture["team_home"], "home"), (fixture["team_away"], "away")]:
            tenure = _tenure_for(team, fixture["_date_str"])
            if tenure is None:
                row[f"{prefix}_manager_tenure_days"] = pd.NA
                row[f"{prefix}_new_manager_this_season"] = pd.NA
                continue
            start = pd.Timestamp(tenure["start_date"])
            row[f"{prefix}_manager_tenure_days"] = (fixture["kickoff_date"] - start).days
            row[f"{prefix}_new_manager_this_season"] = tenure["start_date"] >= close_season_cutoff
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manager_tenure_feature.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/features/manager_tenure.py tests/test_manager_tenure_feature.py
git commit -m "feat: add manager-tenure fixture feature"
```

---

### Task 3: Walk-forward evaluation script

**Files:**
- Create: `src/pl_predictor/evaluate/manager_tenure_prior.py`

**Interfaces:**
- Consumes: `walk_forward.prepare_folds` (existing), `manager_tenure.
  fixture_manager_tenure_features` (Task 2).
- Produces: a `run(seasons=None, min_train_seasons=3) -> dict` result and a
  `__main__` block, mirroring `evaluate/squad_change_prior.py`'s shape
  exactly (no coverage caveat needed here, unlike weather — manager history
  is curated to cover the full training window, not sourced from an API
  with a limited archive).

- [ ] **Step 1: Write the evaluation script**

```python
# src/pl_predictor/evaluate/manager_tenure_prior.py
"""manager_tenure_prior.py — walk-forward evaluation of the manager-tenure
candidate feature (see docs/AI_CONTINUITY.md EXP-2026-23 and
`features/manager_tenure.py`'s own docstring).
"""

from __future__ import annotations

import pandas as pd
import penaltyblog as pb

from ..data import football_data
from ..evaluate import walk_forward
from ..features import manager_tenure
from ..models import ml_scoreline

_RESULT_CODE = {"H": 0, "D": 1, "A": 2}
_FEATURE_COLS = [
    "home_manager_tenure_days",
    "away_manager_tenure_days",
    "home_new_manager_this_season",
    "away_new_manager_this_season",
]


def _extra_feature_frame(seasons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    matches_df = football_data.load_training_data(seasons=seasons)
    extra = manager_tenure.fixture_manager_tenure_features(matches_df)
    return extra, _FEATURE_COLS


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


def _fold_predictions(fold: dict) -> pd.DataFrame:
    home_model, away_model = ml_scoreline.train_goal_regressors(
        fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
    )
    grids = ml_scoreline.predict_grids_batch(home_model, away_model, fold["X_val"])
    val_df = fold["val_df"].reset_index(drop=True)
    return pd.DataFrame(
        {
            "season": val_df["season"].to_numpy(),
            "ftr": val_df["ftr"].to_numpy(),
            "home_win": [g.home_win for g in grids],
            "draw": [g.draw for g in grids],
            "away_win": [g.away_win for g in grids],
        }
    )


def run(seasons: list[str] | None = None, min_train_seasons: int = 3) -> dict:
    seasons = seasons or football_data.default_completed_seasons(n=8)
    extra_frame, extra_cols = _extra_feature_frame(seasons)

    baseline_folds = walk_forward.prepare_folds(seasons, min_train_seasons)
    candidate_folds = walk_forward.prepare_folds(
        seasons, min_train_seasons, extra_feature_frame=extra_frame, extra_feature_cols=extra_cols
    )

    baseline_preds = pd.concat([_fold_predictions(f) for f in baseline_folds], ignore_index=True)
    candidate_preds = pd.concat([_fold_predictions(f) for f in candidate_folds], ignore_index=True)

    return {
        "overall_baseline": _rps_brier(baseline_preds),
        "overall_candidate": _rps_brier(candidate_preds),
        "per_season_baseline": baseline_preds.groupby("season").apply(lambda g: pd.Series(_rps_brier(g))),
        "per_season_candidate": candidate_preds.groupby("season").apply(lambda g: pd.Series(_rps_brier(g))),
    }


if __name__ == "__main__":
    results = run()

    print("=== Overall (every fixture) ===")
    print("baseline: ", results["overall_baseline"])
    print("candidate:", results["overall_candidate"])

    print("\n=== Per-season, overall (most-recent-season corroboration check) ===")
    print("baseline:")
    print(results["per_season_baseline"])
    print("candidate:")
    print(results["per_season_candidate"])
```

- [ ] **Step 2: Commit**

```bash
git add src/pl_predictor/evaluate/manager_tenure_prior.py
git commit -m "feat: add manager-tenure walk-forward evaluation script"
```

---

### Task 4: Run the evaluation and document the result

**Files:**
- Modify: `docs/AI_CONTINUITY.md`

- [ ] **Step 1: Run the evaluation**

Run: `python -m pl_predictor.evaluate.manager_tenure_prior`

- [ ] **Step 2: Apply the promotion gate**

Per `docs/AI_CONTINUITY.md`'s "Non-negotiable research protocol" (item 5):
promote only if the candidate improves the average AND the fixed
most-recent-season (2025-26) holdout, without a material fold regression.

- [ ] **Step 3: Write the EXP-2026-23 entry**

Add a new entry to `docs/AI_CONTINUITY.md`'s "Completed experiment log"
(insert before "## Change checklist for future agents", matching the style
of EXP-2026-18/20/21/22). Include: the question (explicitly link back to
EXP-2026-18's "follow-ups explicitly not started" note, which named this
exact idea), the manager-history table's research sources and coverage,
the full results table (overall and per-season, baseline vs candidate),
and an explicit **Decision:** line. Write this up regardless of outcome.

- [ ] **Step 4: If promoted, wire into production (skip this step if rejected)**

Only if the gate in Step 2 passed: add `manager_tenure.
fixture_manager_tenure_features`'s merge into `features/build.py::
build_training_frame` and `FixtureFeatureContext`/`build_row` (same
pattern as `squad_change`'s existing merge in that file), add the new
columns to `feature_cols`, retrain, and re-run the fixed holdout check to
confirm the production retrain's numbers match the walk-forward's
prediction. No pre-written code here — depends entirely on Step 2's
result.

- [ ] **Step 5: Commit**

```bash
git add docs/AI_CONTINUITY.md
git commit -m "docs: record EXP-2026-23 manager tenure feature result"
```

(If Step 4 also produced changes, include `src/pl_predictor/features/build.py`,
`models/manifest.json`, and any other touched files in that same commit or
a follow-up one, matching EXP-2026-18's commit history for reference.)
