# Bet Recommendation Trustworthiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the live-bet edge threshold by how much real data the underlying prediction rests on (`data_confidence`), validate that this genuinely helps via the existing walk-forward harness, and give model-only markets (BTTS, corners, cards) a distinct visual treatment so a user can't mistake "a plain number" for "the model has a decisive view here."

**Architecture:** A new `required_edge_multiplier(data_confidence)` function in `models/scoreline.py` (next to the `_data_confidence` it consumes) is the single source of truth for the multiplier table, imported by both `odds/value_bets.py` (live serving) and `evaluate/backtest.py` (offline validation) so the same rule governs both. The walk-forward validator's fold adapter (`evaluate/betting_validation.py::_FoldScorelineModel`) currently has no `.context`, so `data_confidence` is silently `None` inside it today — it gets a minimal per-fold `games_played` context so the validation can actually distinguish confidence tiers. Frontend gets one new pure function (`frontend/src/lib/modelCall.ts`) plus small additions to `MarketBar`/`OverUnderRow`.

**Tech Stack:** Python 3.13, FastAPI, pandas, pytest (backend); React 19 + TypeScript + Vite (frontend, no test runner configured — verification is `tsc -b` plus manual browser check, matching this project's existing frontend testing convention).

**Spec:** `docs/superpowers/specs/2026-09-04-bet-recommendation-trustworthiness-design.md`

## Global Constraints

- Multiplier table (from the spec, a starting judgment call, not a measured fact): `established` → 1.0×, `limited` → 1.5×, `new` → 2.5×, `None` → 1.0× (today's flat behavior, unchanged — `None` means a classical Dixon-Coles/Bivariate-Poisson fit is currently served, which has no per-fixture confidence signal at all).
- Base `edge_threshold` stays `0.05` everywhere it already defaults to that — this plan only changes what a fixture's *effective* threshold is, never the base constant itself.
- "Model call" visual treatment must be a genuinely new, distinct style — never reuse the existing pink `flagged`/`valueBet` treatment, which specifically means "there is a live market and the model beats it."
- Decisive one-sided threshold for a "model call": `prob >= 0.60 || prob <= 0.40`.
- Every existing test in `tests/test_betting_guidance.py`, `tests/test_backtest.py`, and `tests/test_betting_validation.py` must keep passing unmodified (verified per-task below) — this is an additive change, not a rewrite.

---

### Task 1: `required_edge_multiplier` in `models/scoreline.py`

**Files:**
- Modify: `src/pl_predictor/models/scoreline.py` (add near `_data_confidence`, around line 115)
- Test: `tests/test_scoreline.py`

**Interfaces:**
- Produces: `scoreline.required_edge_multiplier(data_confidence: str | None) -> float` — used by Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scoreline.py`:

```python
def test_required_edge_multiplier_scales_by_confidence_tier():
    assert scoreline.required_edge_multiplier("established") == pytest.approx(1.0)
    assert scoreline.required_edge_multiplier("limited") == pytest.approx(1.5)
    assert scoreline.required_edge_multiplier("new") == pytest.approx(2.5)


def test_required_edge_multiplier_defaults_to_flat_when_confidence_unknown():
    # None: a classical Dixon-Coles/Bivariate-Poisson fit has no per-fixture
    # confidence signal at all — must not silently require a bigger edge.
    assert scoreline.required_edge_multiplier(None) == pytest.approx(1.0)
    # An unrecognized string must not crash or over/under-penalize silently.
    assert scoreline.required_edge_multiplier("some-future-tier") == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_scoreline.py -k required_edge_multiplier -v`
Expected: FAIL with `AttributeError: module 'pl_predictor.models.scoreline' has no attribute 'required_edge_multiplier'`

- [ ] **Step 3: Write minimal implementation**

In `src/pl_predictor/models/scoreline.py`, immediately after `_data_confidence` (after its closing `return "new"` line):

```python
# A starting judgment call, not a measured fact — see the walk-forward
# validation gate in evaluate/betting_validation.py before this reaches
# the live app. Applied as a multiplier on the existing flat edge
# threshold, not a replacement for it.
CONFIDENCE_EDGE_MULTIPLIERS = {
    "established": 1.0,
    "limited": 1.5,
    "new": 2.5,
}


def required_edge_multiplier(data_confidence: str | None) -> float:
    """How much bigger a model-vs-market edge must be before it's trusted
    enough to recommend, scaled by how much real data the prediction rests
    on (see `_data_confidence`). `None` — a classical Dixon-Coles/
    Bivariate-Poisson fit has no per-fixture confidence signal — and any
    unrecognized tier both keep today's flat 1.0×, never a silent penalty
    for a case this table doesn't know about."""
    if data_confidence is None:
        return 1.0
    return CONFIDENCE_EDGE_MULTIPLIERS.get(data_confidence, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_scoreline.py -k required_edge_multiplier -v`
Expected: PASS (both new tests)

- [ ] **Step 5: Run the full scoreline test file to check for regressions**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_scoreline.py -v`
Expected: all pass (no existing test touches this new function)

- [ ] **Step 6: Commit**

```bash
git add src/pl_predictor/models/scoreline.py tests/test_scoreline.py
git commit -m "feat: add confidence-scaled edge multiplier for bet recommendations"
```

---

### Task 2: Wire the multiplier into `odds/value_bets.py`'s live flagging

**Files:**
- Modify: `src/pl_predictor/odds/value_bets.py:155-159` (the `value_bet_flags` list comprehension)
- Test: `tests/test_betting_guidance.py`

**Interfaces:**
- Consumes: `scoreline.required_edge_multiplier(data_confidence: str | None) -> float` (Task 1). `value_bets.py` already does `from ..models import scoreline` at the top — no new import needed.
- Produces: no change to `build_value_bet_table`'s signature or return shape — only which sides end up in `row["value_bet_flags"]` changes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_betting_guidance.py` (reuses this file's existing `_fixtures_frame`/`_odds_frame` helpers):

```python
def test_low_confidence_fixture_needs_a_bigger_edge_to_flag(monkeypatch):
    # home_win edge here is 0.68 - 0.60 = 0.08 -- clears the flat 5%
    # threshold but not "new" tier's 2.5x-scaled 12.5% requirement.
    monkeypatch.setattr(value_bets, "_devig_h2h", lambda *_: {"home_win": 0.60, "draw": 0.25, "away_win": 0.15})
    monkeypatch.setattr(
        value_bets.scoreline,
        "predict_fixtures_batch",
        lambda *_, **__: [
            {
                "home_win": 0.68,
                "draw": 0.17,
                "away_win": 0.15,
                "btts_yes": 0.5,
                "over_2_5": 0.5,
                "under_2_5": 0.5,
                "top_scorelines": [{"home": 2, "away": 0}],
                "fallback": False,
                "data_confidence": "new",
                "home_goal_expectation": 1.6,
                "away_goal_expectation": 1.0,
                "home_2plus_prob": 0.25,
                "away_2plus_prob": 0.1,
            }
        ],
    )
    row = value_bets.build_value_bet_table(_fixtures_frame(), _odds_frame(), models={"scoreline": object()}).iloc[0]
    assert row["value_bet_flags"] == []
    assert row["recommended_market"] is None


def test_established_confidence_keeps_todays_flat_threshold(monkeypatch):
    # Same 0.08 edge as above, but "established" -- must still flag, exactly
    # like it would have before this change (1.0x multiplier, unchanged).
    monkeypatch.setattr(value_bets, "_devig_h2h", lambda *_: {"home_win": 0.60, "draw": 0.25, "away_win": 0.15})
    monkeypatch.setattr(
        value_bets.scoreline,
        "predict_fixtures_batch",
        lambda *_, **__: [
            {
                "home_win": 0.68,
                "draw": 0.17,
                "away_win": 0.15,
                "btts_yes": 0.5,
                "over_2_5": 0.5,
                "under_2_5": 0.5,
                "top_scorelines": [{"home": 2, "away": 0}],
                "fallback": False,
                "data_confidence": "established",
                "home_goal_expectation": 1.6,
                "away_goal_expectation": 1.0,
                "home_2plus_prob": 0.25,
                "away_2plus_prob": 0.1,
            }
        ],
    )
    row = value_bets.build_value_bet_table(_fixtures_frame(), _odds_frame(), models={"scoreline": object()}).iloc[0]
    assert row["value_bet_flags"] == ["home_win"]
    assert row["recommended_market"] == "home_win"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_betting_guidance.py -k "low_confidence or established_confidence" -v`
Expected: `test_low_confidence_fixture_needs_a_bigger_edge_to_flag` FAILS (today's flat 5% threshold flags it); `test_established_confidence_keeps_todays_flat_threshold` PASSES already (documents current behavior, confirms the baseline before the change).

- [ ] **Step 3: Write minimal implementation**

In `src/pl_predictor/odds/value_bets.py`, replace lines 155-159:

```python
        row["value_bet_flags"] = [] if row["odds_is_stale"] else [
            side
            for side in ["home_win", "draw", "away_win", "over_2_5", "under_2_5"]
            if row[f"{side}_edge"] is not None and row[f"{side}_edge"] > edge_threshold
        ]
```

with:

```python
        required_edge = edge_threshold * scoreline.required_edge_multiplier(row["data_confidence"])
        row["value_bet_flags"] = [] if row["odds_is_stale"] else [
            side
            for side in ["home_win", "draw", "away_win", "over_2_5", "under_2_5"]
            if row[f"{side}_edge"] is not None and row[f"{side}_edge"] > required_edge
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_betting_guidance.py -v`
Expected: all pass, including both new tests.

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/odds/value_bets.py tests/test_betting_guidance.py
git commit -m "feat: scale live value-bet edge threshold by prediction confidence"
```

---

### Task 3: Thread `data_confidence` through `evaluate/backtest.py`

**Files:**
- Modify: `src/pl_predictor/evaluate/backtest.py:69-96` (`_precompute_predictions`) and `:159-176` (`logic()` inside `build_value_bet_backtest`)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `scoreline.required_edge_multiplier` (Task 1). `backtest.py` already does `from ..models import scoreline`.
- Produces: `_precompute_predictions`'s per-fixture dict gains a `"data_confidence"` key in both branches (batch ML path and per-row stat-model path — the stat-model path already returns it via `scoreline.predict_fixture`, no change needed there). `build_value_bet_backtest`'s selection logic now reads it via `pred.get("data_confidence")` — `.get`, not `[...]`, so a caller (like the existing test below) that omits the key entirely keeps working exactly as before.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backtest.py`:

```python
def test_low_confidence_prediction_needs_a_bigger_edge_to_be_selected(monkeypatch):
    fixtures = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-08-01"),
                "team_home": "Arsenal",
                "team_away": "Chelsea",
                "goals_home": 2,
                "goals_away": 1,
                "ftr": "H",
                "b365_h": 2.0,
                "b365_d": 3.5,
                "b365_a": 4.0,
                "b365>2.5": 1.9,
                "b365<2.5": 2.0,
            }
        ]
    )
    # Same 0.08-ish edge shape as test_historical_replay_records_qualified_
    # de_vigged_selection above, but tagged "new" -- must NOT be selected,
    # since 0.08 clears the flat 5% default but not "new"'s 2.5x (12.5%).
    monkeypatch.setattr(
        backtest,
        "_precompute_predictions",
        lambda _model, frame, market_overrides=None: {
            frame.index[0]: {
                "home_win": 0.48,
                "draw": 0.30,
                "away_win": 0.22,
                "over_2_5": 0.75,
                "under_2_5": 0.25,
                "fallback": False,
                "data_confidence": "new",
            }
        },
    )
    selections = []

    replay = backtest.build_value_bet_backtest(
        fixtures,
        model=object(),
        start_date="2025-08-01",
        end_date="2025-08-01",
        staking="flat",
        selections=selections,
    )

    assert replay.results()["Total Bets"] == 0
    assert selections == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backtest.py -k low_confidence_prediction -v`
Expected: FAIL — `replay.results()["Total Bets"]` is `1`, not `0` (today's flat threshold selects it).

- [ ] **Step 3: Write minimal implementation**

In `src/pl_predictor/evaluate/backtest.py`, inside `_precompute_predictions`'s batch branch (the `if hasattr(model, "predict_many_from_rows"):` block, right after the `results = {...}` dict comprehension that builds `home_win`/`draw`/.../`"fallback": False`), add:

```python
        for idx, row in df.iterrows():
            results[idx]["data_confidence"] = scoreline._data_confidence(model, row["team_home"], row["team_away"])
```

(The `else` branch already returns `data_confidence` inside `scoreline.predict_fixture`'s own dict — no change needed there.)

Then, inside `build_value_bet_backtest`'s `logic()` function, replace:

```python
            edge = pred[side] - implied[side]
            if edge > edge_threshold and (best is None or edge > best["edge"]):
```

with:

```python
            edge = pred[side] - implied[side]
            required_edge = edge_threshold * scoreline.required_edge_multiplier(pred.get("data_confidence"))
            if edge > required_edge and (best is None or edge > best["edge"]):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: all pass, including the new test and the pre-existing
`test_historical_replay_records_qualified_de_vigged_selection` (its mock
omits `"data_confidence"` entirely, so `.get` returns `None` →
multiplier `1.0` → identical behavior to before).

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/evaluate/backtest.py tests/test_backtest.py
git commit -m "feat: apply confidence-scaled edge threshold in the value-bet backtest"
```

---

### Task 4: Give the walk-forward validator real per-fold confidence

**Files:**
- Modify: `src/pl_predictor/evaluate/betting_validation.py:12-22` (`_FoldScorelineModel`) and `:84` (its construction call inside `run_walk_forward_value_bet_validation`)
- Test: `tests/test_betting_validation.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks directly — this makes `_FoldScorelineModel` compatible with what `scoreline._data_confidence` already expects (`model.context.games_played`, a `dict[str, int]`), which Task 3's `_precompute_predictions` change now calls for *any* model with `predict_many_from_rows`, including this one.
- Produces: `_FoldScorelineModel(home_model, away_model, feature_cols, train_df)` — one new required constructor argument. Existing callers must be updated (there is exactly one, inside `run_walk_forward_value_bet_validation`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_betting_validation.py`:

```python
import pandas as pd


def test_fold_model_computes_games_played_from_its_own_train_df():
    train_df = pd.DataFrame(
        [
            {"team_home": "Arsenal", "team_away": "Chelsea"},
            {"team_home": "Chelsea", "team_away": "Arsenal"},
            {"team_home": "Arsenal", "team_away": "Fulham"},
        ]
    )
    model = betting_validation._FoldScorelineModel(
        home_model=object(), away_model=object(), feature_cols=[], train_df=train_df
    )
    # Arsenal appears in all 3 rows, Chelsea in 2, Fulham (a newly-promoted
    # side in this toy fold) in only 1 -- exactly the signal
    # scoreline._data_confidence needs to tell them apart.
    assert model.context.games_played == {"Arsenal": 3, "Chelsea": 2, "Fulham": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_betting_validation.py -k games_played -v`
Expected: FAIL with `TypeError: _FoldScorelineModel.__init__() got an unexpected keyword argument 'train_df'`

- [ ] **Step 3: Write minimal implementation**

In `src/pl_predictor/evaluate/betting_validation.py`, add near the top (after the existing imports):

```python
from types import SimpleNamespace
```

Replace the `_FoldScorelineModel` class:

```python
class _FoldScorelineModel:
    """Small adapter so the existing backtest can batch-score a fold."""

    def __init__(self, home_model, away_model, feature_cols: list[str]):
        self.home_model = home_model
        self.away_model = away_model
        self.feature_cols = feature_cols

    def predict_many_from_rows(self, frame: pd.DataFrame):
        features = frame[self.feature_cols].fillna(0)
        return ml_scoreline.predict_grids_batch(self.home_model, self.away_model, features)
```

with:

```python
def _games_played_from_train_df(train_df: pd.DataFrame) -> dict[str, int]:
    """Per-fold analogue of the live `FixtureFeatureContext.games_played`
    `scoreline._data_confidence` reads — how many matches this fold's own
    training window has seen for each team, so a newly-promoted side (few
    or zero rows) scores as low-confidence in validation exactly like it
    would in live serving, instead of `data_confidence` silently being
    `None` for every fold fixture."""
    counts: dict[str, int] = {}
    for column in ("team_home", "team_away"):
        for team, n in train_df[column].value_counts().items():
            counts[team] = counts.get(team, 0) + int(n)
    return counts


class _FoldScorelineModel:
    """Small adapter so the existing backtest can batch-score a fold."""

    def __init__(self, home_model, away_model, feature_cols: list[str], train_df: pd.DataFrame):
        self.home_model = home_model
        self.away_model = away_model
        self.feature_cols = feature_cols
        # `scoreline._data_confidence` only needs `.context.games_played` —
        # a plain namespace is enough, no live FixtureFeatureContext needed.
        self.context = SimpleNamespace(games_played=_games_played_from_train_df(train_df))

    def predict_many_from_rows(self, frame: pd.DataFrame):
        features = frame[self.feature_cols].fillna(0)
        return ml_scoreline.predict_grids_batch(self.home_model, self.away_model, features)
```

Then, inside `run_walk_forward_value_bet_validation`, update the construction call:

```python
        model = _FoldScorelineModel(home_model, away_model, list(fold["X_train"].columns))
```

to:

```python
        model = _FoldScorelineModel(home_model, away_model, list(fold["X_train"].columns), train_df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_betting_validation.py -v`
Expected: all pass, including the new test.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: no new failures anywhere (this touches shared code paths — `_data_confidence`, `_precompute_predictions` — worth a full-suite check, not just the touched files').

- [ ] **Step 6: Commit**

```bash
git add src/pl_predictor/evaluate/betting_validation.py tests/test_betting_validation.py
git commit -m "feat: give the walk-forward fold model real per-fold confidence context"
```

---

### Task 5: Run the required validation gate

Per the spec's mandatory gate: confirm the confidence-weighted threshold reduces the flagged-bet error rate without collapsing recommendation frequency to near-zero, before this reaches the live app.

- [ ] **Step 1: Capture the baseline (pre-change behavior)**

```bash
git stash  # temporarily shelve Tasks 1-4's changes
PYTHONPATH=src .venv/bin/python -c "
from pl_predictor.evaluate import betting_validation
import json
result = betting_validation.run_walk_forward_value_bet_validation()
print(json.dumps({k: v for k, v in result.items() if k != 'selections'}, indent=2, default=str))
"
git stash pop  # restore Tasks 1-4's changes
```

Record the printed summary (bets, wins, win_rate, yield — both overall and
per-fold) somewhere durable (paste into this task's notes, or a scratch
file) — this is the "before" comparison point.

- [ ] **Step 2: Capture the after-change result**

```bash
PYTHONPATH=src .venv/bin/python -c "
from pl_predictor.evaluate import betting_validation
import json
result = betting_validation.run_walk_forward_value_bet_validation()
print(json.dumps({k: v for k, v in result.items() if k != 'selections'}, indent=2, default=str))
"
```

- [ ] **Step 3: Apply the decision rule**

Compare the two summaries:
- **Ship** if `win_rate` improves (or stays flat) while `bets` doesn't
  collapse to a handful across the whole multi-season walk-forward window
  (a near-zero bet count means the thresholds are too strict to ever be
  useful, not "safer").
- **Revisit the multiplier table** (Global Constraints above) if `bets`
  drops sharply with no `win_rate` improvement — the multipliers are a
  judgment call, not sacred; tune `CONFIDENCE_EDGE_MULTIPLIERS` in
  `models/scoreline.py` and re-run from Step 2 rather than shipping
  something the data doesn't support.
- Either way, do **not** treat this as a one-time gate never revisited —
  the live `/api/value-bets/walk-forward` endpoint already surfaces this
  same validation; check it again after a few weeks of live data the same
  way the rest of this project already treats its calibration numbers as
  living, not fire-and-forget.

- [ ] **Step 4: Record the outcome**

If multipliers changed in Step 3, amend Task 1's constants, re-run Task
1-4's tests, and commit:

```bash
git add src/pl_predictor/models/scoreline.py
git commit -m "fix: retune confidence-edge multipliers per walk-forward validation"
```

If no change was needed, no commit here — proceed to Task 6.

---

### Task 6: Frontend — a distinct "model call" treatment

**Files:**
- Create: `frontend/src/lib/modelCall.ts`
- Modify: `frontend/src/components/MarketBar.tsx`
- Modify: `frontend/src/components/FixtureModal.tsx:33-52` (`highlightFor`, `OverUnderRow`), `:365-376` (BTTS row), `:438-439` (corners/cards rows)

**Interfaces:**
- Produces: `isModelCall(prob: number): boolean` (pure function, `frontend/src/lib/modelCall.ts`) — used by both `MarketBar` and `OverUnderRow` call sites, so the `>= 0.60 || <= 0.40` boundary is defined exactly once.
- `MarketBar`'s `highlight` prop type grows from `"flagged" | "hit" | "miss"` to `"flagged" | "hit" | "miss" | "modelCall"`.
- `highlightFor`'s signature grows from `(flagged: boolean, postMatchHit?: boolean)` to `(flagged: boolean, postMatchHit?: boolean, modelCall?: boolean)`, same return type plus `"modelCall"`, same priority order (`postMatchHit` wins if resolved, then `flagged`, then `modelCall`, else `undefined`).
- `OverUnderRow`'s props grow with an optional `modelCall?: boolean`, used the same priority order as above for its background className.

**No test runner is configured for this frontend** (no vitest/jest in `frontend/package.json`, no existing `*.test.*` files) — this plan does not introduce one for a handful of boundary checks; `isModelCall` is pulled into its own pure, exported function specifically so its correctness is legible by inspection and by the manual browser check in Step 5, matching how this project already verifies frontend changes (see this session's own account of prior frontend work being "smoke-tested in a real browser," not unit-tested).

- [ ] **Step 1: Create the pure boundary helper**

Create `frontend/src/lib/modelCall.ts`:

```typescript
// A "model call" is a plain, informational probability decisively on one
// side or the other -- distinct from a value bet (which requires a live
// market the model beats). Threshold is a starting judgment call (see
// docs/superpowers/specs/2026-09-04-bet-recommendation-trustworthiness-design.md),
// open to tuning after real usage.
export function isModelCall(prob: number): boolean {
  return prob >= 0.6 || prob <= 0.4;
}
```

- [ ] **Step 2: Type-check the new file**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors (this file has no dependents yet, so it can't break
anything else — just confirms it compiles standalone).

- [ ] **Step 3: Add the `modelCall` highlight variant to `MarketBar`**

In `frontend/src/components/MarketBar.tsx`, change the `highlight` prop type:

```typescript
  highlight?: "flagged" | "hit" | "miss" | "modelCall";
```

Add to `HIGHLIGHT_CLASSES`:

```typescript
const HIGHLIGHT_CLASSES: Record<string, string> = {
  hit: "bg-win/10 ring-1 ring-win/30",
  miss: "bg-loss/10",
  flagged: "bg-pl-pink/15 ring-2 ring-pl-pink/70",
  // Deliberately not the pink `flagged` treatment -- that specifically
  // means "there is a live market and the model beats it." This means
  // "the model has a decisive view, no market to compare against."
  modelCall: "bg-pl-cyan/10 ring-1 ring-pl-cyan/40",
};
```

- [ ] **Step 4: Wire `highlightFor` and `OverUnderRow`, and apply to BTTS/corners/cards**

In `frontend/src/components/FixtureModal.tsx`:

Add the import (alongside the existing `GLOSSARY` import):

```typescript
import { isModelCall } from "../lib/modelCall";
```

Replace `highlightFor`:

```typescript
function highlightFor(flagged: boolean, postMatchHit?: boolean): "flagged" | "hit" | "miss" | undefined {
  if (postMatchHit === true) return "hit";
  if (postMatchHit === false) return "miss";
  if (flagged) return "flagged";
  return undefined;
}
```

with:

```typescript
function highlightFor(
  flagged: boolean,
  postMatchHit?: boolean,
  modelCall?: boolean
): "flagged" | "hit" | "miss" | "modelCall" | undefined {
  if (postMatchHit === true) return "hit";
  if (postMatchHit === false) return "miss";
  if (flagged) return "flagged";
  if (modelCall) return "modelCall";
  return undefined;
}
```

Replace the BTTS `MarketBar` call (around line 365-376):

```tsx
                      <MarketBar
                        label={
                          <span className="inline-flex items-center gap-1.5">
                            BTTS: Yes <InfoTooltip text={GLOSSARY.btts} align="right" />
                          </span>
                        }
                        prob={detail.btts_yes_prob}
                        highlight={highlightFor(
                          false,
                          postMatchVerdict("BTTS")?.prediction === "yes" ? postMatchVerdict("BTTS")?.hit : undefined
                        )}
                      />
```

with:

```tsx
                      <MarketBar
                        label={
                          <span className="inline-flex items-center gap-1.5">
                            BTTS: Yes <InfoTooltip text={GLOSSARY.btts} align="right" />
                          </span>
                        }
                        prob={detail.btts_yes_prob}
                        highlight={highlightFor(
                          false,
                          postMatchVerdict("BTTS")?.prediction === "yes" ? postMatchVerdict("BTTS")?.hit : undefined,
                          isModelCall(detail.btts_yes_prob)
                        )}
                      />
```

Replace `OverUnderRow`'s definition (around line 40-52):

```tsx
function OverUnderRow({ label, lam, line, over, postMatchHit }: { label: string; lam: number; line: number; over: number; postMatchHit?: boolean }) {
  return (
    <div className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${postMatchHit === true ? "bg-win/10 ring-1 ring-win/30" : postMatchHit === false ? "bg-loss/10" : "bg-pl-850/60"}`}>
```

with:

```tsx
function OverUnderRow({ label, lam, line, over, postMatchHit }: { label: string; lam: number; line: number; over: number; postMatchHit?: boolean }) {
  const rowClass =
    postMatchHit === true
      ? "bg-win/10 ring-1 ring-win/30"
      : postMatchHit === false
        ? "bg-loss/10"
        : isModelCall(over)
          ? "bg-pl-cyan/10 ring-1 ring-pl-cyan/40"
          : "bg-pl-850/60";
  return (
    <div className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${rowClass}`}>
```

(The corners/cards call sites at lines 438-439 need no change — `OverUnderRow` now applies the treatment internally from the `over` prop they already pass.)

- [ ] **Step 5: Type-check and manually verify in the browser**

Run: `cd frontend && npx tsc --noEmit` — expected: no errors.

Then, per this project's standing convention for frontend changes: start the dev server (`npm run dev` in `frontend/`, with the backend running per the README's "Run the dashboard" section) and open a fixture whose `btts_yes_prob` is above 0.60 or below 0.40 — confirm the BTTS row shows the new cyan treatment, distinct from a value-bet row's pink treatment and from a plain unremarkable row for a BTTS probability near 0.50. Check a corners or cards row the same way.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/modelCall.ts frontend/src/components/MarketBar.tsx frontend/src/components/FixtureModal.tsx
git commit -m "feat: give model-only probabilities a distinct visual treatment"
```

---

## Self-Review Notes

- **Spec coverage:** Section 1 (confidence-weighted flagging) → Tasks 1-3. The spec's own "before it ships" evaluation-gate requirement, plus the mid-design discovery that the existing walk-forward harness couldn't actually exercise it → Task 4 (harness fix) + Task 5 (the gate itself). Section 2 (frontend model-call treatment) → Task 6. Testing section's boundary-case requirement (0.60/0.59/0.40/0.41) is satisfied by `isModelCall` being a single pure function simple enough to verify by inspection, per Task 6's no-test-runner rationale.
- **Placeholder scan:** none found — every step has real code.
- **Type consistency:** `required_edge_multiplier(data_confidence: str | None) -> float` used identically in Tasks 2 and 3. `_FoldScorelineModel`'s new `train_df` parameter is positional in both its definition (Task 4) and its one call site (same task) — no drift. `isModelCall(prob: number): boolean` used identically in both `MarketBar`'s BTTS call site and `OverUnderRow` (Task 6).
