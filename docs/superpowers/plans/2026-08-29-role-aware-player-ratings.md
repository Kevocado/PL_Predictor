# Role-Aware Player Ratings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace percentile player ratings with calibrated role-aware Quality, Form, Overall, and Impact scores, and use the historical counterpart only in a gated scoreline research experiment.

**Architecture:** The pure rating module derives Quality and capped Form from the cached FPL bootstrap using a fixed evidence-aware scale. Player Hub serialises the four values. A historical builder creates leakage-safe Quality-plus-Form unit features for research only; it never modifies deployed model features.

**Tech Stack:** Python, pandas, scikit-learn, FastAPI, React/TypeScript, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-role-aware-player-ratings-design.md`

## Global Constraints

- Serving reads only cached FPL bootstrap; no per-player requests or model fitting.
- Quality is 0–100, Form is 0–15, Overall is `min(100, Quality + Form)`, and Impact discounts Overall by availability and expected minutes.
- 90+ requires high Quality plus sustained high-minute, underlying-and-actual form evidence; percentile rank never creates elite ratings.
- Team-unit features remain research-only until a manual promotion decision.

---

### Task 1: Build fixed-scale role-aware player ratings

**Files:**
- Modify: `src/pl_predictor/models/player_ratings.py`
- Modify: `tests/test_player_ratings.py`

**Interfaces:**
- Consumes: `rate_bootstrap_elements(elements: list[dict], positions: dict[int, str])`.
- Produces: per-element `quality_rating`, `form_rating`, `overall_rating`, `current_impact_rating`, `rating_driver`, `rating_expected_minutes`, and `rating_model_source`.

- [x] **Step 1: Write failing fixed-scale tests**

```python
def test_rating_scale_does_not_make_top_rank_elite():
    ratings = rate_bootstrap_elements([ordinary_mid, ordinary_mid_two], {3: "MID"})
    assert max(row["overall_rating"] for row in ratings.values()) < 90

def test_form_needs_minutes_underlying_and_actual_evidence():
    ratings = rate_bootstrap_elements([breakout_mid, finishing_spike_mid], {3: "MID"})
    assert ratings[breakout_mid["id"]]["form_rating"] >= 10
    assert ratings[finishing_spike_mid["id"]]["form_rating"] < 10

def test_availability_changes_impact_only():
    ratings = rate_bootstrap_elements([available_mid, doubtful_mid], {3: "MID"})
    assert ratings[available_mid["id"]]["quality_rating"] == ratings[doubtful_mid["id"]]["quality_rating"]
    assert ratings[available_mid["id"]]["form_rating"] == ratings[doubtful_mid["id"]]["form_rating"]
    assert ratings[doubtful_mid["id"]]["current_impact_rating"] < ratings[available_mid["id"]]["current_impact_rating"]
```

- [x] **Step 2: Confirm the current rank-normalised implementation fails**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_player_ratings.py -q`

Expected: FAIL because it assigns the top player 100 by rank and has no Form or Overall fields.

- [x] **Step 3: Implement role metrics and evidence shrinkage**

```python
ROLE_WEIGHTS = {"GK": {"saves": .40, "clean_sheets": .35, "bps": .25}, "DEF": {"clean_sheets": .35, "defensive_contribution": .20, "expected_goal_involvements": .25, "bps": .20}, "MID": {"expected_goal_involvements": .45, "expected_assists": .20, "threat": .15, "creativity": .20}, "FWD": {"expected_goals": .55, "expected_goal_involvements": .30, "goals_scored": .15}}
def _quality_score(element: dict, position: str) -> tuple[float, str]: ...
def _form_score(element: dict, position: str) -> float: ...
def _shrunk_quality(raw: float, minutes: float, starts: float, position: str) -> float: ...
```

Calculate role inputs per 90 (or per appearance for FPL index measures), blend toward a fixed role prior with `min(1.0, (minutes / 900 + starts / 10) / 2)`, then map to a fixed 0–100 scale. Gate Form below 360 minutes or four starts. Form blends role-specific underlying output, actual output, and starts/minutes, clipped to `[0, 15]`.

- [x] **Step 4: Publish the complete score contract**

```python
overall = round(min(100.0, quality + form), 1)
impact = round(overall * availability * expected_minutes / 90.0, 1)
return {"quality_rating": quality, "form_rating": form, "overall_rating": overall, "current_impact_rating": impact, "rating_driver": driver, "rating_expected_minutes": round(expected_minutes, 1), "rating_model_source": "role_aware_evidence_baseline"}
```

Remove rank sorting. Retain existing availability/expected-minute helpers with effects confined to Impact.

- [x] **Step 5: Verify and commit**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_player_ratings.py -q`

Expected: PASS.

Commit: `git add src/pl_predictor/models/player_ratings.py tests/test_player_ratings.py && git commit -m "feat: add role-aware player rating scale"`

### Task 2: Expose all score dimensions through Player Hub

**Files:**
- Modify: `src/pl_predictor/api/hub_analytics.py:225-273`
- Modify: `tests/test_hub_analytics.py`
- Modify: `tests/test_public_snapshot_routes.py`

**Interfaces:**
- Consumes: Task 1 rating dictionary and current bootstrap.
- Produces: `build_player_hub()` player records with all four scores; leaderboards sorted by Overall.

- [x] **Step 1: Write the failing response-shape test**

```python
report = hub_analytics.build_player_hub(bootstrap)
player = report["players"][0]
assert {"quality_rating", "form_rating", "overall_rating", "current_impact_rating"} <= player.keys()
assert report["leaderboards"]["GK"][0]["overall_rating"] >= report["leaderboards"]["GK"][-1]["overall_rating"]
```

- [x] **Step 2: Confirm the response test fails**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_hub_analytics.py tests/test_public_snapshot_routes.py -q`

Expected: FAIL because Form/Overall are missing and leaderboards sort by Quality.

- [x] **Step 3: Implement Hub serialisation and ordering**

```python
players.sort(key=lambda player: (player.get("overall_rating", 0), player["minutes"]), reverse=True)
leaderboards = {position: sorted(role_players, key=lambda player: player.get("overall_rating", 0), reverse=True)[:5] for position, role_players in grouped_players.items()}
```

Return source `role_aware_evidence_baseline`, preserve cached-bootstrap freshness, and retain the existing 300-second route cache.

- [x] **Step 4: Verify and commit**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_hub_analytics.py tests/test_public_snapshot_routes.py -q`

Expected: PASS.

Commit: `git add src/pl_predictor/api/hub_analytics.py tests/test_hub_analytics.py tests/test_public_snapshot_routes.py && git commit -m "feat: expose player quality form and overall ratings"`

### Task 3: Render Overall, Quality, Form, and Impact separately

**Files:**
- Modify: `frontend/src/types.ts:395-424`
- Modify: `frontend/src/components/PlayerHub.tsx`

**Interfaces:**
- Consumes: `PlayerHubPlayer.overall_rating`, `quality_rating`, `form_rating`, `current_impact_rating`, driver, source, freshness.
- Produces: an Overall-ranked sortable table and per-position top-five Overall leaderboards.

- [x] **Step 1: Add failing type references**

```ts
overall_rating: number;
quality_rating: number;
form_rating: number;
current_impact_rating: number;
```

Add `overall_rating` and `form_rating` to `SortKey`; put score columns Overall, Quality, Form, Impact.

- [x] **Step 2: Check that the old frontend fails to build**

Run: `cd frontend && npm run build`

Expected: FAIL until response types/table cells agree.

- [x] **Step 3: Implement transparent display hierarchy**

```tsx
<p>Overall combines durable role-aware Quality with a capped current Form lift. Impact then accounts for availability and expected minutes.</p>
const [sortKey, setSortKey] = useState<SortKey>("overall_rating");
```

Use `TOP {position} OVERALL` cards and show Overall prominently followed by distinct Quality, Form, Impact cells, retaining driver/source/freshness text.

- [x] **Step 4: Verify and commit**

Run: `cd frontend && npm run build`

Expected: PASS.

Commit: `git add frontend/src/types.ts frontend/src/components/PlayerHub.tsx && git commit -m "feat: display overall quality form and impact ratings"`

### Task 4: Replace the temporary unit proxy with causal role-unit research

**Files:**
- Modify: `src/pl_predictor/evaluate/goal_contribution_research.py:167-320`
- Modify: `tests/test_goal_contribution_research.py`
- Modify: `tests/test_walk_forward.py`
- Modify: `docs/AI_CONTINUITY.md`

**Interfaces:**
- Consumes: historical FPL rows and shifted start/form features.
- Produces: exactly eight home/away role-strength fields plus a research-only walk-forward report.

- [x] **Step 1: Write failing causal expected-XI tests**

```python
features = build_projected_team_player_features(seasons=["2023-24"])
units = [column for column in features if column.endswith("_unit_strength")]
assert len(units) == 8
assert {"home_gk_unit_strength", "away_fwd_unit_strength"} <= set(units)
first = features.sort_values("kickoff_date").iloc[0]
assert first[units].fillna(0).sum() == 0
```

Add a test that mutates current-match realised minutes/goals and proves same-fixture units do not change.

- [x] **Step 2: Confirm the temporary proxy fails the new contract**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_goal_contribution_research.py tests/test_walk_forward.py -q`

Expected: FAIL because current `position_quality` is raw goals/xG and lacks promotion metrics.

- [x] **Step 3: Implement shifted Quality-plus-Form aggregation and report**

```python
rows["historical_quality"] = _historical_role_quality(rows)
rows["historical_form"] = _historical_form_lift(rows).clip(0, 15)
rows["historical_overall"] = (rows["historical_quality"] + rows["historical_form"]).clip(upper=100)
rows["expected_xi_weight"] = rows["starts_last5"].fillna(0).clip(0, 1) * rows["minutes_ema"].fillna(0).clip(0, 90) / 90
rows["weighted_unit_score"] = rows["historical_overall"] * rows["expected_xi_weight"]
```

Use shifted columns only, pivot by home/away and `GK/DEF/MID/FWD`, and extend folds with RPS, Brier, ECE, scoreline log loss, coverage, and importance. Return `promotion_eligible=False`; no training/API code may read it.

- [x] **Step 4: Record mandatory manual gates**

Document the exact eight fields, lower mean RPS/no calibration regression/non-regressing recent fold gates, and `not deployed` status in `docs/AI_CONTINUITY.md`.

- [x] **Step 5: Verify and commit**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_goal_contribution_research.py tests/test_walk_forward.py -q`

Expected: PASS.

Commit: `git add src/pl_predictor/evaluate/goal_contribution_research.py tests/test_goal_contribution_research.py tests/test_walk_forward.py docs/AI_CONTINUITY.md && git commit -m "feat: add causal role-unit rating research"`

### Task 5: Upgrade role-model evaluation and complete verification

**Files:**
- Modify: `src/pl_predictor/models/player_ratings.py`
- Modify: `tests/test_player_ratings.py`
- Modify: `docs/RESEARCH_FINDINGS.md`

**Interfaces:**
- Consumes: historical shifted form rows.
- Produces: `evaluate_role_models(history)` with position, target, MAE/RMSE, selected source, and strongest driver.

- [x] **Step 1: Write failing chronological promotion-gate test**

```python
report = evaluate_role_models(history)
assert {"position", "target", "baseline_mae", "rich_mae", "baseline_rmse", "rich_rmse", "selected_model", "top_driver"} <= set(report.columns)
assert (report.loc[report["selected_model"] == "rich", "rich_mae"] < report.loc[report["selected_model"] == "rich", "baseline_mae"]).all()
```

- [x] **Step 2: Confirm old FPL-points-per-90 evaluator fails**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_player_ratings.py -q`

Expected: FAIL because the old evaluator lacks role-target metadata.

- [x] **Step 3: Implement strict role-target evaluation**

Use shifted historical composites: GK save/clean-sheet/BPS; DEF clean-sheet/defensive/attack; MID xGI/creation/output; FWD xG/xGI/output. Train only prior seasons and label rich only if MAE improves and RMSE does not regress. Do not alter live serving without manual review.

- [x] **Step 4: Document outcomes and verify**

Run: `PYTHONPATH="$(pwd)/src" python -m pytest tests/test_player_ratings.py tests/test_hub_analytics.py tests/test_public_snapshot_routes.py tests/test_goal_contribution_research.py tests/test_walk_forward.py -q && cd frontend && npm run build && git diff --check`

Expected: focused tests pass, frontend builds, diff check is empty. Record selected role source and unit deployability in `docs/RESEARCH_FINDINGS.md`, without claiming promotion if gates fail.

- [x] **Step 5: Commit**

Commit: `git add src/pl_predictor/models/player_ratings.py tests/test_player_ratings.py docs/RESEARCH_FINDINGS.md && git commit -m "docs: record role-aware rating research results"`
