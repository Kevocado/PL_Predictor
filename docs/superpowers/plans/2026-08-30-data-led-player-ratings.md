# Data-Led Player Ratings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build transparent role-aware Ability ratings from observed multi-season Premier League data, mark insufficient-evidence players Provisional, and retain team-strength use as research-only.

**Architecture:** `player_ratings.py` receives a local-file historical-prior builder and a live score contract. `hub_analytics.py` serializes its evidence status, while the React Player Hub handles nullable Overall values explicitly. Documentation records the source limits and preserves the current rejected team-unit decision.

**Tech Stack:** Python, pandas, FastAPI, React/TypeScript, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-data-led-player-ratings-design.md`

## Global Constraints

- Do not use FPL `total_points`, FPL projections, fixture predictions, or current-pool percentiles to calculate Quality.
- Serve only cached official FPL bootstrap and local FPL archive data; no Player Hub visit may fetch, train, or mutate.
- Fewer than 900 qualifying minutes in the latest three completed seasons means `rating_status="provisional"` and no rankable Overall.
- Quality/Overall cap at 95; availability only affects Impact.
- Team-unit fields remain research-only until the documented promotion gate passes.

---

### Task 1: Derive multi-season, context-adjusted historical priors

**Files:**

- Modify: `src/pl_predictor/models/player_ratings.py`
- Modify: `tests/test_player_ratings.py`

**Interfaces:**

- Consumes: cache archive rows with player name, position, team, season, minutes, and observed role statistics.
- Produces: `build_historical_priors(history: pd.DataFrame) -> dict[str, dict[str, float | str]]` and memoized `cached_historical_priors()` records containing `quality_rating`, `evidence_minutes`, `rating_status`, and `rating_driver`.

- [ ] **Step 1: Write failing prior tests**

```python
def test_multi_season_prior_rewards_sustained_elite_role_evidence_not_fpl_points():
    priors = player_ratings.build_historical_priors(history)
    assert priors["elite forward"]["quality_rating"] >= 85
    assert priors["elite forward"]["rating_status"] == "established"

def test_goalkeeper_saves_without_prevention_advantage_cannot_become_elite():
    priors = player_ratings.build_historical_priors(history)
    assert priors["busy keeper"]["quality_rating"] < 70

def test_insufficient_recent_history_is_provisional():
    priors = player_ratings.build_historical_priors(history)
    assert priors["new signing"]["rating_status"] == "provisional"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_player_ratings.py -q`

Expected: FAIL because `build_historical_priors` does not exist and priors contain no evidence status.

- [ ] **Step 3: Implement the prior builder**

```python
def build_historical_priors(history: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    seasons = _latest_positioned_seasons(history, limit=3)
    rows = history[history["season"].isin(seasons) & (history["minutes"] > 0)]
    season_rows = _aggregate_player_seasons(rows)
    return _calibrate_role_priors(season_rows, seasons)
```

Aggregate normalized-name/position/team seasons. Calculate team defensive baselines from goalkeeper minutes, compare GK/DEF goals conceded to that baseline, use xGI/xA/output for MID/FWD, derive fixed robust role anchors from qualifying history, and apply 0.55/0.30/0.15 recency with 900/1,800-minute status/confidence thresholds. Do not access `total_points`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_player_ratings.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/models/player_ratings.py tests/test_player_ratings.py
git commit -m "feat: build data-led multi-season player priors"
```

### Task 2: Apply the Provisional and fixed-scale live score contract

**Files:**

- Modify: `src/pl_predictor/models/player_ratings.py`
- Modify: `tests/test_player_ratings.py`

**Interfaces:**

- Consumes: current FPL element data, position mapping, Task 1 prior records.
- Produces: existing ratings plus `rating_status`, `rating_evidence_minutes`, nullable `overall_rating`, and a fixed 0-95 Quality/Overall scale.

- [ ] **Step 1: Write failing live-contract tests**

```python
def test_provisional_player_has_no_rankable_overall_but_keeps_live_form_and_impact():
    rating = player_ratings.rate_bootstrap_elements([new_player], positions, historical_priors={})[1]
    assert rating["rating_status"] == "provisional"
    assert rating["overall_rating"] is None
    assert rating["live_form_rating"] is not None

def test_availability_only_changes_impact_for_an_established_player():
    assert doubtful["quality_rating"] == available["quality_rating"]
    assert doubtful["overall_rating"] == available["overall_rating"]
    assert doubtful["current_impact_rating"] < available["current_impact_rating"]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_player_ratings.py -q`

Expected: FAIL because every player currently receives numeric Quality and Overall.

- [ ] **Step 3: Implement the live blend**

```python
prior = historical_priors.get(_element_name_key(element))
status = prior.get("rating_status", "provisional") if prior else "provisional"
quality = _blend_prior_with_current_role_evidence(prior, element, position)
overall = _overall_from_quality_and_form(quality, form) if status == "established" else None
```

Keep Live Form and Impact readable for Provisional players. Cap established Quality and Overall at 95, retain the existing evidence gate for current Form, and ensure the blend reads no point/projection fields.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_player_ratings.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/models/player_ratings.py tests/test_player_ratings.py
git commit -m "feat: mark insufficient-evidence players provisional"
```

### Task 3: Publish and render data confidence in Player Hub

**Files:**

- Modify: `src/pl_predictor/api/hub_analytics.py`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/components/PlayerHub.tsx`
- Modify: `tests/test_hub_analytics.py`

**Interfaces:**

- Consumes: Task 2 rating dictionaries.
- Produces: player rows with `rating_status`, `rating_evidence_minutes`, nullable Overall, and unchanged Live Form/Impact fields.

- [ ] **Step 1: Write failing response-shape tests**

```python
report = hub_analytics.build_player_hub(bootstrap)
newcomer = next(player for player in report["players"] if player["name"] == "Newcomer")
assert newcomer["rating_status"] == "provisional"
assert newcomer["overall_rating"] is None
assert newcomer["rating_evidence_minutes"] == 0
assert newcomer not in report["leaderboards"]["MID"]
```

- [ ] **Step 2: Run the focused backend tests and verify RED**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_hub_analytics.py -q`

Expected: FAIL because the current response has neither status nor leader-board exclusion.

- [ ] **Step 3: Serialize status and update the TypeScript display**

```tsx
{player.rating_status === "provisional"
  ? <span className="inline-flex rounded-md border border-pl-pink/50 bg-pl-pink/10 px-2 py-1 text-xs font-bold text-pl-pink">Provisional</span>
  : <RatingCell value={player.overall_rating} tone="text-pl-pink" />}
```

Add a separate boxed Quality cell beside Overall, retain Live Form and vs Quality, and explain that Overall is observed multi-season role evidence plus a capped current-form lift. Render unavailable values as `—`, never zero.

- [ ] **Step 4: Verify backend and frontend**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_hub_analytics.py -q && cd frontend && npm run build`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pl_predictor/api/hub_analytics.py frontend/src/types.ts frontend/src/components/PlayerHub.tsx tests/test_hub_analytics.py
git commit -m "feat: show provisional player ratings"
```

### Task 4: Record provenance and keep unit strength gated

**Files:**

- Modify: `docs/AI_CONTINUITY.md`
- Modify: `docs/RESEARCH_FINDINGS.md`
- Modify: `tests/test_player_ratings.py`

**Interfaces:**

- Consumes: Task 1 provenance and existing EXP-2026-21 result.
- Produces: documented data coverage, rating boundaries, and an explicit no-deployment team-unit decision.

- [ ] **Step 1: Add a forbidden-field regression test**

```python
def test_quality_ignores_fpl_points_and_projection_fields():
    baseline = player_ratings.rate_bootstrap_elements([element], positions, historical_priors=priors)[1]
    changed = player_ratings.rate_bootstrap_elements([{**element, "total_points": 999, "projected_points": 999}], positions, historical_priors=priors)[1]
    assert changed["quality_rating"] == baseline["quality_rating"]
```

- [ ] **Step 2: Run it and verify the score provenance**

Run: `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/test_player_ratings.py -q`

Expected: PASS; any failure requires removing the forbidden input before continuing.

- [ ] **Step 3: Update continuity and research findings**

Record the three-season archive, team-context adjustment, Provisional threshold, lack of per-player live fetches, and that EXP-2026-21 remains rejected. Do not claim a scoreline gain from this Player Hub implementation.

- [ ] **Step 4: Run full verification**

Run: `PYTHONPATH="$(pwd)/src" MPLCONFIGDIR=/tmp/pl-mpl .venv/bin/python -m pytest tests/ -q && cd frontend && npm run build && git diff --check`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/AI_CONTINUITY.md docs/RESEARCH_FINDINGS.md tests/test_player_ratings.py
git commit -m "docs: record data-led player ratings"
```
