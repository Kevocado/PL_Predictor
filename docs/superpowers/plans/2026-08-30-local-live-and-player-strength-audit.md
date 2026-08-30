# Local/Live Audit and Player-Strength Research Plan

**Goal:** Audit the local and public application independently, repair only reproduced defects, then re-test causal player-strength units without changing the live scoreline model.

## Guardrails

- Local FastAPI/Vite and Render are separate products: local builds live state; Render must use `PUBLIC_MODE=true` and `data/public_snapshot.json` only.
- No audit task may retrain, refresh data, mutate tracking, or rewrite the snapshot automatically.
- No player feature may use a final lineup, post-match minute, later season, closing odds, Player Hub rounded score, or a fixture prediction from the future.
- Every model candidate stays research-only until it beats the existing model on the documented gates and the user explicitly approves promotion.

## Phase 1 — Audit the local/public route contract

1. Add a focused public-route matrix test covering Fixtures, fixture detail,
   fixture players, Data Hub, Player Hub, and FPL. In `PUBLIC_MODE`, each
   response must read a controlled snapshot fixture; monkeypatch live builders
   to fail so an accidental live computation is observable.
2. Add public-lifespan coverage: a TestClient startup with `PUBLIC_MODE=true`
   must not schedule `_initial_sync`, tracking, or auto-retraining.
3. Expand snapshot schema tests to require `generated_at`,
   `model_fingerprint`, current gameweek, every public Hub payload, and all
   FPL payloads. Missing snapshot content must return a typed empty/error
   payload, never construct live state on Render.
4. Build the Docker image locally and smoke-test public readiness, fixtures,
   Hub, and FPL responses with `PUBLIC_MODE=true`. Record startup and request
   timings plus available process-memory measurements.
5. Obtain the actual deployed URL and Render logs when available. Compare OOM
   timestamp, process environment, request path, and lifecycle evidence before
   changing the Render plan or calling a restart a memory leak.

## Phase 2 — Audit client behaviour and data freshness

1. Capture a browser request trace for initial Fixtures paint and background
   dashboard preload, then Data Hub, Calibration, and FPL. Verify Vite uses
   `/api` proxy requests locally and the public app uses the same origin.
2. Reproduce any loading/failure state before changing `frontend/src/api/client.ts`.
   Preserve independent panels and never automatically retry a retrain or
   refresh POST.
3. Create `reports/local_live_reliability_audit.md`: endpoint status/source,
   request duration, partial-failure behaviour, snapshot generation time,
   fixture/table freshness, manifest training timestamp, and calibration match
   count. Mark missing Render evidence as blocked, not healthy.
4. Inventory every feature in `features/build.py`: source, cache/freshness,
   earliest forecast-time availability, historical coverage, and leakage risk.
   Add only reproduced safeguards; record data/model findings in
   `docs/AI_CONTINUITY.md`.

## Phase 3 — Corrected player-strength experiment

1. Extract a pure historical identity/eligibility helper from
   `models/player_ratings.py`. It must resolve name variants before aggregation
   using same role, token-subset names, overlapping club, and non-overlapping
   seasons; it must be evaluated at a historical fixture cutoff.
2. Test the helper for GK, DEF, MID, and FWD: sub-900-minute players are
   Provisional, one qualifying season is Limited evidence, and two qualifying
   seasons are Established. New-to-league players receive a neutral role prior
   in research—not invented reputation or an accidental zero.
3. Build legal expected XIs from shifted starts/minutes only. Aggregate eight
   latent (unrounded) expected-minutes-weighted fields: home/away GK, DEF,
   MID, FWD. Report role coverage alongside each row.
4. Run the candidate on the exact chronological folds/fixture universe used by
   EXP-2026-21. Produce `reports/exp-2026-22-player-strength-rerun.csv` and
   Markdown with baseline/candidate RPS, Brier, log loss, ECE, coverage,
   fixture count, and feature importance for every fold and the average.
5. The report may be **accepted for manual review** only with lower average
   RPS, no Brier/log-loss/ECE regression, no most-recent-fold RPS regression,
   and adequate established-player coverage. Otherwise it is explicitly
   rejected, continuity is updated, and `features/build.py`/the manifest stay
   unchanged.

## Required verification

- Focused API/snapshot/client tests for each reproduced finding.
- `PYTHONPATH="$(pwd)/src" .venv/bin/python -m pytest tests/ -q`.
- `npm run build` from `frontend/`.
- Present the complete audit ledger and the full baseline-vs-candidate fold
  table before asking whether a winning player-strength feature should be
  promoted.
