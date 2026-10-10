# Player shots / shots-on-target: fix and re-ship plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (one fresh subagent per task, review between). Steps use `- [ ]`.

**Goal:** replace the heuristic per-player shots / SoT numbers with a model that beats the naive baselines in a walk-forward evaluation, then turn the display back on.

**Why now:** the 2026-10-10 back-test (`src/pl_predictor/evaluate/player_shots_backtest.py`, PL#65) over 34,448 appearances (2023-24 to 2025-26) found the live numbers are not a predictor beyond a rate lookup. The display is hidden (`SHOW_PLAYER_SHOTS = false` in `PlayerScorerList.tsx`). This plan is the way back.

**Spec this satisfies:** `docs/superpowers/specs/2026-09-04-player-shots-market-design.md` section 7 (must beat a position-average baseline on a held-out season before shipping). That evaluation never existed; this plan creates it.

## What the back-test found (the problems to fix)

| Finding | Number |
|---|---|
| Shots abs error vs position-average (model minus baseline) | +0.065 [+0.047, +0.081] (worse) |
| Shots abs error vs player's own trailing per-90 | +0.097 [+0.083, +0.111] (worse) |
| SoT Brier vs position-average | +0.0091 [+0.0070, +0.0113] (worse) |
| Correlation with actual shots | about 0.38, same as position-average |
| Appearances given exactly 0 expected shots | 23%; their actual mean is 0.26 shots, 6.3% had a SoT |
| Players with no earlier same-season game | always 0 (prior-season data carries no shots rate) |
| Top decile predicted vs actual shots | 4.06 vs 2.00 (overpredicted about 2x; about 3x with under 5 prior games) |
| `anytime_shot_on_target_prob` bucket 0-0.05 / 0.6+ | predicted 0.003 vs actual 0.095 / predicted 0.76 vs actual 0.49 |
| Removing `strength_multiplier` / `shots_scale` / both | each improves it; both removed is best (the two are correlated 0.40: team attack counted twice) |

Cause: `predict_player` computes `shots_per90 x strength_multiplier x minutes x availability x shots_scale` from a raw last-N rate. No shrinkage for small samples, two overlapping team factors, and a Poisson tail that is badly calibrated for shots.

## Global constraints

- Offline tests only (the repo's network guard); real data shapes (pandas frames shaped like the cached Understat/FPL files).
- Training and serving use the **same function** (no train/serve skew): the walk-forward tool must call the production code path.
- Evaluation rule is binding: paired bootstrap by match, 2,000 resamples, 95% CI. A change ships only if it beats BOTH naive baselines (position-average per-90 x minutes, and the player's own trailing per-90 x minutes) on the stated metrics with the CI excluding zero. Otherwise the display stays hidden and the result is documented.
- Hold-out discipline: fit any parameter (shrinkage strength, team-factor exponent) on seasons before the evaluated one only.
- Branch + PR per task, never push to main, read every check line before merge, `scripts/coderabbit.sh` gate (exit 0, no pipe).

## Task 1: make the evaluation reusable and tested

**Files:** `src/pl_predictor/evaluate/player_shots_backtest.py` (from PL#65), new `tests/test_player_shots_backtest.py`.

- [ ] Add offline tests with real-shaped frames: metrics are computed correctly on a tiny hand-checked set; the bootstrap is paired by match; a perfect predictor beats both baselines; the position-average baseline never sees the evaluated match.
- [ ] Add a `--predictor` hook so a candidate prediction function can be evaluated against the same appearances and baselines without editing the tool.
- [ ] Make the tool print the gate verdict (pass/fail per metric with CIs) as the last lines.

## Task 2: shrink the per-90 rate (the single most valuable fix)

**Files:** `src/pl_predictor/features/player_form.py` (rate estimation), `src/pl_predictor/models/player_goals.py` (`predict_player` shots lines), tests.

- [ ] Write the failing test first: a player with 1 prior appearance and 5 shots must not predict 5 shots next game; a player with 0 prior same-season appearances must not predict exactly 0 if he has a prior-season or position rate.
- [ ] Empirical-Bayes rate: `rate = (n90 * player_rate + k * prior) / (n90 + k)`, prior = the player's prior-season shots per 90 when he has one, else the position average. Fit `k` (separately for shots and SoT) on seasons before the evaluated one by minimising out-of-sample squared error; record the fitted values in the PR.
- [ ] Compute prior-season shots per 90 from the Understat shots history (currently dropped: "prior-season totals have no shots").
- [ ] Re-run the Task 1 tool with this predictor. Record the numbers.

## Task 3: one team factor, chosen by measurement

**Files:** same as Task 2, plus the evaluation run.

- [ ] Evaluate four variants against the Task 2 rate: no team factor; `strength_multiplier` only; `shots_scale` only; a single fitted factor `team_term ** beta` with `beta` fit on earlier seasons. Pick the variant with the best held-out score; do not keep both factors unless the measurement says so.
- [ ] Remove the losing factor from `predict_player` (and the unused context plumbing, if nothing else reads it).

## Task 4: calibrated SoT probability, no impossible zeros

**Files:** `src/pl_predictor/models/player_goals.py`, calibration helper, tests.

- [ ] SoT expectation = shrunk shots rate x a shrunk SoT-per-shot ratio, so SoT never exceeds shots.
- [ ] Replace `anytime_probability(lam)` (Poisson, `1 - exp(-lam)`) for the SoT probability with a mapping fit on earlier seasons (negative binomial dispersion, or isotonic regression on the predicted rate), then check reliability by bucket.
- [ ] Any player with expected minutes above 0 gets a non-zero expectation; no hard 0.0.

## Task 5: the gate, with live-like inputs

- [ ] Re-run the full walk-forward on all three seasons with the final predictor. Required, all with CIs excluding zero versus both baselines: shots and SoT absolute error, squared error, and SoT Brier.
- [ ] Required calibration: by prediction decile, predicted mean within 15% of actual mean (or an absolute 0.05 where the mean is small); reliability buckets for the SoT probability within 0.10 of actual; no predicted-zero bucket whose actual mean exceeds 0.10.
- [ ] The earlier back-test substituted a trailing goal estimate for the scoreline model's goal expectation and set availability to 1.0. Re-run once with the real scoreline expectation for the most recent full season if it can be reconstructed offline; state exactly what was substituted if not.
- [ ] If the gate fails: keep `SHOW_PLAYER_SHOTS = false`, document the result in `docs/RESEARCH_FINDINGS.md`, stop.

## Task 6: re-enable and verify live

- [ ] Flip `SHOW_PLAYER_SHOTS` to `true`; update `PlayerScorerList.shots.test.tsx` to assert the figures appear.
- [ ] Update the tooltip text so it states what the numbers are (expected shots, expected shots on target, chance of at least one on target).
- [ ] Deploy; check Arsenal v Leeds on a phone and a desktop: no player shows 0.0 shots at 90 expected minutes, forwards are not below midfielders without a stated reason.
- [ ] Update the design spec section 7 with the evaluation that now exists.

## Out of scope

Team-level "Predicted shots on target" (a different model, not audited here); the goals / assists models; a trained gradient-boosted shots model (revisit only if the shrunk-rate model passes the gate by a thin margin and more signal is wanted).
