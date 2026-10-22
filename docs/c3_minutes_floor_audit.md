# C3 minutes-floor audit (walk-forward)

Source: player_shots_backtest output /tmp/psb_out.csv (2024-25, 2025-26).
Floor: GK 60 min, others 45 min if current-season appearances (games_before) >= 6.
Paired match-cluster bootstrap (2000). Research only: no src/ changes.

## Segment finding
- <5-expected-minutes segment: 1378 of 23064 rows (6.0%); mean actual minutes **46.3**.
  Positions: {'MID': 625, 'DEF': 493, 'FWD': 156, 'GK': 104}
- Floor applied to 103 rows (base exp <5, games_before >= 6); those average 37.1 actual minutes.

## Minutes metrics
- MAE all rows: base 24.18 -> floored 24.05; paired d -0.1287 [-0.2349, -0.0239] — **improves, CI excludes 0**.
- MAE appearances: base 24.18 -> floored 24.05; paired d -0.1287 [-0.2349, -0.0239].
- Minutes-calibration max decile gap: base 37.59 -> floored 32.63.

## Scorer/assist invariance
- Cannot fully evaluate: backtest output lacks goal/assist lambdas and floored exp_minutes for shots/sot.
- Requires re-running the backtest with floored exp_minutes plugged into predict_player.
- **Conditional gate**: minutes MAE passes (CI excludes zero), but scorer/assist log-loss invariance NOT YET VERIFIED. Floor should not ship without a full re-run confirming no dNLL worsening (CI > 0) on any leg.

## Gate
**CONDITIONAL PASS** — minutes MAE improves with CI excluding zero; scorer/assist invariance pending full re-run.

## Reviewer live-data leads — tested on MODEL walk-forward (public_snapshot.json, n=70 finished fixtures)

*Full-sample paired-bootstrap (2000 resamples) on OUR MODEL's own predicted probabilities (from fixture_detail + fixtures_by_gameweek). Propose fix only if CI excludes zero.*

See **docs/pl_model_calibration.md** for full model calibration audit (1X2 home/draw/away, O/U 2.5, BTTS, per-decile tables, paired-bootstrap CIs).

**Summary of model calibration (n=70 finished fixtures):**
- **1X2 Home win**: bias +0.0363 CI [-0.066, +0.144] — no problem; slope 1.181 CI [0.454, 1.849] contains 1.0.
- **1X2 Draw**: bias -0.0532 CI [-0.164, +0.047] — no problem; slope 1.546 CI [-2.136, 4.438] contains 1.0.
- **1X2 Away win**: bias +0.0170 CI [-0.088, +0.115] — no problem; slope 1.310 CI [0.597, 1.934] contains 1.0.
- **O/U 2.5**: bias +0.0158 CI [-0.101, +0.141] — no problem; slope 1.564 CI [-2.025, 6.776] contains 1.0.
- **BTTS**: bias +0.0195 CI [-0.095, +0.144] — no problem; slope 3.282 CI [0.578, 5.938] contains 1.0.

**Reviewer 50-60% bucket lead**: The spot-check (8/23 = 35% for match_result 50-60%) is consistent with small-sample noise — full model walk-forward shows no calibration problem in any market. All bias CIs straddle zero; all slope CIs contain 1.0. Per-decile tables in pl_model_calibration.md show no outlier buckets.

**Verdict**: No calibration problem found in our model's own probabilities. No fix proposed.
