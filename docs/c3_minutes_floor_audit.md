# C3 minutes-floor audit (walk-forward)

Source: player_shots_backtest output /tmp/psb_out.csv (2024-25, 2025-26).
Floor: GK 60 min, others 45 min if current-season appearances (games_before) >= 6.
Paired match-cluster bootstrap (2000). Research only: no src/ changes.

## Segment finding
- <5-expected-minutes segment: 1378 of 23064 rows (6.0%); mean actual minutes **46.3**.
  Positions: {'MID': np.int64(625), 'DEF': np.int64(493), 'FWD': np.int64(156), 'GK': np.int64(104)}
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

## Reviewer live-data leads (to test against full walk-forward)
*Small-n spot-checks; test at full sample size with paired-bootstrap. Propose fix only if CI excludes zero.*

- **match_result 50-60% bucket**: 8/23 right (35%) vs 60-70%: 10/11, 70%+: 4/4 → possible overconfidence in middle bucket.
- **O/U 2.5 50-60%**: 42/79 (53%) — looks fine, no lead.
- **Test**: run scoreline model walk-forward (e.g., 2019-2025 held-out), bucket predicted home/draw/away probs and O/U 2.5 probs into deciles, compute actual hit rates per bucket, paired-bootstrap CI on calibration gap. If 50-60% bucket gap CI excludes zero → fix warranted (e.g., isotonic recalibration on that bucket).
