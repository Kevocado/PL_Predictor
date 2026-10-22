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

## Reviewer live-data leads — tested on full walk-forward (closing odds, 2022-23 to 2024-25)

*Full-sample paired-bootstrap (2000 resamples) on closing odds (Bet365). Propose fix only if CI excludes zero.*

- **match_result (1X2) home win**: bias -0.0076, 95% CI [-0.0349, +0.0178] — **no problem** (CI straddles zero). Reliability slope 1.053, CI [-0.150, 0.148] includes 1.0. Decile calibration (min n=5): 0.08→0.06 (n=18) | 0.16→0.12 (n=121) | 0.25→0.25 (n=162) | 0.35→0.38 (n=213) | 0.45→0.44 (n=195) | 0.55→0.61 (n=168) | 0.65→0.66 (n=130) | 0.74→0.68 (n=88) | 0.84→0.91 (n=45). The 50-60% bucket (reviewer spot-check: 8/23 = 35%) is not an outlier in full walk-forward — overall calibration is fair.
- **O/U 2.5**: bias -0.0094, 95% CI [-0.0373, +0.0185] — **no problem**. Slope 1.130, CI [-0.363, 0.368]. Deciles: 0.39→0.40 (n=5) | 0.46→0.45 (n=206) | 0.55→0.56 (n=512) | 0.64→0.65 (n=332) | 0.72→0.72 (n=85). The 50-60% bucket (reviewer: 42/79 = 53%) is consistent with full walk-forward — no problem.
- **BTTS**: no closing odds available in data source.
- **Verdict**: neither match_result nor O/U 2.5 shows a calibration problem on full walk-forward; reviewer's spot-checks (8/23 and 42/79) are consistent with small-sample noise. No fix proposed.
