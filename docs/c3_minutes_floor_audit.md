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

## Scorer/assist invariance (COMPLETED — see docs/c3_minutes_floor_invariance.md)
- Re-run with floored exp_minutes plugged into predict_player across 4 seasons (2021-22..2024-25), 44,780 appearances, 3,040 matches.
- Minutes MAE: base 24.92 -> floored 24.63; d -0.2901 [-0.3730,-0.2117] — improves, CI excludes 0 (PASSES).
- Leg dNLL (corrected - raw): scorer (goals) +0.0006 [-0.0000,+0.0013] (not worse); assist +0.0011 [+0.0005,+0.0018] (WORSE); shots +0.0009 [-0.0011,+0.0029] (not worse); shots-on-target +0.0012 [+0.0003,+0.0021] (WORSE).
- **Verdict: NO-SHIP** — floor breaks on assist and shots-on-target legs. Floor stays behind a flag OFF (research-only; no src/ change).

## Gate
**NO-SHIP** — minutes MAE improves (CI excludes zero), but the full walk-forward
invariance re-run (docs/c3_minutes_floor_invariance.md) shows the assist and
shots-on-target legs worsening (dNLL CI entirely > 0). Floor stays behind a flag
OFF. See docs/c3_minutes_floor_invariance.md.

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
