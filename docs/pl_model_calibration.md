# PL model calibration audit (our model predictions vs actuals)

Generated 2026-10-22. Model predictions from public_snapshot.json (fixtures_by_gameweek + fixture_detail_by_event_id), n=70 finished fixtures. Paired-bootstrap 95% CIs over 2000 resamples (fixture-level bootstrap). Research only: no model output changed, nothing under src/.

## Summary

- **Data source**: OUR MODEL's predicted probabilities (not bookmaker odds) from public_snapshot.json: `fixtures_by_gameweek` (1X2 probs + actuals) joined with `fixture_detail_by_event_id` (O/U 2.5, BTTS).
- **Sample**: n=70 finished fixtures with model predictions and actual results (2026-08 to 2026-10).
- **Probabilities sum to 1**: 1X2 probs sum to 1.000 for all fixtures.

## Per-market results (paired-bootstrap 95% CIs, 2000 resamples)

| Market | Metric | Value | 95% CI | Reading |
|--------|--------|-------|--------|---------|
| Home win (1X2) | Bias (pred − actual) | +0.0363 | [-0.0661, +0.1440] | no problem |
| Home win (1X2) | Reliability slope | 1.181 | [0.454, 1.849] | inconclusive (n=70, interval too wide to judge) |
| Home win (1X2) | Max calibration gap | 0.2387 | — | >0.03 |
| Draw (1X2) | Bias (pred − actual) | -0.0532 | [-0.1637, +0.0465] | no problem |
| Draw (1X2) | Reliability slope | 1.546 | [-2.136, 4.438] | inconclusive (n=70, interval too wide to judge) |
| Draw (1X2) | Max calibration gap | 0.0803 | — | >0.03 |
| Away win (1X2) | Bias (pred − actual) | +0.0170 | [-0.0875, +0.1150] | no problem |
| Away win (1X2) | Reliability slope | 1.310 | [0.597, 1.934] | inconclusive (n=70, interval too wide to judge) |
| Away win (1X2) | Max calibration gap | 0.3506 | — | >0.03 |
| O/U 2.5 | Bias (pred − actual) | +0.0158 | [-0.1013, +0.1407] | inconclusive (n=70, interval too wide to judge) |
| O/U 2.5 | Reliability slope | 1.564 | [-2.025, 6.776] | inconclusive (n=70, interval too wide to judge) |
| O/U 2.5 | Max calibration gap | 0.0207 | — | <0.03 honest |
| BTTS | Bias (pred − actual) | +0.0195 | [-0.0950, +0.1437] | inconclusive (n=70, interval too wide to judge) |
| BTTS | Reliability slope | 3.282 | [0.578, 5.938] | inconclusive (n=70, interval too wide to judge) |
| BTTS | Max calibration gap | 0.1411 | — | >0.03 |

## Per-decile calibration (min n=5, bins with n<10 flagged ⚠)

### Home win (1X2)
0.16→0.40 (n=5) ⚠ | 0.27→0.25 (n=8) ⚠ | 0.35→0.14 (n=14) | 0.45→0.39 (n=18) | 0.53→0.44 (n=16) | 0.65→0.83 (n=6) ⚠

### Draw (1X2)
0.18→0.10 (n=10) | 0.24→0.32 (n=60)

### Away win (1X2)
0.15→0.00 (n=9) ⚠ | 0.25→0.21 (n=24) | 0.36→0.50 (n=18) | 0.44→0.40 (n=10) | 0.55→0.20 (n=5) ⚠

### O/U 2.5
0.55→0.53 (n=60) | 0.62→0.62 (n=8) ⚠

### BTTS
0.47→0.33 (n=9) ⚠ | 0.55→0.54 (n=57)

## Interpretation
- A market reads as **problem** when its bias/slope CI excludes the ideal value (0 for bias, 1.0 for slope).
- A market reads as **inconclusive** when the CI is too wide (>±0.5 half-width around ideal) to distinguish from ideal — this happens at small n.
- A market reads as **no problem** when the CI contains the ideal value and is narrow enough to be informative.
- **Home win**: bias +0.0363 CI [-0.0661, +0.1440] — no problem; slope 1.181 CI [0.454, 1.849] — inconclusive (n=70, interval too wide to judge).
- **Draw**: bias -0.0532 CI [-0.1637, +0.0465] — no problem; slope 1.546 CI [-2.136, 4.438] — inconclusive (n=70, interval too wide to judge).
- **Away win**: bias +0.0170 CI [-0.0875, +0.1150] — no problem; slope 1.310 CI [0.597, 1.934] — inconclusive (n=70, interval too wide to judge).
- **O/U 2.5**: bias +0.0158 CI [-0.1013, +0.1407] — inconclusive (n=70, interval too wide to judge); slope 1.564 CI [-2.025, 6.776] — inconclusive (n=70, interval too wide to judge).
- **BTTS**: bias +0.0195 CI [-0.0950, +0.1437] — inconclusive (n=70, interval too wide to judge); slope 3.282 CI [0.578, 5.938] — inconclusive (n=70, interval too wide to judge).
- **Reviewer 50-60% bucket lead**: The spot-check (8/23 = 35% for match_result 50-60%) was a small-n spot-check. This analysis is NOT a multi-season walk-forward — it is a single-season snapshot (n=70 finished fixtures from 2026-08 to 2026-10). No multi-season walk-forward backtest exists for the model's own probabilities. The spot-check concern remains open.
- **No multi-season walk-forward has been run** for the model's own probabilities. The scoreline model backtest evaluates closing odds, not our model's own probabilities.
- No fix shipped: research only; fix only if paired-bootstrap CI excludes ideal value on re-tested candidate.

## Verdict
Not enough 2026 fixtures to judge; re-run at ~300 fixtures or on the multi-season backtest (which does not yet exist for model probabilities). Current n=70 is too small to distinguish calibration from noise — all slope CIs are inconclusive (span >±0.5 around 1.0), and bias CIs straddle zero but are too wide to be informative.
