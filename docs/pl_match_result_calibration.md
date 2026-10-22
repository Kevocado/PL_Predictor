# PL match_result / O-U 2.5 calibration (closing odds)

Generated 2026-10-22. Closing odds (Bet365), seasons 2022-23 to 2024-25, n=1140 fixtures. Paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

## Summary

- **1X2 (closing odds)**: n=1140, Brier=0.5618, ECE=0.0351
- **O/U 2.5 (closing odds)**: n=1140, Brier=0.2359
- **Home win bias**: -0.0076 [-0.0349, +0.0178]
- **O/U 2.5 bias**: -0.0094 [-0.0373, +0.0185]
- **Home win reliability slope**: 1.053 [-0.150, 0.148]
- **O/U 2.5 slope**: 1.130 [-0.363, 0.368]
- **Max calibration gap (home)**: 0.4143 [0.4036, 0.4250]

## Per-decile calibration (min n=5)

### Home win (1X2)
0.08→0.06 (n=18) | 0.16→0.12 (n=121) | 0.25→0.25 (n=162) | 0.35→0.38 (n=213) | 0.45→0.44 (n=195) | 0.55→0.61 (n=168) | 0.65→0.66 (n=130) | 0.74→0.68 (n=88) | 0.84→0.91 (n=45)

### O/U 2.5
0.39→0.40 (n=5) | 0.46→0.45 (n=206) | 0.55→0.56 (n=512) | 0.64→0.65 (n=332) | 0.72→0.72 (n=85)

## Interpretation
- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.
- **Home win (1X2)**: bias -0.0076 CI [-0.0349, +0.0178] — no problem; slope 1.053 [-0.150, 0.148] — problem.
- **O/U 2.5**: bias -0.0094 CI [-0.0373, +0.0185] — no problem; slope 1.130 [-0.363, 0.368] — problem.
- **No BTTS data** available in closing odds source.
- **Reviewer 50-60% bucket lead**: spot-check had 8/23 (35%) — full walk-forward decile analysis above replaces spot-check with full-sample paired-bootstrap.
- No fix shipped: research only; fix only if paired-bootstrap CI excludes ideal value on re-tested candidate.
