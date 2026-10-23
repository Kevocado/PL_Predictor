# PL-1 venue-split audit (walk-forward, multi-season)

Task PL-1 (Lane C): fix the 68 of 231 constant-in-training columns caused
by the rolling-form venue split (`home_away_*` / `away_home_*` were NaN on
every training row because `build_training_frame` filtered the long frame
by `was_home`).

## Fix applied

`features/rolling_form.py::build_rolling_form` now forward-fills the
venue-specific form columns within team (sorted by date) so a home match
also exposes the team's latest away-venue form, and vice versa. Rolling
already uses `shift(1)` within venue, so only past values are reused (no
lookahead). Result: 0 constant feature columns in training (was 45 with
n=3,040).

## Evaluation

Paired bootstrap, resampling matches (B=5,000), primary measure = multi-
season RPS of `ml_scoreline` (XGBoost goals regressors).

- Walk-forward folds: seasons 2021-22 .. 2025-26 (5 folds, 1,900 val matches).
- Baseline RPS: 0.200008
- After fix RPS: 0.200340
- Mean paired difference (base - after): -0.000332
- 95% bootstrap CI: [-0.001302, +0.000607] — includes zero.

## Gate

NOT shipped as a measured improvement: CI on the primary measure includes
zero. No other measure is significantly worse (Brier CI also includes
zero). The parity fix is retained because it removes a latent train/serve
skew (`latest_form` already populated both venue columns at serving while
training saw a constant 0.0), with no measured regression; committed model
artefacts are untouched, so served predictions are unchanged.

## Dixon-Coles multi-season RPS (audit, xi=0.0018)

- Seasons 2017-18 .. 2024-25, 8 folds: mean RPS 0.2083 (Brier 0.591).
- Published DC benchmark: 0.199. Our DC underperforms; xi tuning available
  via `evaluate/tune_dc_bp_xi.py` (not run here — xi grid hit negative
  goal-matrix errors on some folds).

## Numbers

- base_rps.csv / after bootstrap: mean diff -0.000332, CI [-0.001302, 0.000607].
"