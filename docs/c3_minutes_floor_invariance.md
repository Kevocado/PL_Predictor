# C3 minutes-floor invariance (full walk-forward re-run)

Source: `player_shots_backtest` re-run with floored expected minutes plugged into
`predict_player`, across 4 available seasons (2021-22, 2022-23, 2023-24, 2024-25).
Floored exp minutes = `max(exp_minutes, floor)` where floor = GK 60 / others 45 when
current-season appearances (`games_before`) >= 6, else 0.

Games checked: **44,780 appearances** across **3,040 matches**, 4 seasons.
Floor applied to 32,823 of 44,780 rows (73.3%).

## Minutes (MAE)

- Base MAE 24.92 -> floored 24.63; paired d **-0.2901 [-0.3730, -0.2117]** — **improves, CI excludes 0** (passes).

## Scorer / assist / shots / shots-on-target invariance (log-loss, Poisson NLL)

Corrected = floored minutes plugged into `predict_player`. Raw = uncapped `exp_minutes`.
Paired cluster bootstrap (2,000) over matches.

| Leg | raw NLL | corrected NLL | point delta | dNLL 95% CI | entirely > 0 (worse)? |
|---|---|---|---|---|---|
| scorer (goals) | 0.4629 | 0.4635 | +0.0006 | [-0.0000, +0.0013] | No |
| assist | 0.4849 | 0.4860 | +0.0011 | [+0.0005, +0.0018] | **Yes** |
| shots | 0.7383 | 0.7392 | +0.0009 | [-0.0011, +0.0029] | No |
| shots-on-target | 0.4600 | 0.4612 | +0.0012 | [+0.0003, +0.0021] | **Yes** |

## Gate

- Minutes MAE: **PASSES** (CI excludes zero).
- Scorer/assist/shots/sot invariance: **FAILS** — assist and shots-on-target legs have
  dNLL CIs entirely > 0 (corrected is statistically worse).

## Verdict: **NO-SHIP**

The floor breaks on the **assist** and **shots-on-target** legs: flooring expected
minutes inflates the per-appearance minute fraction for low-minute players, which
inflates the assist and SoT lambdas enough to worsen Poisson log-loss with a CI
that excludes zero on both legs. Scorer and shots legs are not significantly worse
(CIs include zero) but show no improvement either.

## Action

- **Ship nothing.** Floor stays **behind a flag OFF** (research-only; scoring code
  unchanged, but the research harness `src/pl_predictor/evaluate/player_shots_backtest.py`
  was extended to emit floored vs raw lambdas).
- Documented where it broke: assist leg dNLL +0.0011 [+0.0005,+0.0018] and
  shots-on-target leg dNLL +0.0012 [+0.0003,+0.0021].
- Follow-up: a per-leg (or per-position) floor, or a floor that only applies to
  the minutes estimate used for minutes MAE, not the lambdas. Re-run this same
  invariance gate before any ship.