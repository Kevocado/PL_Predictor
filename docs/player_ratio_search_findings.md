# Player ratio-feature search: findings

**Date:** 2026-10-30. **Branch:** `pl-player-ratio-search`.
**Pre-registration:** `docs/player_ratio_search_preregistration.md` (read Amendment 1 first).
**Raw record:** `docs/player_ratio_search_results.json` (run 2, selection only, all 195 candidates x 4 markets).
**Superseded:** `docs/player_ratio_search_run1_superseded.json` (run 1, flawed pipeline, holdout consumed; its holdout numbers are not evidence).

## Verdict

**Nothing is confirmed. No production change, no feature flag, no model file changed.**

- Selection (2022-23, 2023-24, 2024-25; 34,295 appearances per market) produced
  82 candidate-market passes under Holm-adjusted one-sided tests with sign
  stability. The top three per market are kept in the JSON.
- The holdout (2025-26) was read once, in run 1, on a pipeline with an
  asymmetric preprocessing bug (Amendment 1). Run 1's holdout result is void and
  this run does not re-read it. So no kept candidate has a clean out-of-sample
  confirmation.
- A confirmation needs a season that has not been read yet, for example
  2026-27 once complete.

## Selection signals (not confirmed)

Mean paired per-row loss difference, candidate minus baseline (negative = better). Fold means are in the JSON.

| Market | Kept candidate | Mean diff | Holm p |
|---|---|---|---|
| Anytime scorer (log loss, base 0.2613) | `pair_xa_over_xgi_last10` | -0.00053 | 3.5e-6 |
| | `opp_shots_allowed_last3` | -0.00041 | 1e-7 |
| | `opp_shots_allowed_last5` | -0.00039 | 7e-10 |
| Assist (log loss, base 0.2602) | `team_volume_over_opp_allowed_last5` | -0.00076 | 0.027 |
| | `opp_shots_allowed_last5` | -0.00062 | 6e-6 |
| | `opp_shots_allowed_last10` | -0.00050 | 6e-6 |
| Shots (Poisson deviance, base 1.212) | `pair_goals_over_shots_last10` | -0.0223 | 2e-66 |
| | `team_volume_over_opp_allowed_last10` | -0.0194 | 2e-22 |
| | `pair_goals_over_sot_last10` | -0.0194 | 3e-51 |
| SoT (Poisson deviance, base 0.734) | `pair_goals_over_shots_last10` | -0.0075 | 3e-49 |
| | `pair_goals_over_sot_last10` | -0.0066 | 2e-36 |
| | `team_volume_over_opp_allowed_last10` | -0.0060 | 2e-8 |

## Why these are not yet evidence

1. **Baseline omits production's team-volume and team-strength terms.** The
   production shots formula scales by the team's recent shots-for
   (`shots_scale`). Production's scorer and assist formulas scale by the
   scoreline model's team goal expectation. My baseline has neither. So the
   team-volume and opponent-concession candidates may be re-encoding terms
   production already uses. The baseline should be amended to include them,
   and the search re-run with a fresh holdout.
2. **Shot ratios with shots in the denominator.** `goals/shots` and `xG/shots`
   partly re-encode shot volume. A large Poisson gain from them is consistent
   with a functional-form effect (the linear baseline cannot represent an
   inverse-volume relation), not with new player information.
3. **Clustered data, optimistic p-values.** Per-row standard errors ignore
   clustering by player and by season. The p-values are too small. Only three
   selection folds exist, so the sign-stability check is weak.
4. **Duplicate control.** A copy of production's own `shots_per90_last5`
   still shows about -0.001 (shots) and -0.0003 (SoT), from optimiser noise on
   collinear columns. Any selection gain under about 0.002 should be read as
   noise.
5. **Small sample.** Six seasons from 2020-21, because team names in FPL
   history stop before 2020-21 and understat joins need them. 2025-26 holdout
   n = 11,498 appearances, about 900 scorer events. That is adequate for a
   0.001-scale effect, but no holdout remains.
6. **Coverage.** Team-constrained name matching leaves 3.3% of shot rows unmapped
   (down from 12.7%). Shooters left unmapped count as zero shots, which biases
   shot counts slightly down for those players only.

## Not built

- `key_passes`: absent from FPL history for 2020-21 to 2025-26. Key-pass
  candidates dropped.
- npxG/90: no penalty-xG column. xG per 90 used as the stand-in.

## Next step if pursued

1. Amend the baseline to include production's shots-scale team volume (and
   a scoreline-based team strength term for scorer and assist), and re-run
   selection.
2. Hold out a season not yet read: 2026-27 once complete. Apply the
   bootstrap rule as pre-registered.
3. Only a candidate that passes that holdout, with nothing else worse, would go
   behind a feature flag defaulted OFF.
