# Player ratio-feature search: pre-registration

**Registered:** 2026-10-30, before any search code was run against the data.
**Branch:** `pl-player-ratio-search`. **Raw results:** `docs/player_ratio_search_results.json`.
**Implementation:** `src/pl_predictor/evaluate/player_ratio_search.py`.

## Question

Does any ratio feature, built from a player's own strictly pre-match rolling
stats (or a pairwise ratio of them), improve out-of-sample prediction of four
player markets beyond the features the current production player model uses?

Markets: anytime scorer (goal >= 1), assist (assist >= 1), shots, shots on target.

## Data and universe

- Source: FPL gameweek history (`fpl_history`) joined to Understat per-shot
  data (`understat_shots` cache) on fixture.
- Seasons: **2020-21 to 2025-26 only.** The cached FPL history has no team
  names for 2016-17 to 2019-20, so fixtures cannot be joined to Understat
  there. The 2026-27 season is in progress and is excluded.
- Fixture join: FPL home/away canonical team names plus Understat fixture
  date (tolerance one day). Unjoined fixtures are dropped from every market,
  and their count is reported.
- Universe: player appearances with `minutes > 0` in joined fixtures. Shots
  and SoT are 0 for a covered player with no matching Understat shot rows.
  Shooters whose name cannot be crosswalked to an FPL element are counted and
  reported (their shots would be lost as zeros).

## Strict lag

Every candidate, every baseline feature and every team/opponent feature is a
rolling statistic over the **prior** appearances only (`shift(1)`), grouped by
season and player (or by season and team). The current match contributes only
its own target. A unit test checks that changing the current match's shots
does not change its own feature.

## Baseline (the "current production player model features")

- The production position-rate feature set `models/player_goals.RATE_FEATURES`
  (goals, assists, xG, xA, xGI, threat, creativity, rolling windows 3/5/10,
  `was_home`), plus `threat_last10`, `creativity_last10` (the reliability
  extras production blends in), plus the production lineup/minutes features
  (`starts_*`, `sub_rate_*`, `minutes_*`, `minutes_ema`, `start_streak`).
- For shots and SoT only: the production blended shot rates
  `shots_per90_last{3,5,10}` and `shots_on_target_per90_last{3,5,10}`.
- Position dummies.
- Model class: the same calibrated logistic regression used by
  `evaluate/goal_contribution_research.py` for binary markets (balanced class
  weights, Platt-calibrated on the most recent training season), and a Poisson
  GLM for counts. Production's anytime probability is a Poisson-from-rates
  formula, not a classifier, so this class choice is a stated comparison
  device, not a claim about production internals.

## Candidate pool (every candidate is recorded in the JSON, pass or fail)

Built from prior-appearance rolling windows w in {3, 5, 10}:

1. Per-90 rates: shots, SoT, goals, assists, xG, xA, key passes (`key_passes`
   exists in FPL history), xGI, threat, creativity.
2. Ratios: SoT/shots, goals/SoT, goals/shots, xG/shot, goals/xG, assists/xA,
   key passes/xA, assists/key passes.
3. Team-share ratios: player shots and SoT divided by the team's shots in the
   same prior matches. Minutes-weighted versions use the per-match share
   weighted by minutes.
4. Team volume: own team's mean shots for, and the opponent's mean shots
   allowed, both over prior matches.
5. Pairwise ratios: for windows 5 and 10, every ordered pair of per-90 rates
   from {goals, assists, xG, xA, shots, SoT, key passes, threat, creativity,
   xGI}, computed as `a / (b + 0.01)`.

**npxG/90 is not built.** The FPL history carries no penalty-xG column, so
non-penalty xG cannot be derived from these data. The xG-per-90 candidate
stands in for it, and the JSON records this gap.

Ratios are clipped at the 99.5th percentile of the **training fold** only
(no test-set information enters the clip). Missing values are filled with 0,
matching the harness convention.

The pool is capped at 300 candidates. The count is written into the JSON.

## Folds

- Selection (walk-forward, test seasons): 2022-23, 2023-24, 2024-25. Each fold
  trains on every earlier season from 2020-21. Calibration season = the last
  training season.
- Holdout: 2025-26, trained on 2020-21 to 2024-25. **Looked at once, after
  selection is frozen.** No selection step reads holdout rows.

## Selection rule

For each market and each candidate:

- Metric per row: log loss (binary) or Poisson deviance (counts).
- Statistic: mean paired per-row difference (candidate minus baseline) over
  pooled selection rows. The standard error is the per-row standard deviation
  divided by root n. The one-sided p-value tests improvement.
- Family: all candidates within a market. Holm step-down adjustment over the
  whole family.
- Candidate passes if Holm-adjusted p < 0.05 **and** the mean difference is
  negative in every one of the three selection folds (sign-stable).
- Kept set: up to 3 passing candidates per market, ranked by the most negative
  mean difference. If no candidate passes in a market, that market has no
  kept set.

## Holdout rule (evaluated once)

For each kept candidate, paired bootstrap over holdout rows (2000 resamples,
seed 20261030):

- Pass requires the confidence interval for the mean difference to exclude
  zero on the improvement side. The interval is Bonferroni-adjusted over K, the
  total number of kept candidates across markets, so its level is
  1 - 0.05/K.
- "Nothing else worse": the secondary metric must not be worse than baseline.
  Brier for binary markets, and MAE for counts. For binary markets, expected
  calibration error must not be worse either.
- If no candidate is kept in any market, the holdout is **not** evaluated.
  The result is then a docs-only negative.

## Power statement

Holdout n is reported. This sample is small (six seasons, three selection
folds). A null or inconclusive result here does not rule out a small effect.
The report must say so, and must not present a non-pass as evidence of no
effect.

## Amendment 1 (2026-10-30): read this before the results

Run 1 was executed end to end, including the holdout, and then found to be
flawed. The holdout season is therefore **consumed**, and this amendment
records what happened so the results cannot be read as cleaner than they are.

1. **Asymmetric preprocessing (the bug).** Run 1 winsorised candidate columns
   at the training quantile but left baseline columns untouched. A control
   exposed it: `cand_shots_per90_last5` is an exact copy of production's own
   `shots_per90_last5`, and it showed a selection gain of about 0.033 Poisson
   deviance on every shots fold. A duplicate column must show no gain. The
   gain came from the clipped copy stabilising the baseline's extreme values,
   not from ratio information. Fix: every design column, baseline and
   candidate, is now winsorised at the training-fold 0.5th and 99.5th
   percentiles, identically in both arms.
2. **Holdout consumed by run 1.** Run 1 evaluated the holdout on the flawed
   pipeline. Its output is kept, for transparency, as
   `docs/player_ratio_search_run1_superseded.json`. **Its holdout numbers are
   not evidence.** Run 2 (the corrected pipeline) is **selection-only**: the
   holdout is not evaluated again, because re-reading 2025-26 would be a second
   look. A future confirmatory test needs a season that has not been read
   yet, for example 2026-27 once complete.
3. **Crosswalk fallback (data, not outcomes).** The project's Understat to FPL
   crosswalk left 12.7% of shot rows unmapped. The cause is that FPL history
   stores long legal names ("Richarlison de Andrade") and has no `web_name`.
   Added a team-constrained token match: a name matches only when exactly one
   same-team FPL element contains all its tokens. Unmapped share is now 3.3%.
   Found by diagnosing name mismatches, not by looking at any outcome.
4. **key_passes absent.** FPL history for 2020-21 to 2025-26 has no
   `key_passes` column, so the key-pass candidates were dropped. The count is
   recorded in the JSON under `not_built`.

Run 2 is the result of record for selection. Run 2 keeps the same folds, the
same Holm and sign-stability rules, and the same pool.

## What would change production

Nothing ships on a pass alone. A passing candidate would go behind a feature
flag defaulted OFF, and its sign-off would be a separate, reviewed change.
Any production change still needs review before merge.
