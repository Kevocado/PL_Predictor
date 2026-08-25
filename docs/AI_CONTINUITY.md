# PL Predictor: AI Continuity and Improvement Log

**Last reviewed:** 2026-08-24  
**Purpose:** this is the maintained handoff document for any future human or AI
working on the project. Update it when an experiment, data source, model,
validation rule, or production assumption changes. It records evidence rather
than promises.

## System at a glance

PL Predictor is a personal Premier League pre-match probability dashboard.
The FastAPI package in `src/pl_predictor/` serves a React/Vite frontend in
`frontend/`. It has five distinct jobs:

1. Collect and normalize historical/current match, player, lineup, and odds
   data.
2. Build strictly pre-kickoff team and player features.
3. Serve scoreline, player goal/assist/G+A, corners, and cards probabilities,
   then resolve completed-fixture outcomes without changing a live snapshot.
4. Compare only live 1X2 and 2.5-goal prices to the model; never create a
   parlay recommendation.
5. Snapshot predictions before kickoff and reconcile them to final results in
   a local SQLite ledger.

The project is approximately 10,000 Python lines, a React frontend, and 24
targeted pytest modules. The trained artifact is `models/manifest.json`; it
must not be treated as a permanent source of truth because retraining replaces
it.

## Architecture and ownership

| Area | Main modules | Responsibility |
| --- | --- | --- |
| Data | `data/football_data.py`, `pulselive.py`, `fpl_api.py`, `fpl_history.py`, `understat*.py`, `odds_api.py`, `espn.py` | Cache external data and map team names to one canonical convention. |
| Features | `features/build.py` plus rolling form, ratings, xG, rest, referee, player form | Create training and live feature rows. Historical values must only use prior matches. |
| Models | `models/scoreline.py`, `ml_scoreline.py`, `market_models.py`, `player_goals.py`, `manifest.py` | Train, evaluate, persist, load, and score probabilities. |
| Evaluation | `evaluate/calibration.py`, `walk_forward.py`, `backtest.py`, `betting_validation.py`, `goal_contribution_research.py` | Holdout/walk-forward evidence; experiments must stay here until promoted. |
| Betting | `odds/value_bets.py`, `tracking/value_bet_ledger.py` | Shin de-vig, at-most-one single recommendation, and immutable result reconciliation. |
| Serving | `api/routes.py`, `api/schemas.py`, `api/main.py` | Thin HTTP layer, TTL cache, background warming/retraining. |
| UI | `frontend/src/` | Fixtures, detail modal, data hub, calibration/backtest/validation pages. |

### Data Hub display surfaces

- **Team Hub:** on-demand, display-only current-season team summaries: record,
  recent form, underlying xG, shooting, corners, cards, discipline, and set-piece
  xG share. It is for exploration and does not feed predictions or betting.
- **Player Hub:** on-demand official FPL season-to-date player stats in one
  searchable, sortable table. It includes starts/minutes, goals/assists,
  xG/xA/xGI, Threat, Creativity, ICT, BPS, and bonus. It is descriptive and
  does not alter player prediction probabilities.
- **Fixture detail:** a compact rest and match-style comparison shows rest days
  plus each side's rolling five-match xG, corners, cards, and set-piece xG share.
  It is deliberately in the modal rather than crowded onto fixture cards.

### Fixture review and scorer tracking

- Completed fixture details now distinguish the final score and reviewed
  predictions from the pre-match surface. Core result/goals calls remain the
  immutable track-record snapshots already stored before kickoff.
- Corners/cards and confirmed-XI player probabilities are captured in separate
  SQLite tables. Their final totals and FPL scorer/assist outcomes are reconciled
  after the fixture, without changing scoreline or value-bet logic.
- FPL's `/fixtures/` plus `/event/{gameweek}/live/` feed is the first source
  for completed player goals/assists. The client accepts a score-bearing FPL
  fixture even before FPL flips its `finished` flag, caches successful
  responses, and falls back to saved player histories only when needed.
- Historical player/corner rows may be marked **reconstructed** when no
  pre-kickoff snapshot existed. Keep their scorer calibration and hit-rate
  separate from live snapshots; reconstructed rows are descriptive, not
  prospective evidence.
- Startup and five-minute background jobs reconcile completed outcomes and
  warm completed-fixture player payloads. This is best-effort only: detail
  responses must remain usable if an external FPL request fails.

### Player review rules

- Persist every confirmed starter's goal, assist, and G+A probability. A
  player is eligible for tracking if any of the three reaches `20%`; the top
  three confirmed starters by G+A are marked **Recommended**.
- The post-match review intentionally shows one compact, relevant signal per
  player: Goal at `35%+`, Assist at `25%+`, or G+A at `30%+`; a recommended
  player remains visible even below those review tiers. Hits below all tiers
  are labelled **Overperformer**.
- Goal and assist are separate event probabilities. G+A is the probability of
  at least one goal or assist, so it can be higher than either component; do
  not compare a G+A review line to only the goal or assist percentage.
- Snapshot rows are immutable. Reconstructed rows are created only when a
  snapshot is absent; never overwrite a genuine pre-kickoff row with a later
  recomputation.

## Current model and data state

### Inputs already used

- **Match history:** football-data.co.uk results, shots, shots-on-target,
  corners, fouls, cards, referee, and archived odds. It supplies the deep
  historical training window.
- **Fresh current season:** football-data.org where configured, falling back
  to the undocumented Pulselive backend. Pulselive's result detail is cached
  per completed match.
- **Underlying performance:** Understat team xG and cached shot-situation
  aggregates.
- **Player data:** official FPL bootstrap/current player history and the
  vaastav FPL archive for historical player-fixture rows.
- **Lineups:** ESPN confirmed starters, with FPL availability as an additional
  status gate.
- **Odds:** The Odds API bulk `h2h` and `totals` feed. The current integration
  is deliberately limited to 1X2 and goals O/U 2.5.

### Production evaluation snapshot

The manifest generated 2026-08-23 trained on 2,669 matches and held out 380
matches from 2025-26. The currently selected scoreline model is XGBoost goal
regression (the API model selector uses `scoreline.chosen_model`; confirm that
field instead of relying on a display-only summary).

| Model/market | Latest holdout result |
| --- | ---:|
| ML scoreline | RPS `0.2071`, Brier `0.6149` |
| Dixon-Coles | RPS `0.2256`, Brier `0.6541` |
| Bivariate Poisson | RPS `0.2259`, Brier `0.6544` |
| Corners | MAE `2.69`, RMSE `3.35` |
| Cards | MAE `1.64`, RMSE `2.03` |

The player research records that rolling Threat had the largest tested
goal-signal increment (`R² +0.0208`), Creativity the assist increment
(`+0.0170`), and ICT the strongest G+A candidate (`+0.0182`). Influence
slightly regressed G+A (`-0.0008`) and remains excluded. G+A is served from a
chronologically calibrated direct classifier only because it beat the
Poisson-union baseline in the existing study; goal and assist remain their
specialized rate models.

The base G+A calculation is the Poisson union
`1 - exp(-(goal_lambda + assist_lambda))`, equivalently
`1 - (1 - P(goal)) * (1 - P(assist))`. The direct G+A model is only allowed
to supersede that baseline when its calibrated estimate is higher; it is not
an arithmetic addition of displayed percentages. The tracking store persists
the model's supplied G+A value unchanged and only uses the three independent
probabilities for eligibility/review classification.

### Betting status: do not overstate it

`evaluate/betting_validation.py` retrains the production ML scoreline model
on only earlier seasons and replays the subsequent season against archived
Bet365 prices. The completed five-fold result is **not positive**:

| Metric | Result |
| --- | ---:|
| Selections | `1,328` |
| Wins | `499` (`37.6%`) |
| Flat-stake yield | `-4.6%` |
| Bootstrap 95% yield interval | `-11.6%` to `+2.3%` |

Goals O/U 2.5 yielded `-7.1%`; match result yielded `-1.9%`. One fold was
strongly positive, but the aggregate and interval do not establish an edge.
The live recommendation UI is therefore an explanation of the model-vs-price
comparison, **not evidence of a profitable strategy and not betting advice**.
Use paper tracking and the immutable ledger; do not promote or market a
profitability claim.

## Non-negotiable research protocol

1. **Time first.** A feature must have been available before that fixture's
   kickoff. Never use current-season totals, final lineups, updated injuries,
   closing odds, or revised feeds as though they were known earlier.
2. **One source of feature logic.** Notebooks call package functions only.
   Add data/feature/model code under `src/`, then test it; do not reimplement
   training logic in a notebook.
3. **Chronological comparison.** Start with the fixed final completed-season
   holdout *and* multi-season walk-forward folds. Use the same folds for the
   baseline and candidate.
4. **Probability metrics before accuracy.** Score 1X2 with RPS, Brier,
   log/ignorance score, calibration curves/ECE; score count markets with MAE
   and probabilistic Brier/log loss at the displayed line. Report every fold,
   not only an average.
5. **Promotion gate.** A candidate only becomes live when it improves the
   average chronological and walk-forward results, without a material fold
   regression, and has an availability/no-lookahead test. Record the result
   below before changing `manifest.py` or serving code.
6. **Betting is a separate hypothesis.** Backtest a prediction snapshot and
   odds snapshot from the same pre-kickoff time. Include voids, stale quotes,
   commission/limits assumptions, all qualified signals, and confidence
   intervals. Never select thresholds after looking at the same test period.

## Review findings and priorities

### Strengths to preserve

- The split between thin API routes and package logic is healthy, and most
  significant work already has direct tests.
- The feature builders use shifted rolling histories and the project has
  explicit no-lookahead tests. This is the correct foundation.
- The model comparison keeps interpretable Dixon-Coles/Bivariate-Poisson
  baselines rather than assuming XGBoost always wins.
- Live tracking snapshots predictions and separately records confirmed value
  bets. This is much more credible than re-scoring historical fixtures with a
  current model.
- Player research is already separated from production, and confirmed-XI
  information is used as a display/eligibility gate rather than silently
  inventing realized minutes.

### P0 — measurement and reliability before new features

1. **Create an immutable prediction/quote dataset.** Persist fixture ID,
   UTC prediction timestamp, model/manifest hash, full probability vector,
   raw odds by bookmaker, source timestamp, de-vig method, selected side,
   lineup state, and result source. The existing ledgers are a good start but
   need a complete price snapshot and schema version. This makes a future
   replay reproducible instead of dependent on today’s cache.
2. **Separate odds-time tests.** The historical replay uses archived prices;
   live recommendations use a 12-hour disk cache plus an in-process cache.
   Add `quote_fetched_at`, a maximum fixture-relative age (for example 60
   minutes pre-kickoff), an explicit stale state, and a manual refresh that
   visibly reports its timestamp. A closing-price backtest cannot validate a
   recommendation allegedly made much earlier in the day.
3. **Version data and environments.** Save source URLs, response/file hashes,
   row counts, feature list/hash, package versions, random seeds, and commit
   SHA with every model manifest. Pin Python dependencies with a lockfile.
4. **Regression and contract coverage.** Add API schema tests, fixture-detail
   latency/cache tests, frontend component tests, source-normalization tests,
   SQLite migration tests, and a golden prediction fixture. Continue testing
   no-lookahead when adding every source.

### P1 — highest-value modelling experiments

1. **Calibrate the *final* 1X2/total probabilities per fold.** Test Platt,
   beta, and isotonic calibration trained only on an earlier calibration
   slice. Choose with Brier/log loss and reliability diagrams, not raw
   accuracy. Preserve a no-calibration baseline because small samples can
   overfit.
2. **Blend, do not replace, independent models.** Fit non-negative
   out-of-fold weights for ML scoreline, Dixon-Coles, Bivariate-Poisson, and
   optionally a de-vigged market prior. The blend must be trained on prior
   folds only. Test model-only and market-assisted variants separately; a
   market-assisted model is useful for calibrated forecasts but cannot by
   itself demonstrate betting edge.
3. **Make lineup impact a match-level experiment.** Aggregate pre-kickoff
   projected XI attacking/creative strength, top-player concentration,
   availability, penalty taker, and set-piece taker into team features.
   Score two modes: early projection and confirmed-XI update. Do not use raw
   realized player statistics or post-kickoff minutes.
4. **Improve count distributions before adding corners/cards bets.** Validate
   the existing Negative-Binomial/Poisson O/U conversion by line and fold;
   test zero inflation, team/referee random effects, and calibration before
   presenting fair odds or a recommendation.
5. **Promotion/relegation priors.** Test promoted-team estimates from prior
   Championship form and/or ClubElo at the season boundary. This attacks the
   known cold-start fallback rather than papering over it with league average.

### P2 — engineering and UX improvements

- `features/build.py` emits pandas fragmentation warnings during warm/retrain.
  Construct feature blocks in dictionaries/DataFrames and concatenate once;
  profile before/after because fixture detail currently feels slow.
- The startup warmer launches many expensive synchronous tasks through one
  background call. Instrument per-cache duration, failure, data age, and
  cache-hit rate; do not hide a source failure behind a generic “not found.”
- The generated frontend bundle is about 673 kB minified. Lazy-load the
  calibration charts/modal or route-level pages after measuring user impact.
- CORS is intentionally wide open for a private network. If exposing the app
  publicly, add authentication, a restricted origin list, rate limiting, and
  do not expose `/docs` or retrain controls anonymously.
- Document an operational runbook: expected freshness per source, fallback
  order, how to recover corrupt caches, and which database files are personal
  and must not be committed.

## Free data research (verified 2026-08-24)

| Candidate | What it adds | Recommended use | Caveat |
| --- | --- | --- | --- |
| football-data.co.uk opening/closing/max/average odds | More historical books and time-of-market information; current data already contains much of it | P0: replace a Bet365-only historical price view with pre-registered, same-time snapshots; test closing-line movement as an evaluation diagnostic | Do not use closing odds as a feature for an early prediction, or infer profitability from best-of-book prices that were not available then. [Source](https://www.football-data.co.uk/data) |
| Open-Meteo forecast/archive | Forecast temperature, rain, wind, humidity and weather regime at each stadium | P1: train only on archived forecasts available at prediction time; benchmark by season and only retain if calibration improves | Historical reanalysis is hindsight for a live forecast. The forecast archive is the correct data for no-lookahead testing. Attribution is required. [Source](https://open-meteo.com/) |
| ClubElo | Cross-league team strength, manager/last-match context, and long historical coverage including England’s second tier | P1: use as a pre-season/promoted-team prior or ensemble input, shifted to the fixture date | It is results-based, overlaps the project’s own Elo, and needs a date-correct cached extract. Do not add it merely as a duplicate feature. [Source](https://clubelo.com/Data) |
| Official FPL API | Current availability, player positions, per-player xG/xA/ICT/BPS/bonus and team strengths | P1: snapshot it daily/pre-deadline; use only fields demonstrably available then for lineups and player experiments | The project already uses it. Store snapshots; current API values cannot reconstruct historical availability. |
| vaastav FPL archive | Historical player-fixture rows and many player metrics | Keep as the historical player backbone; continue shifting all columns and exclude uncertain `xP` | Weekly updates stop after 2024-25, except periodic updates; the maintainer explicitly flags `xP` timing risk. [Source](https://github.com/vaastav/Fantasy-Premier-League) |
| StatsBomb Open Data | Rich events, lineups, and selected 360 data for research | P2: use for event-feature prototyping or external validity tests, not the PL production pipeline | It covers selected competitions/seasons, not a continuous current Premier League feed. Confirm `competitions.json` and licence before each use; attribute StatsBomb. [Source](https://github.com/statsbomb/open-data) |
| ESPN lineups | Confirmed starters close to kickoff | Keep; record `lineup_confirmed_at` and run a separate confirmed-XI prediction snapshot | It improves late information, but cannot be substituted into early pre-match historical rows without timestamped history. |

Avoid adding unlicensed scraping sources merely because they are accessible.
In particular, do not depend on an unofficial feed unless its stability,
terms, historical availability, and timestamp semantics are documented in this
file. Pulselive is already a conscious, isolated exception for private use.

## Research basis

- Dixon and Coles provide the enduring Poisson/dynamic-team baseline behind
  the project’s interpretable score models. [Publication record](https://www.research.lancs.ac.uk/portal/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-football-betting-market%28d16276a2-d6e0-483b-a708-1d29663f1992%29.html)
- Probability forecast evaluation should include reliability and
  discrimination diagnostics rather than a single score. [Foulley (2021)](https://arxiv.org/abs/2106.14345)
- Player abilities are a plausible extension to hierarchical team scoring
  models, which supports the projected-XI experiment—not a direct injection
  of post-match player stats. [Whitaker et al.](https://arxiv.org/abs/1710.00001)
- xG is useful as a model feature/representation, but it is not a guarantee of
  betting value; test its incremental effect against a strict baseline.
  [PLOS One study](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0282295)

## Experiment log template

Copy this section for every substantive experiment, then keep the completed
entry in this document or link a dated report/notebook. Never delete a failed
experiment; negative evidence prevents repeated work.

```md
### EXP-YYYY-NN — short name
- **Question:**
- **Status:** proposed | running | rejected | promoted
- **Data/availability:** source, coverage, timestamp rule, licence/attribution
- **Baseline:** model, features, folds, frozen commit/manifest hash
- **Candidate:** exact additional feature/model/calibration and hyperparameters
- **Protocol:** chronological folds, calibration split, metrics, promotion gate
- **Results:** per-fold table, mean, uncertainty, runtime
- **Decision:** why rejected/promoted; affected files; follow-up
```

## First experiments to run

1. **EXP-2026-01: quote-time integrity.** Persist raw odds and model snapshots;
   then compare opening/last-refreshed/closing outcomes without altering the
   live recommendation threshold.
2. **EXP-2026-02: final-probability calibration.** Apply leakage-safe Platt,
   beta, and isotonic candidates to 1X2 and O/U 2.5; retain only a stable
   multi-fold Brier/log-loss improvement.
3. **EXP-2026-03: early vs confirmed-XI.** Build the planned player aggregates
   and measure whether a confirmed-XI update improves RPS/calibration compared
   with the early forecast.
4. **EXP-2026-04: weather archive.** Add stadium coordinates and Open-Meteo
   archived forecast features. Start with wind, rain, temperature, and humidity
   interactions; run ablations per market.
5. **EXP-2026-05: promoted-team prior.** Compare existing cold-start blend
   with date-correct ClubElo/Championship priors; require a gain on promoted
   teams without harm elsewhere.

## Completed experiment log

### EXP-2026-05 — seasonal model and consensus study
- **Status:** completed and rejected for production. It does not alter the
  serving scoreline model, 1X2 probabilities, or value-bet selection, and has
  no UI control or scheduled background job.
- **Arms:** frozen historical core (no retrain), a current-season-weighted ML
  regressor (current-season rows have weight `3.0`), and a season-only ML
  regressor. The latter remains experimental because early-season samples are
  inherently small.
- **Cadences:** no retrain, every 10 completed league matches, every 19
  completed league matches, and each time every club reaches another 19-match
  block. Same-day results are never used to score another fixture that day.
- **Consensus:** equal averaging and a non-negative weighted blend are tested.
  Weighted blend coefficients are fit only on earlier completed seasons; no
  evaluated season contributes to its own consensus weights.
- **Metrics:** paired 1X2 RPS is primary; Brier, log loss, ECE, exact-scoreline
  log loss, paired bootstrap intervals, per-season folds, and fixture counts
  are also reported. Every arm scores the same pre-kickoff fixture rows.
- **Prospective record:** `seasonal_study_predictions` stores only the first
  pre-kickoff snapshot per fixture/cadence/arm, then reconciles it to confirmed
  results. It is intentionally separate from the value-bet ledger.
- **Promotion gate:** show results to the user first. Any later promotion
  requires improved average RPS, no material fold regression, and positive
  prospective evidence across three completed live seasons. Never auto-promote.
- **Completed result:** `reports/seasonal-model-study.json` contains 1,403
  chronological checkpoints across 2021-22 through 2025-26. The only average
  RPS gain was season-weighted retraining every 19 matches (`-0.00008` across
  1,888 fixtures), which is negligible and its mean paired interval crosses
  zero (`-0.01344` to `+0.01281`). It improved in only 3 of 5 seasons. Every
  10-match alternative was worse or tied; season-only was materially worse
  (`+0.00891` to `+0.00952` RPS) in every season. Both consensus candidates
  also regressed. In the final 2025-26 fold, every seasonal/consensus arm was
  worse than the frozen core.
- **Decision:** retain frozen production model weights. Continue refreshing
  time-safe form, availability, lineup, and existing momentum features for
  every fixture; do not retrain on a calendar cadence without a new,
  pre-registered experiment that produces a material and stable gain.
- **Next studies:** test shrinkage/residual blending of long-run and seasonal
  strength, event-driven retrains (transfers, manager change, availability),
  early-season/promotion priors, and online calibration. Keep feature refresh
  separate from weight retraining in every design.

### EXP-2026-06 — purpose-built within-season scoreline model
- **Status:** rejected on its first holdout; offline-only.
- **Model:** an online Gamma-Poisson attack/defence model begins each season
  at league-average home/away scoring rates, uses a 12-match empirical-Bayes
  prior, and updates only after all fixtures on a calendar date finish. It has
  no multi-season fit, so it directly tests the "learn this season only"
  hypothesis rather than merely changing the production model's training
  window.
- **2025-26 result (380 fixtures):** frozen historical ML scored RPS
  `0.20656`, Brier `0.61336`, log loss `1.02125`, and scoreline log loss
  `2.88455`. The within-season challenger regressed to RPS `0.21320`, Brier
  `0.62731`, log loss `1.04242`, and scoreline log loss `2.91929` (RPS delta
  `+0.00664`, Brier delta `+0.01395`).
- **Decision:** do not promote the pure within-season model. This does not by
  itself prove that production should never retrain; the separate full
  multi-cadence study must compare pre-registered retrain schedules across
  multiple historical seasons and then prospective live seasons.

### Ranking correction — current-season shrinkage
- The Data Hub formerly ranked current clubs from raw long-window
  Dixon-Coles parameters while treating historic appearances as current-season
  evidence. A newly promoted or returning club could therefore occupy an
  implausible top slot after a handful of high-weight matches.
- The live Data Hub now uses the frozen pre-season Dixon-Coles prior, rather
  than the refit-with-current-results parameters. This prevents a one-match
  result from changing the ordering; it changes display rankings only, never
  scoreline predictions or betting.

### EXP-2026-07 — opponent-adjusted dominance rankings
- **Status:** rejected for live display. It anchors each club to a pre-season
  Dixon-Coles rating, then adds a capped match signal from goal difference
  versus expected goal difference and shots-on-target dominance.
- **Protocol:** five completed seasons (2021-22 to 2025-26), measured after
  10, 50, 100, and 190 league fixtures. The target is points per game in the
  remaining matches; every candidate sees the same historical prefix.
- **Results (20 held-out comparisons):** pre-season rank averaged Spearman
  `0.6385`; dominance `0.6410` (delta `+0.0025`); adding result-streak
  momentum `0.6377` (delta `-0.0008`). These differences are too small and
  dominance regressed in several folds.
- **Decision:** do not add shots-on-target dominance or an extra momentum
  adjustment to the live rankings. Historical match feeds contain no
  possession field, so possession is not included or approximated.

### EXP-2026-08 — history-seeded Elo/Pi display ranking
- **Status:** promoted for display only; it cannot affect scoreline
  probabilities or value bets.
- **Candidate:** average standardised Elo and Pi ratings after replaying only
  matches available at each checkpoint. Elo accounts for opposition strength;
  Pi also responds to goal margin, so an upset does not outweigh several
  seasons of evidence.
- **Result:** average future-PPG Spearman `0.6748`, a `+0.0363` improvement
  over the frozen pre-season ranking across the same 20 comparisons. It was
  nevertheless worse in several individual folds, including early 2023-24
  and much of 2025-26.
- **Follow-up blend test:** fixed live-form blends with 25%, 50%, and 75%
  Elo/Pi weight were replayed across the same 20 held-out comparisons. The
  75% form / 25% pre-season blend was best (`0.6767` future-PPG Spearman,
  delta `+0.0381`), slightly ahead of pure Elo/Pi (`+0.0363`). The 50% and
  25% blends gained `+0.0253` and `+0.0084` respectively. Some folds still
  regress, so this is not evidence for changing the match model.
- **Decision:** use the fixed 75/25 blend for the Data Hub only. It updates
  after every completed match and needs no manual weekly adjustment. Review it
  only as routine monitoring, not as an input to predictions or betting.

### EXP-2026-09 — existing main-model momentum feature
- **Status:** retained in the production ML scoreline feature set.
- **Feature:** `home_current_streak` and `away_current_streak` are signed
  win/loss streaks, shifted before every fixture so a match cannot see its own
  result. They refresh per fixture; this is not a model-weight retrain.
- **2025-26 chronological holdout:** including the feature improved RPS from
  `0.20725` to `0.20656` and Brier from `0.61478` to `0.61336`.
- **Decision:** keep the existing streak feature. A richer possession or
  match-momentum feature requires a timestamped historical source and the
  same multi-season no-lookahead evaluation before it can be considered.

### EXP-2026-10 — power-ranking feature ablation
- **Status:** completed; no production feature change required.
- **Question:** whether the live 75/25 Elo/Pi-plus-preseason display ranking
  should be added as another scoreline-model feature.
- **Finding:** the model already receives strictly pre-match `elo_home`,
  `elo_away`, `elo_diff`, `pi_home`, `pi_away`, `pi_diff`, plus the signed
  win/loss streak features. The display ranking is a deterministic blend of
  those signals and a display-only pre-season prior, so it does not introduce
  independent information.
- **2025-26 chronological ablation:** retaining Elo/Pi scored RPS `0.20732`
  and Brier `0.61508`; removing all six Elo/Pi inputs regressed to RPS
  `0.20866` and Brier `0.61706` (deltas `+0.00134`, `+0.00198`).
- **Decision:** retain the existing Elo/Pi and streak features. Do not add a
  duplicate power-ranking feature; test only genuinely new, time-safe inputs
  such as timestamped possession or event data in a separate experiment.

### OPS-2026-01 — completed-fixture player reconciliation and review
- **Status:** implemented operational reliability change; no production model
  or value-bet decision changed.
- **Problem:** final scores could be present before FPL marked a fixture as
  finished, leaving player reviews blank even though FPL had already published
  goal/assist explanations. On-demand reconstruction also made fixture detail
  slow and could block other completed fixtures behind the first one.
- **Implementation:** cache `/fixtures/` and `/event/{gameweek}/live/`, accept
  score-bearing fixtures, persist every outcome before rebuilding player
  reviews, then warm the completed-fixture player cache in the background.
  `fixture_player_outcomes` is tested for direct event parsing, cached
  fallback, team side mapping, and the score-before-finished case.
- **Provenance:** FPL outcomes are official current-feed data; any prediction
  reconstructed after kickoff stays labelled reconstructed. The stored review
  preserves goals, assists, misses, and overperformers rather than showing
  only successes.

### OPS-2026-02 — club crests and Data Hub presentation
- **Status:** implemented display change only.
- **Club identity:** `TeamBadge` loads a local `/badges/` canonical crest asset and
  retains a club-colour initials fallback when an asset is missing or fails.
  No remote crest request is made while rendering a fixture.
- **Data Hub:** Team Hub defaults to form ordering and shows an improving,
  steady, declining, or insufficient-data marker. Player Hub is one sortable
  table rather than filter chips or a separate Squad Health section. Tooltips
  explain FPL-derived measures and non-obvious team measures.

### EXP-2026-01 — quote-time integrity
- **Status:** promoted as an audit safeguard, not a model change.
- **Implementation:** odds frames now carry cache-file fetch time; quotes older
  than one hour cannot create a value flag/recommendation. The live ledger
  stores the model training timestamp, source timestamp, odds source, and the
  best-quote snapshot for every recorded recommendation.
- **Next:** add a separate raw per-bookmaker snapshot table and use it for a
  same-time, not closing-price-only, betting replay once enough live history
  exists.

### EXP-2026-02 — final-probability calibration and model blend
- **Status:** rejected; no production model change.
- **Protocol:** five outer chronological folds. Each fold fit the three base
  models on the earliest seasons, reserved the latest earlier season for
  calibration, then evaluated an untouched following season. Candidates were
  raw ML scoreline, multinomial Platt scaling of ML probabilities, and a
  non-negative ML/Dixon-Coles/Bivariate-Poisson blend whose weights were fit
  only on the calibration season.
- **Results:** raw ML won every average metric: RPS `0.19991`, Brier
  `0.58059`, log loss `0.97629`, ECE `0.03917`. The blend regressed to RPS
  `0.20100`, Brier `0.58266`, log loss `0.97929`, ECE `0.03956`; Platt scaling
  regressed further (RPS `0.20183`, Brier `0.58515`, log loss `0.98294`, ECE
  `0.04821`).
- **Decision:** retain uncalibrated ML probabilities. Revisit only with more
  live snapshots, a different calibration-window design, or an independent
  pre-kickoff information source; do not tune more variants on these folds.

### EXP-2026-03 — projected player aggregates
- **Status:** promising but unpromoted; expand to walk-forward folds first.
- **Protocol:** existing strictly pre-kickoff player aggregates (prior start
  rate weighting, rolling goal/assist/Threat/Creativity rates, and top-player
  concentration) were added to the production ML feature set for the fixed
  2025-26 holdout only.
- **Result:** a very small improvement: RPS `0.2069397 → 0.2069334`, Brier
  `0.6144294 → 0.6143059` (265 → 281 features). This is far too small and
  single-season-specific to deploy.
- **Decision:** build the same feature comparison into the multi-season
  walk-forward module and add a distinct confirmed-XI snapshot experiment.

### EXP-2026-04 — shot-situation features for corners/cards
- **Status:** corners candidate; cards rejected; neither promoted.
- **Result:** the fixed holdout's set-piece shot-situation features improved
  corners (MAE `2.6860 → 2.6574`, Brier `0.2501 → 0.2460`, log loss
  `0.6937 → 0.6853`) but slightly worsened ECE (`0.0489 → 0.0520`). Cards
  regressed on MAE, Brier, log loss, average precision, and ECE.
- **Decision:** do not add the feature to cards. Run multi-season corners
  calibration before considering it for that model; no fair-odds/corners
  recommendation is permitted from this result alone.

## Change checklist for future agents

- Read this file, `README.md`, and relevant tests before editing.
- Use `rg` to find every consumption point of a changed schema/feature.
- Add source attribution/licence and cache policy for every external input.
- Add a no-lookahead test and a source-normalization test with every feature.
- Run the smallest relevant pytest set, then the frontend build for UI/API
  changes. Record untested work explicitly.
- Update this file, README, notebooks, and manifest metadata together when a
  model is promoted.
- Do not commit keys, local caches, `data/tracking.db`, or personal results.
