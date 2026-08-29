# PL Predictor: AI Continuity and Improvement Log

**Last reviewed:** 2026-08-29  
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
| Data | `data/football_data.py`, `pulselive.py`, `fpl_api.py`, `fpl_history.py`, `understat*.py`, `odds_api.py`, `espn.py`, `clubelo.py` (research-only, unevaluated — see EXP-2026-05) | Cache external data and map team names to one canonical convention. |
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

The manifest generated 2026-08-25 trained on 2,670 matches and held out 380
matches from 2025-26. The currently selected scoreline model is XGBoost goal
regression (the API model selector uses `scoreline.chosen_model`; confirm that
field instead of relying on a display-only summary). As of this retrain,
`manifest.py::MARKET_TRAINING_WINDOWS` gives corners its own 12-season
training window (4,190 matches) per EXP-2026-11 — scoreline and cards stay
on the 8-season default; see `manifest.json`'s `market_training_windows`
and `corners.seasons`/`corners.n_train` fields. Per EXP-2026-16,
`manifest.py::MARKET_MODEL_OVERRIDES` serves Over/Under 2.5 goals from the
covariate-Poisson model instead of `chosen_model` — every other market
still comes from `chosen_model`; see `manifest.json`'s `scoreline.
market_overrides`/`market_metrics` fields for the live-current values.

| Model/market | Latest holdout result |
| --- | ---:|
| ML scoreline (`chosen_model`) | RPS `0.2082`, Brier `0.6171` |
| Dixon-Coles | RPS `0.2257`, Brier `0.6542` |
| Bivariate Poisson | RPS `0.2260`, Brier `0.6545` |
| Covariate-Poisson (serves Over/Under 2.5 only) | RPS `0.2103` |
| Corners (12-season window) | MAE `2.67`, RMSE `3.30` |
| Cards | MAE `1.62`, RMSE `2.01` |

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
   SHA with every model manifest. Python dependencies are now pinned via
   `requirements-lock.txt` (a `pip freeze`, used as a constraints file); the
   per-manifest provenance (source hashes, row counts, commit SHA) is still
   open.
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
6. **Tune Dixon-Coles/Bivariate-Poisson themselves — never done.** Confirmed
   (2026-08-25) via `models/scoreline.py`: both are fit with penaltyblog's
   library defaults and a hardcoded `xi=0.0018` time-decay, evaluated as
   fixed baselines every retrain, but never themselves the subject of a
   tuning pass — unlike `ml_scoreline`, which has a dedicated Optuna search
   (`evaluate/tune_hyperparams.py`). Candidates worth testing on the
   existing chronological/walk-forward folds: `xi` itself (a faster or
   slower recency decay), a team-specific rather than single global
   home-field-advantage constant, and whether either model's role
   (currently: comparison baseline + Dixon-Coles alone feeding the Data Hub
   attack/defence display) would benefit from either change. Since
   `ml_scoreline` currently wins the holdout RPS and is what's actually
   served, an improved DC/BP would need to close most of that gap before it
   changes which model gets served — but Dixon-Coles is served today
   regardless for the Power Rankings display, so even a display-only
   improvement isn't nothing.

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
| ClubElo | Cross-league team strength, manager/last-match context, and long historical coverage including England’s second tier | P1: use as a pre-season/promoted-team prior or ensemble input, shifted to the fixture date | It is results-based, overlaps the project’s own Elo, and needs a date-correct cached extract. Do not add it merely as a duplicate feature. **`api.clubelo.com` has not responded from this project's development environment (see EXP-2026-05) — code is written but unevaluated, and reuse terms are unconfirmed.** [Source](https://clubelo.com/Data) |
| Official FPL API | Current availability, player positions, per-player xG/xA/ICT/BPS/bonus and team strengths | P1: snapshot it daily/pre-deadline; use only fields demonstrably available then for lineups and player experiments | The project already uses it. Store snapshots; current API values cannot reconstruct historical availability. |
| vaastav FPL archive | Historical player-fixture rows and many player metrics | Keep as the historical player backbone; continue shifting all columns and exclude uncertain `xP` | Weekly updates stop after 2024-25, except periodic updates; the maintainer explicitly flags `xP` timing risk. [Source](https://github.com/vaastav/Fantasy-Premier-League) |
| StatsBomb Open Data | Rich events, lineups, and selected 360 data for research | P2: use for event-feature prototyping or external validity tests, not the PL production pipeline | Confirmed (2026-08-25): `competitions.json` lists exactly two men's PL seasons — 2015/16 and 2003/04 — plus several FA WSL seasons. Not remotely continuous with the 8-season training window or the current season, so this cannot supply a possession/event feature for production; only useful as an isolated historical ablation (does possession help *when we happen to have it*), which would not generalize to live serving. [Source](https://github.com/statsbomb/open-data) |
| ESPN lineups | Confirmed starters close to kickoff | Keep; record `lineup_confirmed_at` and run a separate confirmed-XI prediction snapshot | It improves late information, but cannot be substituted into early pre-match historical rows without timestamped history. |
| FBref / Sports Reference | Team and match possession %, touches, and other rich stats, with deep historical coverage | **Rejected — do not use.** | Confirmed (2026-08-25) directly from their own published terms: automated access (scripts, bots, scrapers, data miners) is prohibited without express written permission, sessions get rate-limited/blocked ("jailed" up to a day) past ~10 requests/minute, and Sports Reference states outright they cannot offer an API or downloads because they license the underlying data from third parties who prohibit redistribution. This is a licensing dead end, not a technical one — do not scrape it. [Terms](https://static.fbref.com/termsofuse.html), [Bot policy](https://www.sports-reference.com/bot-traffic.html) |
| Sofascore / FotMob live APIs | Live match possession, momentum, and other rich stats | **Rejected — not the same category as pulselive.py, do not build a bypass.** | Investigated further (2026-08-25) by reading real open-source scrapers' actual request code, not just hitting the endpoints. FotMob now requires an `x-fm-req` header — a signed token its own JS generates per-request, added specifically to block API scraping (confirmed via [probberechts/soccerdata#742](https://github.com/probberechts/soccerdata/issues/742): existing tools broke with 401s the day FotMob shipped this, until someone reverse-engineered the signing algorithm). Sofascore sits behind Cloudflare bot management; a real scraper's source ([tunjayoff/sofascore_scraper](https://github.com/tunjayoff/sofascore_scraper)) uses `curl_cffi` with browser TLS-fingerprint impersonation specifically "to bypass Cloudflare," and checks `cf-ray`/`cf-mitigated` headers to detect when that protection triggers. **This is the real distinction from pulselive.com, and it isn't about licensing or attribution:** pulselive has no anti-bot protection at all, just no published terms. Both of these have an active, deliberate technical control specifically built to stop this kind of access — using either means building a bypass for that control, not just using an undocumented endpoint. Do not implement a `data/sofascore.py` or `data/fotmob.py` module using request-signature forgery or TLS-fingerprint spoofing, regardless of citation/attribution intent. |
| football-data.org statistics | Possession/shots on the fixtures the project already fetches from this source | **Confirmed absent (2026-08-25).** | Directly checked a finished 2026-27 match via both the bulk `/matches` and single `/matches/{id}` endpoints on this project's own configured key — neither returns a `statistics` field or anything beyond score/referee/metadata on the free tier. This source cannot supply possession; the project's only current possession field (`pulselive.py`'s `hp`/`ap`, from `possession_percentage`) already exists but is used for display only (`api/routes.py`'s post-match stats panel) — see the possession note below. |
| football-data.org Champions League (`competitions/2001`) | Fixture-congestion dates for PL clubs still in Europe — see EXP-2026-17 | Keep; `data/other_competitions.py::fetch_champions_league_matches` | Confirmed live (2026-08-28) with this project's existing `FOOTBALL_DATA_KEY`: English club names come back as `"Arsenal FC"` etc., which `team_names.to_canonical(..., source="football_data_org")` already handles — no new alias work. Free-tier `season` param only reaches ~2 completed seasons back plus current (older seasons return 403) — best-effort only, does not cover the full 8-season training window. |
| ESPN site API — Europa League/Conference League/FA Cup/EFL Cup (`uefa.europa`, `uefa.europa.conf`, `eng.fa`, `eng.league_cup` scoreboard slugs) | Fixture-congestion dates for the competitions football-data.org's free tier doesn't cover — see EXP-2026-17 | Keep; `data/other_competitions.py::fetch_cup_matches` | Confirmed live (2026-08-28), same host `data/espn.py` already uses for lineups. **Must pass `limit=1000`** — ESPN's default 100-event cap otherwise returns qualifying-round matches from dozens of countries before ever reaching a Premier League club's fixture, which looks like "no PL data" but is actually a pagination artifact. Current season only (no historical-season param found); every fetch degrades to empty on any failure per this project's usual best-effort discipline. |

**Possession, specifically:** no source checked so far can supply it *continuously* across the actual 8-season training window — the real blocker isn't finding a source with possession at all, it's finding one with possession for both the deep historical seasons and the live current season, on a licence that permits use. `pulselive.py` already fetches current-season possession (`hp`/`ap`) but only surfaces it for display; football-data.co.uk (the historical backbone) has no possession column at all. Adding a feature that's real for ~1 of 8 season-equivalents and blended-to-league-average everywhere else would need the same walk-forward evaluation as everything else — the project's own experience with a similarly sparse/partial feature (EXP-2026-04's shot-situation features, and the excluded `situation_cols`/`stakes_cols` noted directly in `features/build.py`) is that this kind of asymmetric-coverage feature tends to add noise rather than signal, via SHAP-invisible tree-structure changes, not real predictive value. Nothing here should be added without testing it first.

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
3. **EXP-2026-03: early vs confirmed-XI.** The early-projection half is done
   and rejected (see Completed experiment log — regressed on every
   walk-forward fold). Still open: measure whether a confirmed-XI update
   (ESPN lineups, not shifted historical rates) improves RPS/calibration
   over the early forecast — no code exists for this yet.
4. **EXP-2026-04: weather archive.** Add stadium coordinates and Open-Meteo
   archived forecast features. Start with wind, rain, temperature, and humidity
   interactions; run ablations per market.
5. **EXP-2026-05: promoted-team prior.** Code complete
   (`data/clubelo.py`, `features/promoted_team_prior.py`), not yet
   evaluated — see the Completed experiment log entry below for what's
   built, what's untested, and why.

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
- **Status:** completed and rejected for production.
- **Protocol (single-holdout, original):** existing strictly pre-kickoff
  player aggregates (prior start rate weighting, rolling goal/assist/Threat/
  Creativity rates, and top-player concentration) were added to the
  production ML feature set for the fixed 2025-26 holdout only.
- **Result (single-holdout):** a very small improvement: RPS
  `0.2069397 → 0.2069334`, Brier `0.6144294 → 0.6143059` (265 → 281
  features). Flagged as far too small and single-season-specific to deploy
  on its own.
- **Follow-up protocol (walk-forward):** `evaluate.walk_forward.
  prepare_folds` gained an optional `extra_feature_frame`/`extra_feature_cols`
  merge so a candidate feature set can be tested across every walk-forward
  fold, not one fixed season (`evaluate.goal_contribution_research.
  evaluate_scoreline_player_aggregates_walk_forward`). Re-ran the same
  aggregates against all 5 default walk-forward folds (2021-22 through
  2025-26, 8 seasons of history).
- **Follow-up result (walk-forward, RPS / Brier):**

  | Val season | Production | +Aggregates |
  | --- | ---: | ---: |
  | 2021-22 | `0.199186` / `0.574089` | `0.199299` / `0.574584` |
  | 2022-23 | `0.200823` / `0.578529` | `0.202084` / `0.580763` |
  | 2023-24 | `0.190876` / `0.549089` | `0.192080` / `0.551615` |
  | 2024-25 | `0.199535` / `0.583653` | `0.199887` / `0.584327` |
  | 2025-26 | `0.206559` / `0.613363` | `0.207047` / `0.614629` |
  | **Mean** | `0.199396` / `0.579745` | `0.200079` / `0.581183` |

  The candidate regressed in **every single fold** on both RPS and Brier,
  not just on average — a materially cleaner rejection than the original
  single-season result suggested.
- **Decision:** do not add projected player aggregates to the production
  scoreline feature set. This closes out the "expand to walk-forward folds
  first" instruction with a clear negative result. A distinct confirmed-XI
  snapshot experiment (early projection vs. confirmed lineup, P1 #3) remains
  a separate, not-yet-built idea — this result does not speak to it, since
  it never used ESPN's confirmed lineups, only shifted historical rates.

### EXP-2026-04 — shot-situation features for corners/cards
- **Status:** corners candidate; cards rejected; neither promoted.
- **Result:** the fixed holdout's set-piece shot-situation features improved
  corners (MAE `2.6860 → 2.6574`, Brier `0.2501 → 0.2460`, log loss
  `0.6937 → 0.6853`) but slightly worsened ECE (`0.0489 → 0.0520`). Cards
  regressed on MAE, Brier, log loss, average precision, and ECE.
- **Decision:** do not add the feature to cards. Run multi-season corners
  calibration before considering it for that model; no fair-odds/corners
  recommendation is permitted from this result alone.

### EXP-2026-05 — promoted-team ClubElo prior (code complete, pending live data)
- **Status:** proposed. Code and unit tests exist; **no live fetch, no
  evaluation, and no promotion decision have been made.** Do not treat this
  entry as evidence either way.
- **Motivation:** confirmed a real gap, not a hypothetical one —
  `features/ratings.py`'s Elo/Pi replay has no cold-start handling at all
  for a team with zero matches in the loaded football-data window; only
  `features/cold_start.py`'s rolling-form blend does. A promoted club
  already has a real, dated rating from the Championship that this project
  currently discards entirely in favor of a flat league-average default.
- **Blocker:** `api.clubelo.com` has not responded from this project's
  development environment across four separate attempts on different days —
  TCP connects, the server never sends a response, even at 15s timeout. The
  root `clubelo.com` domain does resolve and respond, so this looks like an
  issue with that specific API host, not a blanket network restriction.
  Nothing here has been evaluated against a real fetch.
- **What's built:**
  - `data/clubelo.py::fetch_ratings_asof`/`team_rating_asof` — fetch-or-cache
    a date's ratings snapshot, filtered to England's top two tiers, with
    caching modeled on `pulselive.py`'s per-match cache (a past date's
    snapshot is immutable once fetched). Unit-tested
    (`tests/test_clubelo.py`) against a synthetic CSV shaped like ClubElo's
    documented format — **this shape has not been confirmed against a real
    response.**
  - `data/team_names.py::_CLUBELO_ALIASES` — a small, explicitly
    unverified alias table for the few club names known from general
    ClubElo usage to differ from this project's canonical short form;
    everything else relies on the existing case-insensitive fallback.
  - `features/promoted_team_prior.py::zscore_bridge` — the cross-scale
    rating transfer: expresses a team's ClubElo rating as a z-score against
    other current top-flight teams' ClubElo ratings on the same date, then
    maps that z-score onto this project's own Elo distribution for those
    same teams. Deliberately avoids assuming the two rating systems share a
    scale (they're fit independently, different K-factor/home-field
    constants) — only assumes ClubElo ranks teams sensibly relative to each
    other. Fully unit-tested with synthetic ratings
    (`tests/test_promoted_team_prior.py`), including a no-lookahead check
    (every ClubElo lookup is dated strictly before the fixture).
  - `features/promoted_team_prior.py::clubelo_elo_prior` — ties the above
    together for one team/date; returns `None` (never raises) when ClubElo
    has no rating for that team/date, so a caller's existing flat-average
    fallback is always the safe default.
- **Explicitly NOT done:** not imported by `features/build.py` — neither
  `build_training_frame` nor `FixtureFeatureContext` reference this module.
  No walk-forward evaluation has run (would reuse the
  `extra_feature_frame` mechanism `evaluate/walk_forward.py::prepare_folds`
  gained for EXP-2026-03's follow-up, isolating promoted-team fixtures
  specifically per the promotion-gate requirement below). Licence/terms for
  reuse of ClubElo's data have not been confirmed either — see
  `config.py::CLUBELO_BASE_URL`.
- **Next step:** once `api.clubelo.com` is reachable, (1) fetch a real
  snapshot and confirm the CSV shape and `_CLUBELO_ALIASES` table are
  correct, (2) confirm reuse terms, (3) run the walk-forward evaluation
  isolated to promoted-team fixtures across the standard chronological
  folds, and (4) only then decide promote/reject — same bar as every other
  entry in this log.

### OPS-2026-03 — fixed a one-match-stale cross-source rolling-feature bug
- **Status:** fixed in `features/xg_form.py` and `features/shot_situation.py`
  (and built correctly from the start in the new `features/match_dominance.py`).
  Confirmed real via real 2023-24 Arsenal data, not just synthetic — every
  `home/away_xg_for/against_last_{5,10}` and `set_piece_xg_share_last_{5,10}`
  value was exactly one match staler than intended.
- **Root cause:** `attach_xg_features`/`attach_shot_situation_features` join
  Understat's independently-dated data onto `matches_df` via
  `merge_asof(direction="backward", allow_exact_matches=False)` — that alone
  already guarantees the matched source row is strictly before the target
  fixture, which is the correct and sufficient no-lookahead guarantee for a
  cross-source join. `build_rolling_xg`/`build_rolling_shot_situation` then
  *also* applied `shift(1)` before rolling, so the matched source row's own
  stored value excluded its own match a second time — the result skipped
  the source team's most recent actual match entirely, landing one full
  match short of what the target fixture should see. `rolling_form.py`'s
  own features were never affected — they're built directly on the same
  frame the target row belongs to, no cross-source date-matching involved.
- **Consequence discovered by this**: this was a train/serve inconsistency,
  not just a training artifact — live serving (`FixtureFeatureContext`/
  `latest_xg_form`, a direct `.tail(w).mean()`, no merge_asof) was never
  affected, so the production model was *trained* on one-match-stale xG
  features but *served* with fresh ones.
- **Fix:** removed `shift(1)` from `build_rolling_xg`/
  `build_rolling_shot_situation`'s rolling computation — the stored value
  is now inclusive of the row's own match, and the no-lookahead guarantee
  is documented as belonging entirely to the `merge_asof` call. Added
  regression tests (`tests/test_xg_form.py`, `tests/test_shot_situation.py`,
  `tests/test_match_dominance.py`) that pin the corrected behavior against
  hand-computed expected values.
- **Measured impact on the currently-served ml_scoreline model** (5-fold
  walk-forward, `DEFAULT_HYPERPARAMS` unchanged): mean RPS `0.199396 →
  0.199891` (+0.00050), mean Brier `0.579745 → 0.580968` (+0.00122) — worse
  on 4 of 5 folds, better on only 2024-25. Small, but the same order of
  magnitude as improvements this project has previously kept (e.g.
  EXP-2026-09's streak feature, +0.00069 RPS). Plausible explanation: the
  stale version was accidentally acting as a mild smoother on a noisy xG
  signal, and `DEFAULT_HYPERPARAMS` was tuned against the old (buggy)
  feature distribution, not this one.
- **Re-tuning attempt (to compensate) — rejected.** A fresh 40-trial Optuna
  search against the corrected features found a candidate improving the
  walk-forward mean RPS to `0.19967` (vs `0.19989` for current defaults —
  a 0.11% gain). Per this project's own established two-part corroboration
  bar for `DEFAULT_HYPERPARAMS` (walk-forward win alone is insufficient;
  must also hold on the single fixed chronological holdout), checked the
  2025-26 holdout directly: the candidate is *worse* there too (RPS
  `0.20732 → 0.20824`, Brier `0.61510 → 0.61688`) — a walk-forward-specific
  artifact, not a corroborated win. **`DEFAULT_HYPERPARAMS` is unchanged.**
- **Decision:** keep the correctness fix — a known train/serve inconsistency
  and a real double-exclusion logic error shouldn't ship regardless of a
  small, inconsistent-direction metric wobble that a proper retuning
  attempt could not turn into a corroborated win. `models/manifest.json`
  and the trained model files are untouched by this entry; the fix takes
  effect the next time `models.manifest.train_all()` actually runs.

### EXP-2026-11 — 12-season window and match-dominance features (three arms)
- **Status:** completed. Corners' 12-season window is corroborated and
  **promoted** (see EXP-2026-12 — `manifest.py::MARKET_TRAINING_WINDOWS`,
  live in `models/manifest.json` since the 2026-08-25 retrain: corners MAE
  `2.69 -> 2.67` on the real production holdout). Scoreline and cards
  results are rejected or
  inconclusive. No production feature or manifest change.
- **Protocol:** three arms — A) current 8-season production-equivalent
  baseline, B) 12-season historical window (2014-15 through 2025-26,
  same features otherwise), C) 12-season window plus the new
  `features/match_dominance.py` features (non-penalty xG, shots, xG/shot,
  open-play/set-piece xG share, average shot distance) — evaluated on
  identical validation seasons (2021-22 through 2025-26) for all three arms
  via paired `min_train_seasons` offsets in `evaluate/walk_forward.py::
  prepare_folds`. Scored with RPS (primary), Brier, log loss, ECE,
  exact-scoreline log loss, and per-market (1X2/O-U-2.5/BTTS) log-loss/
  Brier, all from the same fitted grids — see
  `evaluate/scoreline_dominance_arms.py`. Corners and cards repeated
  independently with the same three arms, not assumed to inherit the
  scoreline result.
- **Corroboration bar applied** (same one that just rejected the
  OPS-2026-03 retune): an arm's average-across-folds improvement is
  only trusted if it *also* holds on 2025-26 specifically, the single most
  recent season — an average win driven by older folds while the most
  recent one regresses is treated as a walk-forward-specific artifact, not
  a real gain.
- **Scoreline result (mean RPS — A 0.199891, B 0.199415, C 0.199626):** B
  looks like a small win on average, but **fails corroboration** — on
  2025-26 specifically, A (`0.207318`) beats both B (`0.209354`) and C
  (`0.209425`). C is worse than B on 4 of 5 folds outright. **Reject both
  the 12-season window and the dominance features for scoreline.**
- **Corners result (mean MAE — A 2.761954, B 2.742246, C 2.755298):** B
  beats A on 4 of 5 folds and **passes corroboration** — B also wins on
  2025-26 (`2.655616` vs A's `2.698566`). C adds no further benefit over B
  (worse MAE, effectively tied log-loss). **The 12-season window is a
  genuine, corroborated candidate for corners specifically — dominance
  features are not.**
- **Cards result (mean MAE — A 1.665949, B 1.666199, C 1.655043; mean
  log-loss — A 0.695660, B 0.691690, C 0.688743):** C has the best MAE and
  log-loss on average and wins outright on 3 of 5 folds, but is
  **inconclusive on corroboration** — on 2025-26, B has the best log-loss
  (`0.682874`) with C a close second (`0.684729`), while C's MAE win there
  is by a hair (`1.625610` vs B's `1.628007`). Directionally interesting,
  not a clean win either way.
- **Decision:**
  1. Do not change scoreline's training window or feature set.
  2. Corners' 12-season window is promoted — see EXP-2026-12 for the
     per-market training-window mechanism that made this possible and the
     real production retrain result.
     - **Live check added** (`evaluate/current_season_check.py`): trains
       each arm on strictly pre-season data and scores it against the
       *actual* current (2026-27) season's completed fixtures — a
       genuinely live signal, distinct from the historical walk-forward
       folds above, that grows more informative every gameweek.
       `MIN_FIXTURES_FOR_A_DECISION = 60` is an explicit, printed
       rule-of-thumb floor (this data's typical arm-to-arm MAE gaps are
       small enough that a single-digit-to-low-double-digit fixture count
       can't reliably distinguish them) — the tool reports `n_fixtures`
       and whether that floor is met rather than presenting a small-n
       result as decisive.
     - **First run, 10 completed 2026-27 fixtures** (below the floor, not
       decisive on its own): corners MAE — A `2.562`, B `2.446`, C
       `2.444`. Directionally consistent with the historical corroborated
       finding (B beats A) even at this size, which is reassuring but not
       proof. Re-run as the season progresses.
     - Given the historical result already passes the full corroboration
       bar on its own, this follow-up proceeds: build per-market
       training-window selection (see EXP-2026-12).
  3. Match-dominance features are not promoted for any market from this
     result. Cards' signal is the most interesting remaining thread —
     worth a second look with more folds/seasons before either promoting
     or fully closing it out, since one recent-season near-miss isn't
     enough folds to be confident it's noise.
  4. `features/match_dominance.py` and the underlying
     `understat_shots.py::load_match_dominance_data` stay research-only,
     not imported by `features/build.py`.

### EXP-2026-12 — per-market training-window selection; corners promoted
- **Status:** promoted and live. `models/manifest.json` reflects this from
  the 2026-08-25 retrain onward.
- **Motivation:** EXP-2026-11 corroborated a real corners-specific
  improvement from a 12-season training window (won on the 5-fold
  walk-forward average AND the single most recent completed season), but
  `models/manifest.py::train_all` had only ever trained every market from
  one shared season window — there was no way to give corners a different
  window without also changing scoreline/cards, which the same study
  showed did *not* help (scoreline regressed, cards was inconclusive).
- **Implementation:** `manifest.py::MARKET_TRAINING_WINDOWS = {"scoreline":
  8, "corners": 12, "cards": 8}`. `train_all` builds the default (8-season)
  frame once as before; it only builds a second, corners-specific frame
  when `MARKET_TRAINING_WINDOWS["corners"]` actually differs from the
  default (avoiding a redundant rebuild otherwise). An explicit
  `assert corners_feature_cols == feature_cols` guards the one thing
  serving depends on but never previously checked: `odds/value_bets.py::
  predict_market_models_for_fixture` reindexes one shared feature row onto
  `models["feature_cols"]` for *both* corners and cards, which is only
  safe because `build_training_frame`'s `feature_cols` is a fixed list of
  column names independent of how many seasons were loaded — true today,
  now enforced rather than assumed. `manifest.json` gained
  `market_training_windows` (top-level) and `corners.seasons`/
  `corners.n_train` (so corners' actual training window is auditable, not
  inferred from the shared top-level `seasons` field, which still
  describes scoreline/cards' window only).
- **Verified before promoting:** two unit tests
  (`tests/test_manifest_market_windows.py`) confirm an explicit `seasons`
  override still applies uniformly (one shared build, no redundant corners
  rebuild) and that the default path gives corners a genuinely larger,
  further-back window while scoreline/cards are untouched. A full
  end-to-end serving smoke test (`load_models` -> `predict_market_models_
  for_fixture` -> `predict_fixture`) confirmed no runtime errors and
  sensible output before the real retrain.
- **Real production retrain result (2026-08-25):** corners trained on
  4,190 matches (12-season window + current partial) vs. 2,670 for
  scoreline/cards (8-season window + current partial). Corners MAE on the
  real 2025-26 holdout: `2.69 -> 2.67` (RMSE `3.35 -> 3.30`) — consistent
  with EXP-2026-11's corroborated result. Scoreline/cards numbers are
  otherwise consistent with the prior snapshot (small movement from
  OPS-2026-03's xg_form fix and the current season's 10 additional
  matches folding in, not from this change).
- **Next:** if a future study corroborates a differing window for
  scoreline or cards specifically, add it to `MARKET_TRAINING_WINDOWS`
  the same way — the mechanism is already general, not corners-specific.

### EXP-2026-13 — Dixon-Coles/Bivariate-Poisson xi (recency decay) tuning
- **Status:** rejected. `xi` stays at `0.0018` for both models.
- **Motivation:** P1 #6 flagged that neither model has ever had its `xi`
  tuned — both just inherited penaltyblog's suggested default. This is the
  direct DC/BP analogue of `ml_scoreline`'s existing Optuna search, against
  DC/BP's one real tunable parameter.
- **Protocol:** `evaluate/tune_dc_bp_xi.py` grid-searches
  `xi in [0, 0.0005, 0.001, 0.0018, 0.003, 0.005, 0.008, 0.012, 0.02]` for
  both models independently across the same 5-fold chronological
  walk-forward (2021-22 through 2025-26) used everywhere else.
- **Result:** both models show a smooth, unimodal RPS curve peaking at
  `xi=0.003` (mean RPS: Dixon-Coles `0.20967 -> 0.20946`, Bivariate-Poisson
  `0.20973 -> 0.20954`) — the shared optimum between two independently
  fit models is a good sign this isn't noise. **But it fails the
  most-recent-season corroboration bar**: on 2025-26 specifically, `xi=0.003`
  is *worse* for both models (Dixon-Coles `0.211747 -> 0.212631`,
  Bivariate-Poisson `0.211741 -> 0.212627`) — the average gain is almost
  entirely driven by a large improvement on 2024-25 masking a regression on
  2025-26 (and a smaller one on 2021-22).
- **Pattern worth flagging on its own:** this is the *third* tuning attempt
  in this session with the exact same signature — average-across-folds
  improves, 2025-26 specifically regresses (`ml_scoreline`'s post-OPS-2026-03
  hyperparameter retune, the scoreline 12-season window in EXP-2026-11, and
  now this). Three independent tunings converging on the same failure mode
  is itself evidence that 2025-26 is a genuinely different-shaped season
  for this data (not a bug in any one of these attempts), and reinforces
  that the walk-forward average alone is not a safe promotion signal for
  this project without the fixed-recent-season check every time.
- **Decision:** keep `xi=0.0018` for both models. Do not retune again on
  this same 5-fold protocol without either more seasons in the walk-forward
  set or a different, pre-registered validation design — repeating this
  exact search would just rediscover the same non-corroborated optimum.

### EXP-2026-14 — per-market/per-segment model selection (Part 4 closeout)
- **Status:** rejected. No evidence supports selecting a different
  scoreline model (Dixon-Coles, Bivariate-Poisson, `ml_scoreline`) for any
  market or the one pre-registered segment tested. The single global
  `chosen_model` in `models/manifest.py` stays as-is.
- **Protocol:** `evaluate/model_selection_by_segment.py` fits all three
  models per walk-forward fold (5 folds, 2021-22 through 2025-26) and
  scores each on every market (1X2/BTTS/O-U-2.5/exact-scoreline) via the
  shared `scoreline_dominance_arms.py::_metrics_from_grids` (refactored out
  of that module so it works on any model's grids, not just XGBoost's).
  One segment was pre-registered before looking at results: **cold-start-
  involved fixtures** (either side's `confidence_home`/`confidence_away` is
  not `"current"`) vs. established-only, using the already-computed
  cold-start confidence columns rather than inventing a new threshold.
- **Overall (no segment) result:** `ml_scoreline` wins **every single
  market metric outright** — RPS, Brier, log-loss, ECE, exact-scoreline
  log-loss, and both O-U-2.5/BTTS log-loss and Brier all favor
  `ml_scoreline` over both Dixon-Coles and Bivariate-Poisson, with no
  exception. This closes the per-market question cleanly: there is no
  market where switching away from `ml_scoreline` would help.
- **Cold-start segment result — inspected, not trustworthy.** The mean
  numbers look interesting (`ml_scoreline` RPS `0.2116` vs Dixon-Coles/
  Bivariate-Poisson `0.2279`), and per-fold it flips: `ml_scoreline` wins
  3 of 5 folds by a wide margin, but Dixon-Coles/Bivariate-Poisson win the
  *most recent* fold (2025-26: `0.2279` vs `ml_scoreline`'s `0.2680`) —
  the same average-vs-most-recent disagreement that already rejected two
  other candidates this session. But inspecting *which* fixtures make up
  this segment revealed something more important than the metric itself:
  **every fold's "cold-start-involved" set is just one single newly-
  promoted team's first 10 matches of that season** (confirmed directly —
  2025-26's set is entirely Sunderland). This isn't a general cold-start
  population; it's one team's small sample re-run every season, and the
  bootstrap CIs for this segment overlap heavily across all three models
  in every fold (e.g. 2025-26: `ml_scoreline` `[0.146, 0.284]` overlaps
  Dixon-Coles/Bivariate-Poisson `[0.192, 0.256]`). The apparent
  model-vs-model gap here is noise from n=10, not a real effect — treat
  any future "cold-start" segment analysis on this codebase's data the
  same way: check what teams actually populate it before trusting the
  numbers, not just the segment label.
- **Decision:** do not build per-market or per-segment model-selection
  infrastructure into `models/manifest.py`/`models/scoreline.py`. If a
  genuinely different, larger cold-start population becomes available
  (e.g. after several more seasons accumulate more promoted-team fixtures,
  or if EXP-2026-05's ClubElo prior becomes evaluable and changes the
  picture), revisit with a segment that isn't dominated by a single team.

### EXP-2026-15 — covariate-augmented Poisson goal model (Part 3b)
- **Status:** completed with a clean, fully corroborated result. Research
  code only (`evaluate/covariate_poisson_research.py`) — not yet promoted
  to `models/`; see Decision for what promotion would actually require.
- **Motivation:** penaltyblog's `DixonColesGoalModel`/
  `BivariatePoissonGoalModel` cannot accept covariates at all (confirmed by
  direct inspection of their constructors — only goals/teams/weights/
  neutral_venue). Every prior attempt to improve on DC/BP this session
  (`xi` tuning, EXP-2026-13) worked only within that closed API. This
  builds a genuinely new Dixon-Coles-*style* model from scratch — team
  attack/defence strength via one-hot dummies (the classical
  reshape-to-two-rows-per-match GLM parameterization), fit as a real
  Poisson GLM via `sklearn.linear_model.PoissonRegressor` (already a
  project dependency via scikit-learn — no new dependency needed) with the
  same `dixon_coles_weights(xi=0.0018)` recency weighting DC/BP already
  use — so external covariates (Elo/Pi difference now; the match-dominance
  rolling stats are already wired as an option via `dominance_extra_cols`)
  can sit alongside team effects as ordinary coefficients.
- **Protocol:** two specs on the same 5-fold walk-forward
  (2021-22 through 2025-26) `model_selection_by_segment.py` already uses —
  `team_effects_only` (attack/defence dummies + home indicator, a sanity-
  check baseline structurally closest to vanilla Dixon-Coles) and
  `team_effects_plus_elo_pi` (+ signed Elo/Pi difference, this side minus
  opponent). Scored identically to every other scoreline candidate this
  session via the shared `scoreline_dominance_arms.py::_metrics_from_grids`.
- **Result — team_effects_plus_elo_pi beats both Dixon-Coles and
  Bivariate-Poisson on every single fold, no exceptions**:

  | Val season | Dixon-Coles | Bivariate-Poisson | Covariate-Poisson |
  | --- | ---: | ---: | ---: |
  | 2021-22 | `0.198835` | `0.198766` | `0.196133` |
  | 2022-23 | `0.222608` | `0.222906` | `0.210731` |
  | 2023-24 | `0.197737` | `0.197701` | `0.193499` |
  | 2024-25 | `0.217427` | `0.217528` | `0.202321` |
  | 2025-26 | `0.211747` | `0.211741` | `0.210546` |
  | **Mean** | `0.209671` | `0.209729` | `0.202646` |

  `team_effects_only` (no covariates) is materially *worse* than real
  Dixon-Coles on average (`0.231066`) — expected, since it also lacks
  penaltyblog's own low-score correlation (`rho`) adjustment (fixed at
  `rho=0.0` here, same simplification `ml_scoreline` already uses). The
  entire win comes from the two Elo/Pi covariates, not the reimplementation
  itself being a better Dixon-Coles.
- **Still behind `ml_scoreline`** (mean RPS `0.199891` from EXP-2026-14's
  overall segment) by about `0.0028` — closes roughly 70% of the prior
  gap between vanilla DC/BP and XGBoost using two covariates, but does not
  overtake it.
- **Decision:** this is real, corroborated evidence that Dixon-Coles/
  Bivariate-Poisson specifically benefit from covariates — but promoting
  it to `models/covariate_poisson.py` and wiring it into `manifest.py`'s
  candidate comparison is a bigger step than a parameter change (a new
  model class in production, and a decision about what replaces
  `dixon_coles_for_rankings`'s hard-coded reference for the Power Rankings
  display, which currently always uses raw Dixon-Coles regardless of
  `chosen_model`). Not promoted in this pass — flagged as a strong,
  ready-to-build candidate specifically for that display role (structurally
  closest to what it already shows, empirically stronger), pending an
  explicit decision to take that step.
- **Next, if pursued:** (1) fit `rho` properly instead of fixing it at
  `0.0` (penaltyblog's own DC does; a real low-score correlation term might
  close more of the gap), (2) test whether the match-dominance covariates
  add anything on top of Elo/Pi here specifically — a different question
  than EXP-2026-11's rejection, since those tested XGBoost, not this model
  family, (3) if promoted, decide explicitly whether it *replaces*
  Dixon-Coles for the Power Rankings display, becomes a fourth
  `manifest.py` candidate, or both.

### EXP-2026-16 — promote covariate-Poisson to production, wire up per-market model selection
- **Question:** EXP-2026-15 found a corroborated (if partial) Dixon-Coles
  win via Elo/Pi covariates; EXP-2026-14 found no per-market/per-segment
  edge across DC/BP/`ml_scoreline`. Does the covariate-Poisson model beat
  `ml_scoreline` on any *specific* market once it's actually compared to
  it (not just to DC/BP), and if so, is a general "serve a different model
  per market" mechanism worth building?
- **Status:** promoted, with an explicit caveat (see Decision) — this is a
  softer promotion than most entries in this log.
- **Candidate:** (1) `models/covariate_poisson.py` — `evaluate/
  covariate_poisson_research.py`'s `team_effects_plus_elo_pi` spec,
  productionized as a real module (`fit`/`predict`/`predict_from_row`/
  `predict_grids_batch`/`save`/`load`), a 4th candidate in `manifest.py::
  train_all` alongside Dixon-Coles/Bivariate-Poisson/`ml_scoreline`. (2)
  `MARKET_MODEL_OVERRIDES` in `manifest.py` — a fixed dict mapping market
  name to override model name, applied via a new `market_overrides`
  parameter on `scoreline.py::predict_fixture`/`predict_fixtures_batch`
  (recursively re-invoked per overridden market, so XGBoost's batched fast
  path for every other market is untouched). `load_models()` resolves
  override names into loaded, context-attached model objects
  (`scoreline_market_overrides`), threaded through every serving call site
  (`odds/value_bets.py`, `api/routes.py` — gameweek list, fixture detail,
  further-out lookup, player ranking, team fixtures, tracking backfill,
  `/api/backtest` — plus `tracking/store.py` and `evaluate/backtest.py`).
- **Protocol:** same 5-fold chronological walk-forward
  (2021-22 through 2025-26, `model_selection_by_segment.py::prepare_folds`)
  as every other candidate this session, scored via the shared
  `scoreline.py::evaluate_grids_multi_market` on every market (1X2, BTTS,
  O/U 2.5, exact scoreline) for `covariate_poisson` vs `ml_scoreline`
  specifically (EXP-2026-14 already ruled out DC/BP winning anywhere).
- **Results — Over/Under 2.5 goals, covariate-Poisson vs ml_scoreline:**

  | Val season | CP log-loss | ML log-loss | CP Brier | ML Brier | Winner |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 2021-22 | 0.686906 | 0.700873 | 0.246958 | 0.253621 | CP |
  | 2022-23 | 0.679133 | 0.682485 | 0.243295 | 0.244851 | CP |
  | 2023-24 | 0.672151 | 0.666202 | 0.239607 | 0.236743 | **ML** |
  | 2024-25 | 0.684834 | 0.673814 | 0.245651 | 0.240545 | **ML** |
  | 2025-26 | 0.690040 | 0.691073 | 0.248351 | 0.248971 | CP |
  | **Mean** | 0.682613 | 0.682890 | 0.244772 | 0.244946 | CP |

  CP wins the mean and the single most-recent-season check — this
  project's two-part promotion bar — on both log-loss and Brier. But
  **read the margin honestly**: the mean gap is under 0.0003 (log-loss)
  and 0.0002 (Brier), the 2025-26 gap is under 0.0011/0.0007, and CP
  actually *loses* 2 of the 5 folds (2023-24, 2024-25) by a larger margin
  than it wins by in any single fold. This is a much thinner, less
  consistent signal than "wins on average and on the most recent season"
  sounds like in isolation — it clears the letter of the bar, not a
  strong version of it. Every other market (1X2, BTTS, exact scoreline)
  was checked and CP does not win any of them — consistent with
  EXP-2026-15's finding that CP still trails `ml_scoreline` overall.
- **Decision:** kept live as a deliberate, informed call, not a confident
  win — `MARKET_MODEL_OVERRIDES = {"over_2_5": "covariate_poisson"}`,
  confirmed correctly resolved end-to-end after a real production retrain
  (`manifest.json`'s `scoreline.market_overrides` matches; `load_models()`
  → `predict_fixture()`/`predict_fixtures_batch()` both apply it, changing
  only `over_2_5`/`under_2_5` and leaving `home_win`/`draw`/`away_win`
  untouched, with `market_model_overrides: ["over_2_5"]` flagged in the
  result). The general mechanism (arbitrary per-market override, not just
  this one case) is the more durable value of this experiment — it's
  evidence-gated and reversible (set the dict back to `{}` to fully revert
  to single-model serving) if a future retrain or a tighter promotion bar
  (e.g. requiring the win margin to clear a bootstrap CI, not just be
  directionally positive twice) says this shouldn't have qualified.
- **Follow-up:** (1) don't add another market override on evidence this
  equivocal without first tightening the bar with a margin/CI requirement;
  (2) if a future covariate-Poisson refinement (e.g. fitting `rho` instead
  of fixing it at 0, or adding match-dominance covariates per EXP-2026-15's
  own follow-up note) widens the over_2_5 margin and makes it monotonic
  across folds, that's a much stronger candidate for keeping this override
  long-term; (3) Dixon-Coles still serves the Power Rankings display
  unconditionally regardless of `chosen_model`/`market_overrides` — revisit
  only as its own explicit decision, not implicitly via this change.

### EXP-2026-17 — cross-competition fixture congestion (Champions League/Europa League/Conference League/FA Cup/EFL Cup)
- **Status:** rest_days correction promoted; new congestion features rejected
  (computed but unused).
- **Question:** `rest_days_home`/`rest_days_away` (`features/rest_days.py`)
  was computed from `matches_df` alone — Premier League matches only — so a
  team's true rest before a PL fixture was invisible whenever its previous
  match was actually a Champions League, Europa League, Conference League,
  FA Cup, or EFL Cup fixture. Does correcting this, and/or adding new
  congestion features from the same data, improve `ml_scoreline`?
- **Data/availability:** new `data/other_competitions.py`. Champions League
  via football-data.org (competition id 2001, same key as the existing PL
  integration) — confirmed live, but the free tier only grants `season`
  access ~2 completed seasons back plus current (older seasons 403). Europa
  League/Conference League/FA Cup/EFL Cup via ESPN's site API (same host
  `data/espn.py` already uses for lineups, different league slugs) — only
  the current season, `limit=1000` required (ESPN's default 100-event cap
  otherwise returns qualifying rounds from other countries before ever
  reaching a Premier League club's own fixtures). At measurement time,
  Europa League/Conference League/FA Cup had zero fixtures yet this season
  (not yet started); only EFL Cup and Champions League had real data.
- **Baseline:** `ml_scoreline`, `evaluate/walk_forward.py`, 8-season
  chronological folds (2021-22 through 2025-26 validation seasons), default
  hyperparameters.
- **Candidate 1 (rest_days correction):** `features/rest_days.py::build_rest_days`
  now takes `other_fixtures_df` and folds those dates into each team's match
  calendar before computing rest days — wired unconditionally into
  `build_training_frame`/`FixtureFeatureContext` (not gated; this corrects an
  already-shipped feature rather than adding a new one).
- **Candidate 2 (new congestion features, gated):** new
  `features/fixture_congestion.py` — `games_last_14_days_{home,away}` (match
  count across all competitions in the prior 14 days) and
  `european_fixture_last_4_days_{home,away}` (1/0, played a
  Champions/Europa/Conference League match in the prior 4 days).
- **Results (mean across 5 folds):**
  | Variant | RPS | Brier |
  | --- | ---:| ---:|
  | PL-only rest_days (before) | 0.199891 | 0.580968 |
  | Corrected rest_days (after) | 0.199931 | 0.581041 |
  | + congestion columns | 0.200029 | 0.581281 |
  The rest_days correction is flat (differences only appear in the 2 most
  recent folds, where football-data.org/ESPN actually have coverage, and
  even there the deltas are ~0.00002-0.00018 — noise-level). Adding the two
  congestion columns worsens both metrics further (+0.0001 RPS, +0.0002
  Brier vs. the corrected baseline).
- **Decision:** keep the rest_days correction — it's a correctness fix (a
  team's true rest day count), not a speculative new signal, and it doesn't
  regress the holdout. Do **not** add `games_last_14_days_*`/
  `european_fixture_last_4_days_*` to `feature_cols` (see the dated comment
  in `features/build.py`) — same "keep only if it earns it" discipline as
  `stakes_cols`/`situation_cols`. The likely cause is coverage, not a fake
  signal: both columns are a constant zero for the large majority of the
  8-season training window (football-data.org's 2-season depth limit for
  Champions League; Europa League/Conference League/FA Cup not fetchable at
  all pre-season), which is the same asymmetric-coverage failure mode
  EXP-2026-04's shot-situation features already demonstrated.
- **Follow-up:** revisit the congestion columns once `other_competitions.py`
  has a full season of Europa League/Conference League/FA Cup/EFL Cup data
  (i.e., re-run this check partway through a season once those competitions
  are actually in progress) — the current measurement was taken before any
  of them had kicked off, which is close to a worst case for their coverage.

### EXP-2026-18 — squad continuity as a leading indicator of off-season squad change

- **Status: promoted to production (2026-08-29).** Cleared this project's
  hardest bar (fixed most-recent-season corroboration) despite a real
  fold-level regression and a modest effect size (see Results below); the
  user made the human call explicitly rather than this being
  auto-promoted, per this doc's own protocol — option (a) from the
  Decision section below. Wired into `features/build.py` (`home_squad_
  continuity`/`away_squad_continuity` merged onto both `build_training_
  frame` and `FixtureFeatureContext.build_row`, keyed by
  season+team via `squad_change.team_season_continuity_table`; NaN for a
  team-season with no prior-season data, same as h2h/rest-days elsewhere
  — XGBoost handles it natively) and retrained into `models/manifest.
  json`. Also exposed directly (not just via feature-importance) on the
  frontend Calibration page — `GET /api/squad-continuity` +
  `SquadContinuityPanel.tsx` — so the raw per-team numbers are visible,
  not just the model's aggregate use of them.
- **Question:** prompted directly by a user observation (Newcastle 2026-27
  "looks hot" 2 games in, ahead of what the model expects) — the model's
  only channel for team strength is Elo/Pi carried over from last season
  plus this-season match evidence, which barely exists a few games in.
  Does a leading indicator of squad-strength change — available before a
  ball is kicked, from data outside match results — help, specifically in
  fixtures where a team's actual results end up diverging most from the
  model's own expectation?
- **Explicitly not a re-run of already-rejected ideas:** this project has
  twice rejected general "retrain more" / "weight recent form more"
  candidates (EXP-2026-05's seasonal/consensus study; EXP-2026-06's
  within-season-only model) and once rejected recency-decay tuning
  (EXP-2026-13) — all three failed the same way, a walk-forward average
  improvement that reversed on the fixed most-recent season. This
  candidate is a different category: a new pre-match *feature* (squad
  continuity), not a retrain cadence or decay-rate change.
- **Data/availability:** `data/fpl_history.py` (vaastav's archive, already
  used elsewhere in this project) — no new external source. **Real data
  quality finding along the way:** FPL's numeric `element` (player) id is
  **not stable across seasons** — confirmed directly, Bruno Fernandes is
  element `333` in the 2022-23 archive and `373` in 2023-24, same player,
  same club. Any cross-season player join in this codebase must use
  `name`, not `element` — `features/squad_change.py`'s own docstring
  flags this for whoever touches this data next.
- **Candidate:** `features/squad_change.py::squad_continuity(team,
  start_year, history)` — the fraction of a team's total minutes last
  season played by players still registered to that team this season.
  "Registered" is read from whether a player has *any* row (played or
  not) at the new season's gameweek 1 — confirmed directly that vaastav's
  GW1 slice includes dozens of 0-minute rows per club, the full
  registered squad rather than just who started or was named to a bench
  — so this is genuinely known before a ball is kicked, safe to use for
  every fixture of a season including its very first one. Wired into the
  walk-forward comparison via `evaluate.walk_forward.prepare_folds`'s
  existing `extra_feature_frame` mechanism (`evaluate/
  squad_change_prior.py`), not merged into `features/build.py`.
- **Baseline:** `ml_scoreline`, default hyperparameters, the standard
  8-season chronological walk-forward (2021-22 through 2025-26 validation
  seasons).
- **Segment, pre-registered against EXP-2026-14's mistake:** that
  experiment's "cold-start" segment turned out to be one newly-promoted
  team's first 10 matches, re-run every season (n=10, noise). This one is
  every team-season across all 8 seasons whose actual PPG over its first
  8 matches deviated most (top/bottom 15%, either direction) from the
  *baseline model's own* expected PPG for those same matches — 15 of 100
  team-seasons, spanning 8 different seasons and 15 different clubs (not
  one team dominating): Crystal Palace 2024-25, Liverpool 2022-23, Wolves
  2025-26, Leeds 2021-22, Southampton 2024-25, Luton 2023-24, Newcastle
  2021-22, Nott'm Forest 2025-26, Leicester 2022-23, Sheffield United
  2023-24 (all underperformed expectation); West Ham 2023-24, Chelsea
  2021-22, Arsenal 2022-23, Man United 2022-23, Tottenham 2023-24 (all
  overperformed). Composition checked and reported before trusting any
  metric on it, per that lesson.
- **Results (5-fold walk-forward, `n=1,900` fixtures total):**
  | | RPS | Brier |
  | --- | ---:| ---:|
  | Overall — baseline | 0.19993 | 0.58104 |
  | Overall — candidate | 0.19965 | 0.58046 |
  | Segment (n=536) — baseline | 0.19355 | 0.56944 |
  | Segment (n=536) — candidate | 0.19273 | 0.56771 |

  Per-season RPS (the non-negotiable most-recent-season check):
  | Season | Baseline | Candidate | Delta |
  | --- | ---:| ---:| ---:|
  | 2021-22 | 0.200231 | 0.200769 | +0.00054 (worse) |
  | 2022-23 | 0.201436 | 0.201761 | +0.00033 (worse) |
  | 2023-24 | 0.191604 | 0.191002 | −0.00060 (better) |
  | 2024-25 | 0.198883 | 0.198145 | −0.00074 (better) |
  | **2025-26** | **0.207499** | **0.206594** | **−0.00091 (better)** |

  The candidate improves the overall average, improves more on the
  surprise segment specifically than overall (the hypothesis this
  experiment set out to test), and — unlike EXP-2026-05/06/13 — **holds
  on the fixed 2025-26 holdout**. It regresses on the two oldest folds
  (2021-22, 2022-23), both by a similar small magnitude to the gains
  elsewhere. Every delta here is small in absolute terms, the same order
  of magnitude as EXP-2026-09's streak feature (`+0.00069` RPS average) —
  this is a modest signal, not a dramatic one.
- **Decision: promoted as-is (option (a)).** The 2-of-5 fold regression
  (both oldest folds) and small effect size were disclosed; the
  most-recent-season evidence — the specific bar three prior candidates
  failed — was judged the stronger signal, and this project's own README
  already has a public-model track record collecting live evidence going
  forward. (b) (a second held-out check once 2026-27 completes) and (c)
  (investigating whether the two regressed folds share a cause) were not
  done first — worth revisiting if a future retrain shows this feature
  regressing live, since neither was ruled out, only deprioritized against
  an explicit user decision to ship now.
- **Follow-ups explicitly not started** (see the plan this experiment came
  from): manager-change flag, transfer spend/activity, and preseason
  market movement are all still just candidate ideas — no source has been
  researched or licensed for any of them yet.
- **Real production retrain (2026-08-29), confirming the promotion held
  outside the walk-forward folds too:** `models/manifest.json`'s actual
  8-season fixed 2025-26 holdout moved RPS `0.20815 → 0.20836` and Brier
  `0.61696 → 0.61735` — a small, single-fold worsening (this holdout is
  exactly the "chosen_model" evaluation, not the full walk-forward suite),
  consistent with the small effect size and fold-level variance already
  disclosed above; `ml_scoreline` remained decisively `chosen_model` over
  `covariate_poisson` (`0.2084` vs `0.2103`), unchanged from before this
  feature.
- **Live out-of-sample check (2026-08-29,
  `evaluate/squad_continuity_current_season_check.py`), first 10 completed
  2026-27 fixtures:** RPS `0.236343 -> 0.236246` and Brier
  `0.583906 -> 0.583673` — both effectively tied (a difference far smaller
  than this data's fixture-to-fixture noise), while calibration (ECE)
  moved the other way (`0.1538 -> 0.1935`, worse) on this same tiny
  sample. `n=10` is far below `MIN_FIXTURES_FOR_A_DECISION` (60,
  EXP-2026-11's own precedent) — this is a wash, not a decision either
  way, and does not on its own confirm or contradict the promotion. Re-run
  as the season accumulates more completed fixtures.
- **Two real, pre-existing `models/manifest.py` bugs found and fixed while
  wiring this feature in** (unrelated to squad continuity itself — this
  feature's addition to `ml_scoreline`'s training data happened to be the
  first thing that shifted a small-sample model comparison enough to
  reach them): (1) `_append_history` assumed every `chosen_model`'s
  `metrics` dict has a flat `"brier"` key, but `covariate_poisson`'s own
  stored `metrics` is `evaluate_grids_multi_market`'s dict (`"brier_1x2"`,
  same underlying score, different key) — would `KeyError` on any retrain
  where `covariate_poisson` won the overall argmin outright, not just its
  `over_2_5` override role. (2) `load_models()` had no branch at all for
  `chosen == "covariate_poisson"` — it would silently fall into the
  `ml_scoreline` `else` branch and construct the wrong model class. Both
  fixed directly (`_append_history` reads either key; `load_models` gained
  an explicit `covariate_poisson` branch and extended `needs_context`).
  Confirmed via `tests/test_manifest_market_overrides.py`, whose own
  3-season fixture (760 rows) sits inside this feature's documented
  smallest-training-window regression zone and started actually triggering
  `chosen_model == covariate_poisson` once squad continuity was added —
  bumped to a 5-season fixture (reproduces real production's ranking) so
  the tests verify the override *mechanism* rather than being sensitive to
  which model wins on an arbitrarily small slice.

### EXP-2026-19 — free-data FPL surface and prospective market/lineup research

- **Status: infrastructure shipped; no model promotion.** Added a native React
  FPL tab backed by the official FPL API: a player scout, position/price/
  expected-minutes filters, pagination, an exact legal XI optimiser with
  selectable formations, a £100m 15-player optimiser, and public-entry or
  manual transfer planning. `models/fpl.py` uses a transparent pre-deadline
  baseline with independent scoreline context; it is not presented as a
  historically validated player-points hybrid until such a model beats this
  baseline on chronological player-gameweek tests.
- **Operational fixes:** the gameweek view now falls back to FPL's cached
  full fixture feed (including FPL gameweek ids) when football-data.org is
  unavailable, so future fixtures do not disappear. FPL entry planning falls
  back from unavailable future-GW picks to the latest published public squad;
  the future picks endpoint normally returns 404 until that deadline.
  SQLite tracking now uses WAL/busy timeout and nonessential player-model
  warmups are deferred so startup does not block the first UI request.
- **Confirmed-XI track:** early and confirmed-XI baseline snapshots are now
  stored separately in `tracking.db`. Current count: 0 / 0 — no fixture has
  passed the prospective capture window while this code has been running.
  It remains a data-collection experiment; no historical reconstruction or
  late-model promotion has been claimed.
- **Free live odds track:** raw quote snapshots are saved only in narrow
  T-24h and T-1h windows. Current count: 0 / 0. The historical closing-odds
  benchmark covers 2,870 fixtures with 1X2 Brier `0.561454` and ECE
  `0.011836`; it is explicitly non-deployable because closing prices contain
  late market/team-news information unavailable at those decision times.
  The independent scoreline model remains the only model used for value-bet
  decisions.
- **Covariate-Poisson rho experiment:** each walk-forward fold fitted rho
  from its training rows only, then evaluated it on its untouched validation
  season. Results:

  | Validation season | rho | O/U Brier: rho=0 → fitted | RPS: rho=0 → fitted |
  | --- | ---: | ---: | ---: |
  | 2021-22 | -0.025 | 0.246958 → 0.246956 | 0.196133 → 0.196153 |
  | 2022-23 | -0.025 | 0.243295 → 0.243265 | 0.210731 → 0.210679 |
  | 2023-24 | 0.000 | unchanged | unchanged |
  | 2024-25 | 0.000 | unchanged | unchanged |
  | 2025-26 | 0.000 | unchanged | unchanged |

  **Decision: rejected.** It does not improve O/U Brier in every fold and
  worsens RPS in 2021-22. The production covariate-Poisson model therefore
  remains at `rho=0.0`; this experiment did not alter live predictions.
- **Squad continuity UI:** retained as a model feature per EXP-2026-18, but
  removed from the frontend at the user's request. The first live ten-match
  check remains too small for a decision: RPS/Brier were effectively tied
  while ECE worsened.

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
