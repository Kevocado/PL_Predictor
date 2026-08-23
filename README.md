# PL Predictor

<p>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="React 19" src="https://img.shields.io/badge/frontend-React%2019-61DAFB?logo=react&logoColor=black">
  <img alt="TypeScript" src="https://img.shields.io/badge/frontend-TypeScript-3178C6?logo=typescript&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/ML-XGBoost%20%2B%20penaltyblog-EB5B25">
  <img alt="Status" src="https://img.shields.io/badge/status-personal%20project-lightgrey">
</p>

A self-hosted Premier League prediction dashboard. Match outcomes and
scorelines are the core, on top of which sit corners/cards, live value-bet
detection, and player-level goalscorer/assist predictions — all served
through a FastAPI backend and a React dashboard, with every prediction
tracked against what actually happens so the model can be judged honestly
rather than taken on faith.

**Table of contents:** [Screenshots](#screenshots) · [What it does](#what-it-does) ·
[How it's built](#how-its-built) · [Setup](#setup) · [Train the models](#train-the-models) ·
[Run the dashboard](#run-the-dashboard) · [Using it from your phone](#using-it-from-your-phone-tailscale) ·
[Notebooks](#notebooks) · [Project layout](#project-layout) · [Notes / limitations](#notes--current-limitations)

## Screenshots

<table>
<tr>
<td width="50%">

**Fixtures — one gameweek at a time**
<br>Completed and upcoming matches together, prev/next between gameweeks,
value-bet fixtures flagged automatically.
<br><img src="docs/screenshots/fixtures-gameweek.jpg" alt="Fixtures page showing a full gameweek of matches, finished and upcoming, with a value-bet badge on two fixtures">

</td>
<td width="50%">

**Fixture detail — click through on any match**
<br>Scoreline probability heatmap, market-vs-model edges, corners/cards,
head-to-head, and likely scorers, for finished fixtures too.
<br><img src="docs/screenshots/fixture-detail.jpg" alt="Fixture detail modal showing a scoreline heatmap and match result probabilities with live market comparison">

</td>
</tr>
<tr>
<td width="50%">

**Data Hub — power rankings**
<br>The model's own fitted attack/defence strength per team, not a
league-table proxy.
<br><img src="docs/screenshots/power-rankings.jpg" alt="Power rankings bar chart showing attack and defence strength per team">

</td>
<td width="50%">

**Calibration & backtest**
<br>Model vs. bookmaker RPS/Brier on a genuinely held-out season, plus how
much of the current season has folded into training so far.
<br><img src="docs/screenshots/calibration.jpg" alt="Model calibration dashboard comparing model, bookmaker, and naive baseline RPS and Brier scores">

</td>
</tr>
</table>

## What it does

- **Match outcomes & scorelines** — Dixon-Coles and Bivariate-Poisson goal
  models (via [`penaltyblog`](https://github.com/martineastwood/penaltyblog)),
  plus an Optuna-tuned XGBoost regressor; whichever has the better held-out
  RPS is served. Full scoreline probability grid, not just 1X2.
- **Corners & cards** — XGBoost count regressors on the same feature set,
  since the goal models can't derive these markets on their own.
- **Live value bets** — [The Odds API](https://the-odds-api.com/) odds are
  de-vigged (Shin's method) and compared against the model's own
  probabilities; a fixture is flagged when the model's edge over the
  market clears a threshold.
- **Player-level predictions** — anytime goalscorer/assist probabilities
  per squad, blended from FPL's rolling per-90 rates and the FPL stat
  measures a walk-forward reliability study actually found predictive
  (Threat, Creativity, ICT Index) — not just whichever stat sounded
  plausible. Live injury/suspension status from the official FPL API.
- **Weekly-fresh current season** — the historical training set comes from
  football-data.co.uk (deep, reliable, but slow to publish new-season
  data), supplemented with the Premier League's own backend
  (`pulselive.com`) for fixtures/stats from the *current* season, so a
  result from this weekend can be training data within hours, not whenever
  a third-party CSV catches up.
- **Honest tracking** — every prediction is snapshotted *before* kickoff
  into a local SQLite store and reconciled against results as they land.
  The Track Record and Backtest tabs report what the model actually said
  in advance, not a re-run of its current (possibly retrained) state
  against old data.

## How it's built

| Layer | Stack |
|---|---|
| Match/scoreline models | [`penaltyblog`](https://github.com/martineastwood/penaltyblog) (Dixon-Coles, Bivariate-Poisson, Elo/Pi ratings, de-vigging) + XGBoost, tuned with [Optuna](https://optuna.org/) |
| Corners/cards/player stats | XGBoost regressors, scikit-learn |
| Backend | FastAPI, served with `uvicorn --reload` |
| Frontend | React 19 + TypeScript + Vite + Tailwind |
| Data sources | football-data.co.uk (historical), pulselive.com (fast current-season supplement), FPL API + [vaastav's archive](https://github.com/vaastav/Fantasy-Premier-League) (player data), The Odds API (live odds), football-data.org (fixtures/standings) |
| Prediction tracking | SQLite (snapshot → reconcile), local to this install |

<details>
<summary><strong>Why penaltyblog + XGBoost, not just one or the other?</strong></summary>

<br>Goals follow a Poisson-ish process well enough that a dedicated
statistical model (Dixon-Coles / Bivariate-Poisson) captures most of the
signal with far less data than a tree ensemble needs, and it comes with a
century of established match-modeling theory behind it. But corners and
cards don't fit that same generative story, and a tuned XGBoost regressor
picks up feature interactions (form, rest days, head-to-head, table
context) that a two-parameter attack/defence model can't represent at all.
Both scoreline approaches are fit and evaluated every retrain — whichever
wins on held-out RPS is what actually gets served.

</details>

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# then add:
#   ODDS_API_KEY=...      — free key, no card, from https://the-odds-api.com/
#   FOOTBALL_DATA_KEY=... — free key, no card, from https://www.football-data.org/client/register
```

`ODDS_API_KEY` powers live odds/value-bet detection; without it, predictions
still work but with no market to compare against. `FOOTBALL_DATA_KEY`
powers the current-season fixture list and gameweek grouping the Fixtures
page is built around; without it, the app falls back to the FPL API for a
plain remaining-fixtures list, but the gameweek-organized view needs it.

<details>
<summary><strong>Hit a <code>ModuleNotFoundError</code> right after <code>pip install -e .</code>?</strong></summary>

<br>Some environments silently skip pip's auto-generated
`__editable__*.pth` file (security tooling that filters that naming
pattern was the cause during development of this project). Fix: add a
normally-named `.pth` file yourself:

```bash
echo "$(pwd)/src" > .venv/lib/python3.*/site-packages/pl_predictor.pth
```

If you hit this, also set `export PYTHONPATH=$(pwd)/src` when running
scripts directly (`python -m pl_predictor...`) rather than through
pytest/Jupyter, which don't always pick up freshly-added `.pth` files
mid-session.

</details>

## Train the models

```bash
python -m pl_predictor.models.manifest
```

Fetches the last 8 completed EPL seasons from football-data.co.uk (cached
to `data/cache/` after the first run), builds features, fits the
scoreline model (Dixon-Coles vs. Bivariate-Poisson vs. XGBoost, picks
whichever has the better held-out RPS) and the corners/cards XGBoost
regressors, and writes `models/manifest.json` + the trained model files.

Want to re-tune the XGBoost scoreline model's hyperparameters instead of
using the defaults already checked in? `python -m
pl_predictor.evaluate.tune_hyperparams` runs an Optuna search against
5-fold walk-forward validation (not a single held-out season, to avoid
overfitting the hyperparameters the same way a feature can overfit) and
resumes from where a previous run left off.

## Run the dashboard

Two processes: the API (backend) and the web app (frontend). In one
terminal:

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src   # see the editable-install note above
uvicorn pl_predictor.api.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000
```

<details>
<summary><strong>Why <code>--reload-dir src</code> specifically?</strong></summary>

<br>Without it, uvicorn's file-watcher scans the whole working directory
by default, including `.venv/` — hundreds of thousands of files. Anything
that touches the venv (a `pip install`, even incidental mtime changes)
triggers a reload storm that makes every request hang or time out until
it settles. Scoping the watch to `src/` avoids this entirely.

</details>

In a second terminal (first time only, `npm install`):

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (`http://localhost:5173`). Three pages:

- **Fixtures** — one gameweek at a time (completed and upcoming matches
  together), with prev/next controls to browse other gameweeks. Every
  fixture is clickable — including already-finished ones, which show the
  honest prediction that was actually recorded before kickoff, not a
  live recompute that could leak the result back in — and opens the full
  detail view: scoreline heatmap, full market breakdown with live-odds
  edges, corners/cards, head-to-head, recent form, and likely
  goalscorers/assists per squad with live injury/suspension status.
- **Data Hub** — power rankings (the model's own fitted attack/defence
  strength per team, plus an Elo rating trend chart), a projected final
  table (current standings + each team's expected points summed over
  their remaining fixtures), and a live track record (predictions
  snapshotted before kickoff and checked against results as they come
  in, with a "biggest misses" table for reviewing what to improve).
- **Calibration & Backtest** — model vs. bookmaker RPS/Brier, corners/cards
  model metrics, how much of the current season has folded into training,
  and a value-bet backtest with a bankroll chart, plus buttons to retrain
  models or refresh fixtures/odds.

The first fixture detail you open per team will take a few seconds while
player data is fetched live (cached to disk for 6 hours afterward).

The backend also serves interactive API docs at `http://localhost:8000/docs`.

### Using it from your phone (Tailscale)

<details>
<summary><strong>Expand for phone/Tailscale setup</strong></summary>

<br>The `--host 0.0.0.0` above and `frontend/vite.config.ts`'s `host: true`
make both servers reachable from other devices, not just this machine —
but only over a network that can actually route to it.
[Tailscale](https://tailscale.com/) is the easiest way to do that from
anywhere (not just home wifi), without exposing anything to the public
internet:

1. Install Tailscale on this machine (`brew install tailscale` or the App
   Store app on macOS) and on your phone, and sign into the same account
   on both.
2. Start Tailscale on this machine (menu bar app, or `sudo tailscale up`)
   and find its Tailscale name/IP — the Tailscale app shows it, or run
   `tailscale ip`.
3. Start both servers as above.
4. On your phone, open `http://<that-tailscale-name-or-ip>:5173` in a
   browser.

</details>

## Notebooks

Numbered `notebooks/01`–`06` walk through data exploration → feature
engineering → the scoreline model → corners/cards models → backtest/
calibration → live predictions. Each one only calls functions from the
`pl_predictor` package — never redefines logic inline — so they can't
drift out of sync with what's actually trained/served. Open them in
VSCode's Jupyter extension, or `jupyter lab`.

## Project layout

```
src/pl_predictor/
├── data/            historical match/player data, live odds/FPL/pulselive/football-data.org APIs
├── features/        rolling team/player form, Elo/Pi ratings, h2h, rest days, cold-start, table context
├── models/          scoreline (Dixon-Coles/Bivariate-Poisson/XGBoost), corners/cards, player goals, manifest
├── evaluate/        calibration (RPS/Brier vs. bookmaker), backtest, walk-forward validation, Optuna tuning,
│                    player-stat reliability study
├── odds/            de-vig live odds, surface value bets
├── tracking/        SQLite-backed live prediction track record (snapshot → reconcile)
└── api/             FastAPI app — thin JSON layer over everything above
frontend/            React + TypeScript + Vite + Tailwind dashboard
notebooks/           01-06, see above
tests/               feature-leakage and no-lookahead checks (pytest)
```

## Notes / current limitations

<details>
<summary><strong>Newly-promoted teams</strong></summary>

<br>The goal model can only score teams it saw at fit time. A team with
zero history in the loaded seasons window (e.g. a club promoted for the
first time in years) falls back to a league-average-strength prediction —
flagged as `is_fallback_prediction` / `New-team fallback` in the app and
notebooks — rather than crashing. This is a known approximation, not a
solved problem: it doesn't yet use Championship form to estimate a
newly-promoted team's actual strength.

</details>

<details>
<summary><strong>BTTS/corners/cards have no live market</strong></summary>

<br>The Odds API's bulk `/odds` endpoint only serves "core" markets — 1X2
(`h2h`) and totals. BTTS, corners, and cards are "additional markets" only
available per-event via a different endpoint (not implemented here), so
those three stay model-only predictions with no live edge to compare
against.

</details>

<details>
<summary><strong>Backtest is a sanity check, not a strategy</strong></summary>

<br>A strongly positive ROI on one held-out season is far more likely to
mean overfitting/leakage than a genuine edge. Expect roughly break-even to
slightly negative — that's the healthy outcome, not a bug.

</details>

<details>
<summary><strong>Player predictions have a live-status lag</strong></summary>

<br>Injured/suspended/doubtful players are correctly zeroed/discounted
using the official FPL API's live `status` and
`chance_of_playing_next_round` — but a change from a few hours ago may not
be reflected instantly, and there's no live minutes-per-match feed, so
"did they play" is inferred from recent gameweek history rather than a
real-time lineup. A walk-forward reliability study (`evaluate/
player_stat_reliability.py`) validated which FPL stats predict *future*
output before wiring them into predictions, but the player predictions
themselves aren't calibrated/backtested against a held-out season yet, the
way the match-level model is.

</details>

<details>
<summary><strong>Track record starts from zero, on purpose</strong></summary>

<br>Predictions are snapshotted the moment a fixture first shows up in the
app (into `data/tracking.db`, gitignored — it's this install's own live
history, not something to ship in the repo), then reconciled once results
come in. There's no retroactive backfill of matches already played before
this feature existed on a *fresh* install — the alternative (re-fitting a
model as of each past date to reconstruct what it "would have" predicted)
is a meaningfully bigger undertaking and only an approximation of a real
record anyway.

</details>

<details>
<summary><strong><code>pulselive.com</code> is an undocumented endpoint</strong></summary>

<br>It's the Premier League's own site backend, reachable with no API key
and no published terms for programmatic use — used here only as a private,
non-redistributed supplement to close the freshness gap in
football-data.co.uk's publishing schedule (fixtures and match stats for
the *current* season only; all historical training data still comes from
football-data.co.uk). If that trade-off doesn't sit right with you, it's
isolated to `data/pulselive.py` / `data/football_data.py`'s fallback path
and can be disabled without touching anything else.

</details>
