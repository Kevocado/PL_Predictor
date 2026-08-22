# PL Predictor

A Premier League match predictor: match outcomes and scorelines at its core,
plus other betting-relevant markets (both teams to score, over/under goals,
corners, cards).

The statistical core is [`penaltyblog`](https://github.com/martineastwood/penaltyblog)
(Dixon-Coles / Bivariate-Poisson goal models, Elo/Pi ratings, implied-odds
de-vigging, forecast metrics, backtesting). Corners and cards — which
penaltyblog's goal models can't derive — use XGBoost count regressors on top
of the same features. Live odds come from
[The Odds API](https://the-odds-api.com/) (free tier). Player-level anytime
goalscorer/assist predictions run on top of the match model, using the
official FPL API and its public historical archive for real per-player
minutes/goals/assists and live injury/suspension status.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# then add ODDS_API_KEY=... — free key, no card, from https://the-odds-api.com/
```

> **Editable-install note:** if `python -c "import pl_predictor"` fails with
> `ModuleNotFoundError` right after `pip install -e .`, some environments
> silently skip pip's auto-generated `__editable__*.pth` file (security
> tooling that filters that naming pattern was the cause during development
> of this project). Fix: add a normally-named `.pth` file yourself —
> `echo "$(pwd)/src" > .venv/lib/python3.*/site-packages/pl_predictor.pth`.
> If you hit this, also set `export PYTHONPATH=$(pwd)/src` when running
> scripts directly (`python -m pl_predictor...`) rather than through pytest/
> Jupyter, which don't always pick up freshly-added `.pth` files mid-session.

## Train the models

```bash
python -m pl_predictor.models.manifest
```

Fetches the last 8 completed EPL seasons from football-data.co.uk (cached to
`data/cache/` after the first run), builds features, fits the scoreline
model (Dixon-Coles vs. Bivariate-Poisson, picks whichever has the better
held-out RPS) and the corners/cards XGBoost regressors, and writes
`models/manifest.json` + the trained model files.

## Run the dashboard

Two processes: the API (backend) and the web app (frontend). In one terminal:

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src   # see the editable-install note above
uvicorn pl_predictor.api.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000
```

> **`--reload-dir src` matters**: without it, uvicorn's file-watcher scans
> the whole working directory by default, including `.venv/` — hundreds of
> thousands of files. Anything that touches the venv (a `pip install`, even
> incidental mtime changes) triggers a reload storm that makes every request
> hang or time out until it settles. Scoping the watch to `src/` avoids this
> entirely.

In a second terminal (first time only, `npm install`):

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (`http://localhost:5173`). Three pages: **Fixtures** —
a grid of fixture tiles (probabilities, likely scoreline, BTTS/O2.5, a value-bet
badge when the model's edge over the live line exceeds 5%) that expand into a
detail panel on click (scoreline heatmap, full market breakdown, corners/cards,
head-to-head, recent form, and likely goalscorers/assists per squad with live
injury/suspension status) — **Data Hub** — power rankings (the model's own
fitted attack/defence strength per team, plus an Elo rating trend chart), a
projected final table (current standings + each team's expected points summed
over their remaining fixtures), and a live track record (predictions are
snapshotted before kickoff and checked against results as they come in, with
a "biggest misses" table for reviewing what to improve) — and **Calibration &
Backtest** — model vs. bookmaker RPS/Brier, corners/cards model metrics, and a
value-bet backtest with a bankroll chart, plus buttons to retrain models or
refresh fixtures/odds.

The first fixture detail you open per team will take a few seconds while
player data is fetched live (cached to disk for 6 hours afterward).

The backend also serves interactive API docs at `http://localhost:8000/docs`.

### Using it from your phone (Tailscale)

The `--host 0.0.0.0` above and `frontend/vite.config.ts`'s `host: true` make
both servers reachable from other devices, not just this machine — but only
over a network that can actually route to it. [Tailscale](https://tailscale.com/)
is the easiest way to do that from anywhere (not just home wifi), without
exposing anything to the public internet:

1. Install Tailscale on this Mac (`brew install tailscale` or the App Store
   app) and on your phone, and sign into the same account on both.
2. Start Tailscale on the Mac (menu bar app, or `sudo tailscale up`) and find
   its Tailscale name/IP — the Tailscale app shows it, or run `tailscale ip`.
3. Start both servers as above.
4. On your phone, open `http://<that-tailscale-name-or-ip>:5173` in a
   browser.

This machine already has the Tailscale CLI installed but not currently
running (`tailscale status` reports "stopped") — start it (or install the
app if that's easier) before trying step 4.

## Notebooks

Numbered `notebooks/01`–`06` walk through data exploration → feature
engineering → the scoreline model → corners/cards models → backtest/
calibration → live predictions. Each one only calls functions from the
`pl_predictor` package — never redefines logic inline — so they can't drift
out of sync with what's actually trained/served. Open them in VSCode's
Jupyter extension, or `jupyter lab`.

## Project layout

```
src/pl_predictor/
├── data/            historical match/player data, live odds/FPL API, upcoming fixtures
├── features/        rolling team/player form, Elo/Pi ratings, h2h, rest days, cold-start
├── models/          scoreline (Dixon-Coles/Bivariate-Poisson), corners/cards, player goals, manifest
├── evaluate/         calibration (RPS/Brier vs. bookmaker), backtest
├── odds/            de-vig live odds, surface value bets
├── tracking/        SQLite-backed live prediction track record (snapshot → reconcile)
└── api/             FastAPI app — thin JSON layer over everything above
frontend/            React + TypeScript + Vite + Tailwind dashboard
notebooks/           01-06, see above
tests/               feature-leakage checks (pytest)
```

## Notes / current limitations

- **Newly-promoted teams**: the goal model can only score teams it saw at
  fit time. A team with zero history in the loaded seasons window (e.g. a
  club promoted for the first time in years) falls back to a
  league-average-strength prediction — flagged as `is_fallback_prediction`
  / `New-team fallback` in the app and notebooks — rather than crashing.
  This is a known approximation, not a solved problem: it doesn't yet use
  Championship form to estimate a newly-promoted team's actual strength.
- **BTTS/corners/cards have no live market**: The Odds API's bulk `/odds`
  endpoint only serves "core" markets — 1X2 (`h2h`) and totals. BTTS,
  corners, and cards are "additional markets" only available per-event via
  a different endpoint (not implemented here), so those three stay
  model-only predictions with no live edge to compare against.
- **Backtest is a sanity check, not a strategy**: a strongly positive ROI on
  one held-out season is far more likely to mean overfitting/leakage than a
  genuine edge. Expect roughly break-even to slightly negative.
- **Player predictions have a live-status lag**: injured/suspended/doubtful
  players are correctly zeroed/discounted using the official FPL API's live
  `status` and `chance_of_playing_next_round` — but a change from a few hours
  ago may not be reflected instantly, and there's no live minutes-per-match
  feed, so "did they play" is inferred from recent gameweek history rather
  than a real-time lineup. Not calibrated/backtested yet (no held-out Brier
  check has been run on this part), unlike the match-level model.
- **Track record starts from zero, on purpose**: predictions are snapshotted
  the moment a fixture first shows up in the app (via `data/tracking.db`,
  committed rather than gitignored — unlike everything under `data/cache/`,
  it can't be regenerated by re-fetching), then reconciled once results come
  in. There's no retroactive backfill of matches already played before this
  feature existed — the alternative (re-fitting a model as of each past
  date to reconstruct what it "would have" predicted) is a meaningfully
  bigger undertaking and only an approximation of a real record anyway.
