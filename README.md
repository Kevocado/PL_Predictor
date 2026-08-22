# PL Predictor

A Premier League match predictor: match outcomes and scorelines at its core,
plus other betting-relevant markets (both teams to score, over/under goals,
corners, cards).

The statistical core is [`penaltyblog`](https://github.com/martineastwood/penaltyblog)
(Dixon-Coles / Bivariate-Poisson goal models, Elo/Pi ratings, implied-odds
de-vigging, forecast metrics, backtesting). Corners and cards — which
penaltyblog's goal models can't derive — use XGBoost count regressors on top
of the same features. Live odds come from
[The Odds API](https://the-odds-api.com/) (free tier).

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

```bash
streamlit run app.py
```

Four tabs: **Upcoming Fixtures** (predictions across every market), **Scoreline
Grid** (correct-score heatmap for a picked fixture), **Calibration & Backtest**
(model vs. bookmaker RPS/Brier, value-bet backtest), **Value Bets** (live-odds
edges — needs `ODDS_API_KEY`).

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
├── data/            historical match data, live odds, upcoming fixtures
├── features/        rolling form, Elo/Pi ratings, h2h, rest days, cold-start
├── models/          scoreline (Dixon-Coles/Bivariate-Poisson), corners/cards, manifest
├── evaluate/         calibration (RPS/Brier vs. bookmaker), backtest
└── odds/            de-vig live odds, surface value bets
notebooks/           01-06, see above
app.py               Streamlit dashboard
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
- **Corners/cards have no live market**: The Odds API's free markets cover
  1X2 (`h2h`), totals, and BTTS — not corners or cards — so those two stay
  model-only predictions with no live edge to compare against.
- **Backtest is a sanity check, not a strategy**: a strongly positive ROI on
  one held-out season is far more likely to mean overfitting/leakage than a
  genuine edge. Expect roughly break-even to slightly negative.
