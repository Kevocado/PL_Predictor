# Parlay recommender design

## Purpose

Surface multi-leg parlay recommendations (same-game parlays and cross-match
"best picks") from data this project already produces — scoreline
probabilities, market models, and player-level rate models — and track
whether those recommendations are actually reliable once games resolve.

This is a recommender, not an interactive builder: the system scans a
gameweek's fixtures and proposes parlays; there's no UI for a user to
assemble arbitrary custom legs. `odds/value_bets.py` already made an
explicit, documented decision not to build parlays out of independent
single-market edges, because naively multiplying edges ignores the
correlation between legs on the same match. This design's core job is
doing that correlation accounting properly instead of ignoring it.

## Leg-eligible markets

| Tier | Markets | Pricing basis |
|---|---|---|
| Live-priced | Match result (H/D/A), Over/Under 2.5 goals, BTTS | `live_odds` — real bookmaker price via The Odds API, same source `odds/value_bets.py` already uses |
| Model-only | Corners O/U, cards O/U, player anytime goalscorer, player anytime assist, player goal contribution (G+A), player shots-on-target O/U | `model_only` — Poisson-rate estimate (`models/player_goals.py`, `models/market_models.py`), no live market to compare against |

Every leg in the output carries its `pricing_basis` explicitly. A parlay
containing any `model_only` leg gets `combined_price: null` and is
presented as confidence-ranked rather than EV-ranked (see Ranking below) —
never a synthetic price standing in for a real one.

## Joint probability computation

Two cases, handled differently, each leg tagged with its
`correlation_assumption`:

- **Grid-derivable same-match legs** (`joint_grid`): match result, O/U 2.5,
  and BTTS are all derivable from the fixture's own score grid
  (`FootballProbabilityGrid`, `P(home_goals=i, away_goals=j)`, produced by
  the Dixon-Coles/Bivariate-Poisson models in `models/scoreline.py`). When
  two or more of these are combined in the same SGP, their true joint
  probability is read directly off the grid — summing cells that satisfy
  every leg simultaneously — rather than assumed independent. This is
  exact under the model, no approximation.
- **Everything else** (`independent`): corners/cards (separate Poisson-rate
  models, no shared joint distribution with the scoreline grid) and every
  player-level leg (goals/assists/shots — also separate rate models) are
  priced as independent of any other leg, including team-result legs from
  the same match. This is a known, flagged simplification: a striker's
  anytime-goal leg is *positively* correlated with their own team winning
  or going over 2.5, so independence understates the true joint
  probability — the conservative direction for a recommender, but real,
  and always visible via `correlation_assumption` in the output.
- **Cross-match legs** (`independent_fixtures`): different fixtures have no
  causal link in this project's models, so joint probability is the
  product of each leg's own marginal probability.

## Recommendation algorithm

Runs per gameweek, over that gameweek's fixtures.

1. **Per-fixture leg generation.** For each fixture, compute every eligible
   live-priced leg (reusing `scoreline.predict_fixtures_batch`, joined to
   live odds exactly as `odds/value_bets.py` does today) and every eligible
   model-only leg (`player_goals.predict_player` for each rostered player,
   `market_models.price_over_under` for corners/cards).
2. **Leg filtering.** Drop any leg where `is_fallback_prediction` is true,
   or (for live-priced legs) `odds_is_stale` — the same guardrails
   `value_bets.py` already applies to single recommendations. Also drop
   legs with model probability below 15%, so a parlay can't be assembled
   from several long-shots whose product merely looks selective.
3. **SGP candidate generation.** Within a single fixture, enumerate every
   non-conflicting subset of size 2-5 from that fixture's surviving eligible
   legs (team markets + its players), excluding mutually exclusive pairs
   (e.g. never home_win + away_win, never over_2.5 + under_2.5). The
   eligible-leg count per fixture is small enough (roughly a dozen) that
   exhaustive subset enumeration is cheap — no heuristic search needed.
4. **Cross-match candidate generation.** Take each fixture's single
   strongest-edge live-priced leg (same "best market per fixture" logic
   `value_bets.py` already uses for its single recommendation), then
   combine 2-5 of these across different fixtures into "best picks"
   parlays — at most one leg per fixture, so cross-match parlays can never
   be accidentally correlated through a shared match.
5. **Ranking.**
   - All-live-priced parlays: rank by real EV, `joint_prob * combined_price
     - 1`, using actual bookmaker prices multiplied together (correct
     payout math for independently-settled legs).
   - Any parlay containing a `model_only` leg: no real payout exists, so
     rank by `joint_prob` alone and label the output "no live payout,
     confidence-ranked."
   - Cap combined odds at a parlay-level ceiling (proposed 25.0, vs. the
     single-bet `MAX_RECOMMENDATION_ODDS` of 6.0 — tunable during
     implementation) to exclude implausible tail combinations.
6. **Output.** Surface the top few of each pool per gameweek (proposed: top
   3 SGP + top 3 cross-match, tunable).

## Persistence & tracking (`tracking/parlay_ledger.py`)

Two tables, mirroring `tracking/value_bet_ledger.py`'s snapshot-once /
reconcile-later discipline, but split header/legs since legs resolve
independently before the parlay itself does:

- `parlays`: `id`, `type` (`sgp` | `cross_match`), `combined_prob`,
  `combined_price` (nullable), `pricing_basis` (`all_live` | `mixed`),
  `snapshotted_at`, `resolved`, `won` (AND of every leg), `resolved_at`.
- `parlay_legs`: `parlay_id` (FK), `event_id`, `market`, `outcome_name`,
  `model_prob`, `price` (nullable), `pricing_basis`,
  `correlation_assumption`, `resolved`, `won`.

Reconciliation reuses `tracking/store.py::_actual_outcome` for team-market
legs. Player-market and corners/cards legs need a small addition to that
reconciliation logic (actual goals/assists/shots vs. the leg's line, actual
corners/cards vs. the leg's line) — new code, but the same snapshot-once,
never-recomputed idempotency the rest of the ledger already guarantees.

## Reliability testing (`evaluate/parlay_validation.py`)

Directly answers "how do we know parlay recommendations are trustworthy":

- **Walk-forward backtest**: reuse `evaluate/walk_forward.py`'s fold
  structure (train on strictly earlier seasons, replay a later one) to
  regenerate the same SGP/cross-match candidates the live recommender would
  have produced, using archived closing odds in place of live ones.
- **Calibration check** (the central reliability metric): bucket
  backtested parlays by predicted joint probability (e.g. 10-point
  buckets) and compare against realized hit-rate per bucket. A
  well-calibrated recommender's "50-60% predicted" bucket should win
  roughly 50-60% of the time in the backtest. Systematic miscalibration
  concentrated in buckets containing `independent`-tagged (player or
  corners/cards) legs would directly surface whether that correlation
  simplification is distorting real combined probability.
- **Yield/ROI with bootstrap CI**, same method as
  `evaluate/betting_validation.py`, computed separately for all-live-priced
  parlays (real ROI, real prices) and mixed/model-only parlays
  (probability-only — there's no real price to stake against).

## API output

`GET /api/parlays?season=&week=` returns the current gameweek's top SGP and
top cross-match recommendations. Each parlay includes its legs, combined
probability, combined price (nullable), `pricing_basis`, and each leg's
`correlation_assumption` inline — the same transparency contract the rest
of this project's odds output already follows.

## Out of scope

- Interactive parlay building (user-selected arbitrary legs) — explicitly
  deferred per this design's kickoff conversation; the data isn't rich
  enough yet to make an open-ended builder reliable.
- Same-match joint modeling for corners/cards/player legs beyond
  independence — would need a genuinely joint model (e.g. a copula linking
  the scoreline grid to player/corners rates), not attempted here.
