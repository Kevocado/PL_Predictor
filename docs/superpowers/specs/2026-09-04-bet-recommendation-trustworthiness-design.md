# Bet Recommendation Trustworthiness Design

## Purpose

Make the bet-recommendation surface reflect how much the model's own
prediction should actually be trusted, and stop forcing a user to infer a
market's probability from an unrelated display (the scoreline label) when
the real number is already computed and already sent to the frontend. This
was not hypothetical: a BTTS decision was made by reading a 2-1 scoreline
label instead of the model's actual `btts_yes_prob`, because nothing in the
UI distinguished "the model has a real, decisive view here" from "here is a
number." Two independent changes, one on each side of the stack.

## Scope

**In scope:** confidence-weighted edge thresholds for the existing
market-result/goals-totals value-bet flagging (`odds/value_bets.py`); a
distinct frontend treatment for model-only probabilities (BTTS, corners,
cards) that have no live market to compare against.

**Out of scope for this design:** stake sizing / Kelly criterion (a
separate, larger UX change — turns a flag into a number the user is meant
to act on financially, deserves its own design and explicit go-ahead);
weighting thresholds by each market's own measured calibration error from
`evaluate/calibration.py` (a second, independent signal worth adding later,
deliberately not bundled in here to keep this change reviewable and its
backtest gate legible); the player-level shots/SoT market
(`2026-09-04-player-shots-market-design.md`) — unrelated track.

## 1. Backend — confidence-weighted flagging

`odds/value_bets.py::build_value_bet_table` already computes
`data_confidence` per fixture (from `scoreline._data_confidence`) into the
row dict, but nothing downstream reads it — `value_bet_flags` today applies
one flat `edge_threshold` (5%) to every fixture alike, gated only by the
coarser binary `is_fallback_prediction`.

`data_confidence` is one of `"new"`, `"limited"`, `"established"`, or
`None`. It is `None` whenever the currently-served scoreline model is
Dixon-Coles or Bivariate-Poisson rather than the feature-driven ML model —
`_data_confidence` has no per-fixture feature context to measure from a
Poisson-parameter model, and this can change between retrains (whichever
model wins that retrain's held-out RPS is served). The design must not
assume the ML model is always active.

Required edge scales by tier via a multiplier on the existing
`edge_threshold`, applied only when `data_confidence is not None`:

| `data_confidence` | required edge (multiplier × base) |
|---|---|
| `established` | 1.0× (today's 5%, unchanged) |
| `limited` | 1.5× (7.5%) |
| `new` | 2.5× (12.5%) |
| `None` (stat model served) | 1.0× — today's flat behavior, unchanged |

These starting multipliers are a judgment call, not a measured fact — flag
this explicitly rather than presenting them as validated. **Gate:** before
this reaches the live app, run it through the same walk-forward backtest
(`evaluate/`) already used for every other model change, comparing the
confidence-weighted threshold against today's flat one on a held-out
season: it must reduce the flagged-bet error rate without collapsing
recommendation frequency to near-zero. Ship only if it clears that bar —
same standard the shots/SoT spec holds itself to, same standard
`docs/AI_CONTINUITY.md` already documents project-wide.

Implementation shape: a new `_required_edge(data_confidence, base_threshold)`
helper in `value_bets.py`, called where `edge_threshold` is currently used
directly in the `value_bet_flags` list comprehension and in the
`recommended_market` candidate filter — both read the same per-row
`data_confidence`, so one helper serves both.

## 2. Frontend — a distinct "model call" treatment

`FixtureModal.tsx`'s `MarketBar` already supports a `valueBet` prop that
visually highlights a flagged bet. BTTS (`detail.btts_yes_prob`, line ~365)
renders through the same `MarketBar` component with no such prop — visually
indistinguishable from a plain informational number, next to a bold
scoreline label the eye goes to first. Corners/cards O/U probabilities are
model-only in the same way (no live market — see `value_bets.py`'s own
`corners_market_note`/`cards_market_note`) and inherit the same gap.

This needs a **new, separate** visual signal — not the existing `valueBet`
highlight, which specifically means "there is a live market and the model
beats it." Reusing it for BTTS/corners/cards would claim an edge that does
not exist (no market to have beaten). New `MarketBar` prop, e.g.
`modelCall: boolean`, true when the probability is decisively one-sided:
`prob >= 0.60 || prob <= 0.40` (a plain "meaningfully more likely than a
coinflip" bar — a real threshold that ships, open to tuning after real
usage, not a placeholder). Rendered as a distinctly-styled badge/border
(visually different from the green value-bet highlight — a neutral color
that reads as "the model is confident," not "this beats the market").
Applies to: the BTTS row, and the corners/cards `OverUnderRow`-style rows
wherever they render a model probability without a paired live price.

## Testing

- Backend: unit tests on `_required_edge` covering all four
  `data_confidence` states, and on `build_value_bet_table` asserting a
  fixture that would flag at the flat 5% threshold no longer flags at
  `new`/`limited` tiers when its edge falls between the old and new
  thresholds — mirroring the existing edge-case style in
  `test_betting_validation.py`/`test_value_bet_ledger.py`.
- Frontend: a case for each of `prob` at 0.60, 0.59, 0.40, 0.41 confirming
  `modelCall` flips exactly at the stated boundary — boundary tests, not
  just a happy-path render.

## Rollout

Backend and frontend changes ship independently — the frontend badge has
no dependency on the confidence-weighted threshold landing first, and the
threshold change is invisible to a user until they'd have seen a
newly-suppressed (or newly-required-stronger) flag either way. No new data
source, no new external dependency, no effect on the free-tier memory
story (same fields, no new computation shape).

## Validation gate outcome (recorded after shipping)

The required walk-forward gate (Section 1's "Gate:" paragraph, above) came
back neutral, not a confirmed improvement: win rate moved from 38.17% to
38.01% (a 0.16-point difference, well inside the harness's own ~±7-8pp
bootstrap confidence interval — noise, not signal). The 105 bets the new
threshold suppressed had themselves won at 40.0%, slightly above the
38.17% baseline population — again well within noise at this sample size,
but the opposite direction from the change's premise. The multiplier table
therefore ships on the original judgment-call reasoning from Section 1,
not on validated evidence, and should be revisited once live post-launch
data accumulates rather than treated as backtested-and-confirmed.
