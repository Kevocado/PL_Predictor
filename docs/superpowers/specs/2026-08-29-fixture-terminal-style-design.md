# Fixture data-display redesign: terminal/ticker style for probability & value-bet panels

**Date:** 2026-08-29
**Status:** design approved by user pending spec review
**Scope class:** architectural (touches shared components used across 3 pages)

## Context

The user shared three reference screenshots of a dark, monospace, terminal/
trading-ticker-style sports data display: thin horizontal probability bars,
tight single-line rows, team crests inline with names, and a compact
market-vs-model delta notation. They want this look applied to specific
data-dense panels of PL Predictor — not a full app re-skin — while keeping
the existing purple/pink Premier League theme as the shell.

This is one of three sub-projects split out of a broader "make the FPL
visuals more FIFA-like and improve the whole app's information design"
request (see the conversation this spec came from). The other two —
FIFA-style FPL team-view cards, and a fixture-detail load-performance pass —
are separate efforts; the performance pass is already done (see
`docs/AI_CONTINUITY.md`'s 2026-08-29 entry on `reconcile_fixture_market_predictions`
caching).

## Decisions already made (do not re-litigate these)

1. **Scoped to specific panels, not an app-wide re-skin.** The PL purple/
   pink shell (nav, crests, page chrome, branding) stays as-is.
2. **Panels in scope:** `FixtureModal`'s match-result/goals market rows, its
   "next most likely scorelines" list, and the compact fixture cards on the
   Fixtures tab.
3. **Color palette:** recolor to the existing `pl-pink`/`pl-cyan` palette,
   not the reference images' black-and-green — same monospace/dense-row
   *format*, not the same literal colors.
4. **Delta notation: percentages, not the reference's "¢" shorthand.** The
   user explicitly asked to keep `%` because it's easier to interpret than
   a cents-style abstraction. Round to a whole percentage point for the
   compact delta specifically (matching the reference's visual density);
   this does not change the more precise (`.toFixed(2)`) edge displays
   used elsewhere in the app outside this redesign's scope.
5. **Integrate the value-bet recommendation and flagged-market styling into
   the same visual language**, not left as a visually disconnected callout
   box as it is today.

## Open scope question — flagged for the user, not blocking

"Integrate into the bet recommendation and value bet stuff" is interpreted
here as: **within `FixtureModal`** — the existing "Best value bet" callout
box (`FixtureModal.tsx` lines ~308-320) and each `MarketRow`'s `flagged`/
`postMatchHit` highlight state should visually read as the same design
system as the redesigned probability bars, rather than a separately-styled
box bolted on below them.

This does **not** extend to the three components on the Calibration page
(`LiveValueBetPanel`, `BacktestPanel`, `WalkForwardBettingPanel`) — those
were explicitly excluded when the user picked "Data panels only" /
the 4-option panel list earlier in this same conversation. If the user
actually meant those too, that's a scope amendment to make explicitly
before implementation, not something to infer silently.

## Current state (for reference during implementation)

- `MarketRow` (`FixtureModal.tsx:32-54`): flat text row — label left,
  `prob%` / `mkt%` / signed `edge%` (2 decimals) right. No bar visual. Used
  for home/draw/away, O/U 2.5, BTTS, and team-2+ markets (`FixtureModal.tsx:379-421`).
  `flagged` (a value-bet flag) and `postMatchHit` (win/loss/pending after
  the fact) each apply a different background/ring tint.
- "Next most likely scorelines" (`ScorelineHeatmap.tsx:91-110`): flat text
  row — scoreline left, `prob%` right, top row gets a pink ring highlight.
  No bar visual.
- "Best value bet" callout (`FixtureModal.tsx:308-320`): a separately
  bordered/tinted box (`border-pl-cyan/40 bg-pl-cyan/10`), market label +
  type badge on one line, `+X.X% edge` on the right, best-price detail
  below. Visually disconnected from the `MarketRow` list right above it.
- Fixture list cards: compact home/draw/away split, current implementation
  not yet inspected in detail — to be read during implementation, not
  assumed here.

## Design

### New shared component: `ProbabilityBar`

A single reusable row: `label` (left) — thin horizontal bar (bar width ∝
probability, `pl-pink` fill on a `pl-border`/`pl-850` track) — `value%`
(right, monospace numerals). Optional props:
- `marketPct?: number` — when present, renders a second, muted tick/marker
  on the same bar at the market-implied probability's position (visually
  showing *both* model and market on one bar, closer to the reference
  images than two separate numbers), plus a compact signed `±N%` delta
  label (whole-number rounded) next to the bar.
- `highlight?: "flagged" | "hit" | "miss"` — replaces `MarketRow`'s current
  three-way background/ring tinting, applied consistently to both this
  component and the value-bet callout (see below) so the same visual
  vocabulary (pink ring = flagged value bet, green = hit, red = miss) means
  the same thing everywhere in the modal.
- Monospace font (`ui-monospace, SFMono-Regular, Menlo, monospace`) applied
  to the numeric portions only (label text stays the app's normal sans
  font) — this is what gives the reference images their "data terminal"
  feel without changing the modal's overall typographic voice.

`MarketRow` is replaced by `ProbabilityBar` at every call site; `edge.prob`
→ the bar's value, `edge.implied` → `marketPct`, `edge.edge` → the derived
`±N%` delta (computed from `prob`/`implied` directly so there's exactly one
source of truth, not a separately-rounded duplicate).

### "Next most likely scorelines"

Same `ProbabilityBar`, one row per scoreline entry, `label` = the
`{home} {h}–{a} {away}` scoreline text (kept monospace, as it already is
today), no `marketPct` (there's no per-scoreline market price to compare
against). Top-row pink-ring highlight behavior is preserved via
`highlight="flagged"` on just that first row for visual continuity, even
though it isn't a "value bet" flag in that context — same visual grammar,
different meaning by position, which is acceptable since the surrounding
label already makes the meaning clear ("most likely," not "value bet").

### "Best value bet" callout → folded into the market-row list

Instead of a separately bordered/tinted box below the market rows, the
recommended bet becomes: the matching `ProbabilityBar` row (already shown
above, for whichever market is recommended) gets `highlight="flagged"`
*and* a one-line addendum directly under that specific row — best price,
bookmaker, American-odds conversion — using the same row-width container
so it reads as "this row, expanded," not a disconnected second panel. The
"No value bet qualified" / "odds stale" / "no live odds yet" informational
states (`FixtureModal.tsx:324-333`) stay as their own plain text line
below the market-row list, unchanged (they're not a bet recommendation to
visually integrate, just an explanation of absence).

### Fixture list cards

Same `ProbabilityBar`, rendered smaller (reduced height/padding, no
`marketPct` tick unless the card already has room — read the current
implementation first to confirm what data the card already receives before
deciding whether the market comparison fits at that size). This is the
lowest-certainty part of the design since the current card wasn't read in
detail before this spec was written — implementation should treat this
sub-item as "apply the same bar component, adapt density to fit," not a
pixel-exact match to the modal's version.

## Non-goals (explicit)

- No change to the PL purple/pink shell, nav, page chrome, or branding.
- No change to `Calibration` page panels (`LiveValueBetPanel`,
  `BacktestPanel`, `WalkForwardBettingPanel`) unless the user amends scope
  (see "Open scope question" above).
- No backend/API changes — every field this design needs (`prob`,
  `implied`, `edge`, `recommended_bet`) already exists in `FixtureDetail`/
  `FixtureSummary`.
- No change to the `¢`-per-cent framing from the reference images — this
  project uses `%` throughout, per explicit user instruction.
- No literal color match to the reference images (black/green) — palette
  stays `pl-pink`/`pl-cyan`/existing tokens.

## Testing

- Frontend: `npx tsc -b` must stay clean (no new type errors).
- No backend changes, so no new backend tests are needed for this piece.
- Manual verification (this project has no frontend component test
  harness currently): open a fixture with live odds (bars + market tick +
  delta visible), a fixture with no live odds yet (bars only, no tick), a
  finished fixture with a hit/miss recorded (color-coded highlight), and
  the Fixtures tab list view, comparing against this spec's description
  before calling it done.

## Follow-ups (not part of this pass)

- Whether to extend this treatment to the three Calibration-page value-bet
  panels (see "Open scope question").
- The FIFA-style FPL team-view redesign (separate sub-project, not started).
