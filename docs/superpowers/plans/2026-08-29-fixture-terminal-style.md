# Fixture Terminal-Style Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat text market/scoreline rows in the fixture detail modal with a shared, dense, monospace `MarketBar` component (thin probability bar + market-comparison tick + signed % delta + hit/miss/flagged highlighting), fold the separate "Best value bet" callout into the matching row, and lightly extend the same monospace numeral treatment to the existing fixture-card probability bar.

**Architecture:** One new presentational component (`MarketBar`) replaces `FixtureModal.tsx`'s `MarketRow` and `ScorelineHeatmap.tsx`'s flat scoreline rows at their existing call sites — no state/data-flow changes, no backend changes, every field already exists on `FixtureDetail`/`FixtureSummary`. The existing 3-way `ProbabilityBar` component (fixture list cards) keeps its current shape and gets only a numeral-font tweak.

**Tech Stack:** React + TypeScript (Vite), Tailwind utility classes, existing `pl-*`/`win`/`loss`/`draw` design tokens from `frontend/src/index.css`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-29-fixture-terminal-style-design.md`

## Global Constraints

- No backend/API changes of any kind — every field this plan touches
  (`prob`, `implied`, `edge`, `recommended_bet`, `value_bet_flags`) already
  exists on the frontend types.
- No change to the PL purple/pink shell, nav, page chrome, or branding.
- No change to the Calibration page's `LiveValueBetPanel`, `BacktestPanel`,
  or `WalkForwardBettingPanel` — explicitly out of scope per the spec's
  "Open scope question."
- Percentages only — no `¢`/cents-style notation anywhere.
- Palette stays `pl-pink`/`pl-cyan`/existing tokens — no new colors, no
  literal black/green from the reference images.
- This codebase has no frontend component test harness — verification is
  `npx tsc -b` (must stay clean) plus a manual QA pass against the
  spec's "Testing" checklist, not automated unit tests. Every task below
  ends with a `tsc -b` check instead of a test-runner step.

## Plan-time discovery (not in the spec — found while planning; overrides its literal wording where noted)

The spec assumed the fixture-list-card bar treatment ("Fixture list cards"
section) would reuse the same new bar component "rendered smaller." In
fact **a component named `ProbabilityBar` already exists**
(`frontend/src/components/ProbabilityBar.tsx`) — a 3-segment stacked bar
(home/draw/away all in one bar, `win`/`draw`/`loss` colors) used by
`CurrentGameweekCard.tsx`. This is a different shape from what the spec's
new component needs (one bar per single market probability, with an
optional market-comparison tick), and the name is already taken.

Resolution: the new component is named **`MarketBar`**, not
`ProbabilityBar`. Task 5 below applies only a numeral-font tweak to the
*existing* `ProbabilityBar` (matching the spec's actual intent — "adapt
density to fit," not a pixel-exact match) rather than replacing its shape.
`FinishedFixtureCard.tsx` (the completed-match card) doesn't use
`ProbabilityBar` at all today (plain text percentages) — left untouched;
extending it further would be scope creep beyond what was asked.

---

### Task 1: Create the `MarketBar` component

**Files:**
- Create: `frontend/src/components/MarketBar.tsx`

**Interfaces:**
- Consumes: nothing from other tasks (first task).
- Produces: `MarketBar` React component, used by Tasks 2, 3, and 4.
  ```ts
  interface MarketBarProps {
    label: ReactNode;
    prob: number; // 0-1
    marketProb?: number | null; // 0-1, optional — omit to hide the tick/delta
    highlight?: "flagged" | "hit" | "miss";
    detail?: ReactNode; // optional one-line addendum rendered below the bar
  }
  function MarketBar(props: MarketBarProps): JSX.Element
  ```

- [ ] **Step 1: Write the component**

```tsx
import type { ReactNode } from "react";

interface MarketBarProps {
  label: ReactNode;
  prob: number;
  marketProb?: number | null;
  highlight?: "flagged" | "hit" | "miss";
  detail?: ReactNode;
}

const HIGHLIGHT_CLASSES: Record<string, string> = {
  hit: "bg-win/10 ring-1 ring-win/30",
  miss: "bg-loss/10",
  flagged: "bg-pl-pink/10 ring-1 ring-pl-pink/40",
};

export function MarketBar({ label, prob, marketProb, highlight, detail }: MarketBarProps) {
  const pct = Math.round(prob * 100);
  const marketPct = marketProb !== undefined && marketProb !== null ? Math.round(marketProb * 100) : null;
  const delta = marketPct !== null ? pct - marketPct : null;

  return (
    <div className={`rounded-lg px-3 py-2 text-sm ${highlight ? HIGHLIGHT_CLASSES[highlight] : "bg-pl-850/60"}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-pl-text-dim">{label}</span>
        <div className="flex items-center gap-2 font-mono">
          {delta !== null && (
            <span className={`text-xs font-semibold ${delta > 0 ? "text-pl-cyan" : "text-pl-text-faint"}`}>
              {delta > 0 ? "+" : ""}
              {delta}%
            </span>
          )}
          <span className="font-semibold text-pl-text">{pct}%</span>
        </div>
      </div>
      <div className="relative mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-pl-border/60">
        <div className="absolute inset-y-0 left-0 rounded-full bg-pl-pink" style={{ width: `${pct}%` }} />
        {marketPct !== null && (
          <div className="absolute -inset-y-0.5 w-0.5 bg-pl-text-faint" style={{ left: `${marketPct}%` }} />
        )}
      </div>
      {detail && <p className="mt-1.5 text-xs text-pl-text-dim">{detail}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors (the file isn't imported anywhere yet, so this only
checks the new file's own syntax/types are valid).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/MarketBar.tsx
git commit -m "feat: add MarketBar component for the terminal-style redesign"
```

---

### Task 2: Replace `MarketRow` with `MarketBar` for the core markets, BTTS, and team-2+ rows

**Files:**
- Modify: `frontend/src/components/FixtureModal.tsx:32-54` (remove `MarketRow`)
- Modify: `frontend/src/components/FixtureModal.tsx:373-424` (the "Match result & goals" market list)

**Interfaces:**
- Consumes: `MarketBar` from Task 1 (`frontend/src/components/MarketBar.tsx`).
- Produces: a `highlightFor(flagged, postMatchHit)` helper function, reused
  by Task 3.
  ```ts
  function highlightFor(flagged: boolean, postMatchHit?: boolean): "flagged" | "hit" | "miss" | undefined
  ```

- [ ] **Step 1: Remove the old `MarketRow` function and add the import + helper**

Delete this block (`FixtureModal.tsx:32-54`):

```tsx
function MarketRow({ label, edge, flagged, postMatchHit }: { label: ReactNode; edge: MarketEdge; flagged: boolean; postMatchHit?: boolean }) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
        postMatchHit === true ? "bg-win/10 ring-1 ring-win/30" : postMatchHit === false ? "bg-loss/10" : flagged ? "bg-pl-pink/10 ring-1 ring-pl-pink/40" : "bg-pl-850/60"
      }`}
    >
      <span className="text-pl-text-dim">{label}</span>
      <div className="flex items-center gap-3">
        <span className="font-semibold text-pl-text">{(edge.prob * 100).toFixed(1)}%</span>
        {edge.implied !== null && (
          <>
            <span className="text-xs text-pl-text-faint">mkt {(edge.implied * 100).toFixed(1)}%</span>
            <span className={`text-xs font-semibold ${(edge.edge ?? 0) > 0 ? "text-pl-cyan" : "text-pl-text-faint"}`}>
              {(edge.edge ?? 0) > 0 ? "+" : ""}
              {((edge.edge ?? 0) * 100).toFixed(2)}%
            </span>
          </>
        )}
      </div>
    </div>
  );
}
```

Replace it with:

```tsx
function highlightFor(flagged: boolean, postMatchHit?: boolean): "flagged" | "hit" | "miss" | undefined {
  if (postMatchHit === true) return "hit";
  if (postMatchHit === false) return "miss";
  if (flagged) return "flagged";
  return undefined;
}
```

Separately, add `import { MarketBar } from "./MarketBar";` to the block of
component imports at the top of the file (alongside the existing
`import { InfoTooltip } from "./InfoTooltip";` on line 7) — not inline
where `MarketRow` used to live. `MarketEdge` may become an unused import
once `MarketRow`'s signature is gone — check the rest of the file still
uses it before removing the import; if nothing else references
`MarketEdge`, delete it from the `import type { ... } from "../types"`
line to keep `tsc` clean of unused-import warnings.

- [ ] **Step 2: Replace the core 5-market loop and BTTS/2+ rows**

Find this block (`FixtureModal.tsx:379-424`, inside the "Match result &
goals" section):

```tsx
                    <div className="flex flex-col gap-1.5">
                      {(["home_win", "draw", "away_win", "over_2_5", "under_2_5"] as const).map((k) => (
                        <MarketRow
                          key={k}
                          label={MARKET_LABELS[k]}
                          edge={detail[k]}
                          flagged={detail.value_bet_flags.includes(k)}
                          postMatchHit={(() => {
                            const verdict = postMatchVerdict(k === "home_win" || k === "draw" || k === "away_win" ? "Match result" : "Goals O/U 2.5");
                            const selection = k === "home_win" || k === "draw" || k === "away_win" ? k : k === "over_2_5" ? "over" : "under";
                            return verdict?.prediction === selection ? verdict.hit : undefined;
                          })()}
                        />
                      ))}
                      <MarketRow
                        label={
                          <span className="inline-flex items-center gap-1.5">
                            BTTS: Yes <InfoTooltip text={GLOSSARY.btts} align="right" />
                          </span>
                        }
                        edge={{ prob: detail.btts_yes_prob, implied: null, edge: null }}
                        flagged={false}
                        postMatchHit={postMatchVerdict("BTTS")?.prediction === "yes" ? postMatchVerdict("BTTS")?.hit : undefined}
                      />
                      {detail.home_2plus_prob !== null && (
                        <MarketRow
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_home} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          edge={{ prob: detail.home_2plus_prob, implied: null, edge: null }}
                          flagged={false}
                        />
                      )}
                      {detail.away_2plus_prob !== null && (
                        <MarketRow
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_away} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          edge={{ prob: detail.away_2plus_prob, implied: null, edge: null }}
                          flagged={false}
                        />
                      )}
```

Replace it with (note: `recommendedDetail` here is a placeholder wired up
fully in Task 3 — for this step, pass `undefined` so the file still
compiles standalone):

```tsx
                    <div className="flex flex-col gap-1.5">
                      {(["home_win", "draw", "away_win", "over_2_5", "under_2_5"] as const).map((k) => (
                        <MarketBar
                          key={k}
                          label={MARKET_LABELS[k]}
                          prob={detail[k].prob}
                          marketProb={detail[k].implied}
                          highlight={highlightFor(
                            detail.value_bet_flags.includes(k),
                            (() => {
                              const verdict = postMatchVerdict(k === "home_win" || k === "draw" || k === "away_win" ? "Match result" : "Goals O/U 2.5");
                              const selection = k === "home_win" || k === "draw" || k === "away_win" ? k : k === "over_2_5" ? "over" : "under";
                              return verdict?.prediction === selection ? verdict.hit : undefined;
                            })()
                          )}
                        />
                      ))}
                      <MarketBar
                        label={
                          <span className="inline-flex items-center gap-1.5">
                            BTTS: Yes <InfoTooltip text={GLOSSARY.btts} align="right" />
                          </span>
                        }
                        prob={detail.btts_yes_prob}
                        highlight={highlightFor(
                          false,
                          postMatchVerdict("BTTS")?.prediction === "yes" ? postMatchVerdict("BTTS")?.hit : undefined
                        )}
                      />
                      {detail.home_2plus_prob !== null && (
                        <MarketBar
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_home} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          prob={detail.home_2plus_prob}
                        />
                      )}
                      {detail.away_2plus_prob !== null && (
                        <MarketBar
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_away} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          prob={detail.away_2plus_prob}
                        />
                      )}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors. If `MarketEdge` shows as an unused import, remove it
from the `import type` line at the top of the file.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/FixtureModal.tsx
git commit -m "feat: replace MarketRow with MarketBar in fixture modal"
```

---

### Task 3: Fold the "Best value bet" callout into its matching row

**Files:**
- Modify: `frontend/src/components/FixtureModal.tsx:301-335` (the standalone "Best value bet" section)
- Modify: `frontend/src/components/FixtureModal.tsx:379-386` (the core-markets loop from Task 2 — add the `detail` wiring)
- Modify: `frontend/src/components/FixtureModal.tsx:446` (add the "no value bet" fallback below the market-row list)

**Interfaces:**
- Consumes: `MarketBar`'s `detail` prop (Task 1), `highlightFor` (Task 2).
- Produces: nothing further consumed by later tasks.

- [ ] **Step 1: Remove the standalone "Best value bet" section, drop its live/upcoming branch entirely**

Find this block (`FixtureModal.tsx:301-335`):

```tsx
              {detail.post_match ? (
                detail.pre_match_value_bets?.length > 0 ? <PreMatchValueBets bets={detail.pre_match_value_bets} /> : (
                  <section>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Pre-match value bets</h3>
                    <p className="rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-faint">No value bet qualified before kickoff for this fixture, so no pre-match bet was recorded.</p>
                  </section>
                )
              ) : <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Best value bet</h3>
                {detail.recommended_bet ? (
                  <div className="rounded-xl border border-pl-cyan/40 bg-pl-cyan/10 p-3 text-sm">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <span className="font-semibold text-pl-text">{MARKET_LABELS[detail.recommended_bet.market]}</span>
                        <span className="ml-2 text-[10px] font-semibold uppercase text-pl-text-faint">{marketType(detail.recommended_bet.market)}</span>
                      </div>
                      <span className="font-semibold text-pl-cyan">+{(detail.recommended_bet.edge * 100).toFixed(1)}% edge</span>
                    </div>
                    <p className="mt-1 text-xs text-pl-text-dim">
                      Best observed price: {americanOdds(detail.recommended_bet.price)} at {detail.recommended_bet.bookmaker} · model {(detail.recommended_bet.probability * 100).toFixed(1)}%
                    </p>
                    <p className="mt-2 text-[11px] text-pl-text-faint">This highlights the strongest qualifying row below; it is educational only, never a parlay.</p>
                  </div>
                ) : (
                  <p className="rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-faint">
                    {new Date(detail.commence_time).getTime() <= Date.now()
                      ? "Kickoff has passed, so pre-match odds can no longer be used to calculate a value bet."
                      : detail.odds_is_stale
                        ? `Live odds were last fetched ${detail.odds_fetched_at ? new Date(detail.odds_fetched_at).toLocaleString() : "too long ago"}. Refresh odds before treating an edge as actionable.`
                      : detail.has_live_odds
                        ? "No value bet clears the current 5-point edge and price filters."
                        : "Live match-result and goals odds have not loaded yet, so a value bet cannot be calculated."}
                  </p>
                )}
              </section>}
```

Replace it with — the `detail.post_match` / `PreMatchValueBets` branch
(finished fixtures) is untouched and kept exactly as-is; the live/upcoming
"else" branch is dropped entirely from this location, because per the
spec its content belongs "below the market-row list," not here, above the
whole two-column grid. Its fallback text is moved to Step 3 below, right
next to the market list it actually explains:

```tsx
              {detail.post_match && (
                detail.pre_match_value_bets?.length > 0 ? <PreMatchValueBets bets={detail.pre_match_value_bets} /> : (
                  <section>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Pre-match value bets</h3>
                    <p className="rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-faint">No value bet qualified before kickoff for this fixture, so no pre-match bet was recorded.</p>
                  </section>
                )
              )}
```

- [ ] **Step 2: Wire the recommended-bet detail line onto its matching `MarketBar` row**

In the core-markets loop from Task 2 (now at roughly `FixtureModal.tsx:379-393`),
change:

```tsx
                      {(["home_win", "draw", "away_win", "over_2_5", "under_2_5"] as const).map((k) => (
                        <MarketBar
                          key={k}
                          label={MARKET_LABELS[k]}
                          prob={detail[k].prob}
                          marketProb={detail[k].implied}
                          highlight={highlightFor(
                            detail.value_bet_flags.includes(k),
                            (() => {
                              const verdict = postMatchVerdict(k === "home_win" || k === "draw" || k === "away_win" ? "Match result" : "Goals O/U 2.5");
                              const selection = k === "home_win" || k === "draw" || k === "away_win" ? k : k === "over_2_5" ? "over" : "under";
                              return verdict?.prediction === selection ? verdict.hit : undefined;
                            })()
                          )}
                        />
                      ))}
```

to:

```tsx
                      {(["home_win", "draw", "away_win", "over_2_5", "under_2_5"] as const).map((k) => (
                        <MarketBar
                          key={k}
                          label={MARKET_LABELS[k]}
                          prob={detail[k].prob}
                          marketProb={detail[k].implied}
                          highlight={highlightFor(
                            detail.value_bet_flags.includes(k),
                            (() => {
                              const verdict = postMatchVerdict(k === "home_win" || k === "draw" || k === "away_win" ? "Match result" : "Goals O/U 2.5");
                              const selection = k === "home_win" || k === "draw" || k === "away_win" ? k : k === "over_2_5" ? "over" : "under";
                              return verdict?.prediction === selection ? verdict.hit : undefined;
                            })()
                          )}
                          detail={
                            !detail.post_match && detail.recommended_bet?.market === k ? (
                              <>
                                Best observed price: {americanOdds(detail.recommended_bet.price)} at {detail.recommended_bet.bookmaker} ·{" "}
                                <span className="font-mono font-semibold text-pl-cyan">
                                  +{(detail.recommended_bet.edge * 100).toFixed(1)}%
                                </span>{" "}
                                edge
                              </>
                            ) : undefined
                          }
                        />
                      ))}
```

(`marketType`, imported/defined earlier in the file, is no longer called
anywhere after this change — if `tsc`/`oxlint` flags it as unused, delete
the now-dead `marketType` function together with its call site removal in
this step, not as a separate cleanup task.)

- [ ] **Step 3: Add the "no value bet qualified" fallback below the market-row list**

Find this line (originally `FixtureModal.tsx:446`, immediately before the
"Match result & goals" section's closing `</section>`):

```tsx
                    {!detail.has_live_odds && <p className="mt-2 text-xs text-pl-text-faint">{GLOSSARY.noLiveMarket}</p>}
                  </section>
```

Replace it with:

```tsx
                    {!detail.has_live_odds && <p className="mt-2 text-xs text-pl-text-faint">{GLOSSARY.noLiveMarket}</p>}
                    {!detail.post_match && !detail.recommended_bet && (
                      <p className="mt-2 text-xs text-pl-text-faint">
                        {new Date(detail.commence_time).getTime() <= Date.now()
                          ? "Kickoff has passed, so pre-match odds can no longer be used to calculate a value bet."
                          : detail.odds_is_stale
                            ? `Live odds were last fetched ${detail.odds_fetched_at ? new Date(detail.odds_fetched_at).toLocaleString() : "too long ago"}. Refresh odds before treating an edge as actionable.`
                          : detail.has_live_odds
                            ? "No value bet clears the current 5-point edge and price filters."
                            : "Live match-result and goals odds have not loaded yet, so a value bet cannot be calculated."}
                      </p>
                    )}
                  </section>
```

- [ ] **Step 4: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/FixtureModal.tsx
git commit -m "feat: fold best-value-bet callout into its matching market row"
```

---

### Task 4: Replace the "next most likely scorelines" rows with `MarketBar`

**Files:**
- Modify: `frontend/src/components/ScorelineHeatmap.tsx:91-111` (approximate — verify exact line numbers before editing, since Tasks 1-3 don't touch this file and its line numbers are unaffected by them)

**Interfaces:**
- Consumes: `MarketBar` from Task 1.
- Produces: nothing further consumed by later tasks.

- [ ] **Step 1: Replace the flat scoreline rows**

Find this block in `ScorelineHeatmap.tsx` (top-level structure; re-read the
file first since exact line numbers may have drifted since the spec was
written):

```tsx
      {topScorelines.length > 1 && (
        <div className="mt-4">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
            Next most likely scorelines
          </p>
          <div className="flex flex-col gap-1">
            {topScorelines.map((s, i) => (
              <div
                key={i}
                className={`flex items-center justify-between rounded-lg px-3 py-1.5 text-xs ${
                  i === 0 ? "bg-pl-pink/10 ring-1 ring-pl-pink/30" : "bg-pl-850/60"
                }`}
              >
                <span className="font-mono font-medium text-pl-text">
                  {homeTeam} {s.home}–{s.away} {awayTeam}
                </span>
                <span className="font-semibold text-pl-text-dim">{(s.prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
```

Replace it with:

```tsx
      {topScorelines.length > 1 && (
        <div className="mt-4">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
            Next most likely scorelines
          </p>
          <div className="flex flex-col gap-1">
            {topScorelines.map((s, i) => (
              <MarketBar
                key={i}
                label={
                  <span className="font-mono font-medium text-pl-text">
                    {homeTeam} {s.home}–{s.away} {awayTeam}
                  </span>
                }
                prob={s.prob}
                highlight={i === 0 ? "flagged" : undefined}
              />
            ))}
          </div>
        </div>
      )}
```

Add `import { MarketBar } from "./MarketBar";` near the top of
`ScorelineHeatmap.tsx` alongside its existing imports.

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ScorelineHeatmap.tsx
git commit -m "feat: use MarketBar for the next-most-likely-scorelines list"
```

---

### Task 5: Monospace-numeral tweak to the existing fixture-card `ProbabilityBar`

**Files:**
- Modify: `frontend/src/components/ProbabilityBar.tsx` (entire file — it's 24 lines)

**Interfaces:**
- Consumes: nothing from earlier tasks (independent, cosmetic-only change).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Apply the monospace font to the percentage numerals only**

Replace the full contents of `frontend/src/components/ProbabilityBar.tsx`:

```tsx
interface Props {
  home: number;
  draw: number;
  away: number;
  homeLabel?: string;
  awayLabel?: string;
}

export function ProbabilityBar({ home, draw, away, homeLabel = "H", awayLabel = "A" }: Props) {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

  return (
    <div className="w-full">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-pl-850">
        <div className="bg-win" style={{ width: pct(home) }} />
        <div className="bg-draw" style={{ width: pct(draw) }} />
        <div className="bg-loss" style={{ width: pct(away) }} />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] font-medium text-pl-text-dim">
        <span>
          {homeLabel} <span className="font-mono text-pl-text">{pct(home)}</span>
        </span>
        <span>
          D <span className="font-mono text-pl-text">{pct(draw)}</span>
        </span>
        <span>
          {awayLabel} <span className="font-mono text-pl-text">{pct(away)}</span>
        </span>
      </div>
    </div>
  );
}
```

(Only change from the original: `text-pl-text` → `font-mono text-pl-text`
on the three percentage `<span>`s — the bar shape, colors, and layout are
unchanged, matching the plan-time discovery note above.)

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ProbabilityBar.tsx
git commit -m "style: monospace numerals on the fixture-card probability bar"
```

---

### Task 6: Manual QA pass against the spec's testing checklist

**Files:** none (verification only).

**Interfaces:** none — final task.

- [ ] **Step 1: Confirm the dev servers are running**

The backend (port 8000) and frontend (port 5173) dev servers should
already be running from earlier work this session. If not:

```bash
cd /Users/sigey/Documents/Projects/Prem_Predictor/PL_Predictor && source .venv/bin/activate && PYTHONPATH="$(pwd)/src" uvicorn pl_predictor.api.main:app --reload --port 8000 &
cd frontend && npm run dev &
```

- [ ] **Step 2: Open a fixture with live odds**

In the browser, open the Fixtures tab, click into a fixture that has live
odds (`has_live_odds: true`). Confirm: each market row shows a bar, a
market-comparison tick mark on the bar, and a signed whole-number `%`
delta next to the percentage — not the old `mkt X% / +X.XX%` two-number
text format.

- [ ] **Step 3: Open a fixture with no live odds yet**

Open a fixture further out than the odds window (`has_live_odds: false`).
Confirm: bars render with no tick mark and no delta (since there's no
`marketProb` to compare against), and the `GLOSSARY.noLiveMarket` note
still appears below the list, unchanged.

- [ ] **Step 4: Open a finished fixture with a recorded result**

Confirm the hit/miss coloring (green ring for a correct call, red tint for
a miss) still shows correctly per market row, and that the standalone
"Best value bet" section is gone — the recommended market's own row shows
the price/bookmaker detail line directly beneath it instead.

- [ ] **Step 5: Check the Fixtures tab list view**

Confirm the compact fixture cards' home/draw/away bar still renders
correctly with monospace percentage numerals, unchanged in shape/layout.

- [ ] **Step 6: Final full-repo verification**

Run: `cd frontend && npx tsc -b`
Expected: no errors, confirming the whole redesign compiles cleanly
end-to-end (not just per-task).

- [ ] **Step 7: Update `docs/AI_CONTINUITY.md` with a short entry**

Add a brief note (following this file's existing style — see its most
recent entries for the exact tone/format) recording: `MarketBar` component
added, `MarketRow`/the standalone "Best value bet" section removed, and
the fixture-card `ProbabilityBar` kept its shape (naming collision
discovered during planning, documented here as the historical record of
why two similarly-named-sounding bar components coexist).

- [ ] **Step 8: Commit the documentation update**

```bash
git add docs/AI_CONTINUITY.md
git commit -m "docs: record the fixture terminal-style redesign"
```
