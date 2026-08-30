# FPL page redesign: FIFA-style player cards + 3-way squad-view toggle

**Date:** 2026-08-29
**Status:** design approved by user pending spec review
**Scope class:** architectural (new shared component, page-level restructure)

## Context

Sub-project #3 of a three-way split from a broader "make the FPL visuals
more FIFA-like" request (sub-project #1, fixture-detail load performance,
and #2, the terminal-style fixture/odds redesign, are both already shipped
— see `docs/AI_CONTINUITY.md`'s `OPS-2026-04` entry for #2).

While scoping which sections should get FIFA-card treatment, the user
described a bigger restructure than "just add cards": today's FPL tab
(`frontend/src/pages/FPLPage.tsx`) stacks four sections vertically —
formation picker + starting-XI pitch view, the £100m squad list, the
transfer planner, and the player scout table — which means the two squad
views (this gameweek's XI vs. the theoretical best £100m squad) compete
for scroll space even though a user only wants to look at one at a time.
The request became: turn the XI view, the £100m squad view, and the
transfer planner into one 3-way toggle, and give the transfer planner two
things it's currently missing — a visible transfer count/bank, and a
plain-language reason for each suggested swap.

## Decisions already made (do not re-litigate these)

1. **In scope:** the starting-XI pitch view, the £100m squad list, and the
   transfer planner — restructured into one 3-way toggle. **Out of scope:**
   the player scout table (500+ players) — stays a data table below the
   toggle group, unaffected, per explicit user choice (a card grid at that
   scale would hurt scannability, not help it).
2. **Palette:** FIFA-style card *shape and layout* (rating badge, position
   badge, team crest, price tag), but colored with the existing
   `pl-pink`/`pl-cyan`/`pl-850` tokens — no gold/silver/bronze card-rarity
   tiers, consistent with how the terminal-style redesign (#2) was scoped.
3. **Toggle structure:** one 3-way toggle/tab group — `"This gameweek's
   XI"` / `"Best £100m squad"` / `"My team & transfers"` — replacing
   today's always-stacked layout, at the top of the page above all three
   views.
4. **Transfer explanation depth:** a short, 1-2-line plain-English reason
   per suggested transfer, built from the incoming player's existing
   `drivers` field (e.g. `"3 fixtures · 78 expected minutes"`) — no new
   backend computation, this field already exists on every player object
   the API already returns.
5. **Transfer count must be visible:** once transfer results load, show
   how many free transfers and how much bank the calculation used —
   currently returned by the API (`free_transfers`, `bank`) but never
   displayed anywhere in the UI.

## Current state (for reference during implementation)

All in `frontend/src/pages/FPLPage.tsx` (as of the 2026-08-29 fixture
terminal-style redesign commit — this file has had no changes since):

- `PlayerLine` (lines 8-13): flat text row — name + marker badge, position
  + projected points. Used only in the £100m squad grid (line 110).
- `PitchPlayer` (lines 15-20): small rounded-pill card — name + marker,
  projected points. Used only inside `LineupCard` (line 28), laid out in
  per-position grid rows on a green pitch background.
- `LineupCard` (lines 22-32): renders the starting XI on a pitch, using
  `PitchPlayer`. Takes `data: FPLRecommendation`, `title`, `requestedFormation`.
- Page body (lines 105-115): five always-rendered sections in document
  order — header, formation picker + `LineupCard`, £100m squad grid
  (`PlayerLine`s), transfer planner (inputs + results table), player scout
  table. No page-level "mode" state exists today; all sections always
  render if their data has loaded.
- Transfer planner results table (line 112): columns Out / In / Cost /
  Projected gain / Net gain. `transfers.free_transfers` and
  `transfers.bank` (both already in the `FPLTransfersResponse` type,
  `frontend/src/types.ts:600-608`) are received but never rendered
  anywhere on the page today.
- `FPLPlayerProjection` (`types.ts:549-565`) already has everything a
  card needs: `team`, `team_id`, `position`, `price`, `status`,
  `projected_points`, `drivers: string[]`. No backend or type changes
  needed for this redesign.
- `PlayerLine`/`PitchPlayer` are used nowhere else in the codebase
  (confirmed via repo-wide search) — safe to delete entirely once
  replaced, not just deprecated in place.
- `TeamBadge` (`frontend/src/components/TeamBadge.tsx`) already exists
  and takes `{ team: string; size?: "sm" | "md" | "lg" }` — reused as-is,
  no changes needed.
- The player scout table's existing availability-dot convention (line 114):
  `{p.status !== "a" && <span className="ml-1 text-loss">●</span>}` —
  reused verbatim for the new card's availability indicator.

## Design

### New shared component: `PlayerCard`

`frontend/src/components/PlayerCard.tsx`. Props:

```ts
interface PlayerCardProps {
  player: FPLPlayerProjection;
  marker?: string; // "C" / "VC", same convention PitchPlayer/PlayerLine already use
  size?: "sm" | "md"; // sm = pitch view, md = squad grid. Default "md".
}
```

Layout (both sizes share the same content, `sm` just compresses it):
- Top row: `<TeamBadge team={player.team} size="sm" />`, a position badge
  (`GK`/`DEF`/`MID`/`FWD`, small pill), and the availability dot when
  `player.status !== "a"`.
- Center: `player.projected_points.toFixed(1)` as the card's headline
  number — the FIFA "OVR" slot — large, bold, `pl-pink`.
- Bottom: `player.web_name || player.name`, and (size `"md"` only —
  `"sm"` omits it, there isn't room in a pitch-formation row) the price
  tag `£{player.price.toFixed(1)}`.
- The existing `marker` ("C"/"VC") renders as a small badge overlapping
  the top-right corner, same visual role `PitchPlayer`'s inline marker
  and `PlayerLine`'s marker span already play — not a new interaction,
  just carried over.

**Lowest-certainty part of this design:** `size="sm"` packs meaningfully
more content (crest, position badge, availability dot, rating, name) into
the same compact pitch-row space `PitchPlayer` used for just name+points.
This may need a tighter sub-layout (e.g. crest/badge/dot condensed into
one small icon row, name truncated more aggressively) than a naive
shrink-everything approach once actually laid out at real pitch widths —
treat "all five elements fit legibly in a formation row on a typical
screen" as something to verify and adjust during implementation, not an
assumption this spec guarantees.

### `LineupCard` and the £100m squad grid: use `PlayerCard`

- `LineupCard` (`FPLPage.tsx:22-32`): replace every `<PitchPlayer .../>`
  call with `<PlayerCard size="sm" .../>`, passing through the same
  `player`/`marker` props unchanged. The per-position grid layout
  (`gridTemplateColumns` sizing) is untouched — only the rendered card
  changes.
- £100m squad grid (currently `FPLPage.tsx:110`'s
  `grid gap-1 sm:grid-cols-2`, `PlayerLine` rows): becomes a card grid —
  `grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2`, rendering
  `<PlayerCard size="md" .../>` for each of the 15 squad players.
- Delete `PlayerLine` and `PitchPlayer` entirely once both call sites are
  migrated — confirmed no other file references them.

### Page restructure: 3-way toggle

Add `mode` state to `FPLPage`: `useState<"gameweek" | "squad" |
"transfers">("gameweek")`. Render a 3-button tab group at the top of the
page (same visual convention as the existing formation-picker buttons:
`bg-pl-pink text-white` when active, `border border-pl-border
text-pl-text-dim hover:text-pl-text` otherwise), labeled `"This gameweek's
XI"` / `"Best £100m squad"` / `"My team & transfers"`.

- `mode === "gameweek"`: renders the formation picker + `LineupCard`
  (today's lines 109), and nothing else from the old squad/transfer
  sections.
- `mode === "squad"`: renders the £100m squad card grid (today's line 110
  content, restyled per above) and nothing else from the XI/transfer
  sections.
- `mode === "transfers"`: renders the transfer planner section (today's
  line 112) — inputs plus results — and nothing else from the XI/squad
  sections.
- The header section (today's line 106) and the player scout table
  (today's line 114) are **not** part of the toggle — both always render,
  above and below the toggle group respectively, exactly as they do
  today. Switching `mode` never affects data fetching (`load()` still
  fetches projections/XI/squad together on mount, same as today) — it
  only changes which already-loaded section is visible, so switching
  tabs is instant, no new loading state.

### Transfer planner: visible count + per-row explanation

- Once `transfers` loads, render a header line above the results table:
  `"You have {transfers.free_transfers} free transfer(s) · £{transfers.bank.toFixed(1)}m in the bank"`.
- Each result row (`transfers.recommendations`) gains a second line below
  the existing Out/In/Cost/Gain row: the incoming player's `drivers`,
  joined as a short reason — e.g. `idea.in.drivers.slice(0, 2).join(" · ")`
  (cap at 2 driver strings so the reason stays one line; `drivers` is
  already ordered most-relevant-first by the backend, so taking the first
  two is a safe, no-new-logic truncation).

## Non-goals (explicit)

- No change to the player scout table (data, layout, or card treatment).
- No gold/silver/bronze FIFA card-rarity color system — existing palette
  only.
- No backend or type changes — every field this design needs already
  exists on `FPLPlayerProjection`/`FPLTransfersResponse`.
- No new data fetching or loading states — the toggle only changes which
  already-fetched section is visible.
- The separate player-rating/team-unit-strength research idea raised
  alongside this request is explicitly a different, later effort (its own
  brainstorm, its own spec) — not part of this plan. If it's eventually
  built, the FPL card's rating number could switch from `projected_points`
  to that new rating, but that's a future decision, not this one.

## Testing

- Frontend: `npx tsc -b` must stay clean.
- No backend changes, so no new backend tests.
- Manual verification (no frontend component test harness exists in this
  project): switch between all three tabs and confirm each shows the
  right section with no flash-of-missing-data (since nothing needs to
  re-fetch); confirm the pitch view's `PlayerCard size="sm"` cards still
  fit the formation grid at typical viewport widths; confirm the squad
  grid's `PlayerCard size="md"` cards show price and name correctly;
  confirm a transfer-planner run shows the free-transfers/bank line and a
  reason line under each recommendation.

## Follow-ups (not part of this pass)

- The player-rating / team-unit-strength research idea (see Non-goals) —
  a separate, later brainstorm; touches the core prediction model and
  needs this project's full walk-forward evaluation protocol before any
  promotion, same rigor as `EXP-2026-18` (squad continuity).
- Whether the player scout table ever gets a card-grid alternative view —
  explicitly deferred, not decided against permanently, just out of scope
  here.
