# FPL FIFA-Style Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the FPL page's plain player rows (`PlayerLine`, `PitchPlayer`) with one shared FIFA-style `PlayerCard` component, restructure the page's three squad views (gameweek XI, £100m squad, transfers) into a single 3-way toggle instead of always-stacked sections, and make the transfer planner show its transfer count/bank and a one-line reason per suggested swap.

**Architecture:** One new presentational component (`PlayerCard`) replaces two existing ones at their exact call sites; the page gains a `mode` state that gates which of three existing sections renders, with no new data fetching. No backend or type changes — every field this design needs already exists on `FPLPlayerProjection`/`FPLTransfersResponse`.

**Tech Stack:** React + TypeScript (Vite), Tailwind utility classes, existing `pl-*`/`win`/`loss` design tokens.

**Spec:** `docs/superpowers/specs/2026-08-29-fpl-fifa-style-design.md`

## Global Constraints

- No backend or type changes — `FPLPlayerProjection` and `FPLTransfersResponse` already have every field this plan needs.
- No gold/silver/bronze FIFA card-rarity colors — existing `pl-pink`/`pl-cyan`/`pl-850` palette only.
- The player scout table is out of scope — stays exactly as it is today, below the toggle group.
- Switching `mode` must never trigger new data fetching — it only changes which already-loaded section is visible.
- This codebase has no frontend component test harness — verification is `npx tsc -b` (must stay clean) plus a manual QA pass, not automated unit tests.

---

### Task 1: Create the `PlayerCard` component

**Files:**
- Create: `frontend/src/components/PlayerCard.tsx`

**Interfaces:**
- Consumes: `TeamBadge` (`frontend/src/components/TeamBadge.tsx`, unchanged — `{ team: string; size?: "sm" | "md" | "lg" }`), `FPLPlayerProjection` (`frontend/src/types.ts:549-565`, unchanged).
- Produces: `PlayerCard` React component, used by Tasks 2 and 3.
  ```ts
  interface PlayerCardProps {
    player: FPLPlayerProjection;
    marker?: string; // "C" / "VC"
    size?: "sm" | "md"; // default "md"
  }
  function PlayerCard(props: PlayerCardProps): JSX.Element
  ```

- [ ] **Step 1: Write the component**

```tsx
import type { FPLPlayerProjection } from "../types";
import { TeamBadge } from "./TeamBadge";

interface PlayerCardProps {
  player: FPLPlayerProjection;
  marker?: string;
  size?: "sm" | "md";
}

export function PlayerCard({ player, marker, size = "md" }: PlayerCardProps) {
  const isSmall = size === "sm";
  return (
    <div
      className={`relative min-w-0 rounded-xl border border-pl-border bg-pl-900/90 text-center shadow-sm ${
        isSmall ? "px-1.5 py-1.5" : "px-3 py-3"
      }`}
    >
      {marker && (
        <span className="absolute -right-1 -top-1 rounded-full bg-pl-pink px-1.5 py-0.5 text-[9px] font-bold text-white">
          {marker}
        </span>
      )}
      <div className="flex items-center justify-center gap-1">
        <TeamBadge team={player.team} size="sm" />
        <span
          className={`rounded bg-pl-850 px-1 py-0.5 font-semibold text-pl-text-dim ${
            isSmall ? "text-[8px]" : "text-[9px]"
          }`}
        >
          {player.position}
        </span>
        {player.status !== "a" && <span className="text-loss">●</span>}
      </div>
      <p className={`mt-1 font-bold text-pl-pink ${isSmall ? "text-sm" : "text-lg"}`}>
        {player.projected_points.toFixed(1)}
      </p>
      <p className={`truncate font-semibold text-white ${isSmall ? "text-[10px]" : "text-xs"}`}>
        {player.web_name || player.name}
      </p>
      {!isSmall && <p className="text-[10px] text-pl-text-faint">£{player.price.toFixed(1)}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors (the file isn't imported anywhere yet).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/PlayerCard.tsx
git commit -m "feat: add PlayerCard component for the FPL FIFA-style redesign"
```

---

### Task 2: Use `PlayerCard` in the starting-XI pitch view; delete `PitchPlayer`

**Files:**
- Modify: `frontend/src/pages/FPLPage.tsx:15-20` (delete `PitchPlayer`)
- Modify: `frontend/src/pages/FPLPage.tsx:28` (the pitch grid's player render)

**Interfaces:**
- Consumes: `PlayerCard` from Task 1.
- Produces: nothing further consumed by later tasks.

- [ ] **Step 1: Add the import and delete `PitchPlayer`**

Add near the top of `FPLPage.tsx`, alongside the existing `api`/`types` imports:

```tsx
import { PlayerCard } from "../components/PlayerCard";
```

Delete this block entirely (`FPLPage.tsx:15-20`):

```tsx
function PitchPlayer({ player, marker }: { player: FPLPlayerProjection; marker?: string }) {
  return <div className="min-w-0 rounded-full border border-white/20 bg-pl-900/90 px-2 py-1 text-center shadow-sm">
    <p className="truncate text-[10px] font-bold text-white">{marker && <span className="mr-0.5 text-pl-pink">{marker}</span>}{player.web_name}</p>
    <p className="text-[9px] text-emerald-200">{player.projected_points.toFixed(1)} pts</p>
  </div>;
}
```

- [ ] **Step 2: Use `PlayerCard` inside the pitch grid**

Find this line inside `LineupCard` (`FPLPage.tsx:28`):

```tsx
      {(["FWD", "MID", "DEF", "GK"] as const).map((position) => <div key={position} className="mx-auto grid w-full max-w-md gap-2" style={{ gridTemplateColumns: `repeat(${Math.max(positions(position).length, 1)}, minmax(0, 1fr))` }}>{positions(position).map((player) => <PitchPlayer key={player.player_id} player={player} marker={data.captain?.player_id === player.player_id ? "C" : data.vice_captain?.player_id === player.player_id ? "VC" : undefined} />)}</div>)}
```

Replace it with:

```tsx
      {(["FWD", "MID", "DEF", "GK"] as const).map((position) => <div key={position} className="mx-auto grid w-full max-w-md gap-2" style={{ gridTemplateColumns: `repeat(${Math.max(positions(position).length, 1)}, minmax(0, 1fr))` }}>{positions(position).map((player) => <PlayerCard key={player.player_id} player={player} size="sm" marker={data.captain?.player_id === player.player_id ? "C" : data.vice_captain?.player_id === player.player_id ? "VC" : undefined} />)}</div>)}
```

(Only change: `PitchPlayer` → `PlayerCard`, with `size="sm"` added — same `key`/`player`/`marker` props otherwise.)

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors. If `FPLPlayerProjection` shows as an unused import in `FPLPage.tsx`, do not remove it — it's still used elsewhere in this file (e.g. `positionFilter` state, `players` filtering).

- [ ] **Step 4: Visually check the pitch view fits**

With the dev server running (`npm run dev` in `frontend/`, backend on :8000), open the FPL tab and look at the starting-XI pitch view. The spec flags this as its lowest-certainty part: `PlayerCard`'s `size="sm"` packs a team crest, position badge, availability dot, rating, and name into the same compact space `PitchPlayer` used for just name+points. If any element is illegibly cramped or overflows its card at typical viewport widths, reduce `TeamBadge`'s size further (it's already `size="sm"`, the smallest `TeamBadge` supports — instead, tighten the card's own padding/gap or drop the position-badge text to an even smaller size) rather than redesigning the layout wholesale — note whatever adjustment was needed in the commit message for Task 6's docs entry.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/FPLPage.tsx
git commit -m "feat: use PlayerCard in the starting-XI pitch view"
```

---

### Task 3: Use `PlayerCard` in the £100m squad grid; delete `PlayerLine`

**Files:**
- Modify: `frontend/src/pages/FPLPage.tsx:8-13` (delete `PlayerLine`)
- Modify: `frontend/src/pages/FPLPage.tsx:110` (the squad grid's player render)

**Interfaces:**
- Consumes: `PlayerCard` from Task 1.
- Produces: nothing further consumed by later tasks.

- [ ] **Step 1: Delete `PlayerLine`**

Delete this block entirely (`FPLPage.tsx:8-13`):

```tsx
function PlayerLine({ player, marker }: { player: FPLPlayerProjection; marker?: string }) {
  return <div className="flex items-center justify-between gap-2 rounded-md border border-pl-border/70 bg-pl-850/40 px-2 py-1.5 text-xs">
    <span className="min-w-0 truncate font-semibold text-pl-text">{marker && <span className="mr-1 rounded bg-pl-pink/20 px-1 text-pl-pink">{marker}</span>}{player.web_name || player.name}</span>
    <span className="shrink-0 text-pl-text-dim">{player.position} · {player.projected_points.toFixed(1)}</span>
  </div>;
}
```

- [ ] **Step 2: Replace the squad grid**

Find this line (`FPLPage.tsx:110`):

```tsx
    <div className="grid gap-4 xl:grid-cols-2">{squad ? <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4"><div className="mb-3 flex justify-between"><div><h3 className="font-semibold text-pl-text">Best £100m squad</h3><p className="text-xs text-pl-text-dim">£{squad.spent.toFixed(1)}m spent · £{squad.remaining.toFixed(1)}m remaining</p></div><span className="text-xs text-pl-text-faint">Normal gameweek only</span></div><div className="grid gap-1 sm:grid-cols-2">{squad.squad.map((player) => <PlayerLine key={player.player_id} player={player} marker={squad.captain?.player_id === player.player_id ? "C" : undefined} />)}</div></section> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the £100m squad…</section>}</div>
```

Replace it with:

```tsx
    <div>{squad ? <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4"><div className="mb-3 flex justify-between"><div><h3 className="font-semibold text-pl-text">Best £100m squad</h3><p className="text-xs text-pl-text-dim">£{squad.spent.toFixed(1)}m spent · £{squad.remaining.toFixed(1)}m remaining</p></div><span className="text-xs text-pl-text-faint">Normal gameweek only</span></div><div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">{squad.squad.map((player) => <PlayerCard key={player.player_id} player={player} size="md" marker={squad.captain?.player_id === player.player_id ? "C" : undefined} />)}</div></section> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the £100m squad…</section>}</div>
```

(Changes: outer `xl:grid-cols-2` two-column wrapper dropped since a 15-card
grid needs its own width, not a half-width column next to nothing else —
`div` kept as the outer wrapper for layout-diff minimalism, just without
the old grid classes; inner grid changed from `grid gap-1 sm:grid-cols-2`
row-list sizing to a `grid-cols-2 sm:grid-cols-3 lg:grid-cols-5` card
grid; `PlayerLine` → `PlayerCard` with `size="md"`.)

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/FPLPage.tsx
git commit -m "feat: use PlayerCard in the best-squad grid"
```

---

### Task 4: Add the 3-way mode toggle

**Files:**
- Modify: `frontend/src/pages/FPLPage.tsx` (state, tab buttons, section gating)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `mode` state (`"gameweek" | "squad" | "transfers"`), consumed by no later task directly but changes which sections Task 5 touches are visible.

- [ ] **Step 1: Add `mode` state and a labels map**

Find this line (near the other `useState` declarations):

```tsx
  const [formation, setFormation] = useState("Auto");
```

Replace it with:

```tsx
  const [formation, setFormation] = useState("Auto");
  const [mode, setMode] = useState<"gameweek" | "squad" | "transfers">("gameweek");
```

Add this constant near the top of the file, alongside `FORMATIONS`/`PAGE_SIZE`:

```tsx
const MODE_LABELS: Record<"gameweek" | "squad" | "transfers", string> = {
  gameweek: "This gameweek's XI",
  squad: "Best £100m squad",
  transfers: "My team & transfers",
};
```

- [ ] **Step 2: Add the tab buttons and gate the three sections**

Find this block (`FPLPage.tsx:108-112`):

```tsx
    {error && <p className="rounded-lg border border-loss/40 bg-loss/10 px-3 py-2 text-sm text-loss">{error}</p>}
    <section><div className="mb-2 flex flex-wrap gap-1">{FORMATIONS.map((item) => <button key={item} onClick={() => chooseFormation(item)} className={`rounded-md px-2.5 py-1 text-xs font-semibold ${formation === item ? "bg-pl-pink text-white" : "border border-pl-border text-pl-text-dim hover:text-pl-text"}`}>{item}</button>)}</div>{xi ? <LineupCard data={xi} title="Best starting XI" requestedFormation={formation} /> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the best starting XI…</section>}</section>
    <div>{squad ? <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4"><div className="mb-3 flex justify-between"><div><h3 className="font-semibold text-pl-text">Best £100m squad</h3><p className="text-xs text-pl-text-dim">£{squad.spent.toFixed(1)}m spent · £{squad.remaining.toFixed(1)}m remaining</p></div><span className="text-xs text-pl-text-faint">Normal gameweek only</span></div><div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">{squad.squad.map((player) => <PlayerCard key={player.player_id} player={player} size="md" marker={squad.captain?.player_id === player.player_id ? "C" : undefined} />)}</div></section> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the £100m squad…</section>}</div>

    <section className="rounded-xl border border-pl-border bg-pl-900/50 p-5"><h3 className="font-semibold text-pl-text">Transfer planner</h3>
```

Replace it with:

```tsx
    {error && <p className="rounded-lg border border-loss/40 bg-loss/10 px-3 py-2 text-sm text-loss">{error}</p>}

    <div className="flex flex-wrap gap-1 rounded-lg border border-pl-border p-1">
      {(["gameweek", "squad", "transfers"] as const).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={`rounded-md px-3 py-1.5 text-xs font-semibold ${
            mode === m ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"
          }`}
        >
          {MODE_LABELS[m]}
        </button>
      ))}
    </div>

    {mode === "gameweek" && (
      <section><div className="mb-2 flex flex-wrap gap-1">{FORMATIONS.map((item) => <button key={item} onClick={() => chooseFormation(item)} className={`rounded-md px-2.5 py-1 text-xs font-semibold ${formation === item ? "bg-pl-pink text-white" : "border border-pl-border text-pl-text-dim hover:text-pl-text"}`}>{item}</button>)}</div>{xi ? <LineupCard data={xi} title="Best starting XI" requestedFormation={formation} /> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the best starting XI…</section>}</section>
    )}

    {mode === "squad" && (
      <div>{squad ? <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4"><div className="mb-3 flex justify-between"><div><h3 className="font-semibold text-pl-text">Best £100m squad</h3><p className="text-xs text-pl-text-dim">£{squad.spent.toFixed(1)}m spent · £{squad.remaining.toFixed(1)}m remaining</p></div><span className="text-xs text-pl-text-faint">Normal gameweek only</span></div><div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">{squad.squad.map((player) => <PlayerCard key={player.player_id} player={player} size="md" marker={squad.captain?.player_id === player.player_id ? "C" : undefined} />)}</div></section> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the £100m squad…</section>}</div>
    )}

    {mode === "transfers" && (
      <section className="rounded-xl border border-pl-border bg-pl-900/50 p-5"><h3 className="font-semibold text-pl-text">Transfer planner</h3><p className="mt-1 text-sm text-pl-text-dim">Use a public FPL entry ID or paste a 15-player element-ID list. No FPL account credentials or squad data are stored.</p><div className="mt-4 grid gap-3 lg:grid-cols-[1fr_110px_110px_auto]"><label className="text-xs font-semibold text-pl-text-faint">Public entry ID<input value={entryId} onChange={(e) => setEntryId(e.target.value)} inputMode="numeric" placeholder="e.g. 12345" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><label className="text-xs font-semibold text-pl-text-faint">Free transfers<input value={freeTransfers} onChange={(e) => setFreeTransfers(e.target.value)} inputMode="numeric" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><div className="flex items-end"><button disabled={busy || !entryId} onClick={() => requestTransfers("entry")} className="w-full rounded-lg bg-pl-pink px-3 py-2 text-sm font-bold text-white disabled:opacity-40">Use entry</button></div></div><div className="mt-3 grid gap-3 lg:grid-cols-[1fr_110px_auto]"><label className="text-xs font-semibold text-pl-text-faint">Manual element IDs (15 comma-separated)<input value={manualIds} onChange={(e) => setManualIds(e.target.value)} placeholder="1, 2, 3, …" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><label className="text-xs font-semibold text-pl-text-faint">Bank (£m)<input value={bank} onChange={(e) => setBank(e.target.value)} inputMode="decimal" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><div className="flex items-end"><button disabled={busy || !manualIds} onClick={() => requestTransfers("manual")} className="w-full rounded-lg border border-pl-pink px-3 py-2 text-sm font-bold text-pl-pink disabled:opacity-40">Use manual squad</button></div></div>{transfers && <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="border-b border-pl-border text-pl-text-faint"><tr><th className="p-2">Out</th><th className="p-2">In</th><th className="p-2 text-right">Cost</th><th className="p-2 text-right">Projected gain</th><th className="p-2 text-right">Net gain</th></tr></thead><tbody>{transfers.recommendations.map((idea) => <tr key={`${idea.out.player_id}-${idea.in.player_id}`} className="border-b border-pl-border/60 text-pl-text-dim"><td className="p-2">{idea.out.web_name}</td><td className="p-2 font-semibold text-pl-text">{idea.in.web_name}</td><td className="p-2 text-right">{idea.cost < 0 ? `-£${Math.abs(idea.cost).toFixed(1)}m` : `£${idea.cost.toFixed(1)}m`}</td><td className="p-2 text-right">+{idea.projected_gain.toFixed(1)}</td><td className="p-2 text-right font-semibold text-pl-pink">+{idea.net_gain.toFixed(1)}</td></tr>)}</tbody></table>{transfers.recommendations.length === 0 && <p className="py-3 text-sm text-pl-text-faint">No positive like-for-like transfer is available under the supplied constraints.</p>}</div>}</section>
    )}
```

This is the full, unmodified transfer-planner section from today's
`FPLPage.tsx:112`, just wrapped in the new `{mode === "transfers" && (...)}`
block — Task 5 below edits its *inside* separately, once this task's own
change is already complete and compiling on its own.

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/FPLPage.tsx
git commit -m "feat: gate FPL squad views behind a 3-way mode toggle"
```

---

### Task 5: Transfer planner — visible count + per-row reason

**Files:**
- Modify: `frontend/src/pages/FPLPage.tsx:1` (add `Fragment` import)
- Modify: `frontend/src/pages/FPLPage.tsx` (the transfer results table, now inside the `mode === "transfers"` block Task 4 added)

**Interfaces:**
- Consumes: nothing from earlier tasks beyond the already-compiling file Task 4 left behind.
- Produces: nothing further consumed by later tasks.

- [ ] **Step 1: Import `Fragment`**

Find this line (`FPLPage.tsx:1`):

```tsx
import { useCallback, useEffect, useMemo, useState } from "react";
```

Replace it with:

```tsx
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
```

- [ ] **Step 2: Add the free-transfers/bank line and per-row reason**

Find this exact substring inside the transfer-planner section Task 4 added
(it starts right after the "Use manual squad" button's closing `</div>`):

```tsx
{transfers && <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="border-b border-pl-border text-pl-text-faint"><tr><th className="p-2">Out</th><th className="p-2">In</th><th className="p-2 text-right">Cost</th><th className="p-2 text-right">Projected gain</th><th className="p-2 text-right">Net gain</th></tr></thead><tbody>{transfers.recommendations.map((idea) => <tr key={`${idea.out.player_id}-${idea.in.player_id}`} className="border-b border-pl-border/60 text-pl-text-dim"><td className="p-2">{idea.out.web_name}</td><td className="p-2 font-semibold text-pl-text">{idea.in.web_name}</td><td className="p-2 text-right">{idea.cost < 0 ? `-£${Math.abs(idea.cost).toFixed(1)}m` : `£${idea.cost.toFixed(1)}m`}</td><td className="p-2 text-right">+{idea.projected_gain.toFixed(1)}</td><td className="p-2 text-right font-semibold text-pl-pink">+{idea.net_gain.toFixed(1)}</td></tr>)}</tbody></table>{transfers.recommendations.length === 0 && <p className="py-3 text-sm text-pl-text-faint">No positive like-for-like transfer is available under the supplied constraints.</p>}</div>}
```

Replace it with:

```tsx
{transfers && <div className="mt-4"><p className="mb-2 text-xs text-pl-text-dim">You have <span className="font-semibold text-pl-text">{transfers.free_transfers}</span> free transfer{transfers.free_transfers === 1 ? "" : "s"} · <span className="font-semibold text-pl-text">£{transfers.bank.toFixed(1)}m</span> in the bank</p><div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="border-b border-pl-border text-pl-text-faint"><tr><th className="p-2">Out</th><th className="p-2">In</th><th className="p-2 text-right">Cost</th><th className="p-2 text-right">Projected gain</th><th className="p-2 text-right">Net gain</th></tr></thead><tbody>{transfers.recommendations.map((idea) => <Fragment key={`${idea.out.player_id}-${idea.in.player_id}`}><tr className="border-b border-pl-border/20 text-pl-text-dim"><td className="p-2">{idea.out.web_name}</td><td className="p-2 font-semibold text-pl-text">{idea.in.web_name}</td><td className="p-2 text-right">{idea.cost < 0 ? `-£${Math.abs(idea.cost).toFixed(1)}m` : `£${idea.cost.toFixed(1)}m`}</td><td className="p-2 text-right">+{idea.projected_gain.toFixed(1)}</td><td className="p-2 text-right font-semibold text-pl-pink">+{idea.net_gain.toFixed(1)}</td></tr><tr className="border-b border-pl-border/60"><td colSpan={5} className="px-2 pb-2 text-[11px] text-pl-text-faint">{idea.in.drivers.slice(0, 2).join(" · ")}</td></tr></Fragment>)}</tbody></table>{transfers.recommendations.length === 0 && <p className="py-3 text-sm text-pl-text-faint">No positive like-for-like transfer is available under the supplied constraints.</p>}</div></div>}
```

(Changes: added the free-transfers/bank `<p>` before the table, inside
the same `{transfers && ...}` conditional the table already used; each
result row is now a `Fragment` containing the original row plus a new
reason row — `colSpan={5}`, `idea.in.drivers.slice(0, 2).join(" · ")`.)

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc -b`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/FPLPage.tsx
git commit -m "feat: add 3-way FPL mode toggle, visible transfer count, and per-row reasons"
```

---

### Task 6: Manual QA pass and docs update

**Files:** none for QA; modify `docs/AI_CONTINUITY.md` for the docs step.

**Interfaces:** none — final task.

- [ ] **Step 1: Confirm the dev servers are running**

Backend (`:8000`) and frontend (`:5173`) should already be running from
earlier work this session. If not, start them the same way as before.

- [ ] **Step 2: Check all three tabs**

Open the FPL tab. Confirm: it defaults to "This gameweek's XI" showing
the pitch view with `PlayerCard`s (crest, position badge, rating, name —
price omitted at this size per the spec). Click "Best £100m squad" —
confirm a 15-card grid renders with crest, position, rating, name, *and*
price (the `size="md"` card). Click "My team & transfers" — confirm the
entry-ID/manual-squad inputs are visible and the player scout table below
is unaffected by whichever tab is selected (it should always be visible,
regardless of `mode`).

- [ ] **Step 3: Run a real transfer request**

Enter a public FPL entry ID (or a manual 15-ID list) and submit. Confirm:
the "You have N free transfer(s) · £Xm in the bank" line appears above
the results table, and each recommendation row has a second, smaller line
underneath it with 1-2 driver phrases (e.g. "3 fixtures · 78 expected
minutes") explaining the incoming player.

- [ ] **Step 4: Final compile check**

Run: `cd frontend && npx tsc -b`
Expected: no errors, confirming the whole redesign compiles cleanly
end-to-end.

- [ ] **Step 5: Update `docs/AI_CONTINUITY.md`**

Add a short entry (matching this file's existing style — see its most
recent entries, e.g. `OPS-2026-04`, for tone/format) recording: `PlayerCard`
added, `PitchPlayer`/`PlayerLine` removed, the FPL page restructured into
a 3-way mode toggle (gameweek XI / £100m squad / transfers), and the
transfer planner's free-transfers/bank line and per-row driver-based
reason. If Task 2's Step 4 required any adjustment to fit the pitch-view
card's content, note what was changed and why.

- [ ] **Step 6: Commit the documentation update**

```bash
git add docs/AI_CONTINUITY.md
git commit -m "docs: record the FPL FIFA-style redesign"
```
