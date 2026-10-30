import type { ReactNode } from "react";

interface MarketBarProps {
  label: ReactNode;
  prob: number;
  marketProb?: number | null;
  highlight?: "flagged" | "hit" | "miss" | "modelCall";
  // Distinct from `highlight` on purpose: `highlight="flagged"` is also
  // reused by the scoreline list for "most likely" (a different meaning,
  // same ring styling) — the literal "Value" badge must only ever appear
  // for an actual value-bet flag, never piggyback on that shared ring.
  valueBet?: boolean;
  detail?: ReactNode;
}

const HIGHLIGHT_CLASSES: Record<string, string> = {
  hit: "bg-win/10 ring-1 ring-win/30",
  miss: "bg-loss/10",
  // Bolder than hit/miss on purpose: this is the one row a glance should
  // land on. A thin same-weight ring made the old fold-in of the "Best
  // value bet" callout too easy to miss in a list of 5-7 rows — this is
  // deliberately louder, closer to the standalone box's visual weight.
  flagged: "bg-pr-accent/15 ring-2 ring-pr-accent/70",
  // Deliberately not the pink `flagged` treatment -- that specifically
  // means "there is a live market and the model beats it." This means
  // "the model has a decisive view, no market to compare against."
  modelCall: "bg-pr-lean/10 ring-1 ring-pr-lean/40",
};

export function MarketBar({ label, prob, marketProb, highlight, valueBet, detail }: MarketBarProps) {
  const pct = Math.round(prob * 100);
  const marketPct = marketProb !== undefined && marketProb !== null ? Math.round(marketProb * 100) : null;
  // Computed from the raw fractional probabilities, not from the two
  // already-rounded whole-percent values — matches the backend's own
  // `edge = prob - implied` exactly rather than introducing up to 1 point
  // of rounding error from rounding each side before subtracting.
  const delta =
    marketProb !== undefined && marketProb !== null ? Math.round((prob - marketProb) * 100) : null;

  return (
    <div className={`rounded-lg px-3 py-2 text-sm ${highlight ? HIGHLIGHT_CLASSES[highlight] : "bg-pr-panel/60"}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-1.5 text-pr-text-dim">
          {label}
          {valueBet && (
            <span className="rounded bg-pr-accent/20 px-1.5 py-0.5 text-xs font-semibold uppercase tracking-wide text-pr-accent">
              Value
            </span>
          )}
        </span>
        <div className="flex items-center gap-2 font-mono">
          {delta !== null && (
            <span className={`text-xs font-semibold ${delta > 0 ? "text-pr-lean" : "text-pr-text-faint"}`}>
              {delta > 0 ? "+" : ""}
              {delta}%
            </span>
          )}
          <span className="font-semibold text-pr-text">{pct}%</span>
        </div>
      </div>
      <div className="relative mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-pr-rule/60">
        <div className="absolute inset-y-0 left-0 rounded-full bg-pr-accent" style={{ width: `${pct}%` }} />
        {marketPct !== null && (
          <div className="absolute -inset-y-0.5 w-0.5 bg-pr-text-faint" style={{ left: `${marketPct}%` }} />
        )}
      </div>
      {detail && <p className="mt-1.5 text-xs text-pr-text-dim">{detail}</p>}
    </div>
  );
}
