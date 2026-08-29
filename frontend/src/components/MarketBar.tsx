import type { ReactNode } from "react";

interface MarketBarProps {
  label: ReactNode;
  prob: number;
  marketProb?: number | null;
  highlight?: "flagged" | "hit" | "miss";
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
  flagged: "bg-pl-pink/15 ring-2 ring-pl-pink/70",
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
    <div className={`rounded-lg px-3 py-2 text-sm ${highlight ? HIGHLIGHT_CLASSES[highlight] : "bg-pl-850/60"}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-1.5 text-pl-text-dim">
          {label}
          {valueBet && (
            <span className="rounded bg-pl-pink/20 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-pl-pink">
              Value
            </span>
          )}
        </span>
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
