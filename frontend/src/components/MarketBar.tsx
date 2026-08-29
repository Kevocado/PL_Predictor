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
