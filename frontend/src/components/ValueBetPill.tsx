const LABELS: Record<string, string> = {
  home_win: "Home",
  draw: "Draw",
  away_win: "Away",
  over_2_5: "O2.5",
  under_2_5: "U2.5",
};

interface Props {
  flags: string[];
  bestEdge?: number;
}

export function ValueBetPill({ flags, bestEdge }: Props) {
  if (flags.length === 0) return null;

  return (
    <div className="flex items-center gap-1 rounded-full bg-pl-pink/15 px-2.5 py-1 text-[11px] font-semibold text-pl-pink-soft ring-1 ring-pl-pink/40">
      <span className="h-1.5 w-1.5 rounded-full bg-pl-pink-soft" />
      {flags.map((f) => LABELS[f] ?? f).join(", ")}
      {bestEdge !== undefined && <span className="text-pl-cyan">+{(bestEdge * 100).toFixed(1)}%</span>}
    </div>
  );
}
