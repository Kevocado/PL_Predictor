const COLORS: Record<string, string> = {
  W: "bg-win text-pr-stage",
  D: "bg-draw text-pr-stage",
  L: "bg-loss text-pr-stage",
};

export function FormStrip({ results }: { results: string[] }) {
  if (results.length === 0) {
    return <span className="text-xs text-pr-text-faint">No recent matches</span>;
  }
  return (
    <div className="flex gap-1">
      {results.map((r, i) => (
        <span
          key={i}
          className={`flex h-5 w-5 items-center justify-center rounded text-xs font-bold ${COLORS[r] ?? "bg-pr-panel-2"}`}
        >
          {r}
        </span>
      ))}
    </div>
  );
}
