const COLORS: Record<string, string> = {
  W: "bg-win text-white",
  D: "bg-draw text-pl-950",
  L: "bg-loss text-white",
};

export function FormStrip({ results }: { results: string[] }) {
  if (results.length === 0) {
    return <span className="text-[11px] text-pl-text-faint">No recent matches</span>;
  }
  return (
    <div className="flex gap-1">
      {results.map((r, i) => (
        <span
          key={i}
          className={`flex h-5 w-5 items-center justify-center rounded text-[10px] font-bold ${COLORS[r] ?? "bg-pl-700"}`}
        >
          {r}
        </span>
      ))}
    </div>
  );
}
