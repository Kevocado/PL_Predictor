import type { GameweekResult } from "../types";
import { FinishedFixtureCard } from "./FinishedFixtureCard";

export function GameweekResultsGrid({ results, onSelect }: { results: GameweekResult[]; onSelect: (eventId: string) => void }) {
  if (results.length === 0) {
    return (
      <p className="text-xs text-pl-text-faint">
        No resolved fixtures yet — this fills in automatically as matches kick off and finish.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {results.map((r) => (
        <FinishedFixtureCard key={r.event_id} {...r} onClick={() => onSelect(r.event_id)} />
      ))}
    </div>
  );
}
