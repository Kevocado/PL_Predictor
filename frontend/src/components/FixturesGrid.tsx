import type { FixtureSummary } from "../types";
import { FixtureCard } from "./FixtureCard";

interface Props {
  fixtures: FixtureSummary[];
  onSelect: (eventId: string) => void;
}

export function FixturesGrid({ fixtures, onSelect }: Props) {
  if (fixtures.length === 0) {
    return <div className="py-16 text-center text-pl-text-faint">No fixtures match the current filters.</div>;
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {fixtures.map((f) => (
        <FixtureCard key={f.event_id} fixture={f} onClick={() => onSelect(f.event_id)} />
      ))}
    </div>
  );
}
