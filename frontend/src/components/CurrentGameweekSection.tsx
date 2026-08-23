import type { CurrentGameweekResponse } from "../types";
import { CurrentGameweekCard } from "./CurrentGameweekCard";

interface Props {
  data: CurrentGameweekResponse;
  onSelect: (eventId: string) => void;
  onNavigate: (gameweek: number) => void;
}

export function CurrentGameweekSection({ data, onSelect, onNavigate }: Props) {
  const nCompleted = data.fixtures.filter((f) => f.finished).length;
  const nHit = data.fixtures.filter((f) => f.finished && f.hit).length;

  const canGoPrev = data.gameweek !== null && data.min_gameweek !== null && data.gameweek > data.min_gameweek;
  const canGoNext = data.gameweek !== null && data.max_gameweek !== null && data.gameweek < data.max_gameweek;

  return (
    <div className="mb-8">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            disabled={!canGoPrev}
            onClick={() => data.gameweek !== null && onNavigate(data.gameweek - 1)}
            aria-label="Previous gameweek"
            className="rounded-lg border border-pl-border bg-pl-850/70 px-2.5 py-1.5 text-sm text-pl-text-dim transition hover:text-pl-text disabled:cursor-not-allowed disabled:opacity-30"
          >
            ←
          </button>
          <h2 className="flex items-baseline gap-2 text-lg font-semibold text-pl-text">
            {data.gameweek !== null ? `Gameweek ${data.gameweek}` : "No gameweek data yet"}
            {!data.is_current && (
              <span className="rounded bg-pl-700/60 px-1.5 py-0.5 text-xs font-normal text-pl-text-dim">
                not current
              </span>
            )}
          </h2>
          <button
            disabled={!canGoNext}
            onClick={() => data.gameweek !== null && onNavigate(data.gameweek + 1)}
            aria-label="Next gameweek"
            className="rounded-lg border border-pl-border bg-pl-850/70 px-2.5 py-1.5 text-sm text-pl-text-dim transition hover:text-pl-text disabled:cursor-not-allowed disabled:opacity-30"
          >
            →
          </button>
        </div>
        <span className="text-xs font-normal text-pl-text-faint">
          {data.fixtures.length} fixture{data.fixtures.length === 1 ? "" : "s"}
          {nCompleted > 0 && ` · ${nHit}/${nCompleted} called correctly so far`}
        </span>
      </div>

      {data.fixtures.length === 0 ? (
        <div className="py-16 text-center text-pl-text-faint">No fixtures found for this gameweek.</div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {data.fixtures.map((f) => (
            <CurrentGameweekCard
              key={f.event_id}
              fixture={f}
              onClick={() => onSelect(f.event_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
