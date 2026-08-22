import type { FixtureSummary } from "../types";
import { TeamBadge } from "./TeamBadge";
import { ProbabilityBar } from "./ProbabilityBar";
import { ValueBetPill } from "./ValueBetPill";

interface Props {
  fixture: FixtureSummary;
  onClick: () => void;
}

function bestEdge(f: FixtureSummary): number | undefined {
  const edges = [f.home_win.edge, f.draw.edge, f.away_win.edge, f.over_2_5.edge, f.under_2_5.edge].filter(
    (e): e is number => e !== null && e > 0
  );
  return edges.length ? Math.max(...edges) : undefined;
}

function formatKickoff(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }),
    time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
  };
}

export function FixtureCard({ fixture, onClick }: Props) {
  const { date, time } = formatKickoff(fixture.commence_time);

  return (
    <button
      onClick={onClick}
      className="clip-corner group relative flex flex-col gap-3 rounded-xl border border-pl-border bg-pl-850/70 p-4 text-left backdrop-blur transition hover:-translate-y-0.5 hover:border-pl-pink/50 hover:bg-pl-800 hover:shadow-lg hover:shadow-pl-pink/10"
    >
      <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-pl-text-faint">
        <span>
          {date} &middot; {time}
        </span>
        {fixture.data_confidence === "new" && (
          <span
            className="rounded bg-pl-700/60 px-1.5 py-0.5 text-pl-text-dim"
            title="One side has little-to-no match history the model has seen yet — see the fixture detail for more"
          >
            new team
          </span>
        )}
        {fixture.data_confidence === "limited" && (
          <span
            className="rounded bg-pl-700/60 px-1.5 py-0.5 text-pl-text-dim"
            title="One side only has a few recent matches on record so far — see the fixture detail for more"
          >
            limited data
          </span>
        )}
      </div>

      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={fixture.team_home} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{fixture.team_home}</span>
        </div>
        <div className="flex flex-col items-center gap-0.5 px-1">
          <span className="text-[10px] font-medium uppercase text-pl-text-faint">Likely</span>
          <span className="rounded-lg bg-pl-700/50 px-2 py-1 font-mono text-base font-bold text-pl-text">
            {fixture.top_scoreline}
          </span>
        </div>
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={fixture.team_away} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{fixture.team_away}</span>
        </div>
      </div>

      <ProbabilityBar home={fixture.home_win.prob} draw={fixture.draw.prob} away={fixture.away_win.prob} />

      <div className="flex items-center justify-between gap-2 border-t border-pl-border/70 pt-2.5 text-[11px] text-pl-text-dim">
        <span>
          BTTS <span className="font-semibold text-pl-text">{(fixture.btts_yes_prob * 100).toFixed(0)}%</span>
        </span>
        <span>
          O2.5 <span className="font-semibold text-pl-text">{(fixture.over_2_5.prob * 100).toFixed(0)}%</span>
        </span>
        <ValueBetPill flags={fixture.value_bet_flags} bestEdge={bestEdge(fixture)} />
      </div>
    </button>
  );
}
