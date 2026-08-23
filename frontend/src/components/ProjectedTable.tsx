import type { ProjectedTableResponse } from "../types";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

const ZONE_STYLE: Record<string, string> = {
  title: "border-l-2 border-l-win",
  europe: "border-l-2 border-l-pl-cyan",
  relegation: "border-l-2 border-l-loss",
};

function zoneFor(position: number): string | null {
  if (position === 1) return "title";
  if (position <= 5) return "europe";
  if (position >= 18) return "relegation";
  return null;
}

function ordinal(n: number): string {
  const rem100 = n % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1:
      return `${n}st`;
    case 2:
      return `${n}nd`;
    case 3:
      return `${n}rd`;
    default:
      return `${n}th`;
  }
}

function PositionDelta({ position, delta }: { position: number | null; delta: number | null }) {
  if (position === null || delta === null) {
    return <span className="text-pl-text-faint">—</span>;
  }
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-pl-text-dim">{ordinal(position)}</span>
      {delta === 0 ? (
        <span className="text-pl-text-faint">on track</span>
      ) : (
        <span className={`inline-flex items-center gap-0.5 font-semibold ${delta > 0 ? "text-win" : "text-loss"}`}>
          {delta > 0 ? "▲" : "▼"} {Math.abs(delta)}
        </span>
      )}
    </span>
  );
}

export function ProjectedTable({ data }: { data: ProjectedTableResponse }) {
  return (
    <div>
      <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
        Projected final table — {data.season}
        <InfoTooltip text={GLOSSARY.projectedTable} align="left" />
      </h3>
      <div className="overflow-x-auto rounded-xl border border-pl-border">
        <table className="w-full min-w-[480px] text-sm">
          <thead>
            <tr className="border-b border-pl-border bg-pl-850/70 text-left text-[11px] uppercase tracking-wide text-pl-text-faint">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">Team</th>
              <th className="px-3 py-2 text-right">Played</th>
              <th className="px-3 py-2 text-right">Pts</th>
              <th className="px-3 py-2 text-right">Proj. Pts</th>
              <th className="px-3 py-2 text-right">Proj. GD</th>
              <th className="px-3 py-2 text-right">
                <span className="inline-flex items-center gap-1">
                  vs. projection
                  <InfoTooltip
                    text="Their real current league position, plus how many places that differs from the model's end-of-season projection for them — a green ▲ means they're currently sitting better than the model expects them to finish, a red ▼ means worse. Blank until they've played their first match this season."
                    align="left"
                  />
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {data.table.map((row) => {
              const zone = zoneFor(row.projected_position);
              return (
                <tr
                  key={row.team}
                  className={`border-b border-pl-border/60 bg-pl-850/40 ${zone ? ZONE_STYLE[zone] : ""}`}
                >
                  <td className="px-3 py-2 text-pl-text-faint">{row.projected_position}</td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <TeamBadge team={row.team} size="sm" />
                      <span className="text-pl-text">{row.team}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right text-pl-text-dim">{row.played}</td>
                  <td className="px-3 py-2 text-right text-pl-text-dim">{row.current_points}</td>
                  <td className="px-3 py-2 text-right font-semibold text-pl-text">{row.projected_points}</td>
                  <td className="px-3 py-2 text-right text-pl-text-dim">
                    {row.projected_goal_diff > 0 ? "+" : ""}
                    {row.projected_goal_diff}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <PositionDelta position={row.current_position} delta={row.position_delta} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex gap-4 text-[11px] text-pl-text-faint">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-win" /> Title
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-pl-cyan" /> Europe
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-loss" /> Relegation
        </span>
      </div>
    </div>
  );
}
