import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { SquadContinuityResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

// Below this, a team lost enough of last season's minutes to flag as a
// real squad-strength change worth the eye's attention (the Newcastle
// 2026-27 case that prompted EXP-2026-18 sits well under this line).
const NOTABLE_CHANGE_THRESHOLD = 0.75;

interface Entry {
  team: string;
  squad_continuity: number;
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: { payload: Entry }[] }) {
  if (!active || !payload || payload.length === 0) return null;
  const entry = payload[0].payload;
  return (
    <div className="max-w-xs rounded-lg border border-pl-border bg-pl-900 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-semibold text-pl-text">{entry.team}</div>
      <div className="text-pl-text-dim">
        <span className="font-semibold text-pl-text">{(entry.squad_continuity * 100).toFixed(0)}%</span> of last
        season's minutes retained
      </div>
    </div>
  );
}

export function SquadContinuityPanel() {
  const [data, setData] = useState<SquadContinuityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .squadContinuity()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return null; // informational panel — don't let a data gap break the calibration page
  if (!data || data.teams.length === 0) return null;

  const entries: Entry[] = [...data.teams].sort((a, b) => a.squad_continuity - b.squad_continuity);
  const height = Math.max(220, entries.length * 26);

  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5">
        <h2 className="text-lg font-semibold text-pl-text">Squad continuity this season</h2>
        <InfoTooltip text={GLOSSARY.squadContinuity} align="left" />
      </div>
      <p className="mb-3 max-w-2xl text-xs text-pl-text-faint">
        Off-season squad turnover for {data.season} — how much of each team's playing time last season is still on
        the books. Fed into the live scoreline model as its own feature (see the feature-importance charts below);
        lowest-continuity teams are the ones whose real strength this season the model is most likely still
        catching up on.
      </p>
      <div className="rounded-xl border border-pl-border bg-pl-850/70 p-3" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={entries} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 10 }}>
            <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fill: "var(--color-pl-text-faint)", fontSize: 10 }}
            />
            <YAxis
              type="category"
              dataKey="team"
              width={120}
              tick={{ fill: "var(--color-pl-text-dim)", fontSize: 10 }}
            />
            <Tooltip
              content={(props) => <ChartTooltip active={props.active} payload={props.payload as unknown as { payload: Entry }[] | undefined} />}
              cursor={{ fill: "var(--color-pl-border)", opacity: 0.15 }}
            />
            <Bar dataKey="squad_continuity" radius={[0, 4, 4, 0]}>
              {entries.map((entry) => (
                <Cell
                  key={entry.team}
                  fill={entry.squad_continuity < NOTABLE_CHANGE_THRESHOLD ? "var(--color-loss)" : "var(--color-pl-pink)"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex items-center gap-4 text-[10px] text-pl-text-faint">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm" style={{ background: "var(--color-loss)" }} /> notable turnover (&lt;{Math.round(NOTABLE_CHANGE_THRESHOLD * 100)}% retained)
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm" style={{ background: "var(--color-pl-pink)" }} /> squad largely unchanged
        </span>
      </div>
    </div>
  );
}
