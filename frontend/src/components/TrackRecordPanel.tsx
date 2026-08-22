import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TrackRecordResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { MissesTable } from "./MissesTable";
import { GLOSSARY } from "../lib/glossary";

function StatCard({ label, value, info }: { label: string; value: string; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className="mt-1 text-2xl font-bold text-pl-text">{value}</div>
    </div>
  );
}

export function TrackRecordPanel({ data }: { data: TrackRecordResponse }) {
  const { summary, biggest_misses } = data;

  if (summary.n_resolved === 0) {
    return (
      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          Live track record
          <InfoTooltip text={GLOSSARY.trackRecordScore} align="left" />
        </h3>
        <p className="text-xs text-pl-text-faint">
          No resolved predictions yet — this builds up automatically as fixtures kick off and results come in.
          Predictions are snapshotted the moment a fixture first appears in the app, before it's played.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Predictions resolved" value={String(summary.n_resolved)} />
        <StatCard label="Fixtures covered" value={String(summary.n_fixtures ?? "—")} />
        <StatCard label="RPS (result market)" value={summary.rps !== null ? summary.rps.toFixed(4) : "—"} info={GLOSSARY.trackRecordScore} />
        <StatCard label="Brier (goals/BTTS)" value={summary.brier !== null ? summary.brier.toFixed(4) : "—"} info={GLOSSARY.trackRecordScore} />
      </div>

      {summary.weekly_trend.length > 1 && (
        <div className="clip-corner-lg h-64 rounded-xl border border-pl-border bg-pl-850/70 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={summary.weekly_trend} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-pl-border)" strokeDasharray="3 3" />
              <XAxis dataKey="week" tick={{ fill: "var(--color-pl-text-faint)", fontSize: 10 }} />
              <YAxis tick={{ fill: "var(--color-pl-text-faint)", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "var(--color-pl-900)", border: "1px solid var(--color-pl-border)", borderRadius: 8 }}
                labelStyle={{ color: "var(--color-pl-text-faint)" }}
              />
              <Line type="monotone" dataKey="mean_squared_error" stroke="var(--color-pl-pink)" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
          Biggest misses
          <InfoTooltip text={GLOSSARY.biggestMisses} align="left" />
        </h3>
        <MissesTable misses={biggest_misses} />
      </div>
    </div>
  );
}
