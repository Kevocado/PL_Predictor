import type { ReactNode } from "react";
import type { CalibrationResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

function StatCard({ label, value, sub, info }: { label: string; value: string; sub?: string; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-pl-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className="mt-1 text-2xl font-bold text-pl-text">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-pl-text-faint">{sub}</div>}
    </div>
  );
}

function Row({
  label,
  info,
  model,
  bookmaker,
  naive,
}: {
  label: ReactNode;
  info?: string;
  model: number;
  bookmaker?: number | null;
  naive: number;
}) {
  const max = Math.max(model, bookmaker ?? 0, naive);
  const bar = (v: number, color: string) => (
    <div className="flex items-center gap-2">
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-pl-850">
        <div className="h-full rounded-full" style={{ width: `${(v / max) * 100}%`, background: color }} />
      </div>
      <span className="w-14 text-right text-xs font-semibold text-pl-text">{v.toFixed(4)}</span>
    </div>
  );

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <span className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
        {label} (lower is better)
        {info && <InfoTooltip text={info} align="left" />}
      </span>
      <div className="grid grid-cols-[70px_1fr] items-center gap-x-3 gap-y-1.5 text-xs text-pl-text-dim">
        <span>Model</span>
        {bar(model, "var(--color-pl-pink)")}
        <span>Bookmaker</span>
        {bookmaker != null ? bar(bookmaker, "var(--color-pl-cyan)") : <span className="text-pl-text-faint">n/a</span>}
        <span>Naive</span>
        {bar(naive, "var(--color-pl-text-faint)")}
      </div>
    </div>
  );
}

export function CalibrationPanel({ data }: { data: CalibrationResponse }) {
  return (
    <div>
      <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Held-out season" value={data.season ?? "—"} />
        <StatCard label="Matches evaluated" value={String(data.model.n_matches)} />
        <StatCard label="Model RPS" value={data.model.rps.toFixed(4)} info={GLOSSARY.rps} />
        <StatCard label="Model Brier" value={data.model.brier.toFixed(4)} info={GLOSSARY.brier} />
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Row label="RPS" info={GLOSSARY.rps} model={data.model.rps} bookmaker={data.bookmaker?.rps} naive={data.naive.rps} />
        <Row
          label="Brier score"
          info={GLOSSARY.brier}
          model={data.model.brier}
          bookmaker={data.bookmaker?.brier}
          naive={data.naive.brier}
        />
      </div>
    </div>
  );
}
