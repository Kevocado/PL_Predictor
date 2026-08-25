import { useMemo, useState } from "react";
import type { PlayerHubPlayer, PlayerHubResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

type SortKey = keyof Pick<PlayerHubPlayer, "name" | "team" | "position" | "minutes" | "starts" | "goals" | "assists" | "xg" | "xa" | "xgi" | "threat" | "creativity" | "ict" | "bps" | "bonus">;

const COLUMNS: Array<{ key: SortKey; label: string; tooltip?: string; numeric?: boolean }> = [
  { key: "name", label: "Player" },
  { key: "team", label: "Team" },
  { key: "position", label: "Pos" },
  { key: "starts", label: "Starts", numeric: true },
  { key: "minutes", label: "Min", tooltip: GLOSSARY.playerMinutes, numeric: true },
  { key: "goals", label: "Goals", numeric: true },
  { key: "assists", label: "Assists", numeric: true },
  { key: "xg", label: "xG", tooltip: GLOSSARY.playerXg, numeric: true },
  { key: "xa", label: "xA", tooltip: GLOSSARY.playerXa, numeric: true },
  { key: "xgi", label: "xGI", tooltip: GLOSSARY.playerXgi, numeric: true },
  { key: "threat", label: "Threat", tooltip: GLOSSARY.playerThreat, numeric: true },
  { key: "creativity", label: "Creativity", tooltip: GLOSSARY.playerCreativity, numeric: true },
  { key: "ict", label: "ICT", tooltip: GLOSSARY.playerIct, numeric: true },
  { key: "bps", label: "BPS", tooltip: GLOSSARY.playerBps, numeric: true },
  { key: "bonus", label: "Bonus", tooltip: GLOSSARY.playerBonus, numeric: true },
];

function display(value: number | string | null) {
  if (value === null) return "—";
  return typeof value === "number" && !Number.isInteger(value) ? value.toFixed(2) : String(value);
}

export function PlayerHub({ data }: { data: PlayerHubResponse }) {
  const [sortKey, setSortKey] = useState<SortKey>("xgi");
  const [descending, setDescending] = useState(true);
  const [query, setQuery] = useState("");
  const players = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return [...data.players]
      .filter((player) => !normalizedQuery || `${player.name} ${player.team} ${player.position}`.toLowerCase().includes(normalizedQuery))
      .sort((left, right) => {
        const leftValue = left[sortKey] ?? (typeof left[sortKey] === "string" ? "" : -Infinity);
        const rightValue = right[sortKey] ?? (typeof right[sortKey] === "string" ? "" : -Infinity);
        const comparison = typeof leftValue === "string" && typeof rightValue === "string"
          ? leftValue.localeCompare(rightValue)
          : Number(leftValue) - Number(rightValue);
        return descending ? -comparison : comparison;
      });
  }, [data.players, descending, query, sortKey]);

  const selectSort = (key: SortKey) => {
    if (key === sortKey) setDescending((value) => !value);
    else {
      setSortKey(key);
      setDescending(key !== "name" && key !== "team" && key !== "position");
    }
  };

  return (
    <section className="flex flex-col gap-6">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-pl-text">Player form</h3>
          <p className="mt-1 text-sm text-pl-text-dim">Season-to-date FPL performance. Sort any column to compare players on the measure that matters to you.</p>
        </div>
        <label className="text-xs font-semibold text-pl-text-faint">
          Search players
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name, team, position" className="mt-1 block rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm font-normal text-pl-text outline-none placeholder:text-pl-text-faint focus:border-pl-pink" />
        </label>
      </div>
      <div className="overflow-x-auto rounded-xl border border-pl-border">
        <table className="w-full min-w-[1260px] text-left text-xs">
          <thead className="bg-pl-850 text-[10px] uppercase tracking-wide text-pl-text-faint">
            <tr>{COLUMNS.map((column) => <th key={column.key} className={`whitespace-nowrap px-2 py-2 ${column.numeric ? "text-right" : ""}`}><button onClick={() => selectSort(column.key)} className="inline-flex items-center gap-1 hover:text-pl-text">{column.label}{column.tooltip && <InfoTooltip text={column.tooltip} align="left" />}{sortKey === column.key && <span>{descending ? "↓" : "↑"}</span>}</button></th>)}</tr>
          </thead>
          <tbody>{players.map((player) => <tr key={player.id} className="border-t border-pl-border/70 text-pl-text-dim hover:bg-pl-850/50"><td className="px-2 py-2 font-semibold text-pl-text">{player.name}</td><td className="px-2 py-2">{player.team}</td><td className="px-2 py-2">{player.position}</td><td className="px-2 py-2 text-right">{player.starts}</td><td className="px-2 py-2 text-right">{player.minutes}</td><td className="px-2 py-2 text-right">{player.goals}</td><td className="px-2 py-2 text-right">{player.assists}</td><td className="px-2 py-2 text-right">{display(player.xg)}</td><td className="px-2 py-2 text-right">{display(player.xa)}</td><td className="px-2 py-2 text-right font-semibold text-pl-text">{display(player.xgi)}</td><td className="px-2 py-2 text-right">{display(player.threat)}</td><td className="px-2 py-2 text-right">{display(player.creativity)}</td><td className="px-2 py-2 text-right">{display(player.ict)}</td><td className="px-2 py-2 text-right">{player.bps}</td><td className="px-2 py-2 text-right">{player.bonus}</td></tr>)}</tbody>
        </table>
      </div>
      {players.length === 0 && <p className="mt-3 text-sm text-pl-text-faint">No player matches that search.</p>}
    </section>
  );
}
