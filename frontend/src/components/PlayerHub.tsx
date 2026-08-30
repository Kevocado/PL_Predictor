import { Fragment, useMemo, useState } from "react";
import type { PlayerHubPlayer, PlayerHubResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

type SortKey = keyof Pick<PlayerHubPlayer, "name" | "team" | "position" | "overall_rating" | "quality_rating" | "form_rating" | "live_form_rating" | "live_form_vs_quality" | "current_impact_rating" | "minutes" | "starts" | "goals" | "assists" | "xg" | "xa" | "xgi" | "threat" | "creativity" | "ict" | "bps" | "bonus">;

const COLUMNS: Array<{ key: SortKey; label: string; tooltip?: string; numeric?: boolean }> = [
  { key: "name", label: "Player" },
  { key: "team", label: "Team" },
  { key: "position", label: "Pos" },
  { key: "overall_rating", label: "Overall", tooltip: "Overall combines multi-season observed, role-aware Quality with a capped current Form lift. It never uses FPL point projections or match predictions; players with limited evidence are labelled rather than ranked.", numeric: true },
  { key: "quality_rating", label: "Quality", tooltip: "Quality is durable, multi-season role evidence on a common 0–95 scale. 85+ is rare sustained elite performance; 70–84 is established good Premier League performance.", numeric: true },
  { key: "live_form_rating", label: "Live form", tooltip: "Live Form scales the official FPL recent-form signal against the player's role, while capping early-season scores until minutes and starts provide enough evidence.", numeric: true },
  { key: "live_form_vs_quality", label: "vs Quality", tooltip: "Live Form minus the player's underlying Quality. Positive numbers identify players currently outperforming their established level; negative numbers indicate cooler current form.", numeric: true },
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

const PAGE_SIZE = 20;
const POSITION_ORDER: Record<string, number> = { GK: 0, DEF: 1, MID: 2, FWD: 3 };
type GroupBy = "none" | "team" | "position";

function RatingCell({ value, tone = "text-pl-text", signed = false }: { value: number | null | undefined; tone?: string; signed?: boolean }) {
  if (value === null || value === undefined) {
    return <td className="px-1.5 py-2 text-right"><span className="inline-flex min-w-12 justify-center rounded-md border border-pl-border bg-pl-900/70 px-2 py-1 font-bold tabular-nums text-pl-text-faint">—</span></td>;
  }
  return (
    <td className="px-1.5 py-2 text-right">
      <span className={`inline-flex min-w-12 justify-center rounded-md border border-pl-border bg-pl-900/70 px-2 py-1 font-bold tabular-nums ${tone}`}>
        {signed && value > 0 ? "+" : ""}{value.toFixed(0)}
      </span>
    </td>
  );
}

function OverallCell({ player }: { player: PlayerHubPlayer }) {
  if (player.rating_status !== "established") {
    const label = player.rating_status === "limited" ? "Limited evidence" : "Provisional";
    return <td className="px-1.5 py-2 text-right"><span className="inline-flex rounded-md border border-pl-pink/50 bg-pl-pink/10 px-2 py-1 text-[10px] font-bold text-pl-pink">{label}</span></td>;
  }
  return <RatingCell value={player.overall_rating} tone="text-pl-pink" />;
}

export function PlayerHub({ data }: { data: PlayerHubResponse }) {
  const [sortKey, setSortKey] = useState<SortKey>("overall_rating");
  const [descending, setDescending] = useState(true);
  const [query, setQuery] = useState("");
  const [positionFilter, setPositionFilter] = useState("ALL");
  const [teamFilter, setTeamFilter] = useState("ALL");
  const [availabilityFilter, setAvailabilityFilter] = useState("ALL");
  const [minimumOverall, setMinimumOverall] = useState("");
  const [groupBy, setGroupBy] = useState<GroupBy>("none");
  const [page, setPage] = useState(1);
  const teams = useMemo(() => [...new Set(data.players.map((player) => player.team))].sort(), [data.players]);
  const players = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return [...data.players]
      .filter((player) =>
        (!normalizedQuery || `${player.name} ${player.team} ${player.position}`.toLowerCase().includes(normalizedQuery))
        && (positionFilter === "ALL" || player.position === positionFilter)
        && (teamFilter === "ALL" || player.team === teamFilter)
        && (availabilityFilter === "ALL" || (availabilityFilter === "READY" ? player.status === "a" : player.status !== "a"))
        && (!minimumOverall || (player.overall_rating !== null && player.overall_rating >= Number(minimumOverall))),
      )
      .sort((left, right) => {
        if (groupBy === "team" && left.team !== right.team) return left.team.localeCompare(right.team);
        if (groupBy === "position" && left.position !== right.position) return (POSITION_ORDER[left.position] ?? 9) - (POSITION_ORDER[right.position] ?? 9);
        const leftValue = left[sortKey] ?? (typeof left[sortKey] === "string" ? "" : -Infinity);
        const rightValue = right[sortKey] ?? (typeof right[sortKey] === "string" ? "" : -Infinity);
        const comparison = typeof leftValue === "string" && typeof rightValue === "string"
          ? leftValue.localeCompare(rightValue)
          : Number(leftValue) - Number(rightValue);
        return descending ? -comparison : comparison;
      });
  }, [availabilityFilter, data.players, descending, groupBy, minimumOverall, positionFilter, query, sortKey, teamFilter]);
  const pageCount = Math.max(1, Math.ceil(players.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount);
  const visiblePlayers = players.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const selectSort = (key: SortKey) => {
    if (key === sortKey) setDescending((value) => !value);
    else {
      setSortKey(key);
      setDescending(key !== "name" && key !== "team" && key !== "position");
    }
    setPage(1);
  };
  const updateFilter = (setValue: (value: string) => void, value: string) => { setValue(value); setPage(1); };

  return (
    <section className="flex flex-col gap-6">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-pl-text">Player ratings &amp; form</h3>
          <p className="mt-1 text-sm text-pl-text-dim">Overall is durable Quality plus a capped current-form lift. Read Quality beside it, then Live Form and vs Quality to see who is outperforming their usual level. Provisional means fewer than 900 recent Premier League minutes; Limited evidence means only one qualifying season. Neither is rankable yet. {(data.rating_model_source ?? "season form").replaceAll("_", " ")} · {data.data_freshness ?? "cached FPL data"}</p>
        </div>
        <label className="text-xs font-semibold text-pl-text-faint">
          Search players
          <input value={query} onChange={(event) => updateFilter(setQuery, event.target.value)} placeholder="Name, team, position" className="mt-1 block rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm font-normal text-pl-text outline-none placeholder:text-pl-text-faint focus:border-pl-pink" />
        </label>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {Object.entries(data.leaderboards ?? {}).map(([position, leaders]) => <div key={position} className="rounded-xl border border-pl-border bg-pl-850/40 p-3"><p className="flex items-center gap-1 text-[10px] font-bold tracking-wide text-pl-text-faint">TOP {position} FORM<InfoTooltip text="Sorted by Live Form, not Overall. Players with Provisional or Limited evidence can still feature here because this panel reports what they are doing now, but they do not receive an Overall rank until their role evidence is sustained." align="left" /></p>{leaders.map((player) => <div key={player.id} className="mt-2 flex items-center justify-between text-xs"><span className="min-w-0 truncate font-semibold text-pl-text">{player.name}</span><span className="shrink-0 text-pl-text-dim">{player.rating_status !== "established" ? <span className="text-pl-pink">{player.rating_status === "limited" ? "Limited" : "Provisional"}</span> : <>Q {(player.quality_rating ?? 0).toFixed(0)} · </>}<span className="text-pl-pink">{(player.live_form_rating ?? 0).toFixed(0)}</span></span></div>)}</div>)}
      </div>
      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-pl-border bg-pl-850/40 p-3 text-xs text-pl-text-dim">
        <div className="flex gap-1 rounded-lg border border-pl-border p-1">
          {["ALL", "GK", "DEF", "MID", "FWD"].map((position) => <button key={position} onClick={() => updateFilter(setPositionFilter, position)} className={`rounded-md px-2 py-1.5 font-semibold ${positionFilter === position ? "bg-pl-pink text-white" : "hover:text-pl-text"}`}>{position === "ALL" ? "All positions" : position}</button>)}
        </div>
        <label>Team <select value={teamFilter} onChange={(event) => updateFilter(setTeamFilter, event.target.value)} className="ml-1 rounded border border-pl-border bg-pl-900 px-2 py-1.5 text-pl-text"><option value="ALL">All</option>{teams.map((team) => <option key={team} value={team}>{team}</option>)}</select></label>
        <label>Status <select value={availabilityFilter} onChange={(event) => updateFilter(setAvailabilityFilter, event.target.value)} className="ml-1 rounded border border-pl-border bg-pl-900 px-2 py-1.5 text-pl-text"><option value="ALL">All</option><option value="READY">Available</option><option value="FLAGGED">Flagged</option></select></label>
        <label>Min overall <input value={minimumOverall} onChange={(event) => updateFilter(setMinimumOverall, event.target.value)} inputMode="numeric" placeholder="Any" className="ml-1 w-14 rounded border border-pl-border bg-pl-900 px-2 py-1.5 text-pl-text" /></label>
        <div className="ml-auto flex gap-1 rounded-lg border border-pl-border p-1"><span className="px-1 py-1.5 text-pl-text-faint">Group</span>{(["none", "team", "position"] as GroupBy[]).map((mode) => <button key={mode} onClick={() => { setGroupBy(mode); setPage(1); }} className={`rounded-md px-2 py-1.5 font-semibold capitalize ${groupBy === mode ? "bg-pl-pink text-white" : "hover:text-pl-text"}`}>{mode}</button>)}</div>
      </div>
      <div className="overflow-x-auto rounded-xl border border-pl-border">
        <table className="w-full min-w-[1220px] text-left text-xs">
          <thead className="bg-pl-850 text-[10px] uppercase tracking-wide text-pl-text-faint">
            <tr>{COLUMNS.map((column) => <th key={column.key} className={`whitespace-nowrap px-2 py-2 ${column.numeric ? "text-right" : ""}`}><button onClick={() => selectSort(column.key)} className="inline-flex items-center gap-1 hover:text-pl-text">{column.label}{column.tooltip && <InfoTooltip text={column.tooltip} align="left" />}{sortKey === column.key && <span>{descending ? "↓" : "↑"}</span>}</button></th>)}</tr>
          </thead>
          <tbody>{visiblePlayers.map((player, index) => <Fragment key={player.id}>{groupBy !== "none" && (index === 0 || (groupBy === "team" ? player.team !== visiblePlayers[index - 1].team : player.position !== visiblePlayers[index - 1].position)) && <tr className="border-t border-pl-border bg-pl-900/80"><td colSpan={COLUMNS.length} className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-pl-pink">{groupBy === "team" ? player.team : player.position}</td></tr>}<tr className="border-t border-pl-border/70 text-pl-text-dim hover:bg-pl-850/50"><td className="px-2 py-2 font-semibold text-pl-text">{player.name}<span className="ml-1 text-[10px] font-normal text-pl-text-faint">{player.rating_driver ?? ""}</span></td><td className="px-2 py-2">{player.team}</td><td className="px-2 py-2">{player.position}</td><OverallCell player={player} /><RatingCell value={player.quality_rating} /><RatingCell value={player.live_form_rating} /><RatingCell value={player.live_form_vs_quality} signed tone={(player.live_form_vs_quality ?? 0) > 0 ? "text-win" : "text-pl-text-faint"} /><td className="px-2 py-2 text-right">{player.starts}</td><td className="px-2 py-2 text-right">{player.minutes}</td><td className="px-2 py-2 text-right">{player.goals}</td><td className="px-2 py-2 text-right">{player.assists}</td><td className="px-2 py-2 text-right">{display(player.xg)}</td><td className="px-2 py-2 text-right">{display(player.xa)}</td><td className="px-2 py-2 text-right font-semibold text-pl-text">{display(player.xgi)}</td><td className="px-2 py-2 text-right">{display(player.threat)}</td><td className="px-2 py-2 text-right">{display(player.creativity)}</td><td className="px-2 py-2 text-right">{display(player.ict)}</td><td className="px-2 py-2 text-right">{player.bps}</td><td className="px-2 py-2 text-right">{player.bonus}</td></tr></Fragment>)}</tbody>
        </table>
      </div>
      {players.length === 0 ? <p className="mt-3 text-sm text-pl-text-faint">No player matches those filters.</p> : <div className="flex items-center justify-between text-xs text-pl-text-dim"><span>{players.length} players · showing {visiblePlayers.length} · page {safePage} of {pageCount}</span><div className="flex gap-2"><button disabled={safePage <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))} className="rounded border border-pl-border px-2 py-1 disabled:opacity-30">Previous</button><button disabled={safePage >= pageCount} onClick={() => setPage((value) => Math.min(pageCount, value + 1))} className="rounded border border-pl-border px-2 py-1 disabled:opacity-30">Next</button></div></div>}
    </section>
  );
}
