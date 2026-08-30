import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { PlayerHubResponse, ProjectedTableResponse, RankingsResponse, TeamHubResponse, TrackRecordResponse } from "../types";
import { PowerRankings } from "../components/PowerRankings";
import { ProjectedTable } from "../components/ProjectedTable";
import { TrackRecordPanel } from "../components/TrackRecordPanel";
import { TeamHub } from "../components/TeamHub";
import { PlayerHub } from "../components/PlayerHub";
import { PUBLIC_MODE } from "../lib/publicMode";

const TABS = ["Team Hub", "Player Hub", "Power Rankings", "Projected Table", "Track Record"] as const;
type Tab = (typeof TABS)[number];
type Panel = "rankings" | "table" | "trackRecord" | "teamHub" | "playerHub";

// The live-results cache is refreshed by the backend every minute. Matching
// that interval keeps the actual table current without continuously
// rebuilding the much more expensive projected table.
const REFRESH_INTERVAL_MS = 60_000;

function PanelUnavailable({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-loss/40 bg-loss/10 px-3 py-3 text-sm text-loss">
      <span>{message}</span>
      <button onClick={onRetry} className="shrink-0 rounded border border-loss/50 px-2 py-1 text-xs font-semibold hover:bg-loss/10">Retry</button>
    </div>
  );
}

export function DataHubPage() {
  const [rankings, setRankings] = useState<RankingsResponse | null>(null);
  const [table, setTable] = useState<ProjectedTableResponse | null>(null);
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null);
  const [teamHub, setTeamHub] = useState<TeamHubResponse | null>(null);
  const [playerHub, setPlayerHub] = useState<PlayerHubResponse | null>(null);
  const [errors, setErrors] = useState<Partial<Record<Panel, string>>>({});
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [tab, setTab] = useState<Tab>("Team Hub");

  const load = useCallback((force = false) => {
    setRefreshing(true);
    return Promise.allSettled([
      api.rankings(force),
      api.projectedTable(force),
      api.trackRecord(force),
      api.teamHub(force),
      api.playerHub(force),
    ]).then(([rankingsResult, tableResult, trackResult, teamResult, playerResult]) => {
      if (rankingsResult.status === "fulfilled") setRankings(rankingsResult.value);
      if (tableResult.status === "fulfilled") setTable(tableResult.value);
      if (trackResult.status === "fulfilled") setTrackRecord(trackResult.value);
      if (teamResult.status === "fulfilled") setTeamHub(teamResult.value);
      if (playerResult.status === "fulfilled") setPlayerHub(playerResult.value);
      const message = (result: PromiseSettledResult<unknown>) => result.status === "rejected"
        ? (result.reason instanceof Error ? result.reason.message : "This dashboard panel could not load.")
        : undefined;
      setErrors({
        rankings: message(rankingsResult),
        table: message(tableResult),
        trackRecord: message(trackResult),
        teamHub: message(teamResult),
        playerHub: message(playerResult),
      });
    }).finally(() => setRefreshing(false));
  }, []);

  useEffect(() => {
    void load();
    const id = window.setInterval(() => { void load(true); }, REFRESH_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const refresh = async () => {
    setRefreshError(null);
    try {
      if (!PUBLIC_MODE) await api.refreshFixtures();
      await load(true);
    } catch (error) {
      setRefreshError(error instanceof Error ? error.message : "Could not refresh the live fixtures.");
      await load(true);
    }
  };

  if (!rankings && !table && !trackRecord && !teamHub && !playerHub && refreshing) {
    return <div className="py-16 text-center text-pl-text-faint">Loading data hub…</div>;
  }

  return (
    <div>
      {refreshError && <p className="mb-3 rounded-lg border border-loss/40 bg-loss/10 px-3 py-2 text-sm text-loss">{refreshError}</p>}
      <div className="mb-6 flex items-center justify-between gap-3">
        <div className="flex gap-1 rounded-lg border border-pl-border bg-pl-850/50 p-1 w-fit">
          {TABS.map((item) => (
            <button
              key={item}
              onClick={() => setTab(item)}
              className={`rounded-md px-4 py-2 text-sm font-semibold transition ${tab === item ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"}`}
            >
              {item}
            </button>
          ))}
        </div>
        <button
          onClick={() => { void refresh(); }}
          disabled={refreshing}
          className="rounded-lg border border-pl-border bg-pl-850/50 px-3 py-2 text-xs font-semibold text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
        >
          {refreshing ? "Refreshing…" : "Refresh live data"}
        </button>
      </div>

      {tab === "Team Hub" && <section><h2 className="mb-3 text-lg font-semibold text-pl-text">Team Hub</h2>{errors.teamHub ? <PanelUnavailable message={errors.teamHub} onRetry={() => { void load(true); }} /> : teamHub ? <TeamHub data={teamHub} /> : <p className="py-10 text-center text-sm text-pl-text-faint">Loading team analytics…</p>}</section>}
      {tab === "Player Hub" && <section><h2 className="mb-3 text-lg font-semibold text-pl-text">Player Hub</h2>{errors.playerHub ? <PanelUnavailable message={errors.playerHub} onRetry={() => { void load(true); }} /> : playerHub ? <PlayerHub data={playerHub} /> : <p className="py-10 text-center text-sm text-pl-text-faint">Loading player analytics…</p>}</section>}
      {tab === "Power Rankings" && <section><h2 className="mb-3 text-lg font-semibold text-pl-text">Power Rankings</h2>{errors.rankings ? <PanelUnavailable message={errors.rankings} onRetry={() => { void load(true); }} /> : rankings ? <PowerRankings data={rankings} /> : <p className="py-10 text-center text-sm text-pl-text-faint">Loading power rankings…</p>}</section>}
      {tab === "Projected Table" && <section><h2 className="mb-3 text-lg font-semibold text-pl-text">Projected Table</h2>{errors.table ? <PanelUnavailable message={errors.table} onRetry={() => { void load(true); }} /> : table ? <ProjectedTable data={table} /> : <p className="py-10 text-center text-sm text-pl-text-faint">Loading current table…</p>}</section>}
      {tab === "Track Record" && <section><h2 className="mb-3 text-lg font-semibold text-pl-text">Track Record</h2>{errors.trackRecord ? <PanelUnavailable message={errors.trackRecord} onRetry={() => { void load(true); }} /> : trackRecord ? <TrackRecordPanel data={trackRecord} /> : <p className="py-10 text-center text-sm text-pl-text-faint">Loading track record…</p>}</section>}
    </div>
  );
}
