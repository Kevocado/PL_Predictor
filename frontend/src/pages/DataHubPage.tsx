import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { PlayerHubResponse, ProjectedTableResponse, RankingsResponse, TeamHubResponse, TrackRecordResponse } from "../types";
import { PowerRankings } from "../components/PowerRankings";
import { ProjectedTable } from "../components/ProjectedTable";
import { TrackRecordPanel } from "../components/TrackRecordPanel";
import { TeamHub } from "../components/TeamHub";
import { PlayerHub } from "../components/PlayerHub";

const TABS = ["Track Record", "Team Hub", "Player Hub", "Power Rankings", "Projected Table"] as const;
type Tab = (typeof TABS)[number];

// Data Hub pages stay mounted once loaded (see App.tsx) rather than
// refetching on every tab switch, so without this the very first fetch
// (e.g. before a match kicks off, or before the backfill has run) would be
// the only one that ever happens — new results/rankings would never appear
// without a full page reload. Matches the backend's own live-data cache
// window (`_LIVE_CACHE_TTL_SECONDS` in routes.py), so polling faster than
// this wouldn't surface anything newer anyway.
const REFRESH_INTERVAL_MS = 5 * 60 * 1000;

export function DataHubPage() {
  const [rankings, setRankings] = useState<RankingsResponse | null>(null);
  const [table, setTable] = useState<ProjectedTableResponse | null>(null);
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null);
  const [teamHub, setTeamHub] = useState<TeamHubResponse | null>(null);
  const [playerHub, setPlayerHub] = useState<PlayerHubResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [teamHubError, setTeamHubError] = useState<string | null>(null);
  const [playerHubError, setPlayerHubError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [tab, setTab] = useState<Tab>("Track Record");

  const load = useCallback(() => {
    setRefreshing(true);
    return Promise.all([api.rankings(), api.projectedTable(), api.trackRecord()])
      .then(([r, t, tr]) => {
        setRankings(r);
        setTable(t);
        setTrackRecord(tr);
      })
      .catch((e) => setError(e.message))
      .finally(() => setRefreshing(false));
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [load]);

  useEffect(() => {
    if (tab === "Team Hub" && !teamHub) {
      api.teamHub().then(setTeamHub).catch((e) => setTeamHubError(e.message));
    }
    if (tab === "Player Hub" && !playerHub) {
      api.playerHub().then(setPlayerHub).catch((e) => setPlayerHubError(e.message));
    }
  }, [tab, teamHub, playerHub]);

  if (error) {
    return <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>;
  }

  if (!rankings || !table || !trackRecord) {
    return <div className="py-16 text-center text-pl-text-faint">Loading data hub…</div>;
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between gap-3">
        <div className="flex gap-1 rounded-lg border border-pl-border bg-pl-850/50 p-1 w-fit">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-md px-4 py-2 text-sm font-semibold transition ${
                tab === t ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <button
          onClick={() => load()}
          disabled={refreshing}
          className="rounded-lg border border-pl-border bg-pl-850/50 px-3 py-2 text-xs font-semibold text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
        >
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {tab === "Track Record" && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">Track Record</h2>
          <TrackRecordPanel data={trackRecord} />
        </section>
      )}

      {tab === "Power Rankings" && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">Power Rankings</h2>
          <PowerRankings data={rankings} />
        </section>
      )}

      {tab === "Team Hub" && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">Team Hub</h2>
          {teamHubError && <p className="text-sm text-loss">{teamHubError}</p>}
          {!teamHub && !teamHubError && <p className="py-10 text-center text-sm text-pl-text-faint">Loading team analytics…</p>}
          {teamHub && <TeamHub data={teamHub} />}
        </section>
      )}

      {tab === "Player Hub" && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">Player Hub</h2>
          {playerHubError && <p className="text-sm text-loss">{playerHubError}</p>}
          {!playerHub && !playerHubError && <p className="py-10 text-center text-sm text-pl-text-faint">Loading player analytics…</p>}
          {playerHub && <PlayerHub data={playerHub} />}
        </section>
      )}

      {tab === "Projected Table" && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-pl-text">Projected Table</h2>
          <ProjectedTable data={table} />
        </section>
      )}
    </div>
  );
}
