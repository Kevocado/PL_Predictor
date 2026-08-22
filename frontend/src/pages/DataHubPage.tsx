import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ProjectedTableResponse, RankingsResponse, TrackRecordResponse } from "../types";
import { PowerRankings } from "../components/PowerRankings";
import { ProjectedTable } from "../components/ProjectedTable";
import { TrackRecordPanel } from "../components/TrackRecordPanel";

export function DataHubPage() {
  const [rankings, setRankings] = useState<RankingsResponse | null>(null);
  const [table, setTable] = useState<ProjectedTableResponse | null>(null);
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.rankings(), api.projectedTable(), api.trackRecord()])
      .then(([r, t, tr]) => {
        setRankings(r);
        setTable(t);
        setTrackRecord(tr);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return <div className="rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>;
  }

  if (!rankings || !table || !trackRecord) {
    return <div className="py-16 text-center text-pl-text-faint">Loading data hub…</div>;
  }

  return (
    <div className="flex flex-col gap-10">
      <section>
        <h2 className="mb-3 text-lg font-semibold text-pl-text">Power Rankings</h2>
        <PowerRankings data={rankings} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold text-pl-text">Projected Table</h2>
        <ProjectedTable data={table} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold text-pl-text">Track Record</h2>
        <TrackRecordPanel data={trackRecord} />
      </section>
    </div>
  );
}
