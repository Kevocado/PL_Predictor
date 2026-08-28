import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CurrentGameweekResponse } from "../types";
import { FixtureModal } from "../components/FixtureModal";
import { CurrentGameweekSection } from "../components/CurrentGameweekSection";
import { PUBLIC_MODE } from "../lib/publicMode";

export function FixturesPage() {
  const [gameweek, setGameweek] = useState<CurrentGameweekResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  // undefined = "current gameweek" (server decides); once the user
  // navigates, this pins to whichever gameweek they're browsing.
  const [viewGameweek, setViewGameweek] = useState<number | undefined>(undefined);

  const load = (gw?: number) => {
    setLoading(true);
    setError(null);
    api
      .currentGameweek(gw)
      .then(setGameweek)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => load(viewGameweek), [viewGameweek]);

  useEffect(() => {
    if (!gameweek?.fixtures.some((fixture) => fixture.finished && fixture.player_events_pending)) return;
    const timer = window.setTimeout(() => load(viewGameweek), 5000);
    return () => window.clearTimeout(timer);
  }, [gameweek, viewGameweek]);

  const navigate = (gw: number) => setViewGameweek(gw);

  const runAction = async (key: string, fn: () => Promise<unknown>) => {
    setBusy(key);
    try {
      await fn();
      load(viewGameweek);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        {!PUBLIC_MODE && (
          <div className="ml-auto flex gap-2">
            <button
              disabled={busy !== null}
              onClick={() => runAction("fixtures", api.refreshFixtures)}
              className="rounded-lg border border-pl-border bg-pl-850/70 px-3 py-2 text-sm text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
            >
              {busy === "fixtures" ? "Refreshing…" : "Refresh fixtures"}
            </button>
            <button
              disabled={busy !== null}
              onClick={() => runAction("odds", api.refreshOdds)}
              className="rounded-lg border border-pl-border bg-pl-850/70 px-3 py-2 text-sm text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
            >
              {busy === "odds" ? "Refreshing…" : "Refresh odds"}
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>
      )}

      {loading && !gameweek ? (
        <div className="py-16 text-center text-pl-text-faint">Loading fixtures…</div>
      ) : (
        gameweek && <CurrentGameweekSection data={gameweek} onSelect={setSelected} onNavigate={navigate} />
      )}

      {selected && <FixtureModal eventId={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
