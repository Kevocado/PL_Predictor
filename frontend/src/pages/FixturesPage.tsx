import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { FixtureSummary } from "../types";
import { FixturesGrid } from "../components/FixturesGrid";
import { FixtureModal } from "../components/FixtureModal";
import { InfoTooltip } from "../components/InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

export function FixturesPage() {
  const [fixtures, setFixtures] = useState<FixtureSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [valueBetsOnly, setValueBetsOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    api
      .fixtures()
      .then((data) => setFixtures([...data].sort((a, b) => a.commence_time.localeCompare(b.commence_time))))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const filtered = useMemo(() => {
    return fixtures.filter((f) => {
      if (valueBetsOnly && f.value_bet_flags.length === 0) return false;
      if (search) {
        const s = search.toLowerCase();
        if (!f.team_home.toLowerCase().includes(s) && !f.team_away.toLowerCase().includes(s)) return false;
      }
      return true;
    });
  }, [fixtures, valueBetsOnly, search]);

  const runAction = async (key: string, fn: () => Promise<unknown>) => {
    setBusy(key);
    try {
      await fn();
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search team…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="rounded-lg border border-pl-border bg-pl-850/70 px-3 py-2 text-sm text-pl-text placeholder:text-pl-text-faint focus:border-pl-pink focus:outline-none"
        />
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setValueBetsOnly((v) => !v)}
            className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
              valueBetsOnly
                ? "bg-pl-pink text-white"
                : "border border-pl-border bg-pl-850/70 text-pl-text-dim hover:text-pl-text"
            }`}
          >
            Value bets only
          </button>
          <InfoTooltip text={GLOSSARY.valueBetThreshold} />
        </div>
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
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-loss/40 bg-loss/10 px-4 py-3 text-sm text-loss">{error}</div>
      )}

      {loading ? (
        <div className="py-16 text-center text-pl-text-faint">Loading fixtures…</div>
      ) : (
        <FixturesGrid fixtures={filtered} onSelect={setSelected} />
      )}

      {selected && <FixtureModal eventId={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
