import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { PlayerCard } from "../components/PlayerCard";
import type { FPLPlayerProjection, FPLProjectionsResponse, FPLRecommendation, FPLSquadResponse, FPLTransfersResponse } from "../types";

const FORMATIONS = ["Auto", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-2-3", "5-3-2", "5-4-1"];
const PAGE_SIZE = 25;
const MODE_LABELS: Record<"gameweek" | "squad" | "transfers", string> = {
  gameweek: "This gameweek's XI",
  squad: "Best £100m squad",
  transfers: "My team & transfers",
};

function LineupCard({ data, title, requestedFormation }: { data: FPLRecommendation; title: string; requestedFormation: string }) {
  const positions = (position: FPLPlayerProjection["position"]) => data.starting_xi.filter((player) => player.position === position);
  const formation = `${positions("DEF").length}-${positions("MID").length}-${positions("FWD").length}`;
  return <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4">
    <div className="mb-3 flex items-start justify-between gap-2"><div><h3 className="font-semibold text-pl-text">{title}</h3><p className="text-xs text-pl-text-dim">{requestedFormation === "Auto" ? `Optimizer chose ${formation}` : `${formation} selected`} · {data.projected_points.toFixed(1)} points before captaincy</p></div><span className="rounded bg-pl-pink/15 px-2 py-1 text-xs font-bold text-pl-pink">C {data.captain?.web_name ?? "—"}</span></div>
    <div className="space-y-4 rounded-lg border border-emerald-300/30 bg-[repeating-linear-gradient(0deg,rgba(255,255,255,.08)_0,rgba(255,255,255,.08)_1px,transparent_1px,transparent_25%)] bg-emerald-800/80 px-3 py-4">
      {(["FWD", "MID", "DEF", "GK"] as const).map((position) => <div key={position} className="mx-auto grid w-full max-w-md gap-2" style={{ gridTemplateColumns: `repeat(${Math.max(positions(position).length, 1)}, minmax(0, 1fr))` }}>{positions(position).map((player) => <PlayerCard key={player.player_id} player={player} size="sm" marker={data.captain?.player_id === player.player_id ? "C" : data.vice_captain?.player_id === player.player_id ? "VC" : undefined} />)}</div>)}
    </div>
    <p className="mt-3 text-xs text-pl-text-faint">Bench order: {data.bench.map((p) => p.web_name).join(" · ") || "—"}</p>
  </section>;
}

export function FPLPage() {
  const [projections, setProjections] = useState<FPLProjectionsResponse | null>(null);
  const [xi, setXi] = useState<(FPLRecommendation & { gameweek: number; model_source: string }) | null>(null);
  const [squad, setSquad] = useState<FPLSquadResponse | null>(null);
  const [transfers, setTransfers] = useState<FPLTransfersResponse | null>(null);
  const [entryId, setEntryId] = useState("");
  const [manualIds, setManualIds] = useState("");
  const [bank, setBank] = useState("0.0");
  const [freeTransfers, setFreeTransfers] = useState("1");
  const [query, setQuery] = useState("");
  const [positionFilter, setPositionFilter] = useState<"ALL" | FPLPlayerProjection["position"]>("ALL");
  const [sortBy, setSortBy] = useState<"projected_points" | "price" | "expected_minutes" | "fixture_count" | "availability">("projected_points");
  const [maxPrice, setMaxPrice] = useState("");
  const [minMinutes, setMinMinutes] = useState("");
  const [page, setPage] = useState(1);
  const [formation, setFormation] = useState("Auto");
  const [mode, setMode] = useState<"gameweek" | "squad" | "transfers">("gameweek");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setError(null);
    setXi(null); setSquad(null);
    // Render the scout as soon as the lightweight projection response is
    // available. The two exact optimisation requests can then finish in the
    // background instead of leaving the entire page blank behind Promise.all.
    return api.fplProjections()
      .then((p) => {
        setProjections(p);
        return Promise.allSettled([api.fplOptimalXi(), api.fplSquad()]).then(([xiResult, squadResult]) => {
          if (xiResult.status === "fulfilled") setXi(xiResult.value);
          if (squadResult.status === "fulfilled") setSquad(squadResult.value);
          const failure = xiResult.status === "rejected" ? xiResult.reason : squadResult.status === "rejected" ? squadResult.reason : null;
          if (failure) setError(failure instanceof Error ? failure.message : "The optimiser is temporarily unavailable.");
        });
      })
      .catch((e: Error) => setError(e.message));
  }, []);
  useEffect(() => { load(); }, [load]);

  const players = useMemo(() => {
    const term = query.trim().toLowerCase();
    return (projections?.players ?? []).filter((p) =>
      (!term || `${p.name} ${p.team} ${p.position}`.toLowerCase().includes(term))
      && (positionFilter === "ALL" || p.position === positionFilter)
      && (!maxPrice || p.price <= Number(maxPrice))
      && (!minMinutes || p.expected_minutes >= Number(minMinutes))
    ).sort((a, b) => sortBy === "price" ? a.price - b.price : b[sortBy] - a[sortBy]);
  }, [projections, query, positionFilter, maxPrice, minMinutes, sortBy]);
  const pageCount = Math.max(1, Math.ceil(players.length / PAGE_SIZE));
  const visiblePlayers = players.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const chooseFormation = (next: string) => {
    setFormation(next); setXi(null); setError(null);
    api.fplOptimalXi(next === "Auto" ? undefined : next).then(setXi).catch((e) => setError(e.message));
  };

  const requestTransfers = async (mode: "entry" | "manual") => {
    setBusy(true); setError(null);
    try {
      const ft = Math.max(0, Number.parseInt(freeTransfers, 10) || 0);
      const result = mode === "entry"
        ? await api.fplEntryTransfers(Number.parseInt(entryId, 10), ft)
        : await api.fplManualTransfers(manualIds.split(",").map((id) => Number.parseInt(id.trim(), 10)).filter(Number.isFinite), Number.parseFloat(bank) || 0, ft);
      setTransfers(result);
    } catch (e) { setError(e instanceof Error ? e.message : "Could not build transfer recommendations."); }
    finally { setBusy(false); }
  };

  if (error && !projections) return <div className="rounded-lg border border-loss/40 bg-loss/10 p-4 text-sm text-loss">{error}</div>;
  if (!projections) return <div className="py-16 text-center text-pl-text-faint">Building FPL projections…</div>;

  return <div className="space-y-6">
    <section className="rounded-xl border border-pl-border bg-pl-900/50 p-5"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-wider text-pl-pink">Gameweek {projections.gameweek}</p><h2 className="mt-1 text-xl font-bold text-pl-text">FPL Scout &amp; Optimizer</h2><p className="mt-1 max-w-3xl text-sm text-pl-text-dim">Independent match forecasts inform fixture context; player form, availability and expected minutes determine the FPL projection. Value-bet decisions never use these market-free projections.</p></div><button onClick={() => load()} className="rounded-lg border border-pl-border px-3 py-2 text-xs font-semibold text-pl-text-dim hover:text-pl-text">Refresh</button></div><p className="mt-3 text-xs text-pl-text-faint">{projections.data_freshness} · {new Date(projections.generated_at).toLocaleString()}</p></section>

    {error && <p className="rounded-lg border border-loss/40 bg-loss/10 px-3 py-2 text-sm text-loss">{error}</p>}

    <div className="flex flex-wrap gap-1 rounded-lg border border-pl-border p-1">
      {(["gameweek", "squad", "transfers"] as const).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={`rounded-md px-3 py-1.5 text-xs font-semibold ${
            mode === m ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"
          }`}
        >
          {MODE_LABELS[m]}
        </button>
      ))}
    </div>

    {mode === "gameweek" && (
      <section><div className="mb-2 flex flex-wrap gap-1">{FORMATIONS.map((item) => <button key={item} onClick={() => chooseFormation(item)} className={`rounded-md px-2.5 py-1 text-xs font-semibold ${formation === item ? "bg-pl-pink text-white" : "border border-pl-border text-pl-text-dim hover:text-pl-text"}`}>{item}</button>)}</div>{xi ? <LineupCard data={xi} title="Best starting XI" requestedFormation={formation} /> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the best starting XI…</section>}</section>
    )}

    {mode === "squad" && (
      <div>{squad ? <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4"><div className="mb-3 flex justify-between"><div><h3 className="font-semibold text-pl-text">Best £100m squad</h3><p className="text-xs text-pl-text-dim">£{squad.spent.toFixed(1)}m spent · £{squad.remaining.toFixed(1)}m remaining</p></div><span className="text-xs text-pl-text-faint">Normal gameweek only</span></div><div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">{squad.squad.map((player) => <PlayerCard key={player.player_id} player={player} size="md" marker={squad.captain?.player_id === player.player_id ? "C" : undefined} />)}</div></section> : <section className="rounded-xl border border-pl-border bg-pl-900/50 p-4 text-sm text-pl-text-faint">Optimising the £100m squad…</section>}</div>
    )}

    {mode === "transfers" && (
      <section className="rounded-xl border border-pl-border bg-pl-900/50 p-5"><h3 className="font-semibold text-pl-text">Transfer planner</h3><p className="mt-1 text-sm text-pl-text-dim">Use a public FPL entry ID or paste a 15-player element-ID list. No FPL account credentials or squad data are stored.</p><div className="mt-4 grid gap-3 lg:grid-cols-[1fr_110px_110px_auto]"><label className="text-xs font-semibold text-pl-text-faint">Public entry ID<input value={entryId} onChange={(e) => setEntryId(e.target.value)} inputMode="numeric" placeholder="e.g. 12345" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><label className="text-xs font-semibold text-pl-text-faint">Free transfers<input value={freeTransfers} onChange={(e) => setFreeTransfers(e.target.value)} inputMode="numeric" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><div className="flex items-end"><button disabled={busy || !entryId} onClick={() => requestTransfers("entry")} className="w-full rounded-lg bg-pl-pink px-3 py-2 text-sm font-bold text-white disabled:opacity-40">Use entry</button></div></div><div className="mt-3 grid gap-3 lg:grid-cols-[1fr_110px_auto]"><label className="text-xs font-semibold text-pl-text-faint">Manual element IDs (15 comma-separated)<input value={manualIds} onChange={(e) => setManualIds(e.target.value)} placeholder="1, 2, 3, …" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><label className="text-xs font-semibold text-pl-text-faint">Bank (£m)<input value={bank} onChange={(e) => setBank(e.target.value)} inputMode="decimal" className="mt-1 block w-full rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label><div className="flex items-end"><button disabled={busy || !manualIds} onClick={() => requestTransfers("manual")} className="w-full rounded-lg border border-pl-pink px-3 py-2 text-sm font-bold text-pl-pink disabled:opacity-40">Use manual squad</button></div></div>{transfers && <div className="mt-4"><p className="mb-2 text-xs text-pl-text-dim">You have <span className="font-semibold text-pl-text">{transfers.free_transfers}</span> free transfer{transfers.free_transfers === 1 ? "" : "s"} · <span className="font-semibold text-pl-text">£{transfers.bank.toFixed(1)}m</span> in the bank</p><div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="border-b border-pl-border text-pl-text-faint"><tr><th className="p-2">Out</th><th className="p-2">In</th><th className="p-2 text-right">Cost</th><th className="p-2 text-right">Projected gain</th><th className="p-2 text-right">Net gain</th></tr></thead><tbody>{transfers.recommendations.map((idea) => <Fragment key={`${idea.out.player_id}-${idea.in.player_id}`}><tr className="border-b border-pl-border/20 text-pl-text-dim"><td className="p-2">{idea.out.web_name}</td><td className="p-2 font-semibold text-pl-text">{idea.in.web_name}</td><td className="p-2 text-right">{idea.cost < 0 ? `-£${Math.abs(idea.cost).toFixed(1)}m` : `£${idea.cost.toFixed(1)}m`}</td><td className="p-2 text-right">+{idea.projected_gain.toFixed(1)}</td><td className="p-2 text-right font-semibold text-pl-pink">+{idea.net_gain.toFixed(1)}</td></tr><tr className="border-b border-pl-border/60"><td colSpan={5} className="px-2 pb-2 text-[11px] text-pl-text-faint">{idea.in.drivers.slice(0, 2).join(" · ")}</td></tr></Fragment>)}</tbody></table>{transfers.recommendations.length === 0 && <p className="py-3 text-sm text-pl-text-faint">No positive like-for-like transfer is available under the supplied constraints.</p>}</div></div>}</section>
    )}

    <section className="rounded-xl border border-pl-border bg-pl-900/50 p-5"><div className="mb-3 flex flex-wrap items-end justify-between gap-3"><div><h3 className="font-semibold text-pl-text">Player scout</h3><p className="text-sm text-pl-text-dim">Doubles are summed; blanks show 0 fixtures.</p></div><label className="text-xs font-semibold text-pl-text-faint">Search<input value={query} onChange={(e) => { setQuery(e.target.value); setPage(1); }} placeholder="Player or team" className="mt-1 block rounded-lg border border-pl-border bg-pl-850 px-3 py-2 text-sm text-pl-text" /></label></div><div className="mb-4 flex flex-wrap items-end gap-3"><div className="flex gap-1 rounded-lg border border-pl-border p-1">{(["ALL", "GK", "DEF", "MID", "FWD"] as const).map((position) => <button key={position} onClick={() => { setPositionFilter(position); setPage(1); }} className={`rounded-md px-2.5 py-1.5 text-xs font-semibold ${positionFilter === position ? "bg-pl-pink text-white" : "text-pl-text-dim hover:text-pl-text"}`}>{position === "ALL" ? "All" : position}</button>)}</div><label className="text-xs font-semibold text-pl-text-faint">Sort<select value={sortBy} onChange={(e) => { setSortBy(e.target.value as typeof sortBy); setPage(1); }} className="ml-1 rounded border border-pl-border bg-pl-850 px-2 py-1.5 text-xs text-pl-text"><option value="projected_points">Projected points</option><option value="price">Lowest price</option><option value="expected_minutes">Expected minutes</option><option value="fixture_count">Fixture count</option><option value="availability">Availability</option></select></label><label className="text-xs font-semibold text-pl-text-faint">Max £<input value={maxPrice} onChange={(e) => { setMaxPrice(e.target.value); setPage(1); }} inputMode="decimal" placeholder="Any" className="ml-1 w-16 rounded border border-pl-border bg-pl-850 px-2 py-1.5 text-xs text-pl-text" /></label><label className="text-xs font-semibold text-pl-text-faint">Min mins<input value={minMinutes} onChange={(e) => { setMinMinutes(e.target.value); setPage(1); }} inputMode="numeric" placeholder="Any" className="ml-1 w-16 rounded border border-pl-border bg-pl-850 px-2 py-1.5 text-xs text-pl-text" /></label></div><div className="overflow-x-auto"><table className="w-full min-w-[900px] text-left text-xs"><thead className="border-b border-pl-border text-pl-text-faint"><tr><th className="p-2">Player</th><th className="p-2">Team</th><th className="p-2">Pos</th><th className="p-2 text-right">Price</th><th className="p-2 text-right">Proj.</th><th className="p-2 text-right">Min</th><th className="p-2">Fixtures</th><th className="p-2">Why</th></tr></thead><tbody>{visiblePlayers.map((p) => <tr key={p.player_id} className="border-b border-pl-border/60 text-pl-text-dim"><td className="p-2 font-semibold text-pl-text">{p.web_name}{p.status !== "a" && <span className="ml-1 text-loss">●</span>}</td><td className="p-2">{p.team}</td><td className="p-2">{p.position}</td><td className="p-2 text-right">£{p.price.toFixed(1)}</td><td className="p-2 text-right font-bold text-pl-pink">{p.projected_points.toFixed(1)}</td><td className="p-2 text-right">{p.expected_minutes.toFixed(0)}</td><td className="p-2">{p.fixtures.map((f) => `${f.opponent}${f.was_home ? " (H)" : " (A)"}`).join(", ") || "Blank"}</td><td className="p-2">{p.drivers.join(" · ")}</td></tr>)}</tbody></table></div><div className="mt-4 flex items-center justify-between text-xs text-pl-text-dim"><span>{players.length} players · page {page} of {pageCount}</span><div className="flex gap-2"><button disabled={page <= 1} onClick={() => setPage((value) => value - 1)} className="rounded border border-pl-border px-2 py-1 disabled:opacity-30">Previous</button><button disabled={page >= pageCount} onClick={() => setPage((value) => value + 1)} className="rounded border border-pl-border px-2 py-1 disabled:opacity-30">Next</button></div></div></section>
  </div>;
}
