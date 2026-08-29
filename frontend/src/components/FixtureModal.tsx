import { useEffect, useState, type ReactNode } from "react";
import type { FixtureDetail, FixturePlayerReview, FixturePlayers, FixturePostMatch, MarketEdge } from "../types";
import { api } from "../api/client";
import { TeamBadge } from "./TeamBadge";
import { ScorelineHeatmap } from "./ScorelineHeatmap";
import { FormStrip } from "./FormStrip";
import { InfoTooltip } from "./InfoTooltip";
import { PlayerHighlights, PlayerScorerList } from "./PlayerScorerList";
import { GLOSSARY } from "../lib/glossary";

interface Props {
  eventId: string;
  onClose: () => void;
}

const MARKET_LABELS: Record<string, string> = {
  home_win: "Home win",
  draw: "Draw",
  away_win: "Away win",
  over_2_5: "Over 2.5",
  under_2_5: "Under 2.5",
};

function americanOdds(decimalOdds: number) {
  return decimalOdds >= 2 ? `+${Math.round((decimalOdds - 1) * 100)}` : `${Math.round(-100 / (decimalOdds - 1))}`;
}

function marketType(market: string) {
  return ["home_win", "draw", "away_win"].includes(market) ? "Match result" : "Goals total";
}

function MarketRow({ label, edge, flagged, postMatchHit }: { label: ReactNode; edge: MarketEdge; flagged: boolean; postMatchHit?: boolean }) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
        postMatchHit === true ? "bg-win/10 ring-1 ring-win/30" : postMatchHit === false ? "bg-loss/10" : flagged ? "bg-pl-pink/10 ring-1 ring-pl-pink/40" : "bg-pl-850/60"
      }`}
    >
      <span className="text-pl-text-dim">{label}</span>
      <div className="flex items-center gap-3">
        <span className="font-semibold text-pl-text">{(edge.prob * 100).toFixed(1)}%</span>
        {edge.implied !== null && (
          <>
            <span className="text-xs text-pl-text-faint">mkt {(edge.implied * 100).toFixed(1)}%</span>
            <span className={`text-xs font-semibold ${(edge.edge ?? 0) > 0 ? "text-pl-cyan" : "text-pl-text-faint"}`}>
              {(edge.edge ?? 0) > 0 ? "+" : ""}
              {((edge.edge ?? 0) * 100).toFixed(2)}%
            </span>
          </>
        )}
      </div>
    </div>
  );
}

function OverUnderRow({ label, lam, line, over, postMatchHit }: { label: string; lam: number; line: number; over: number; postMatchHit?: boolean }) {
  return (
    <div className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${postMatchHit === true ? "bg-win/10 ring-1 ring-win/30" : postMatchHit === false ? "bg-loss/10" : "bg-pl-850/60"}`}>
      <span className="text-pl-text-dim">{label}</span>
      <div className="flex items-center gap-3">
        <span className="text-xs text-pl-text-faint">exp. {lam.toFixed(1)}</span>
        <span className="font-semibold text-pl-text">
          O{line} {(over * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function fixtureMetric(value: number | null, suffix = "") {
  return value === null ? "—" : `${value.toFixed(1)}${suffix}`;
}

function reportedMetric(value: number | null | undefined, label: string) {
  if (value === null || value === undefined) return "—";
  const rounded = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return label.includes("%") ? `${rounded}%` : rounded;
}

function PostMatchReview({ review }: { review: FixturePostMatch }) {
  const correct = review.verdicts.filter((verdict) => verdict.hit).length;
  return (
    <section className="rounded-xl border border-win/30 bg-win/5 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div><h3 className="text-sm font-semibold text-pl-text">Prediction review · final {review.final_score}</h3><p className="mt-0.5 text-xs text-pl-text-dim">{correct}/{review.verdicts.length} match calls correct</p></div>
        <span className={`rounded px-2 py-1 text-[10px] font-semibold uppercase ${review.provenance === "snapshot" ? "bg-win/20 text-win" : "bg-pl-700/60 text-pl-text-dim"}`}>{review.provenance === "snapshot" ? "Pre-match snapshot" : "Reconstructed"}</span>
      </div>
      <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
        {review.verdicts.map((verdict) => <div key={verdict.label} className={`flex items-center justify-between rounded-lg px-3 py-2 text-xs ${verdict.hit ? "bg-win/10 text-win" : "bg-pl-850/70 text-pl-text-dim"}`}><span className="font-semibold">{verdict.hit ? "✓" : "×"} {verdict.label}</span><span><span className="text-pl-text-faint">{verdict.prediction}</span><span className="mx-1">→</span><span>{verdict.actual}</span></span></div>)}
      </div>
      {review.provenance === "reconstructed" && <p className="mt-2 text-[11px] text-pl-text-faint">Reconstructed after the match from saved inputs where available; it is shown for consistency, not counted as prospective proof.</p>}
    </section>
  );
}

function PlayerCallReview({ review, loading, error }: { review: FixturePlayerReview | null; loading: boolean; error: string | null }) {
  const outcome = (player: FixturePlayerReview["correct"][number]) => {
    if (player.goals > 0 && player.assists > 0) return `${player.goals} goal${player.goals > 1 ? "s" : ""} · ${player.assists} assist${player.assists > 1 ? "s" : ""}`;
    return player.goals > 0 ? `${player.goals} goal${player.goals > 1 ? "s" : ""}` : `${player.assists} assist${player.assists > 1 ? "s" : ""}`;
  };
  const rows = (players: FixturePlayerReview["correct"], state: "hit" | "miss" | "longShot") => players.map((player) => {
    const hit = state !== "miss";
    const tone = state === "hit" ? "bg-win/10" : state === "miss" ? "bg-loss/10" : "bg-pl-blue/10";
    const signal = `${(player.review_probability * 100).toFixed(0)}% ${player.review_market} chance`;
    return <div key={`${player.team}-${player.name}`} className={`flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-xs ${tone}`}><span className="min-w-0 font-semibold text-pl-text">{state === "hit" ? "✓" : state === "miss" ? "×" : "↑"} {player.name} <span className="font-normal text-pl-text-faint">{player.team}</span>{player.is_recommended && <span className="ml-2 rounded bg-pl-pink/20 px-1.5 py-0.5 text-[10px] font-semibold text-pl-pink">Recommended</span>}</span><span className={`shrink-0 text-right ${hit ? state === "longShot" ? "text-pl-blue" : "text-win" : "text-pl-text-dim"}`}><span className="block">{hit ? outcome(player) : "No goal involvement"}</span><span className="text-pl-text-faint">{signal}</span></span></div>;
  });
  return <section className="rounded-xl border border-pl-border bg-pl-850/50 p-4">
    <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold text-pl-text">Player call review</h3>{review && <span className="text-[10px] font-semibold uppercase text-pl-text-faint">{review.provenance === "snapshot" ? "Pre-match snapshot" : "Reconstructed"}</span>}</div>
    {loading && <p className="mt-2 text-xs text-pl-text-faint">Reconstructing confirmed player calls…</p>}
    {error && <p className="mt-2 text-xs text-loss">{error}</p>}
    {!loading && !error && !review && <p className="mt-2 text-xs text-pl-text-faint">Official player outcomes are not available for this fixture yet.</p>}
    {review && <div className="mt-3 grid gap-3 xl:grid-cols-3"><div><p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-pl-text-faint">Correct calls</p><div className="flex flex-col gap-1">{review.correct.length ? rows(review.correct, "hit") : <p className="rounded-lg bg-pl-900/50 px-3 py-2 text-xs text-pl-text-faint">No tiered call recorded a goal involvement.</p>}</div></div><div><p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-pl-text-faint">Confident calls that missed</p><div className="flex flex-col gap-1">{review.missed.length ? rows(review.missed, "miss") : <p className="rounded-lg bg-pl-900/50 px-3 py-2 text-xs text-pl-text-faint">No confident calls missed.</p>}</div></div><div><p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-pl-text-faint">Overperformers</p><div className="flex flex-col gap-1">{review.overperformed.length ? rows(review.overperformed, "longShot") : <p className="rounded-lg bg-pl-900/50 px-3 py-2 text-xs text-pl-text-faint">No low-probability player outperformed the thresholds.</p>}</div></div></div>}
  </section>;
}

export function FixtureModal({ eventId, onClose }: Props) {
  const [detail, setDetail] = useState<FixtureDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<FixturePlayers | null>(null);
  const [playersError, setPlayersError] = useState<string | null>(null);
  const [playerReview, setPlayerReview] = useState<FixturePlayerReview | null>(null);
  const [playerReviewError, setPlayerReviewError] = useState<string | null>(null);
  const [playerReviewLoading, setPlayerReviewLoading] = useState(false);
  const postMatchVerdict = (label: string) => detail?.post_match?.verdicts.find((verdict) => verdict.label === label);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    setPlayers(null);
    setPlayersError(null);
    setPlayerReview(null);
    setPlayerReviewError(null);
    setPlayerReviewLoading(false);
    api.fixtureDetail(eventId).then(async (fixtureDetail) => {
      if (cancelled) return;
      setDetail(fixtureDetail);
      try {
        const fixturePlayers = await api.fixturePlayers(eventId);
        if (!cancelled) setPlayers(fixturePlayers);
      } catch (playersLoadError) {
        if (!cancelled) setPlayersError(playersLoadError instanceof Error ? playersLoadError.message : String(playersLoadError));
      }
      if (fixtureDetail.post_match) {
        setPlayerReviewLoading(true);
        try {
          const review = await api.fixturePlayerReview(eventId);
          if (!cancelled) setPlayerReview(review);
        } catch (reviewError) {
          if (!cancelled) setPlayerReviewError(reviewError instanceof Error ? reviewError.message : String(reviewError));
        } finally {
          if (!cancelled) setPlayerReviewLoading(false);
        }
      }
    }).catch((detailError) => {
      if (!cancelled) setError(detailError.message);
    });
    return () => { cancelled = true; };
  }, [eventId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      <div className="animate-modal-in relative flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-pl-border bg-pl-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-pl-border px-6 py-4">
          <span className="text-sm font-semibold text-pl-text-dim">Fixture detail</span>
          <button
            onClick={onClose}
            className="rounded-full p-1.5 text-pl-text-dim transition hover:bg-pl-800 hover:text-pl-text"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="overflow-y-auto px-6 py-6">
          {error && <div className="text-sm text-loss">{error}</div>}
          {!detail && !error && <div className="flex h-64 items-center justify-center text-pl-text-faint">Loading…</div>}

          {detail && (
            <div className="flex flex-col gap-6">
              <div className="flex items-center justify-center gap-10">
                <div className="flex flex-col items-center gap-2">
                  <TeamBadge team={detail.team_home} size="lg" />
                  <span className="text-base font-semibold text-pl-text">{detail.team_home}</span>
                  <div className="flex items-center gap-1.5">
                    <FormStrip results={detail.home_recent_form} />
                    <InfoTooltip text={GLOSSARY.recentForm} />
                  </div>
                </div>
                <div className="flex flex-col items-center gap-1 text-center">
                  <span className="text-xs font-medium uppercase text-pl-text-faint">
                    {new Date(detail.commence_time).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}
                  </span>
                  <span className="text-2xl font-black text-pl-text-faint">vs</span>
                  <span className="text-xs text-pl-text-faint">
                    {new Date(detail.commence_time).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <TeamBadge team={detail.team_away} size="lg" />
                  <span className="text-base font-semibold text-pl-text">{detail.team_away}</span>
                  <FormStrip results={detail.away_recent_form} />
                </div>
              </div>

              {detail.data_confidence === "new" && (
                <div className="flex items-center gap-2 rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-dim">
                  <span className="rounded bg-pl-700/60 px-1.5 py-0.5 font-semibold text-pl-text-dim">new team</span>
                  {GLOSSARY.dataConfidenceNew}
                </div>
              )}
              {detail.data_confidence === "limited" && (
                <div className="flex items-center gap-2 rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-dim">
                  <span className="rounded bg-pl-700/60 px-1.5 py-0.5 font-semibold text-pl-text-dim">limited data</span>
                  {GLOSSARY.dataConfidenceLimited}
                </div>
              )}

              {detail.post_match && <PostMatchReview review={detail.post_match} />}
              {detail.post_match && <PlayerCallReview review={playerReview} loading={playerReviewLoading} error={playerReviewError} />}

              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                  {detail.actual_stats ? "Reported match statistics" : detail.post_match ? "Match statistics" : "Rest & match style"}
                </h3>
                {detail.post_match && !detail.actual_stats ? (
                  <p className="rounded-xl border border-pl-border bg-pl-850/50 px-3 py-2 text-xs text-pl-text-faint">
                    The box score (shots, corners, cards) for this match hasn&apos;t been reported by the match-data feed yet — it
                    usually lands within a day of kickoff. The final score and result above are already confirmed.
                  </p>
                ) : (
                  <div className="overflow-hidden rounded-xl border border-pl-border bg-pl-850/50 text-xs">
                    <div className="grid grid-cols-[1fr_auto_1fr] border-b border-pl-border bg-pl-900/60 px-3 py-2 font-semibold text-pl-text">
                      <span>{detail.team_home}</span><span className="px-4 text-pl-text-faint">{detail.actual_stats ? "Final" : "Context"}</span><span className="text-right">{detail.team_away}</span>
                    </div>
                    {(detail.actual_stats ? Object.keys(detail.actual_stats.home).map((label) => [
                      label,
                      reportedMetric(detail.actual_stats!.home[label], label),
                      reportedMetric(detail.actual_stats!.away[label], label),
                    ]) : [
                      ["Rest days", fixtureMetric(detail.home_context.rest_days), fixtureMetric(detail.away_context.rest_days)],
                      ["xG for", fixtureMetric(detail.home_context.xg_for_last_5), fixtureMetric(detail.away_context.xg_for_last_5)],
                      ["xG against", fixtureMetric(detail.home_context.xg_against_last_5), fixtureMetric(detail.away_context.xg_against_last_5)],
                      ["Corners", fixtureMetric(detail.home_context.corners_last_5), fixtureMetric(detail.away_context.corners_last_5)],
                      ["Cards", fixtureMetric(detail.home_context.cards_last_5), fixtureMetric(detail.away_context.cards_last_5)],
                      ["Set-piece xG", fixtureMetric(detail.home_context.set_piece_xg_share_last_5 === null ? null : detail.home_context.set_piece_xg_share_last_5 * 100, "%"), fixtureMetric(detail.away_context.set_piece_xg_share_last_5 === null ? null : detail.away_context.set_piece_xg_share_last_5 * 100, "%")],
                    ]).map(([label, homeValue, awayValue]) => (
                      <div key={label} className="grid grid-cols-[1fr_auto_1fr] border-b border-pl-border/60 px-3 py-2 last:border-0">
                        <span className="font-semibold text-pl-text">{homeValue}</span><span className="px-4 text-pl-text-faint">{label}</span><span className="text-right font-semibold text-pl-text">{awayValue}</span>
                      </div>
                    ))}
                  </div>
                )}
                {!(detail.post_match && !detail.actual_stats) && (
                  <p className="mt-2 text-[11px] text-pl-text-faint">{detail.actual_stats ? "Final team totals reported by the match-data feed. Possession is shown whenever the source provides it." : "Rest is fixture congestion; the other rows are each team&apos;s rolling per-match profile, not live betting lines."}</p>
                )}
              </section>

              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Best value bet</h3>
                {detail.recommended_bet ? (
                  <div className="rounded-xl border border-pl-cyan/40 bg-pl-cyan/10 p-3 text-sm">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <span className="font-semibold text-pl-text">{MARKET_LABELS[detail.recommended_bet.market]}</span>
                        <span className="ml-2 text-[10px] font-semibold uppercase text-pl-text-faint">{marketType(detail.recommended_bet.market)}</span>
                      </div>
                      <span className="font-semibold text-pl-cyan">+{(detail.recommended_bet.edge * 100).toFixed(1)}% edge</span>
                    </div>
                    <p className="mt-1 text-xs text-pl-text-dim">
                      Best observed price: {americanOdds(detail.recommended_bet.price)} at {detail.recommended_bet.bookmaker} · model {(detail.recommended_bet.probability * 100).toFixed(1)}%
                    </p>
                    <p className="mt-2 text-[11px] text-pl-text-faint">This highlights the strongest qualifying row below; it is educational only, never a parlay.</p>
                  </div>
                ) : (
                  <p className="rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-faint">
                    {new Date(detail.commence_time).getTime() <= Date.now()
                      ? "Kickoff has passed, so pre-match odds can no longer be used to calculate a value bet."
                      : detail.odds_is_stale
                        ? `Live odds were last fetched ${detail.odds_fetched_at ? new Date(detail.odds_fetched_at).toLocaleString() : "too long ago"}. Refresh odds before treating an edge as actionable.`
                      : detail.has_live_odds
                        ? "No value bet clears the current 5-point edge and price filters."
                        : "Live match-result and goals odds have not loaded yet, so a value bet cannot be calculated."}
                  </p>
                )}
              </section>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="flex flex-col gap-6">
                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Scoreline probability
                      <InfoTooltip text={GLOSSARY.scorelineGrid} align="left" />
                    </h3>
                    <ScorelineHeatmap
                      grid={detail.score_grid}
                      homeTeam={detail.team_home}
                      awayTeam={detail.team_away}
                      topScorelines={detail.top_scorelines}
                    />
                  </section>

                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Head-to-head
                    </h3>
                    {detail.head_to_head.length === 0 ? (
                      <p className="text-xs text-pl-text-faint">No meetings in the loaded seasons.</p>
                    ) : (
                      <div className="flex flex-col gap-1">
                        {detail.head_to_head.map((m, i) => (
                          <div key={i} className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-1.5 text-xs">
                            <span className="text-pl-text-faint">{m.date}</span>
                            <span className="font-medium text-pl-text">
                              {m.team_home} {m.goals_home}-{m.goals_away} {m.team_away}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </section>
                </div>

                <div className="flex flex-col gap-6">
                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Match result &amp; goals
                      <InfoTooltip text={GLOSSARY.edge} align="right" />
                    </h3>
                    <div className="flex flex-col gap-1.5">
                      {(["home_win", "draw", "away_win", "over_2_5", "under_2_5"] as const).map((k) => (
                        <MarketRow
                          key={k}
                          label={MARKET_LABELS[k]}
                          edge={detail[k]}
                          flagged={detail.value_bet_flags.includes(k)}
                          postMatchHit={(() => {
                            const verdict = postMatchVerdict(k === "home_win" || k === "draw" || k === "away_win" ? "Match result" : "Goals O/U 2.5");
                            const selection = k === "home_win" || k === "draw" || k === "away_win" ? k : k === "over_2_5" ? "over" : "under";
                            return verdict?.prediction === selection ? verdict.hit : undefined;
                          })()}
                        />
                      ))}
                      <MarketRow
                        label={
                          <span className="inline-flex items-center gap-1.5">
                            BTTS: Yes <InfoTooltip text={GLOSSARY.btts} align="right" />
                          </span>
                        }
                        edge={{ prob: detail.btts_yes_prob, implied: null, edge: null }}
                        flagged={false}
                        postMatchHit={postMatchVerdict("BTTS")?.prediction === "yes" ? postMatchVerdict("BTTS")?.hit : undefined}
                      />
                      {detail.home_2plus_prob !== null && (
                        <MarketRow
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_home} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          edge={{ prob: detail.home_2plus_prob, implied: null, edge: null }}
                          flagged={false}
                        />
                      )}
                      {detail.away_2plus_prob !== null && (
                        <MarketRow
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_away} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          edge={{ prob: detail.away_2plus_prob, implied: null, edge: null }}
                          flagged={false}
                        />
                      )}
                      {detail.predicted_total_goals !== null && (
                        <OverUnderRow
                          label="Total goals"
                          lam={detail.predicted_total_goals}
                          line={2.5}
                          over={detail.over_2_5.prob}
                        />
                      )}
                      {detail.predicted_margin !== null && (
                        <div className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm">
                          <span className="text-pl-text-dim">Predicted margin</span>
                          <span className="font-semibold text-pl-text">
                            {detail.predicted_margin === 0
                              ? "Even"
                              : detail.predicted_margin > 0
                                ? `${detail.team_home} by ${detail.predicted_margin.toFixed(1)}`
                                : `${detail.team_away} by ${Math.abs(detail.predicted_margin).toFixed(1)}`}
                          </span>
                        </div>
                      )}
                    </div>
                    {!detail.has_live_odds && <p className="mt-2 text-xs text-pl-text-faint">{GLOSSARY.noLiveMarket}</p>}
                  </section>

                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                      Corners &amp; cards
                      <InfoTooltip text={GLOSSARY.noLiveMarket} align="right" />
                    </h3>
                    <div className="flex flex-col gap-1.5">
                      <OverUnderRow label="Total corners" lam={detail.corners.lambda_} line={detail.corners.line} over={detail.corners.over} postMatchHit={postMatchVerdict(`Corners O/U ${detail.corners.line}`)?.hit} />
                      <OverUnderRow label="Total cards" lam={detail.cards.lambda_} line={detail.cards.line} over={detail.cards.over} postMatchHit={postMatchVerdict(`Cards O/U ${detail.cards.line}`)?.hit} />
                    </div>
                    <p className="mt-2 text-[11px] text-pl-text-faint">Use these as match-context signals (for example, a high Over chance suggests a busier game), not as verified betting edges.</p>
                  </section>
                </div>
              </div>

              <section>
                <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
                  Likely scorers &amp; assists
                  <InfoTooltip text={GLOSSARY.anytimeScorer} align="left" />
                  <InfoTooltip text={GLOSSARY.playerAvailability} align="left" />
                </h3>
                {playersError && <p className="text-xs text-loss">{playersError}</p>}
                {!players && !playersError && <p className="text-xs text-pl-text-faint">Loading player predictions…</p>}
                {players && (
                  <PlayerScorerList
                    homeTeam={detail.team_home}
                    awayTeam={detail.team_away}
                    homePlayers={players.home_players}
                    awayPlayers={players.away_players}
                  />
                )}
              </section>

              {players && <PlayerHighlights homePlayers={players.home_players} awayPlayers={players.away_players} />}

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
