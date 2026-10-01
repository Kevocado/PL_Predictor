import { useEffect, useMemo, useState } from "react";
import type { Explanation } from "../predictor-ui";
import { FixtureExplainer } from "../predictor-ui";
import { panelFacts } from "../predictor-ui/lib/panelFacts";
import type { FixtureDetail, FixturePlayerReview, FixturePlayers, FixturePostMatch, FixtureValueBetSnapshot, TrackRecordSummary } from "../types";
import { api } from "../api/client";
import { TeamBadge } from "./TeamBadge";
import { ScorelineHeatmap } from "./ScorelineHeatmap";
import { FormStrip } from "./FormStrip";
import { InfoTooltip } from "./InfoTooltip";
import { MarketBar } from "./MarketBar";
import { PlayerHighlights, PlayerScorerList } from "./PlayerScorerList";
import { GLOSSARY } from "../lib/glossary";
import { isModelCall } from "../lib/modelCall";
import { resolvedPick } from "../lib/pick";

interface Props {
  eventId: string;
  onClose: () => void;
  // Fetches the plain-English summary. Optional on purpose: a fixture the
  // explainer has nothing for, or a site deployed before the service exists,
  // must still open this modal and show everything else in it.
  explain?: (sport: string, id: string) => Promise<Explanation>;
  sport?: string;
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

function OverUnderRow({ label, lam, line, over, postMatchHit, modelCall }: { label: string; lam: number; line: number; over: number; postMatchHit?: boolean; modelCall?: boolean }) {
  const rowClass =
    postMatchHit === true
      ? "bg-win/10 ring-1 ring-win/30"
      : postMatchHit === false
        ? "bg-loss/10"
        : modelCall
          ? "bg-pl-cyan/10 ring-1 ring-pl-cyan/40"
          : "bg-pl-850/60";
  return (
    <div className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${rowClass}`}>
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

// A field that is ABSENT is `undefined`, and `undefined !== null` is true, so a
// `!== null` guard waves it straight through to `undefined.toFixed(...)`.
// 30 of the rows in data/public_snapshot.json are backfilled finished fixtures
// that carry no shot projections and no predicted_result key at all, and they
// blanked the whole modal. Ask whether there IS a number rather than whether it
// is something other than null, so absent and null land in the same honest
// branch: the row is left out. Never filled in, never coerced to 0.
function hasNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined;
}

function reportedMetric(value: number | null | undefined, label: string) {
  if (value === null || value === undefined) return "—";
  const rounded = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return label.includes("%") ? `${rounded}%` : rounded;
}

function PostMatchReview({ review }: { review: FixturePostMatch }) {
  const correct = review.verdicts.filter((verdict) => verdict.hit).length;
  // No provenance chip here any more. The panel's instant block carries the
  // timing once, as a badge or the quiet chip, and that is the one place a
  // reader learns how to read every figure below it -- so this review states
  // only what it reviewed, and the block states the timing. The paragraph that
  // used to sit under the verdicts went with it: it claimed a rebuilt pick is
  // "counted in the track record like any other pick", which is the opposite
  // of what the record counts (types.ts:517-521) and of what the block says.
  return (
    <section className="rounded-xl border border-win/30 bg-win/5 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div><h3 className="text-sm font-semibold text-pl-text">Prediction review, final {review.final_score.replace("-", "–")}</h3><p className="mt-0.5 text-xs text-pl-text-dim">{correct}/{review.verdicts.length} match calls correct</p></div>
      </div>
      <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
        {review.verdicts.map((verdict) => <div key={verdict.label} className={`flex items-center justify-between rounded-lg px-3 py-2 text-xs ${verdict.hit ? "bg-win/10 text-win" : "bg-pl-850/70 text-pl-text-dim"}`}><span className="font-semibold">{verdict.hit ? "✓" : "×"} {verdict.label}</span><span><span className="text-pl-text-faint">{verdict.prediction}</span><span className="mx-1">→</span><span>{verdict.actual}</span></span></div>)}
      </div>
    </section>
  );
}

function PreMatchValueBets({ bets }: { bets: FixtureValueBetSnapshot[] }) {
  return (
    <section>
      <div className="mb-2 flex flex-wrap items-end justify-between gap-1">
        <h3 className="text-sm font-semibold text-pl-text-dim">Pre-match value bets</h3>
        <p className="text-xs text-pl-text-faint">Saved when first flagged; prices and edges never change afterwards.</p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {bets.map((bet) => {
          const status = !bet.resolved ? "Awaiting result" : bet.won ? "Won" : "Lost";
          const statusClass = !bet.resolved ? "bg-pl-700/60 text-pl-text-dim" : bet.won ? "bg-win/20 text-win" : "bg-loss/15 text-loss";
          return (
            <div key={bet.market} className={`rounded-xl border p-3 ${bet.won === true ? "border-win/30 bg-win/5" : bet.won === false ? "border-loss/25 bg-loss/5" : "border-pl-cyan/30 bg-pl-cyan/5"}`}>
              <div className="flex items-center justify-between gap-2">
                <div><span className="font-semibold text-pl-text">{MARKET_LABELS[bet.market] ?? bet.market}</span><span className="ml-2 text-xs font-semibold uppercase text-pl-text-faint">{marketType(bet.market)}</span></div>
                <span className={`rounded px-2 py-1 text-xs font-semibold uppercase ${statusClass}`}>{status}</span>
              </div>
              <p className="mt-1 text-xs text-pl-text-dim">
                {americanOdds(bet.price)}{bet.bookmaker ? ` at ${bet.bookmaker}` : ""}, model {(bet.probability * 100).toFixed(1)}% vs market {(bet.implied_probability * 100).toFixed(1)}%
              </p>
              <div className="mt-2 flex items-center justify-between text-xs">
                <span className="font-semibold text-pl-cyan">+{(bet.edge * 100).toFixed(1)}% edge</span>
                {bet.clv_pct !== null && (
                  <span className={`font-semibold ${bet.clv_pct >= 0 ? "text-win" : "text-loss"}`} title={GLOSSARY.clv}>
                    {bet.clv_pct >= 0 ? "+" : ""}{bet.clv_pct.toFixed(1)}% CLV
                  </span>
                )}
                {bet.final_score && <span className={bet.won ? "text-win" : "text-loss"}>Final {bet.final_score}</span>}
              </div>
            </div>
          );
        })}
      </div>
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
    return <div key={`${player.team}-${player.name}`} className={`flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-xs ${tone}`}><span className="min-w-0 font-semibold text-pl-text">{state === "hit" ? "✓" : state === "miss" ? "×" : "↑"} {player.name} <span className="font-normal text-pl-text-faint">{player.team}</span>{player.is_recommended && <span className="ml-2 rounded bg-pl-pink/20 px-1.5 py-0.5 text-xs font-semibold text-pl-pink">Recommended</span>}</span><span className={`shrink-0 text-right ${hit ? state === "longShot" ? "text-pl-blue" : "text-win" : "text-pl-text-dim"}`}><span className="block">{hit ? outcome(player) : "No goal involvement"}</span><span className="text-pl-text-faint">{signal}</span></span></div>;
  });
  return <section className="rounded-xl border border-pl-border bg-pl-850/50 p-4">
    {/* The heading is not a repeat: it names this block, which the instant
        block's badge does not. The chip beside it was, so it is gone -- the
        badge states the timing once for the whole panel. */}
    <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold text-pl-text">Player call review</h3></div>
    {loading && <p className="mt-2 text-xs text-pl-text-faint">Reconstructing confirmed player calls…</p>}
    {error && <p className="mt-2 text-xs text-loss">{error}</p>}
    {!loading && !error && !review && <p className="mt-2 text-xs text-pl-text-faint">Official player outcomes are not available for this fixture yet.</p>}
    {review && <div className="mt-3 grid gap-3 xl:grid-cols-3"><div><p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Correct calls</p><div className="flex flex-col gap-1">{review.correct.length ? rows(review.correct, "hit") : <p className="rounded-lg bg-pl-900/50 px-3 py-2 text-xs text-pl-text-faint">No tiered call recorded a goal involvement.</p>}</div></div><div><p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Confident calls that missed</p><div className="flex flex-col gap-1">{review.missed.length ? rows(review.missed, "miss") : <p className="rounded-lg bg-pl-900/50 px-3 py-2 text-xs text-pl-text-faint">No confident calls missed.</p>}</div></div><div><p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Overperformers</p><div className="flex flex-col gap-1">{review.overperformed.length ? rows(review.overperformed, "longShot") : <p className="rounded-lg bg-pl-900/50 px-3 py-2 text-xs text-pl-text-faint">No low-probability player outperformed the thresholds.</p>}</div></div></div>}
  </section>;
}

export function FixtureModal({ eventId, onClose, explain, sport = "pl" }: Props) {
  const [detail, setDetail] = useState<FixtureDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<FixturePlayers | null>(null);
  const [playersError, setPlayersError] = useState<string | null>(null);
  const [playerReview, setPlayerReview] = useState<FixturePlayerReview | null>(null);
  const [playerReviewError, setPlayerReviewError] = useState<string | null>(null);
  const [playerReviewLoading, setPlayerReviewLoading] = useState(false);
  const [recordSummary, setRecordSummary] = useState<TrackRecordSummary | null>(null);
  const postMatchVerdict = (label: string) => detail?.post_match?.verdicts.find((verdict) => verdict.label === label);

  // The panel's figures, derived rather than fetched, from the SHARED adapter.
  // The pick translation the bar depends on lives in the shared component now,
  // applied against the same segments it draws.
  const panel = useMemo(
    () =>
      detail
        ? panelFacts({ kind: "PL", fixture: detail })
        : { tiles: [], segments: [], legend: [] as never[] },
    [detail],
  );

  // The record strip's two numbers, read out of the track record rather than
  // computed by this page. `RecordStrip` prints `hits/settled` and reads the
  // proportion off the bar's width, so a percentage is never derived here: the
  // strip exists because the facts carry two counts and no third number.
  //
  // Both counts come from `summary`, and the `TrackRecordSummary` comment at
  // types.ts:517-521 is the reason this strip may call itself the record at
  // all: that headline counts ONLY picks made before kickoff, and a pick
  // rebuilt after the match cannot be one the model would have made on the
  // night. So the figure the badge refuses to count is also the figure this
  // strip does count, and the label says which.
  //
  // `null` until the fetch resolves, so the strip never flashes 0/0 at a
  // record that has not arrived yet.
  const record = useMemo(() => {
    if (!recordSummary) return null;
    const settled = recordSummary.n_resolved_fixtures;
    return {
      label: "Picks made before kickoff",
      hits: Math.round((recordSummary.pct_correct_overall ?? 0) * settled),
      settled,
    };
  }, [recordSummary]);

  // The flow's facts: what this modal already holds, no request. PL carries no
  // market line, so the flow says the pick and stops; the finished flow reads
  // the final score and whether the pick was right, both from the detail, and
  // nothing is claimed before the data carries it.
  const finite = (x: unknown): number | undefined =>
    typeof x === "number" && Number.isFinite(x) ? x : undefined;
  const flowBundle = useMemo(() => {
    if (!detail) return null;
    const probs = {
      home_win: finite(detail.home_win?.prob),
      draw: finite(detail.draw?.prob),
      away_win: finite(detail.away_win?.prob),
    };
    // Which side the model picked, resolved by the shared rule rather than
    // re-derived here: the review's side first, the display variant only where
    // there is no review, and a label in the vocabulary the bar's segments use.
    // `resolvedPick`'s comment carries the measurements and the two fallbacks;
    // this comment only says why the rule is not written out again.
    const pick = resolvedPick(detail);
    const postMatch = detail.post_match;
    const score = postMatch
      ? (() => {
          const m = postMatch.final_score.match(/(\d+)\s*-\s*(\d+)/);
          return m ? { home: Number(m[1]), away: Number(m[2]) } : undefined;
        })()
      : undefined;
    return {
      home_team: detail.team_home,
      away_team: detail.team_away,
      home_win_prob: probs.home_win,
      away_win_prob: probs.away_win,
      // When this pick was made, in the one vocabulary the block reads. A stored
      // pre-match snapshot IS a pick made before kickoff, so it carries no flag
      // and the block's quiet chip is the whole statement; a reconstructed
      // review is a pick rebuilt after the match, which the block badges and
      // refuses to count.
      pick_timing: detail.post_match?.provenance === "reconstructed" ? "rebuilt" : undefined,
      pick,
      score,
      result: !score
        ? undefined
        : score.home === score.away
          ? "draw"
          : score.home > score.away
            ? "home_win"
            : "away_win",
    };
  }, [detail]);
  const flowState = detail?.post_match ? "finished" : "pre-game";

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

  // The track record is the site's, not this fixture's, so it is fetched once
  // when the modal opens and never re-fetched as the reader moves between
  // fixtures. One request, and it is not the explainer's: it asks for a record
  // the site already publishes, and the block is on screen before it lands.
  // Guarded by the same `cancelled` flag the detail fetch uses, so an unmount
  // mid-flight cannot setState. A failure here costs one strip -- the modal is
  // complete without it, so it says nothing rather than reporting an error
  // about a figure the reader never asked for.
  useEffect(() => {
    let cancelled = false;
    api
      .trackRecord()
      .then((response) => {
        if (!cancelled) setRecordSummary(response.summary);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      <div className="animate-modal-in relative flex max-h-[90vh] w-full max-w-7xl flex-col overflow-hidden rounded-2xl border border-pl-border bg-pl-900 shadow-2xl">
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
          {/* In plain English, first: it is the one-screen answer the rest of
              this modal is the evidence for. The flow renders from facts this
              modal already holds, with no request; the AI summary sits behind
              the button and costs nothing until a reader asks. */}
          {explain && detail && (
            <div className="mb-6">
              <FixtureExplainer
                sport={sport}
                state={flowState}
                bundle={flowBundle}
                request={() => explain(sport, eventId)}
                extras={{ tiles: panel.tiles, segments: panel.segments, legend: panel.legend, record: record ?? undefined, moment: "kickoff" }}
              />
            </div>
          )}

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
                  {detail.post_match ? (
                    <><span className="text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Final</span><span className="font-display text-3xl font-semibold tracking-wide text-pl-text">{detail.post_match.final_score.replace("-", "–")}</span></>
                  ) : <span className="font-display text-2xl font-semibold tracking-wide text-pl-text-faint">vs</span>}
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
                <h3 className="mb-2 text-sm font-semibold text-pl-text-dim">
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
                  <p className="mt-2 text-xs text-pl-text-faint">{detail.actual_stats ? "Final team totals reported by the match-data feed. Possession is shown whenever the source provides it." : "Rest is fixture congestion; the other rows are each team&apos;s rolling per-match profile, not live betting lines."}</p>
                )}
              </section>

              {detail.post_match && (
                detail.pre_match_value_bets?.length > 0 ? <PreMatchValueBets bets={detail.pre_match_value_bets} /> : (
                  <section>
                    <h3 className="mb-2 text-sm font-semibold text-pl-text-dim">Pre-match value bets</h3>
                    <p className="rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-faint">No value bet qualified before kickoff for this fixture, so no pre-match bet was recorded.</p>
                  </section>
                )
              )}

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="flex flex-col gap-6">
                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
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
                    <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
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
                  {/* The market bars that stood here are gone, and only those.
                      The 1x2 block is the `result` tile and the three-way bar in
                      the instant block above, the O/U 2.5 pair is the same bar's
                      `total goals` tile plus the goals it quotes, and BTTS is
                      the `both score` tile -- each of those figures was on this
                      page twice, in two different components, and the panel's
                      own copy is the one a reader sees before spending a
                      request. What is left here is what no tile or bar carries:
                      the team scoring 2+, the margin, corners, cards and shots. */}
                  <section>
                    <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
                      More model calls
                      {detail.draw_signal && (
                        <span
                          className="rounded-full bg-pl-cyan/10 px-2 py-0.5 text-xs font-semibold normal-case tracking-normal text-pl-cyan"
                          title="The scoreline model's top pick and the win/draw/loss percentages both lean draw. Informational only — not used to score accuracy."
                        >
                          Model leans draw
                        </span>
                      )}
                    </h3>
                    <div className="flex flex-col gap-1.5">
                      {!detail.post_match && detail.recommended_bet && (
                        <div className="flex flex-col gap-0.5 rounded-lg bg-pl-850/60 px-3 py-2 text-sm">
                          <span className="text-pl-text-dim">Best observed price</span>
                          <span className="font-semibold text-pl-text">
                            {americanOdds(detail.recommended_bet.price)} at {detail.recommended_bet.bookmaker} ·{" "}
                            <span className="font-mono font-semibold text-pl-cyan">
                              +{(detail.recommended_bet.edge * 100).toFixed(1)}%
                            </span>{" "}
                            edge
                          </span>
</div>
                      )}
                      {hasNumber(detail.home_2plus_prob) && (
                        <MarketBar
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_home} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          prob={detail.home_2plus_prob}
                        />
                      )}
                      {hasNumber(detail.away_2plus_prob) && (
                        <MarketBar
                          label={
                            <span className="inline-flex items-center gap-1.5">
                              {detail.team_away} to score 2+ <InfoTooltip text={GLOSSARY.teamTwoPlus} align="right" />
                            </span>
                          }
                          prob={detail.away_2plus_prob}
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
                      <OverUnderRow label="Total corners" lam={detail.corners.lambda_} line={detail.corners.line} over={detail.corners.over} postMatchHit={postMatchVerdict(`Corners O/U ${detail.corners.line}`)?.hit} modelCall={isModelCall(detail.corners.over)} />
                      <OverUnderRow label="Total cards" lam={detail.cards.lambda_} line={detail.cards.line} over={detail.cards.over} postMatchHit={postMatchVerdict(`Cards O/U ${detail.cards.line}`)?.hit} modelCall={isModelCall(detail.cards.over)} />
                      {hasNumber(detail.home_shots) && hasNumber(detail.away_shots) && (
                        <div className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm">
                          <span className="text-pl-text-dim">Predicted shots</span>
                          <span className="font-semibold text-pl-text">
                            {detail.team_home} {detail.home_shots.toFixed(1)} · {detail.team_away} {detail.away_shots.toFixed(1)}
                          </span>
                        </div>
                      )}
                      {hasNumber(detail.home_shots_on_target) && hasNumber(detail.away_shots_on_target) && (
                        <div className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm">
                          <span className="text-pl-text-dim">Predicted shots on target</span>
                          <span className="font-semibold text-pl-text">
                            {detail.team_home} {detail.home_shots_on_target.toFixed(1)} · {detail.team_away} {detail.away_shots_on_target.toFixed(1)}
                          </span>
                        </div>
                      )}
                    </div>
                    {!detail.has_live_odds && <p className="mt-2 text-xs text-pl-text-faint">{GLOSSARY.noLiveMarket}</p>}
                    {!detail.post_match && !detail.recommended_bet && (
                      <p className="mt-2 text-xs text-pl-text-faint">
                        {new Date(detail.commence_time).getTime() <= Date.now()
                          ? "Kickoff has passed, so pre-match odds can no longer be used to calculate a value bet."
                          : detail.odds_is_stale
                            ? `Live odds were last fetched ${detail.odds_fetched_at ? new Date(detail.odds_fetched_at).toLocaleString() : "too long ago"}. Refresh odds before treating an edge as actionable.`
                          : detail.has_live_odds
                            ? "No value bet clears the current edge and price filters."
                            : "Live match-result and goals odds have not loaded yet, so a value bet cannot be calculated."}
                      </p>
                    )}
                    <p className="mt-2 text-xs text-pl-text-faint">Corners/cards/shots are match-context signals (for example, a high Over chance suggests a busier game), not verified betting edges.</p>
                  </section>
                </div>
              </div>

              <section>
                <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
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
