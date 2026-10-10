import { useEffect, useMemo, useState } from "react";
import type { Explanation, MatchupContext } from "../predictor-ui";
import { FixtureExplainer, SignalRows, type Signal } from "../predictor-ui";
import { panelFacts } from "../predictor-ui/lib/panelFacts";
import type { FixtureDetail, FixturePlayerReview, FixturePlayers, FixturePostMatch, FixtureValueBetSnapshot, TrackRecordSummary } from "../types";
import {
  fetchSignals,
  api,
} from "../api/client";
import { TeamBadge } from "./TeamBadge";
import { ScorelineHeatmap } from "./ScorelineHeatmap";
import { FormStrip } from "./FormStrip";
import { InfoTooltip } from "./InfoTooltip";
import { MarketBar } from "./MarketBar";
import { ModelTopCalls, PlayerScorerList } from "./PlayerScorerList";
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
  // The Matchup section's data loader, beside `explain`; same optionality.
  loadContext?: (id: string) => Promise<MatchupContext>;
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
  // used to sit under the verdicts went with it: it described a rebuilt pick's
  // standing in the record, and since the 2026-10-01 reversal a pick made after
  // the start IS counted (see `TrackRecordSummary` in types.ts), so the block's
  // own moment badge is the whole of what a reader needs here.
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

export function FixtureModal({ eventId, onClose, explain, loadContext, sport = "pl" }: Props) {
  const [detail, setDetail] = useState<FixtureDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<FixturePlayers | null>(null);
  const [playersError, setPlayersError] = useState<string | null>(null);
  const [playerReview, setPlayerReview] = useState<FixturePlayerReview | null>(null);
  const [playerReviewError, setPlayerReviewError] = useState<string | null>(null);
  const [playerReviewLoading, setPlayerReviewLoading] = useState(false);
  const [recordSummary, setRecordSummary] = useState<TrackRecordSummary | null>(null);
  // Spec §4's signal rows. `[]` and never null once settled, and `[]` is also what
  // every failure leaves behind -- see the effect below.
  const [signals, setSignals] = useState<Signal[]>([]);
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
  // It reads `pre_kickoff`, NOT the headline. The label says "made before
  // kickoff" and that is a claim: since the 2026-10-01 reversal
  // `n_resolved_fixtures` / `pct_correct_overall` count every recorded pick,
  // whenever it was made, so reading the headline here would put picks made
  // after the start under a heading saying they were made before it — on the
  // surface a reader trusts most, beside a started fixture's stored pick.
  //
  // `null` until the fetch resolves, so the strip never flashes 0/0 at a
  // record that has not arrived yet, and `null` when `pre_kickoff` is absent
  // (a snapshot baked before the field existed) rather than falling back to the
  // headline — a fallback here is precisely the mislabel above.
  const record = useMemo(() => {
    const pre = recordSummary?.pre_kickoff;
    if (!recordSummary || !pre) return null;
    const settled = pre.n_resolved_fixtures;
    return {
      label: "Picks made before kickoff",
      hits: Math.round((pre.pct_correct_overall ?? 0) * settled),
      settled,
    };
  }, [recordSummary]);

  // The flow's state, derived ONCE and used for BOTH the bundle below and the
  // `state` prop, because the rule this PR adds is a rule about the PAIR: the
  // bundle withholds the team names when the flow has nothing to say, and the
  // prop has to be asking the flow for that same nothing. Two separate
  // derivations of "is this finished?" is how the live-game case drifts, so it
  // is written once.
  //
  // This site has no `in-play` state to ask for, and that is measured rather
  // than assumed: `FixtureDetail` carries no live score field at all -- its one
  // score is `post_match.final_score` -- so a fixture that has kicked off but
  // has no `post_match` yet has no score sentence to say and would render the
  // in-play branch empty. Mapping it to `pre-game` is therefore not a state this
  // site is hiding; it is the same empty flow, reached honestly. What was NOT
  // honest before this change was that the empty flow still carried a heading.
  const flowState: "pre-game" | "finished" = detail?.post_match ? "finished" : "pre-game";

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
    /* THE BARE HEADING, and why the sides are spelled two ways here.
       Measured per state by rendering the modal and dumping the flow's rows,
       not by reading a screenshot:

         - pre-game — the flow's ONLY row was the fixture's own name, rendered
                      as an `H4`, so the live modal showed `Tottenham vs Aston
                      Villa` with nothing under it, directly above the AI button.
         - in-play  — `FixtureDetail` has no live score, so a kicked-off fixture
                      with no `post_match` yet takes this same branch and
                      rendered the SAME bare heading.
         - finished — TWO real sentences (the result, and the pick's rightness).
                      Live content, and it stays.

       So this is not a deletion of the flow; it is the heading made conditional
       on rows existing beneath it, expressed in the only place a site may
       express it -- the bundle it hands over, because `predictor-ui/` is
       vendored and must not change.

       `FixtureFlow` reads the sides under `home_team`/`away_team` for BOTH its
       pre-game name row and its finished result sentence, so one spelling
       cannot suppress the first without breaking the second. `bundleFacts` --
       the single place the block's verdict sentence goes through -- reads
       `home_team ?? team_home`, and both spellings are that package's own
       documented contract. So:

         - ALWAYS carry `team_home`/`team_away`, which `fullTeamName` reads to
           resolve the pick's label to a side. Without it the block would fall
           back to nothing and "Tottenham is the pick." would be lost.
         - carry `home_team`/`away_team` ONLY once a result makes a sentence
           that needs them. Pre-game there is no such sentence, and withholding
           the keys the name row reads is what makes that row impossible rather
           than merely unlikely.

       The block's figures depend on neither key: `panelFacts` above is handed
       this site's own fixture, not the bundle, so the tiles and the bar are
       identical either way. */
    const namesForSentences = flowState !== "pre-game";
    return {
      team_home: detail.team_home,
      team_away: detail.team_away,
      ...(namesForSentences
        ? { home_team: detail.team_home, away_team: detail.team_away }
        : {}),
      home_win_prob: probs.home_win,
      away_win_prob: probs.away_win,
      // When this pick was made, in the one vocabulary the block reads. A stored
      // pre-match snapshot IS a pick made before kickoff, so it carries no flag
      // and the block's quiet chip is the whole statement; a reconstructed
      // review is a pick made after the match, which the block badges. The badge
      // says WHEN and nothing about the pick's standing in the record — it is
      // counted, one per (fixture, market) from the earliest recorded one,
      // whenever it was made (spec 2026-10-01-track-record-counts-every-pick).
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
  }, [detail, flowState]);

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
  // The signal rows (spec §4). Fetched here, on open, rather than from behind any
  // paid control: §2 makes signals INSTANT — computed from stored data, "no model
  // call" — so putting them behind a click would make a free computed row a paid
  // one.
  //
  // EVERY failure is silence, and the failures are deliberately indistinguishable:
  // a 404, a network error and an honest empty list all leave `[]`. Spec §2's "no
  // data, no row" forbids a placeholder, and a signal is an enhancement on this
  // page — it must never become the page's error state. Sports' `GameDetailModal`
  // does the same at the same place, so "a fixture page shows its signals" is one
  // implementation and not one per sport.
  //
  // `fetchSignals`, which bypasses the 45 s `readCache` every other read on `api`
  // goes through. See the client's own note for why this endpoint has no `api`
  // entry at all.
  //
  // NOT fetched for a finished gameweek. `/api/signals/{event_id}` answers `[]` for
  // one, because `facts.game_context` empties the player pool once a fixture is
  // live or final — the squad list it holds is the one known BEFORE kick-off, and
  // quoting it afterwards is hindsight. Asking for an answer already known is one
  // request per finished gameweek for nothing, so this asks only when it can differ.
  //
  // GATED ON `detail` BEING LOADED, and that is the fix for a bug this effect had
  // on its first run: `detail?.post_match` is `undefined` on the first render,
  // which is falsy, so the fetch fired for EVERY fixture including finished ones
  // and only the re-run after the detail landed suppressed it. One wasted request
  // per finished gameweek, and — worse — a row that could arrive before the page
  // knew whether it was allowed to show one.
  useEffect(() => {
    let cancelled = false;
    setSignals([]);
    // `detail.event_id === eventId`, NOT `!detail`. Caught by CodeRabbit on #55.
    //
    // When the reader switches fixtures the modal KEEPS the previous `detail` in
    // state until the new one lands (the reset is a `setDetail(null)` inside
    // another effect, and an effect reads the value captured by ITS OWN render).
    // So on the render where `eventId` has just changed, `detail` is the OLD
    // fixture's -- and a gate of `!detail` or `detail.post_match` would read the
    // previous fixture's status, fetch for the NEW id, and could put the OLD
    // fixture's absence row on the new fixture's page.
    //
    // Comparing the id the detail was FOR against the id being asked about is the
    // only gate that cannot be stale.
    if (!detail || detail.event_id !== eventId) return;
    if (detail.post_match) return;
    fetchSignals(eventId)
      .then((response) => {
        if (!cancelled) setSignals(response?.signals ?? []);
      })
      // Swallow, and do NOT repeat the clear: the head of this effect has already
      // set `[]`, on this run and on every re-run, so a rejection lands on `[]`
      // either way. Writing `setSignals([])` here as well would be a second place
      // to keep the same rule, and a mutation deleting it proved it changed
      // nothing -- 8 tests green either way. The `.catch` exists only so a 404 is
      // not an unhandled rejection.
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [eventId, detail]);

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
                loadContext={loadContext}
                fixtureId={eventId}
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

              {/* The signal rows (spec §4), above the explainer and before the
                  panels -- the same position Sports' `GameDetailModal` puts them,
                  so a fixture page reads the same way in both places.

                  `{signals.length > 0 && ...}` rather than a guard around the whole
                  block: `SignalRows` already drops undrawable rows and throws on a
                  figure it cannot render, and spec §2 says a fixture with nothing to
                  say renders NO rows -- not an empty section, not a heading. */}
              {signals.length > 0 && (
                <section className="flex flex-col gap-2">
                  <SignalRows signals={signals} />
                </section>
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

              {players && (
                <ModelTopCalls
                  homeTeam={detail.team_home}
                  awayTeam={detail.team_away}
                  homePlayers={players.home_players}
                  awayPlayers={players.away_players}
                  kickoff={detail.commence_time}
                />
              )}


            </div>
          )}
        </div>
      </div>
    </div>
  );
}
