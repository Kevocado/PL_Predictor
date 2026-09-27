// Synced from predictor-ui@52a31840951f. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useEffect, useId, useState } from "react";
import { ErrorState, Skeleton } from "./States";
import { StatusBadge, type Moment } from "./StatusBadge";
import { BandChip, PanelHeading, type Band } from "./ExplainerVerdict";
import { KeyNumberTile, type MarketTile } from "./KeyNumberTile";
import { FactorList, type Factor } from "./FactorList";
import { RecordStrip } from "./RecordStrip";
import { ProbabilityBar, type PickRef, type Segment } from "./ProbabilityBar";

/** What a v2 answer looks like: a verdict, a computed band, and factors that
 *  *reference* a market by key rather than carrying a figure.
 *
 *  `pick` is absent when there is no pick, which is the case the panel has to
 *  honour twice: no bar segment is accented, and nothing says "for the pick". */
export type Verdict = { verdict: string; band: Band; factors: Factor[]; pick?: PickRef };

/** The shape v1 returned. Still accepted, and removed in v2 phase 4.
 *
 *  It is here because `main` has to stay green between phases: Sports and PL are
 *  wired to v2 in phase 4, and until then their sites pass a v1 body. Rendering
 *  it is the same "do not fail a reader over a shape we did not expect" instinct
 *  the footer already follows — and it is bounded, because the branch is keyed on
 *  the field actually being present rather than on a version string a site might
 *  not send. */
export type LegacyExplanation = {
  headline: string;
  sections: { market: string; title: string; text: string }[];
};

/** Everything both shapes of answer carry. Exported so a caller that *builds* an
 *  answer can name the v2 arm (`Common & Verdict`) rather than the union: on
 *  the union, `.factors` does not exist, which is a type error at every call
 *  site that legitimately has a v2 answer in hand. */
export type Common = {
  /** "llm" means a model wrote these words; "template" means the site's own
   *  copy did, from the same numbers. The panel never blurs the two. */
  source: "llm" | "template";
  model: string;
  generated_at: string;
  /** The sport, so the panel words the rebuilt status for that moment. */
  sport: string;
  /** When the pick itself was made. "rebuilt" means the model was asked again
   *  after the event had begun, so the pick is shown but never counted. This
   *  travels with the answer rather than being read out of the prose: neither
   *  the model nor the template is a reliable source for a status label. */
  pick_timing: "pre_kickoff" | "rebuilt" | "none";
};

export type Explanation = Common & (Verdict | LegacyExplanation);

/** The moment an F1 pick has to beat is the session, not a kick-off. */
const MOMENT_OF: Record<string, Moment> = { f1: "the session", nba: "tip-off" };

/** What "after the moment" means in a sentence. StatusBadge owns the label;
 *  this owns why the pick is not being counted. */
const STARTED: Record<Moment, string> = {
  kickoff: "after the game started",
  "tip-off": "after the game started",
  "the session": "after the session started",
};

/** How long ago a summary was written, in the reader's own units. A summary
 *  from 20 seconds ago is "just now", not "0 min ago" — nobody writes that. */
function ago(iso: string, now: number): string {
  const then = Date.parse(iso);
  // Unparseable, or a clock skew so large the age would be negative. The
  // caller treats an empty age as "no age to quote" and falls back to the
  // template footer, rather than claiming a recency it cannot compute.
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} ${hours === 1 ? "hr" : "hrs"} ago`;
  const days = Math.round(hours / 24);
  return `${days} ${days === 1 ? "day" : "days"} ago`;
}

/**
 * The panel's footer is the honesty contract, and it is the one claim in the
 * product that has no fact behind it: it asserts who wrote these words. So it
 * fails CLOSED. The model-written footer is rendered only when the response
 * actually carries the two things that footer asserts — the model and a
 * timestamp we can read — and anything else (a templated summary, a missing
 * field, a renamed value on the wire, a clock we cannot parse) falls back to the
 * sentence that claims nothing. An "AI" label with a blank model beside
 * it, or with no age, is worse than no label: it looks verified and is not.
 */
function footer(data: Explanation, now: number): string {
  const TEMPLATE = "Summary written from the model's numbers";
  if (data.source !== "llm" || !data.model) return TEMPLATE;
  const age = ago(data.generated_at, now);
  if (!age) return TEMPLATE;
  return `AI summary of the model's numbers · ${data.model} · ${age}`;
}

const toggleClass =
  "font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 transition-colors hover:text-pr-text focus-visible:text-pr-text";

export function ExplainerPanel({
  data,
  loading,
  error,
  onRetry,
  collapsed = false,
  moment,
  tiles = [],
  segments,
  legend,
  expandable = false,
  record,
  players,
}: {
  data: Explanation | null;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
  collapsed?: boolean;
  /** Overrides the moment the sport implies. Sites pass their own so the
   *  status reads in the words that site already uses everywhere else. */
  moment?: Moment;
  /** One per market the facts carry. An absent market is simply not passed, and
   *  `KeyNumberTile` renders nothing for an empty one — never a dash. */
  tiles?: MarketTile[];
  segments?: Segment[];
  /** The market's own split, rendered only when it covers every outcome
   *  `segments` has (§13b). */
  legend?: Segment[];
  /** For a surface that can be too narrow for a row of figures — the market
   *  row's labels collapse behind a real `aria-expanded` button, the same shape
   *  `FactorList` uses for a clamped sentence. Off by default, so a wide panel
   *  never hides text behind a control that adds nothing. */
  expandable?: boolean;
  record?: { label: string; hits: number | null; settled: number };
  /** NFL's top player projections (§6). A list the facts already carry, so it
   *  costs the panel nothing to show. */
  players?: { name: string; projection: string }[];
}) {
  const [opened, setOpened] = useState(false);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  // useId, not a literal: two panels on one page (a fixture and a track-record
  // row, say) would otherwise share an id, and aria-labelledby would point at
  // the wrong one.
  const headingId = useId();
  const showBody = !collapsed || opened;
  const v2 = !!data && "factors" in data;

  // Escape clears a highlight, because a highlight that cannot be dismissed is
  // a stuck state and a keyboard reader has no other way out of it.
  useEffect(() => {
    if (!highlighted) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setHighlighted(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [highlighted]);

  if (loading) return <Skeleton label="Writing the summary…" />;
  if (error) {
    // The panel cannot promise the numbers below are unaffected: the summary
    // may have failed because the same fetch that serves them did. Say what is
    // true — the summary is missing, here is another go, the rest is below.
    return (
      <ErrorState
        message="The summary didn't come through. Try again, or carry on with the numbers below."
        onRetry={onRetry}
      />
    );
  }
  if (!data) return null;

  const now = Date.now();
  const when = moment ?? MOMENT_OF[data.sport] ?? "kickoff";
  const rebuilt = data.pick_timing === "rebuilt";
  const keyFor = (section: { market: string; title: string }, index: number) =>
    `${section.market}-${section.title}-${index}`;

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-3">
      <PanelHeading id={headingId}>In plain English</PanelHeading>

      <div className="flex items-start justify-between gap-3">
        <p className="min-w-0 max-w-[70ch] text-lg font-medium leading-snug text-pr-text">
          {v2 ? data.verdict : data.headline}
        </p>
        {/* A rebuilt pick shows NO band (spec §13e), and the decision is this one
            line. The band is still computed by `band_for` and still in the
            response — withholding a value the panel cannot act on is a rendering
            decision, and keeping it computed keeps it available to whatever reads
            the row next. It is withheld rather than softened because the number it
            describes came from a model asked after the event began: it had seen
            the score, so the band measures confidence in a number produced with
            the answer already known, and the panel is telling the reader two lines
            below not to count or grade it. "STRONG" beside that asks the reader to
            resolve a contradiction this panel created. Hiding it fails closed,
            which is the direction the footer already fails in. */}
        {v2 && !rebuilt && <BandChip band={data.band} />}
      </div>

      {rebuilt && (
        <p className="flex max-w-[70ch] flex-wrap items-center gap-2 text-sm text-pr-text-dim">
          <StatusBadge status="rebuilt" moment={when} />
          <span>
            This pick was made {STARTED[when]}, so it is shown for reference and not counted.
          </span>
        </p>
      )}

      {collapsed && (
        <button
          type="button"
          onClick={() => setOpened((was) => !was)}
          aria-expanded={showBody}
          className={`mt-1 self-start underline ${toggleClass}`}
        >
          {showBody ? "Hide the race story" : "Read the race story"}
        </button>
      )}

      {showBody && v2 && (tiles.length > 0 || segments) && (
        <div className="flex flex-col gap-3">
          {tiles.length > 0 && (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {tiles.map((tile) => (
                <KeyNumberTile key={tile.market} tile={tile} highlighted={highlighted === tile.market} />
              ))}
            </div>
          )}
          {segments && segments.length > 0 && (
            <ProbabilityBar
              segments={segments}
              legend={legend}
              minSegmentPx={2}
              expandable={expandable}
              // The answer's pick, never worked out here. Absent, the bar accents
              // nothing, which is the honest rendering of a bar with no pick.
              pick={v2 ? data.pick : null}
              highlightKey={highlighted}
              onSegmentFocus={(_, market) => market && setHighlighted(market)}
            />
          )}
        </div>
      )}

      {showBody && v2 && (
        <div className="flex flex-col gap-2 border-t border-pr-rule pt-3">
          <h4 className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
            Why it matters
          </h4>
          <FactorList
            factors={data.factors}
            onSelect={(key) => setHighlighted((was) => (was === key ? null : key))}
            expanded={expanded}
            onToggle={(i) => setExpanded((was) => (was === i ? null : i))}
          />
        </div>
      )}

      {showBody && !v2 && data.sections.length > 0 && (
        <div className="flex max-w-[70ch] flex-col gap-4 border-t border-pr-rule pt-4">
          {data.sections.map((section, index) => (
            <div key={keyFor(section, index)} className="flex flex-col gap-1">
              {/* _clean coerces a missing title to "", and validate.py bounds a
                  title's length without requiring one, so an empty heading is
                  reachable. A heading with no words is not a heading. */}
              {section.title && (
                <h4 className="font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">
                  {section.title}
                </h4>
              )}
              <p className="text-sm leading-relaxed text-pr-text-dim">{section.text}</p>
            </div>
          ))}
        </div>
      )}

      {/* The record strip and the player list are panel EXTRAS (§6), not
          factors, so they are not the v2 body's business — but they are not the
          v1 body's exclusive property either. This was gated on `!v2` and the
          only thing that noticed was a screenshot: every v2 panel silently
          dropped its record, and no test asserted it, because the component test
          rendered RecordStrip directly and never through the panel. */}
      {showBody && (record || players) && (
        <div className="flex flex-wrap items-start gap-x-8 gap-y-3 border-t border-pr-rule pt-3">
          {players && players.length > 0 && (
            <div className="flex min-w-0 flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">Top projections</span>
              <ul className="flex flex-col gap-0.5">
                {players.slice(0, 3).map((pl) => (
                  <li key={pl.name} className="flex items-baseline justify-between gap-4 text-xs">
                    <span className="truncate text-pr-text">{pl.name}</span>
                    <span className="shrink-0 tabular-nums text-pr-text-dim">{pl.projection}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {record && <RecordStrip label={record.label} hits={record.hits} settled={record.settled} />}
        </div>
      )}

      <p className="mt-1 text-xs text-pr-text-faint">{footer(data, now)}</p>
    </section>
  );
}
