// Synced from predictor-ui@761c65a6c5f8. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** FixtureExplainer — the panel, three states.
 *
 *  The default state is the flow: an instant, local description built from facts
 *  the site already has, with no request made. The AI summary is behind a
 *  button, so nothing is spent until a reader asks. When the button's request
 *  resolves, the summary swaps in over a flow that never unmounts — there is no
 *  frame in which the panel is empty. When the request fails, the flow is still
 *  on screen and the button offers a retry.
 *
 *  The flow is always mounted. It is the one state that is always true, and
 *  keeping it mounted is what "the flow does not flicker away" means: the
 *  summary appears above it, not instead of it after a gap.
 */
import { useState } from "react";
import { FixtureFlow, type FlowState } from "./FixtureFlow";
import { SummaryButton } from "./SummaryButton";
import { ExplainerPanel, type Explanation } from "./ExplainerPanel";
import type { Moment } from "./StatusBadge";
import type { MarketTile } from "./KeyNumberTile";
import type { Segment } from "./ProbabilityBar";
import { barPick } from "../lib/panelFacts";

/** A summary as the service sends it: a v2 answer, plus the optional news date.
 *  The news date arrives with the summary body; when it is absent the element is
 *  omitted entirely, so the panel ships before the service sends the field. */
export type Summary = Explanation & { news_date?: string };

/** The site's own figures, drawn under the AI prose in the summary state.
 *
 *  The panel renders no figure of its own: the summary carries the words and
 *  the site carries the numbers. A site that already draws tiles, a bar, a
 *  record strip or a player list beside its panel passes them here and the
 *  summary state keeps them; a site with nothing to draw passes nothing and
 *  the summary renders bare. Every field is optional for exactly that reason.
 *
 *  `segments` does double duty: it draws the bar, and it is what the answer's
 *  pick is joined against. The service words a pick in its own vocabulary
 *  ("Ravens win", the shape PL ships) while a site labels its segments in
 *  theirs ("Ravens"), so the pick is translated through `barPick` before it
 *  reaches the panel — exactly label matches pass through, a trailing " win"
 *  is dropped onto a matching segment, and anything unplaceable is returned
 *  unchanged so the bar fails closed (nothing accented) rather than accenting
 *  the nearest segment to a claim nobody made.
 */
export interface FixtureExtras {
  tiles?: MarketTile[];
  segments?: Segment[];
  legend?: Segment[];
  record?: { label: string; hits: number | null; settled: number };
  players?: { name: string; projection: string }[];
  moment?: Moment;
}

/** The panel's three states. `unavailable` differs from `flow` only in what the
 *  button offers: a retry, rather than a first ask. */
type PanelState = "flow" | "summary" | "unavailable";

export interface FixtureExplainerProps {
  sport: string;
  state: FlowState;
  bundle: any;
  /** The panel's request function for the AI summary. Passed to both children;
   *  the flow never calls it. */
  request: () => Promise<unknown>;
  /** The site's own figures for the summary state. Absent by default. */
  extras?: FixtureExtras;
}

/** The AI prose, rendered from the summary the button fetched. Reuses the
 *  existing panel so the verdict, band, factors and footer are the same ones
 *  every site already shows, with the site's own figures underneath. */
function SummaryView({ summary, extras }: { summary: Summary; extras?: FixtureExtras }) {
  const segments = extras?.segments ?? [];
  const pick = summary.pick ? barPick(summary.pick, segments) : undefined;
  const data = pick ? { ...summary, pick } : summary;
  return (
    <div className="flex flex-col gap-3" data-testid="fixture-summary">
      <ExplainerPanel
        data={data}
        loading={false}
        error={false}
        onRetry={() => {}}
        tiles={extras?.tiles}
        segments={extras?.segments}
        legend={extras?.legend}
        record={extras?.record}
        players={extras?.players}
        moment={extras?.moment}
      />
      {summary.news_date && (
        <p className="text-xs text-pr-text-faint" data-testid="news-date">
          Team news from {summary.news_date}
        </p>
      )}
    </div>
  );
}

export function FixtureExplainer({ sport, state, bundle, request, extras }: FixtureExplainerProps) {
  const [panelState, setPanelState] = useState<PanelState>("flow");
  const [summary, setSummary] = useState<Summary | null>(null);

  const handleSummary = (s: Explanation) => {
    // The button has already validated the shape; the news date rides along.
    setSummary(s as Summary);
    setPanelState("summary");
  };

  const handleUnavailable = () => {
    setPanelState("unavailable");
  };

  return (
    <div className="flex flex-col gap-3" data-testid="fixture-explainer">
      {panelState === "summary" && summary && <SummaryView summary={summary} extras={extras} />}
      <FixtureFlow sport={sport} state={state} bundle={bundle} request={request} />
      {panelState !== "summary" && (
        <SummaryButton
          request={request}
          onSummary={handleSummary}
          onUnavailable={handleUnavailable}
          label={panelState === "unavailable" ? "Try again" : "Get the AI summary"}
        />
      )}
    </div>
  );
}
