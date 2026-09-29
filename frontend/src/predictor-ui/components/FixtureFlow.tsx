// Synced from predictor-ui@56ad800924dc. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** FixtureFlow — instant, local, state-appropriate description.
 *
 *  Pure function of its props.  No useEffect, no fetch, no timer.
 *  Renders the game flow for the given sport and state.
 *
 *  Honour assertions (must bite):
 *  - No line named that is not there (test on rendered strings, not keys).
 *  - No direction marker, no "the market disagrees", no derived percentage.
 *  - F1 draws no market-line tile and no split-bar market row.
 *  - Every sentence is true for every bundle that reaches it.
 */

export type FlowState = "pre-game" | "in-play" | "finished";

export interface FixtureFlowProps {
  sport: string;
  state: FlowState;
  bundle: any;
  /** The panel's request function for the AI summary.  The flow is a pure
   *  function of the facts and never calls it — it is accepted so the parent
   *  can pass one prop to both children. */
  request: () => Promise<any>;
}

/** The moment each sport's pick has to beat, in that sport's own words. */
const MOMENT: Record<string, string> = {
  f1: "the session",
  nba: "tip-off",
  nfl: "kickoff",
  cfb: "kickoff",
  pl: "kickoff",
  sp: "kickoff",
};

/** Whether the bundle carries a market line to stand against. */
function hasLine(bundle: any): boolean {
  const line = bundle?.market_line;
  return line !== null && line !== undefined && line !== "";
}

/** Whole percent, matching the panel's own formatting. */
function pct(p: number): string {
  if (p <= 0.005) return "<1%";
  if (p >= 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}

/** The model's pick probability, or null when the facts don't carry one. */
function pickProb(bundle: any): number | null {
  const p = bundle?.pick?.prob;
  return typeof p === "number" && Number.isFinite(p) ? p : null;
}

/** When the pick was made, from the facts' own `pick_timing`. A rebuilt pick
 *  was made after the event began; the moment is worded per sport because F1
 *  has sessions, not kickoffs, and "after the game started" on a race page
 *  names a moment that sport does not have. */
function whenMade(sport: string, bundle: any): string {
  const timing = bundle?.pick_timing;
  if (timing === "rebuilt") return sport === "f1" ? "after the session started" : "after the game started";
  const moment = MOMENT[sport] ?? "kickoff";
  return `before ${moment}`;
}

/** The score, when the facts carry one. */
function scoreSentence(bundle: any): string | null {
  const score = bundle?.score;
  if (!score || typeof score.home !== "number" || typeof score.away !== "number") return null;
  const home = bundle?.home_team ?? "Home";
  const away = bundle?.away_team ?? "Away";
  return `The score is ${home} ${score.home}, ${away} ${score.away}.`;
}

/** The state of the game, for a sport with no score to show (F1). */
function stateSentence(sport: string, bundle: any): string | null {
  if (sport !== "f1") return null;
  const label = bundle?.pick?.label ?? bundle?.driver;
  if (!label) return null;
  const prob = pickProb(bundle);
  return prob !== null
    ? `The state of the race: the model's pick is ${label}, at ${pct(prob)} win probability.`
    : `The state of the race: the model's pick is ${label}.`;
}

/** The market's line, when the facts carry one. */
function lineSentence(bundle: any): string | null {
  if (!hasLine(bundle)) return null;
  return `The market's line is ${bundle.market_line}.`;
}

/** How the score stands against the line, when there is both. */
function standingSentence(bundle: any): string | null {
  const score = bundle?.score;
  if (!score || typeof score.home !== "number" || typeof score.away !== "number") return null;
  if (!hasLine(bundle)) return null;
  const home = bundle?.home_team ?? "Home";
  const away = bundle?.away_team ?? "Away";
  const diff = score.home - score.away;
  const lead =
    diff > 0 ? `${home} lead by ${diff}` : diff < 0 ? `${away} lead by ${-diff}` : "The scores are level";
  return `${lead}; the market's line is ${bundle.market_line}.`;
}

/** The result, in the reader's own words. */
function resultSentence(sport: string, bundle: any): string | null {
  const result = bundle?.result;
  if (result === "draw") return "The result is a draw.";
  if (result === "home_win") return `The result is a win for ${bundle?.home_team ?? "the home side"}.`;
  if (result === "away_win") return `The result is a win for ${bundle?.away_team ?? "the away side"}.`;
  if (result === "home_cover") return `The result is a cover for ${bundle?.home_team ?? "the home side"}.`;
  if (result === "away_cover") return `The result is a cover for ${bundle?.away_team ?? "the away side"}.`;
  if (result) return `The result is ${result}.`;
  // F1 carries no result field — the pick's rightness is the result.
  if (sport === "f1") {
    const wasRight = bundle?.pick?.was_right;
    const label = bundle?.pick?.label ?? bundle?.driver;
    if (typeof wasRight === "boolean" && label) {
      return wasRight
        ? `The result is a win for ${label}.`
        : `The result is not a win for ${label}.`;
    }
  }
  return null;
}

/** Whether the model's pick was right. */
function rightnessSentence(bundle: any): string | null {
  const wasRight = bundle?.pick?.was_right;
  if (typeof wasRight !== "boolean") return null;
  return wasRight
    ? "The pick rightness: the model's pick was right."
    : "The pick rightness: the model's pick was wrong.";
}

/** One row of the flow. `heading` rows are the fixture's own name. */
type Row = { text: string; heading?: boolean };

/** Build the flow's rows for the given sport and state. */
function buildRows(sport: string, state: FlowState, bundle: any): Row[] {
  const rows: Row[] = [];
  const isF1 = sport === "f1";

  if (state === "pre-game") {
    if (isF1) {
      // F1: the driver, the win probability and the pick are one fact.
      const label = bundle?.pick?.label ?? bundle?.driver;
      const prob = pickProb(bundle);
      if (label) {
        rows.push({
          heading: true,
          text:
            prob !== null
              ? `The model picks ${label} to win — win probability ${pct(prob)}.`
              : `The model picks ${label} to win.`,
        });
      }
      rows.push({ text: `The pick was made ${whenMade(sport, bundle)}.` });
    } else {
      const home = bundle?.home_team;
      const away = bundle?.away_team;
      if (home && away) rows.push({ heading: true, text: `${home} vs ${away}` });
      const hw = bundle?.home_win_prob;
      const aw = bundle?.away_win_prob;
      const hasMoneyline = typeof hw === "number" && typeof aw === "number" && !!home && !!away;
      if (hasMoneyline) {
        rows.push({ text: `Win probabilities: ${home} ${pct(hw)}, ${away} ${pct(aw)}.` });
      }
      rows.push({ text: `The pick was made ${whenMade(sport, bundle)}.` });
      const label = bundle?.pick?.label;
      const prob = pickProb(bundle);
      if (label) {
        rows.push(
          hasMoneyline
            ? { text: `The model picks ${label}.` }
            : { text: prob !== null ? `The model picks ${label} at ${pct(prob)}.` : `The model picks ${label}.` },
        );
      }
    }
    const line = lineSentence(bundle);
    if (line) rows.push({ text: line });
  } else if (state === "in-play") {
    const score = scoreSentence(bundle);
    if (score) rows.push({ text: score });
    const st = stateSentence(sport, bundle);
    if (st) rows.push({ text: st });
    const standing = standingSentence(bundle);
    if (standing) rows.push({ text: standing });
  } else if (state === "finished") {
    const result = resultSentence(sport, bundle);
    if (result) rows.push({ text: result });
    const rightness = rightnessSentence(bundle);
    if (rightness) rows.push({ text: rightness });
  }

  return rows;
}

export function FixtureFlow({ sport, state, bundle }: FixtureFlowProps) {
  const rows = buildRows(sport, state, bundle);
  return (
    <div className="flex flex-col gap-2" data-testid="fixture-flow">
      {rows.map((row, i) =>
        row.heading ? (
          <h4 key={i} className="font-pr-display text-lg font-semibold text-pr-text">
            {row.text}
          </h4>
        ) : (
          <p key={i} className="text-sm leading-relaxed text-pr-text-dim">
            {row.text}
          </p>
        ),
      )}
    </div>
  );
}
