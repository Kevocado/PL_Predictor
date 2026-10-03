// Synced from predictor-ui@af3a01f66dd2. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * `Signal[]` as the facts block renders it: one compact row per finding, a
 * headline of at most 12 words, and a collapsed evidence line a keyboard can
 * open.
 *
 * Spec `2026-10-01-fixture-signals-design.md` §3 (the contract), §6 (the look).
 * The type below is §3 **verbatim in field and meaning**; everything after it is
 * this component's own account of what the fields may not be used to do.
 *
 * ## The one rule everything else serves
 *
 * **A figure may only reach the page if the same figure is in the row's own
 * words.** A signal is code-computed (spec §2: "Code computes the insight; the
 * AI only chooses and words it"), so the temptation is to let the visual carry a
 * number the headline does not mention — a bar 74% wide under words that never
 * said 74%. That is the "the model supplies words and ordering, never a number
 * the panel renders" rule from explainer-v2, one level up, and it is why
 * `headline` is **two** fields and not one:
 *
 *   - `headline.text` — the one line a reader reads, at most 12 words.
 *   - `headline.figures` — the numbers the visual draws.
 *
 * They are separate so the two can disagree, and the test file asserts they do
 * not by reading both back out of the rendered DOM rather than by trusting that
 * they came from one variable. Merging them into one formatted string was
 * considered and rejected: a component that *parsed* its own headline to find
 * the figure would make that assertion vacuous — the number on the chip and the
 * number in the sentence would be the same string by construction, so "they are
 * equal" would hold for every possible implementation, including a wrong one.
 *
 * ## What this component refuses, and why
 *
 * Three refusals, all of the same kind: a mistake a caller can make that would
 * otherwise reach the page as a *false statement* rather than as an error.
 *
 *  1. **A rate outside [0, 1]** (`SignalFigureError`). The "9600%" defect in
 *     this contract's clothes: `figures.rate` is a share, and a yardage figure
 *     read as one draws a bar that runs off its own track.
 *  2. **A figure the visual needs and the row does not carry**
 *     (`SignalFigureError`). A `reliability_bar` with no `figures.rate` would
 *     draw an empty track, and an empty track reads as 0%.
 *  3. **A visual this package cannot draw** (`UndrawableSignalVisualError`).
 *     Phase 1 draws `delta_chip` and `reliability_bar`. `absence_strip` (phase 2)
 *     and `projected_vs_actual` (phase 3) are refused rather than rendered
 *     without their marker, which is the one failure mode that both tests and
 *     screenshots miss.
 *
 * ## What this component drops rather than refuses
 *
 * **A rate whose `n` is under the floor.** Spec §4: "Renders only at n >= 30;
 * below that the row says nothing, never a rate from a handful of games." So
 * the whole row goes, not just the rate — a headline reading "has been right
 * 74% of the time" with the bar and the `n` taken away is a worse row than no
 * row. `n: null` on a rate-drawing row is dropped for the same reason and is
 * **not** a special case of the arithmetic: there is no count, so there is
 * nothing that could have cleared a floor.
 *
 * ## Rejected: ranking and capping here
 *
 * Spec §2 puts the fixed rule in the signals layer — "a fixed rule (not the
 * model) ranks them by `strength` and keeps the top 2-3" — and the plan puts it
 * on the endpoint (`GET /signals/{game_id}` returns the top 3 by `strength`).
 * `PicksList` sets the opposite precedent and caps three rows per category
 * inside the component, because that is where it draws the rank numbers and
 * where a caller had already broken the rule. Neither transfers here. A
 * `SignalRows` row draws no position, so a cap in this component would silently
 * drop a row the endpoint ranked into its top three for a reason this component
 * cannot see, and a second copy of the ranking rule is a second rule — the
 * argument `scripts/ui-package.mjs` is written on. Rows render in the order they
 * arrive, and a test pins that.
 */
import { useId, useState } from "react";
import { kickoff, pct, signed } from "../fmt";

/** Spec §3's `kind`. All four are in the contract from phase 1; see
 *  `UndrawableSignalVisualError` for which of them this package can draw. */
export type SignalKind = "trust" | "absence" | "line_gap" | "post_game";

/** Spec §3's `visual`. */
export type SignalVisual = "delta_chip" | "reliability_bar" | "absence_strip" | "projected_vs_actual";

/**
 * Spec §3, field for field. The three places §3 leaves a shape open — `sport`,
 * `headline` and `as_of` — are resolved here, and the resolution is stated
 * rather than smuggled:
 *
 *  - `sport` and `as_of` are `string`. §3 writes `source: str, as_of: iso`, so
 *    `as_of` is an ISO-8601 string by the spec's own notation and is put into
 *    reader's words by `fmt.kickoff`, which every other date on these sites
 *    goes through. A template-literal type (`${number}-${number}-…`) was
 *    considered and rejected: it rejects dates that are valid and accepts
 *    strings that are not, because it never parses.
 *  - `headline` is `{ text, figures }`. §3 writes `headline: {figures...}`, an
 *    open object; the two halves are split for the reason in the file header.
 *  - `figures` is `Record<string, number>` rather than a union of per-visual
 *    shapes, because §3 leaves the figure names open and a caller is allowed to
 *    carry more than its visual draws. `signalFigure` below is what reads them,
 *    and it names the one it needs rather than reaching for whatever is there.
 */
export type Signal = {
  kind: SignalKind;
  sport: string;
  game_id: string;
  /** The one line, and the figures the visual draws. See the file header. */
  headline: { text: string; figures: Record<string, number> };
  /** Sample size where this signal states a rate; `null` where it states none. */
  n: number | null;
  /** Where the claim came from, in the adapter's own words. Rendered verbatim. */
  source: string;
  /** ISO 8601. "Attributable and dated" is §3's phrase, so a row carries both
   *  or neither; `source` is rendered in words rather than as a machine key. */
  as_of: string;
  /** How much the reader should care, 0..1. Set and tested by the adapter. */
  strength: number;
  /** False only for `post_game` (spec §3). */
  pre_kickoff_only: boolean;
  visual: SignalVisual;
};

/**
 * The floor on a rate's sample size.
 *
 * **This identifier is new here.** `TRUST_MIN_N` appears nowhere in this
 * repository: the plan names it as a constant each *sport adapter* declares in
 * Python (`docs/superpowers/plans/2026-10-02-fixture-signals-phases-1-4.md`,
 * "Constants per sport: `TRUST_MIN_N = 30`"), and no `Signal` code exists for
 * it to live in yet. So the number is named for where it comes from rather than
 * for what the plan calls it: spec §4's "`trust` … Renders only at n >= 30", and
 * spec §9 decision 3 ("trust needs n >= 30", reversible by "one constant per
 * sport").
 *
 * It lives in the shared component and not only in each adapter because the rule
 * is about what may be **drawn**, not about what may be computed. An adapter can
 * honestly hold a bucket of 12 games and report it; what it may not do is put a
 * rate built from it on the page. This is the last place a renderer can say no,
 * which is the argument `PicksList` makes with `OutPlayerInRankingError`.
 */
export const SPEC_MIN_N = 30;

/** Spec §6: "Each row has one line (<= 12 words)". */
export const MAX_HEADLINE_WORDS = 12;

/**
 * Which visual draws a RATE, and therefore which one the floor gates.
 *
 * Keyed by `visual`, not by `kind`, and that is the load-bearing choice: the
 * question the floor answers is "is a rate on the page", and `visual` is what
 * decides that. `kind` is the adapter's name for the finding and this component
 * cannot see how an adapter chose to draw it. A `delta_chip` row carries
 * magnitudes — a model's margin and a quoted line, both from this game — so the
 * historical disagreement count `n` is not what licenses them, and a chip row
 * with `n: null` is a complete row.
 *
 * No default branch, per `FactorList.MARK`: a fifth visual fails to compile
 * rather than falling through to "does not draw a rate", which is the reading
 * that would put a rate on the page unexamined.
 */
const DRAWS_A_RATE: Record<SignalVisual, boolean> = {
  reliability_bar: true,
  delta_chip: false,
  absence_strip: false,
  projected_vs_actual: false,
};

/**
 * The figure a visual names for itself, or `null` when this package draws no
 * marker for it at all.
 *
 * `delta_chip` draws the disagreement in the sport's own units; `reliability_bar`
 * draws the share of picks that hit. The two phase-2/3 visuals are `null` rather
 * than carrying a placeholder figure, because that is what makes "not drawable
 * yet" a fact this file can test rather than a comment. One map, so a caller
 * cannot reach a figure for a visual that has none.
 */
const FIGURE: Record<SignalVisual, "gap" | "rate" | null> = {
  delta_chip: "gap",
  reliability_bar: "rate",
  absence_strip: null,
  projected_vs_actual: null,
};

/** A visual this package cannot draw. */
export class UndrawableSignalVisualError extends Error {
  constructor(gameId: string, visual: string, kind: string) {
    super(
      `SignalRows: signal ${gameId} (${kind}) asks for visual "${visual}", which this package does ` +
        `not draw. Rendering the row without its marker would put a headline on the page with the ` +
        `figure beside it missing, so it is refused instead. Drawable today: delta_chip and ` +
        `reliability_bar. absence_strip arrives with the absence signal, projected_vs_actual with ` +
        `post_game.`,
    );
    this.name = "UndrawableSignalVisualError";
  }
}

/** A figure that cannot honestly draw its visual, or is not there at all. */
export class SignalFigureError extends Error {
  constructor(gameId: string, kind: string, visual: string, name: string, detail: string) {
    super(
      `SignalRows: signal ${gameId} (${kind}) draws "${visual}" from headline.figures.${name}, ` +
        `${detail}.`,
    );
    this.name = "SignalFigureError";
  }
}

/**
 * The figure a visual draws, or a throw naming what was wrong with it.
 *
 * Checked before anything renders, so a caller that made the mistake gets a
 * named error naming the game and the figure instead of a bar running off its
 * own track — the same ordering `PicksList.checkRow` uses.
 */
export function signalFigure(signal: Signal, visual: SignalVisual): number {
  const name = FIGURE[visual];
  if (name === null) throw new UndrawableSignalVisualError(signal.game_id, visual, signal.kind);
  const value = signal.headline.figures[name];
  if (value === undefined) {
    throw new SignalFigureError(
      signal.game_id, signal.kind, visual, name,
      "which the signal does not carry, and an empty visual reads as zero",
    );
  }
  if (!Number.isFinite(value)) {
    throw new SignalFigureError(
      signal.game_id, signal.kind, visual, name,
      `which is not a finite number (got ${value})`,
    );
  }
  // Only a rate has a range. `gap` is a margin in the sport's own units and is
  // checked as a magnitude: a yardage figure is a perfectly good gap, and the
  // "9600%" defect is a share read off the wrong field, not a large number.
  if (name === "rate" && (value < 0 || value > 1)) {
    throw new SignalFigureError(
      signal.game_id, signal.kind, visual, name,
      `which must be a share in [0,1] — got ${value}. A yardage figure read as a hit rate is the ` +
        `defect this guard exists for`,
    );
  }
  return value;
}

/**
 * THE floor a page will not go below, whatever a caller asks for.
 *
 * One place, and it is deliberately *not* the component: `rateIsDrawable` is
 * exported, so a site gating its own copy of a list through it must get the same
 * answer `SignalRows` would give. Clamping in the component as well would leave
 * two rules to keep in step — the thing `scripts/ui-package.mjs` is written
 * about. A caller may therefore **raise** the floor and can never lower it.
 */
const floorOf = (minN: number): number => Math.max(SPEC_MIN_N, minN);

/**
 * May this signal put a rate on the page?
 *
 * Three ways to answer no, and all three have been a real shape of the mistake:
 * the visual draws no rate (nothing to gate); the row states no `n`; the `n` is
 * under the floor. A `delta_chip` answers true whatever its `n`, because the
 * figures it draws are this game's own.
 */
export function rateIsDrawable(signal: Signal, minN: number = SPEC_MIN_N): boolean {
  if (!DRAWS_A_RATE[signal.visual]) return true;
  const n = signal.n;
  // `Number.isInteger`, not `n >= floor`: an adapter that put a float where a
  // count belongs has not reported a sample size, and `29.5 >= 30` is false by
  // luck rather than by rule.
  return Number.isInteger(n) && (n as number) >= floorOf(minN);
}

/**
 * Is this signal a row at all?
 *
 * Three of the spec's own rules, all "no data, no row": a blank headline says
 * nothing; a rate under the floor says nothing (spec §4); and a visual this
 * package cannot draw is not a row **this** component is willing to draw, so it
 * throws rather than answering false. Silently omitting that last one would be a
 * signal vanishing for no stated reason — the failure `KeyNumberTile`'s comment
 * about a machine key printed in the wrong slot describes.
 */
export function signalIsDrawn(signal: Signal, minN: number = SPEC_MIN_N): boolean {
  if (FIGURE[signal.visual] === null) {
    throw new UndrawableSignalVisualError(signal.game_id, signal.visual, signal.kind);
  }
  if (signal.headline.text.trim() === "") return false;
  if (!rateIsDrawable(signal, minN)) return false;
  return true;
}

/**
 * Clip a headline to `max` words, and say whether it was.
 *
 * **The rule: count whitespace-delimited words, keep the first `max`, and mark
 * the row `data-clipped`.** A word is a whitespace-delimited token because that
 * is what fits on one line — "−4.5:" is one word to a reader and two to a
 * tokenizer that also splits on punctuation.
 *
 * **Why clipping and not refusing, when a wrong number is not acceptable.**
 * Because clipping removes *words* and never edits a *figure*. The figures this
 * component draws come from `headline.figures`, which clipping does not touch,
 * so a clipped row still shows exactly the numbers it claims and loses only
 * prose — and the sample size survives regardless, because `n` is on the evidence
 * line whether or not the headline is clipped. A refused row, by contrast, loses
 * its whole finding because an adapter miscounted a word, which is a worse
 * outcome than a shorter sentence and a `data-clipped="true"` in the DOM.
 *
 * **A drawn figure is PROTECTED, and this was a real defect until it was.**
 * The first version claimed a clip "cannot corrupt a figure" on the grounds that
 * the figures are not in the text this function rewrites. That was wrong, and
 * CodeRabbit review comment 4171773475 caught it: the headline *does* speak them
 * — that is the whole honesty rule above — and "was right 74% of the time" can
 * sit at word 14. Clipping to 12 then left a bar at 74% under a sentence that no
 * longer mentioned 74%: exactly the dishonesty the split `headline` exists to
 * make impossible, reintroduced by the cap.
 *
 * So the rule is now: **keep the first `max` ORDINARY words, plus every word
 * that states a protected figure, in document order.** The protected forms are
 * passed in by the caller from the figure it is about to draw — this function
 * does not format, because the two formatters (`fmt.signed` for a gap,
 * `fmt.pct` for a rate) differ and guessing here would protect the wrong string.
 *
 * **THREE forms are protected for a rate, because `fmt.pct` has boundaries.** At
 * `p <= 0.005` it prints `<1%` and at `p >= 0.995` it prints `>99%`, never `0%`
 * and `100%`. A headline is written in plain words, so it can say `0%`, `100%`,
 * `+1%` or `<1%` where the formatter says something else, and protecting only
 * `pct(figure)` would leave the boundary rate as the one thing the clip removes.
 * `pct(figure)`, the rounded percent and the floored percent are all protected,
 * and BOTH sides of the comparison are normalised — a leading `+`, `−`, `<`, `>`
 * and any trailing punctuation are dropped — so a `<1%` in the text matches a
 * protected `<1%` instead of stripping one side only (CodeRabbit review comments
 * 4171835614 and 4171902960).
 *
 * The match is on a WHOLE word, never a substring: protecting `1%` must not also
 * protect `61%`, which would keep a clause unrelated to the figure and could
 * still drop the one that carries it. `norm` strips surrounding punctuation, so
 * `(74%`, `74%)` and `74%.` are all the protected word; it can over-protect, which
 * costs a little prose, and that is the safe direction — under-protecting is the
 * defect this rule exists to prevent.
 *
 * **What is given up.** A clipped row can exceed `max` words, so the cap is
 * "12 ordinary words" rather than "12 words", and a headline that names its
 * figure on every one of forty words stays forty words long. That is an adapter
 * copy bug rather than something this component can repair, and `data-clipped`
 * is the signal for it — the fix is the adapter's shorter wording.
 *
 * **What would make this a refusal instead: a NEGATION.** "Out: J. Jacobs is
 * **not** projected" clipped to "Out: J. Jacobs is" inverts the claim, and no
 * amount of figure-protection helps. None of the four spec §4 signals can produce
 * one from their own fields — `kind` says which claim a row is making — so it is
 * not reachable here. The note is here so the next person checks it before
 * adding a `kind` that can.
 */
export function clipHeadline(
  text: string,
  max: number = MAX_HEADLINE_WORDS,
  protectedWords: readonly string[] = [],
): { text: string; clipped: boolean } {
  const trimmed = text.trim();
  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length <= max) return { text: trimmed, clipped: false };
  // WHOLE-WORD matching, not `includes`.
  //
  // Substring matching protects the wrong word: protecting the rate `1%` would
  // also protect `61%`, `21%` and every other word ending in it, so a headline
  // could keep a clause that has nothing to do with the figure and lose the one
  // that does. (CodeRabbit review comment 4171835614 raised this and the boundary
  // form below; both are here.)
  // **BOTH SIDES go through `norm`, or the match fails on its own output.**
  // Normalising only the token was a real defect (CodeRabbit review comment
  // 4171902960): `<1%` strips its angle bracket to `1%` while the protected form
  // kept its own, so the drawn boundary rate stopped matching itself. Same for a
  // signed `+1.6` in the text against an unsigned protected `1.6`.
  //
  // `norm` therefore strips surrounding punctuation — a leading sign or boundary
  // symbol or open bracket, and any trailing punctuation — and leaves digits, the
  // decimal point and `%` alone.
  //
  // **It can over-protect, and that is the direction that is safe.** Dropping the
  // sign means a headline containing both `+1.6` and `−1.6` would protect both,
  // so a row could keep a word it did not need to. Keeping too much prose is a
  // cosmetic cost; keeping too little is the honesty defect this whole rule
  // exists to prevent. The alternative — keeping the sign, so only `+1.6` matches
  // an unsigned `1.6` — is what 4171902960 was about, and it fails the wrong way.
  const norm = (w: string) => w.trim().replace(/^[^\w.%]+/, "").replace(/[^\w%]+$/, "");
  const marks = new Set(protectedWords.filter(Boolean).map(norm));
  const protectedIndex = (word: string): boolean => marks.has(norm(word));
  const kept: string[] = [];
  let ordinary = 0;
  for (const word of words) {
    // A protected word is kept wherever it is and does not spend the budget: the
    // cap is on the sentence AROUND the figure, not on the figure.
    if (protectedIndex(word)) {
      kept.push(word);
      continue;
    }
    if (ordinary < max) kept.push(word);
    ordinary++;
  }
  return { text: kept.join(" "), clipped: ordinary > max };
}

/** The heading a list of rows carries. Plain words and no developer term:
 *  PRODUCT.md requires plain-language copy on public pages, so this is not
 *  "Signals". */
export const DEFAULT_SIGNAL_TITLE = "Context";

export interface SignalRowsProps {
  signals: Signal[];
  /** The heading the rows sit under. Overridable because a site may already
   *  have one in view, and two stacked headings is noise. */
  title?: string;
  /**
   * The floor, for a sport whose own buckets sit under the spec's.
   *
   * **It can only raise it**, because `floorOf` clamps it — the spec's 30 is
   * the floor the page will not go below, so a prop that lowered it would make
   * the guarantee this package exists to provide opt-out-able by the caller with
   * the most reason to want it out. Spec §9 decision 3 reverses that floor "by
   * one constant per sport", and a sport needing a *higher* bar is exactly what
   * this prop is for.
   */
  minN?: number;
}

/** One row. Not exported, so every path into it goes through `signalIsDrawn`
 *  first and no caller can mount a row the floor has not cleared. */
function SignalRow({ signal, rowId }: { signal: Signal; rowId: string }) {
  const [open, setOpen] = useState(false);
  const isBar = signal.visual === "reliability_bar";
  const n = signal.n;
  // The count as evidence words. Empty when the row states none, so a label can
  // never read "of null games"; the floor means a bar-drawing row always has one.
  const count = n === null ? "" : `n=${n}`;
  const figure = signalFigure(signal, signal.visual);
  // Clipped with the figure PROTECTED, and the protected forms are the ones this
  // row is about to draw.
  //
  // A rate has THREE forms, because `fmt.pct` is not injective: `0.004` draws
  // as `<1%`, `1` draws as `>99%`, and a headline written by a human says `0%`
  // or `100%` for those. Protecting only `pct(figure)` would let the drawn
  // boundary rate be the one thing the clip removes. So the rounded plain form
  // and the floored percent are protected alongside the formatter's own, and the
  // match is whole-word so `1%` cannot also match `61%` (CodeRabbit review
  // comments 4171835614 and 4171902960).
  //
  // A gap has ONE: the sign is dropped because a headline says "1.6" where the
  // chip says "+1.6", and `clipHeadline`'s `norm` strips a leading sign anyway, so
  // a headline that writes `+1.6` matches the same figure (4171902960).
  const protectedForms: string[] =
    signal.visual === "reliability_bar"
      ? [pct(figure), `${Math.round(figure * 100)}%`, `${Math.floor(figure * 100)}%`]
      : [signed(figure).replace(/^[+−-]/, "")];
  const { text, clipped } = clipHeadline(signal.headline.text, MAX_HEADLINE_WORDS, protectedForms);

  return (
    <li
      data-testid="signal-row"
      data-row={rowId}
      data-kind={signal.kind}
      data-visual={signal.visual}
      className="flex min-w-0 flex-col gap-1.5 border-t border-pr-rule pt-3 first:border-t-0 first:pt-0"
    >
      <div className="flex min-w-0 items-baseline gap-2">
        {signal.visual === "delta_chip" && (
          // Neutral, and never `pr-win` / `pr-loss`. The spec's own case-8 mock
          // painted this chip green (`text-pr-win`), and a disagreement with the
          // line is not an outcome: green beside red would be two rows claiming
          // the model was right and wrong about the same kind of thing. The sign
          // carries the direction, which is also `FactorList`'s rule — a
          // direction is never carried by colour.
          <span
            data-testid="signal-delta"
            className="shrink-0 rounded-pr border border-pr-rule px-1.5 py-0.5 font-pr-display text-xs font-semibold tabular-nums text-pr-text"
          >
            {signed(figure)}
          </span>
        )}
        <p
          data-testid="signal-headline"
          data-clipped={clipped ? "true" : "false"}
          className="min-w-0 flex-1 text-sm font-medium leading-snug text-pr-text"
        >
          {text}
          {clipped && <span aria-hidden="true"> …</span>}
        </p>
      </div>

      {isBar && (
        // `role="img"` with the figures in words, as `RecordStrip` does and for
        // the same reason: a screen reader cannot see a track's width. It is not
        // `aria-hidden` the way `PicksList`'s bar is, because a headline may be
        // clipped (see `clipHeadline`) and this number has to survive that.
        <span
          role="img"
          aria-label={`${pct(figure)} right${count ? ` of ${String(n)} games` : ""}`}
          className="flex h-1.5 w-full overflow-hidden rounded-pr bg-pr-panel-2"
        >
          <span data-testid="signal-bar-fill" className="h-full bg-pr-accent" style={{ width: `${figure * 100}%` }} />
        </span>
      )}

      {/* The evidence line, collapsed.
          *
          * A real `<button>`, not a `title` and not a hover-only line: a `title`
          * never appears on the device this is mostly read on and is not reliably
          * announced from a span, and evidence nobody can reach is not evidence.
          * No `aria-controls` — the family's other disclosures (`FactorList`,
          * `ProbabilityBar`) carry none either, `aria-expanded` is the part that
          * is load-bearing, and the element it would name does not exist while
          * the panel is closed.
          *
          * `n` and the date are ALWAYS visible and OUTSIDE the control, in the
          * panel's own small faint type rather than the button's uppercase
          * tracking. Two reasons, and the first is the whole one: `n` is the
          * sample size, so uppercasing it prints `N=61` and a reader takes that
          * for a name rather than a count — and a count is the one thing on this
          * row a sceptic needs without opening anything.
          *
          * **WHAT IS AMBIGUOUS IN THE SPEC, and the reading taken.** §6 asks for
          * "a collapsed evidence line (source, n, date)", and its parenthetical
          * names three fields while "collapsed" names a hidden state — so it
          * cannot mean all three are behind the control. This reads it as: the
          * line carries all three, and what collapses is the LONG one. `source`
          * is a free string from another repository ("stored pre-kickoff picks,
          * 0.5-0.6 bucket") and is the field that makes a row two or three lines
          * deep at 390px; `n` and a formatted date are short and cost one line.
          * So `n` and the date sit in the collapsed row and `source` is what the
          * button reveals. A reading where all three hide was rejected: it makes
          * the sample size unavailable without a pointer's worth of interaction
          * on the exact claim a reader most wants to check. */}
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <button
          type="button"
          data-testid="signal-evidence-toggle"
          aria-expanded={open}
          onClick={() => setOpen((was) => !was)}
          className="shrink-0 text-xs font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 hover:underline"
        >
          {open ? "Hide source" : "Source"}
        </button>
        <p data-testid="signal-evidence" className="min-w-0 text-xs leading-snug text-pr-text-faint">
          {[count, kickoff(signal.as_of)].filter(Boolean).join(" · ")}
          {open && signal.source ? ` · ${signal.source}` : ""}
        </p>
      </div>
    </li>
  );
}

export function SignalRows({ signals, title = DEFAULT_SIGNAL_TITLE, minN }: SignalRowsProps) {
  const headingId = useId();
  // The raw caller value; `rateIsDrawable` is what clamps it. See `minN`.
  const floor = minN ?? SPEC_MIN_N;
  // Every row is checked before any of them renders, so one bad row cannot
  // leave a half-drawn list on the page.
  const drawn = signals.filter((signal) => signalIsDrawn(signal, floor));

  // An empty list renders nothing at all: no heading, no panel, no "no signals",
  // no zero. Spec §2 — "No data, no row. No empty states, no 'unknown' rows, no
  // filler" — and a list whose every row was dropped for want of a sample is the
  // same empty list, so it renders nothing too.
  if (drawn.length === 0) return null;

  return (
    <section aria-labelledby={headingId} className="flex min-w-0 flex-col gap-3">
      {/* `<h3>`, the panel's own section level (`PanelHeading`), and not the
          `<p>` `PicksList` uses for its title: a `<p>` gives a screen reader no
          landmark to navigate to, and this list is the thing a reader of the
          facts block is looking for. */}
      <h3
        id={headingId}
        data-testid="signal-rows-title"
        className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        {title}
      </h3>
      <ul data-testid="signal-rows" className="flex min-w-0 flex-col gap-3">
        {/* The key is `game_id:kind:index`, and the index is in it on purpose.
         *
         * Spec §3 carries no row id, and this component does not add one: the
         * type is the contract and inventing a field no adapter has to populate
         * would be a field every real payload leaves undefined.
         *
         * `game_id:kind` alone is not a key. CodeRabbit review comment
         * 4171773477 is right that §3 does not make those unique, and two
         * signals sharing one is a real possibility (two `trust` buckets on one
         * game is plausible). React would then reuse one `SignalRow`'s
         * `useState` for both, which is the row's disclosure state showing on
         * the wrong signal.
         *
         * The index's own weakness is named here rather than glossed: a reorder
         * would carry the open disclosure with the position rather than the
         * signal. That is accepted because rows render in the order the endpoint
         * ranked them (see the file header), a reorder between two renders of
         * the same fixture is not a case this component produces, and the
         * alternative — a key of `(game_id, kind)` alone — is wrong NOW rather
         * than wrong in a case that has not happened. What would remove both
         * costs is an `id` in §3; that is a spec change, so it is reported
         * rather than taken here. */}
        {drawn.map((signal, i) => (
          <SignalRow
            key={`${signal.game_id}:${signal.kind}:${i}`}
            signal={signal}
            rowId={`${signal.game_id}:${signal.kind}:${i}`}
          />
        ))}
      </ul>
    </section>
  );
}