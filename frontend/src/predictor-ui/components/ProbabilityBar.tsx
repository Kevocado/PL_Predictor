// Synced from predictor-ui@d167633be70a. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { contrast, parseHex } from "../contrast";
import { pct } from "../fmt";

export type Segment = {
  label: string;
  prob: number;
  color?: string;
  /** The facts' `market` key, when this segment IS a market. A factor's `key`
   *  refers to it, which is what makes §13c's highlight a lookup rather than a
   *  second naming scheme. Optional, so the match-card call sites that predate
   *  the panel are unaffected. */
  market?: string;
};

// Default fills when a segment has no team colour, never good/bad green/red:
// two-way reads accent / dim; three-way reads home / draw / away as
// accent / faint / dim, three distinct tones that all show on the panel.
const FALLBACK_2 = ["var(--color-pr-accent)", "var(--color-pr-text-dim)"];
const FALLBACK_3 = ["var(--color-pr-accent)", "var(--color-pr-text-faint)", "var(--color-pr-text-dim)"];

// A team colour that nearly matches the panel (Browns brown, Ravens purple)
// would make its share of the bar disappear; those fall back to a neutral
// that reads. 1.6:1 against the panel is the floor.
// Kept equal to --color-pr-panel by tokens.test.ts.
export const PANEL = "#151920";
function fill(color: string | undefined, i: number, count: number): string {
  const fallback = count === 2 ? FALLBACK_2 : FALLBACK_3;
  const hex = color ? parseHex(color) : null;
  return hex && contrast(hex, PANEL) >= 1.6 ? hex : fallback[i % fallback.length];
}

/** Whether the market's own split covers every outcome the model's split has.
 *
 *  Spec §13b. PL's `implied` carries only the two sides, so with a three-way
 *  model split a market row would sit under a draw segment with nothing above
 *  it: two bars covering different outcomes, inviting a comparison that cannot
 *  be made. Omission is the honest answer.
 */
function covers(segments: Segment[], legend: Segment[]): boolean {
  const have = new Set(legend.map((s) => s.label));
  return segments.every((s) => have.has(s.label));
}

/**
 * Segments render in the order given, each labelled in words.
 *
 * The graphic keeps its `role="img"` and its single label, which is the right
 * summary for a reader who is not going to step through it. The per-segment
 * labels are then real buttons carrying the same figures, so a reader who
 * *does* step through it gets them one at a time. That is summary plus detail,
 * and it is why nothing here is `aria-hidden`: a focusable element inside an
 * `aria-hidden` container is reachable by keyboard and invisible to a screen
 * reader, which is worse than either alone.
 */
export function ProbabilityBar({
  segments,
  legend,
  minSegmentPx = 0,
  highlightKey = null,
  onSegmentFocus,
}: {
  segments: Segment[];
  legend?: Segment[];
  /** A 3% probability still has to render as a sliver rather than vanishing.
   *  §6a: the label goes in the legend rather than shrinking below the 12px
   *  floor, so visibility and the floor are not in tension. */
  minSegmentPx?: number;
  /** The market key a selected factor points at (§13c). */
  highlightKey?: string | null;
  onSegmentFocus?: (label: string, market?: string) => void;
}) {
  const described = segments.map((s) => `${s.label} ${pct(s.prob)}`);
  const showLegend = !!legend && covers(segments, legend);
  const dim = (s: Segment) => (highlightKey && s.market && s.market !== highlightKey ? 0.4 : 1);

  return (
    <div className="flex flex-col gap-1.5">
      <div role="img" aria-label={described.join(", ")}>
        <span aria-hidden="true" className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-pr">
          {segments.map((s, i) => (
            <span
              key={i}
              data-testid="pbar-fill"
              data-min-width={minSegmentPx ? `${minSegmentPx}px` : undefined}
              className="h-full transition-opacity duration-150"
              style={{
                flexGrow: Math.max(s.prob, 0.01),
                minWidth: minSegmentPx ? `${minSegmentPx}px` : undefined,
                backgroundColor: fill(s.color, i, segments.length),
                opacity: dim(s),
              }}
            />
          ))}
        </span>
      </div>

      <div className="flex items-baseline justify-between gap-2">
        {segments.map((s, i) => (
          <button
            key={i}
            type="button"
            data-testid="pbar-label"
            data-seg={s.label}
            data-highlighted={highlightKey && s.market === highlightKey ? "true" : "false"}
            onClick={() => onSegmentFocus?.(s.label, s.market)}
            onFocus={() => onSegmentFocus?.(s.label, s.market)}
            className="min-w-0 rounded-pr text-xs tabular-nums text-pr-text-dim underline-offset-4 transition-opacity duration-150 hover:text-pr-text focus-visible:text-pr-text"
            style={{ opacity: dim(s) }}
          >
            {s.label} {pct(s.prob)}
          </button>
        ))}
      </div>

      {showLegend && (
        <div data-testid="pbar-legend" className="flex flex-col gap-1 border-t border-pr-rule pt-1.5">
          <span className="flex items-center gap-2 text-xs text-pr-text-faint">
            <span className="w-16 shrink-0 uppercase tracking-wide">market</span>
            <span aria-hidden="true" className="flex h-1 w-full gap-0.5 overflow-hidden rounded-pr">
              {legend!.map((s, i) => (
                <span key={i} style={{ flexGrow: Math.max(s.prob, 0.01), backgroundColor: "var(--color-pr-text-faint)" }} />
              ))}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}
