// Synced from predictor-ui@797bbd0ef9a8. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { contrast, parseHex } from "../contrast";
import { pct } from "../fmt";

export type Segment = { label: string; prob: number; color?: string };

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

/** Segments render in the order given, each labelled in words. */
export function ProbabilityBar({ segments }: { segments: Segment[] }) {
  const described = segments.map((s) => `${s.label} ${pct(s.prob)}`);
  return (
    <span role="img" aria-label={described.join(", ")} className="flex flex-col gap-1.5">
      <span aria-hidden="true" className="flex h-2 w-full gap-0.5 overflow-hidden rounded-pr">
        {segments.map((s, i) => (
          <span key={i} data-testid="pbar-fill" style={{ flexGrow: Math.max(s.prob, 0.01), backgroundColor: fill(s.color, i, segments.length) }} />
        ))}
      </span>
      <span aria-hidden="true" className="flex justify-between gap-2 text-xs text-pr-text-dim">
        {described.map((text, i) => (
          <span key={i} data-testid="pbar-label">
            {text}
          </span>
        ))}
      </span>
    </span>
  );
}
