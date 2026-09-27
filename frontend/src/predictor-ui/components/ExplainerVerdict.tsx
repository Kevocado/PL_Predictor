// Synced from predictor-ui@52a31840951f. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** The confidence word, as a word.
 *
 *  `band` arrives from the service already computed from `pick.prob` (spec
 *  §13a), and this component **renders what it is given and never recomputes
 *  one**. The thresholds live in one Python function; a second copy in
 *  TypeScript is a second thing to keep right, for no benefit, and the whole
 *  point of computing it server-side is that a model cannot call a 52% pick
 *  strong.
 *
 *  It is a chip rather than a coloured bar because a colour here would be a
 *  confidence nobody can check, and the words are what make it checkable. Uppercase
 *  and tracked out, matching the family’s status labels.
 */
export type Band = "leaning" | "moderate" | "strong";

const WORDS: Record<Band, string> = {
  leaning: "Leaning",
  moderate: "Moderate",
  strong: "Strong",
};

export function BandChip({ band }: { band: Band }) {
  return (
    <span
      data-testid="band-chip"
      className="inline-flex shrink-0 items-center whitespace-nowrap rounded-pr border border-pr-rule bg-pr-panel px-2 py-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim"
    >
      {WORDS[band] ?? WORDS.moderate}
    </span>
  );
}

/** The panel's own title. A real heading, so the panel is a landmark a screen
 *  reader can navigate to — not a small label sitting above a bigger one, which
 *  is the eyebrow the family does not use anywhere. */
export function PanelHeading({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h3 id={id} className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
      {children}
    </h3>
  );
}
