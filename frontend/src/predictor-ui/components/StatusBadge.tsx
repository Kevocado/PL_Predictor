// Synced from predictor-ui@797bbd0ef9a8. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
export type Status = "next" | "live" | "called" | "missed" | "nopick" | "rebuilt";
/** The moment a pick must beat: "kickoff" for football, "tip-off" for
 *  basketball, "the session" for F1 (qualifying and races alike). */
export type Moment = "kickoff" | "tip-off" | "the session";

// Status always carries words; colour only reinforces them.
const LOOK: Record<Status, { words: string; tone: string }> = {
  next: { words: "Next up", tone: "bg-pr-accent text-pr-accent-ink" },
  live: { words: "Live", tone: "border border-pr-loss text-pr-loss" },
  called: { words: "Called it ✓", tone: "text-pr-win" },
  missed: { words: "Missed ✗", tone: "text-pr-loss" },
  nopick: { words: "No pick yet", tone: "text-pr-text-dim" },
  rebuilt: { words: "Rebuilt after {moment}", tone: "border border-pr-rule text-pr-text-dim" },
};

/** The exact words each status reads as, for accessible names elsewhere. */
export const statusWords = (status: Status, moment: Moment = "kickoff") => LOOK[status].words.replace("{moment}", moment);

export function StatusBadge({ status, moment = "kickoff" }: { status: Status; moment?: Moment }) {
  const { tone } = LOOK[status];
  const words = statusWords(status, moment);
  return (
    <span className={`inline-flex items-center whitespace-nowrap rounded-pr px-1.5 py-0.5 font-pr-display text-xs font-semibold uppercase tracking-wide ${tone}`}>
      {words}
    </span>
  );
}
