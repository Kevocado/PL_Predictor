// Synced from predictor-ui@d167633be70a. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { record as fmtRecord } from "../fmt";

/**
 * The record strip: the two numbers the facts carry, and the proportion as
 * geometry.
 *
 * Spec §5a-bis, and this is the component that rule exists for. `record` is
 * `hits: 41, settled: 68`; the facts contain no `60.3`. So the text reads
 * `41/68` and the bar's width carries the proportion. Writing "60.3%" here would
 * be a third number this product invented by arithmetic on two real ones — the
 * same class of error as a model inventing a figure, and the reason the rule is
 * written down rather than left to whoever renders it.
 *
 * The accessible name says "41 of 68" for the same reason. A screen reader
 * cannot see a bar's width, so the numbers have to be in words; those words are
 * the facts' own, not a percentage derived from them.
 */
export function RecordStrip({
  label,
  hits,
  settled,
}: {
  label: string;
  hits: number | null;
  settled: number;
}) {
  const counts = fmtRecord(hits ?? Number.NaN, settled);
  const has = counts !== "—" && settled > 0;
  const share = has ? Math.max(0, Math.min(1, (hits ?? 0) / settled)) : 0;

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">{label}</span>
      {has ? (
        <>
          <span
            role="img"
            aria-label={`${hits} of ${settled}`}
            className="flex h-1.5 w-full overflow-hidden rounded-pr bg-pr-panel-2"
          >
            <span data-testid="record-fill" className="h-full bg-pr-accent" style={{ width: `${share * 100}%` }} />
          </span>
          <span className="font-pr-display text-sm font-semibold text-pr-text">{counts}</span>
        </>
      ) : (
        // Nothing settled reads as a dash, never 0/0 — "0/0" is a claim about a
        // record that does not exist yet.
        <span className="font-pr-display text-sm font-semibold text-pr-text-faint">—</span>
      )}
    </div>
  );
}
