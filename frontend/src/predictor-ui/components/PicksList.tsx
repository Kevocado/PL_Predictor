// Synced from predictor-ui@b2040a541944. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The model's own top calls, ranked per category.
 *
 * Three rules are enforced HERE rather than left to every caller, because each
 * one has already been broken at least once by a caller that was being careful:
 *
 *  1. **A row's visual must match what its number IS.** `kind: "probability"`
 *     draws a share bar and `kind: "projection"` draws a key number with a ±.
 *     Passing a yardage figure as a probability once rendered "9600%", so the
 *     two kinds have disjoint ranges and a value outside its own is refused by
 *     name rather than drawn.
 *  2. **An out player leaves the ranking entirely.** A row marked `out` is
 *     refused. There is no greyed-out row and no "flagged in place" state to
 *     opt into — the caller passes them in `out`, where they are shown once,
 *     below the lists, with the source and the date.
 *  3. **Three rows per category, and three is a ceiling.** Fewer when fewer
 *     exist. Nothing is padded to fill the list.
 *
 * Provenance is the caller's own words, verbatim. This component composes no
 * confidence language at all: a probability, a bucket context and a ± are all
 * it renders, so there is no sentence here that could claim an edge, a price
 * or a guarantee — no odds feed exists in any of these repos.
 */
import { pct, stat } from "../fmt";
import { StatusBadge } from "./StatusBadge";

/** A player out for this fixture, shown once below the lists and never ranked. */
export type OutPlayer = {
  name: string;
  team?: string;
  /** Where the out claim came from, in words. Never an id. */
  source: string;
  /** When it was known, in words. Never an epoch. */
  dated: string;
};

export type PickRow = {
  /** Stable per player+category; the tile's identity and the React key. */
  key: string;
  name: string;
  team?: string;
  /** The category, in words: "Anytime TD", "Rush yds". */
  detail: string;
  /** The model's own number for this category. Never derived here. */
  value: number;
  /** "probability" draws a share bar; "projection" draws a key number + ±. */
  kind: "probability" | "projection";
  /** Only for `kind: "projection"`: the error margin, rendered as "± N".
   *  Absent means there is no error estimate yet, which is said in words
   *  rather than drawn as "± 0" — a zero margin would be a claim. */
  margin?: number;
  /** Accepted for compatibility and NOT rendered. Kevin, 2026-10-01: the best calls are
   *  the player and the prediction, nothing else (no record, Brier, calibration or error
   *  prose per row). Callers may stop passing it. */
  provenance?: string;
  /** True when this player is out. If true the row must NOT be passed — see
   *  `out`. The field exists so the mistake can be caught by name instead of
   *  rendering a forbidden ranking. */
  out?: boolean;
};

export interface PicksListProps {
  title?: string;
  categories: { category: string; rows: PickRow[] }[];
  out?: OutPlayer[];
}

/** At most this many rows per category. A cap, never a quota. */
export const MAX_ROWS_PER_CATEGORY = 3;

/** A row marked `out` was passed as a ranked row: "removed, not flagged" is a
 *  rule about what reaches the page, and this is the last place that can say no. */
export class OutPlayerInRankingError extends Error {
  constructor(name: string) {
    super(
      `PicksList: "${name}" is marked out and was passed as a ranked row. An out player ` +
        `leaves the ranking entirely — pass them in \`out\` instead, where they appear once ` +
        `below the lists with their source and date.`,
    );
    this.name = "OutPlayerInRankingError";
  }
}

/** A row's `value` is outside the range its `kind` can honestly draw. */
export class RowKindMismatchError extends Error {
  constructor(player: string, detail: string, kind: "probability" | "projection", value: number) {
    super(
      kind === "probability"
        ? `PicksList: "${player}" is a ${detail} probability row, so its value must be in [0,1] — got ${value}. ` +
            `A yardage figure read as a share is the defect this guard exists for.`
        : `PicksList: "${player}" is a ${detail} projection row, so its value must be finite and not negative — got ${value}.`,
    );
    this.name = "RowKindMismatchError";
  }
}

/**
 * Refuse a row whose number cannot honestly draw the visual its `kind` asks for.
 *
 * Runs before anything renders, so a caller that made the mistake gets a named
 * error naming the player and the category instead of a bar reading 9600%.
 */
function checkRow(row: PickRow): void {
  if (row.out) throw new OutPlayerInRankingError(row.name);
  if (!Number.isFinite(row.value)) {
    throw new RowKindMismatchError(row.name, row.detail, row.kind, row.value);
  }
  if (row.kind === "probability") {
    if (row.value < 0 || row.value > 1) throw new RowKindMismatchError(row.name, row.detail, row.kind, row.value);
  } else {
    // A projection is a count or a distance: finite and not negative. It MAY be a
    // fraction (0.8 steals, 0.6 touchdowns). The row's `kind` is explicit, so a
    // magnitude test is not how a mislabelled row is caught here; refusing [0,1]
    // threw during render for legitimate rows and took the whole list down.
    if (row.value < 0) throw new RowKindMismatchError(row.name, row.detail, row.kind, row.value);
  }
}

/** The figure a row shows. A probability is a percentage; a projection is a plain
 *  number with no % anywhere, because a count read as a share is the "9600%" defect. */
function figure(row: PickRow): string {
  return row.kind === "probability" ? pct(row.value) : stat(row.value);
}

/** A thin bar for a probability only. Floored so a small probability stays visible. */
function Bar({ value }: { value: number }) {
  const width = Math.max(value * 100, 3);
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-pr-rule" aria-hidden="true">
      <div data-testid="picks-bar" className="h-full rounded-full bg-pr-accent" style={{ width: `${width}%` }} />
    </div>
  );
}

export function PicksList({ title = "Model's top calls", categories, out }: PicksListProps) {
  // Every row is checked before any of them renders, so one bad row cannot
  // leave a half-drawn list on the page.
  for (const { rows } of categories) for (const row of rows) checkRow(row);

  if (categories.length === 0 && !out?.length) return null;

  return (
    <div data-testid="picks-list" className="flex flex-col gap-4">
      <p
        data-testid="picks-title"
        className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        {title}
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        {categories.map(({ category, rows }) => {
          // The ceiling. `slice` rather than a filter, so an out player could not be
          // swapped in to backfill a dropped row; `checkRow` has already refused one.
          const shown = rows.slice(0, MAX_ROWS_PER_CATEGORY);
          return (
            <section
              key={category}
              data-testid="picks-category"
              className="flex flex-col gap-2 rounded-pr border border-pr-rule bg-pr-surface/40 p-3"
            >
              <h3
                data-testid="picks-category-heading"
                className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim"
              >
                {category}
              </h3>
              {shown.length === 0 ? (
                // The existing empty-state vocabulary, never a zero: a figure nobody
                // produced must not be drawn as one.
                <StatusBadge status="nopick" />
              ) : (
                shown.map((row, i) => (
                  <div
                    key={row.key}
                    data-testid="picks-row"
                    data-kind={row.kind}
                    data-category={category}
                    className="flex flex-col gap-1.5 border-t border-pr-rule pt-2 first:border-t-0 first:pt-0"
                  >
                    <div className="flex items-baseline gap-2">
                      <span data-testid="picks-rank" className="w-4 shrink-0 font-pr-display text-xs font-semibold text-pr-text-faint tabular-nums">
                        {i + 1}
                      </span>
                      <p className="min-w-0 flex-1 truncate text-sm font-medium leading-snug text-pr-text">
                        {row.name}
                        {row.team && <span className="font-normal text-pr-text-dim"> · {row.team}</span>}
                      </p>
                      <span
                        data-testid="picks-value"
                        className={`font-pr-display text-xl font-semibold leading-none tabular-nums ${i === 0 ? "text-pr-accent" : "text-pr-text"}`}
                      >
                        {figure(row)}
                      </span>
                    </div>
                    {row.kind === "probability" && <Bar value={row.value} />}
                  </div>
                ))
              )}
            </section>
          );
        })}
      </div>

      {/* §D: an out player is shown ONCE, here, and nowhere in a ranking — no
          bar, no tile, no rank position, attributed and dated. */}
      {!!out?.length && (
        <div data-testid="picks-out" className="flex flex-col gap-1 border-t border-pr-rule pt-2">
          {out.map((p) => (
            <p key={`${p.name}-${p.dated}`} className="text-xs leading-snug text-pr-text-faint">
              Out: {p.name}
              {p.team && ` · ${p.team}`} — not ranked. {p.source} · {p.dated}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

/** The percentage a probability row shows, for a caller that wants the same
 *  rounding the bar uses. Exported so a site cannot re-derive it differently. */
export { pct as picksPct };