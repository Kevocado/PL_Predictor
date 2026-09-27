// Synced from predictor-ui@d167633be70a. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState } from "react";

/**
 * The "why it matters" rows: one per factor, each naming the market it is about.
 *
 * Three things this component is careful about, and each is a rule rather than a
 * preference:
 *
 * 1. **The direction is never carried by colour.** A drawn triangle and a word —
 *    "for the pick" / "against it" — so the meaning survives with no colour
 *    perception, no greyscale print and no stylesheet. The triangle is
 *    `aria-hidden` and the words are the accessible truth.
 * 2. **The marks are drawn, not borrowed.** The §7a mock uses ▲ and ▼, which are
 *    unicode characters standing in for an icon system: they render at the
 *    viewer's font, at the viewer's weight, and they are not a consistent set.
 *    These are two authored triangles on one 24-unit grid, so up and down are
 *    visibly the same mark turned over.
 * 3. **A row is a real button.** §13c links a factor to the figure it names, and
 *    the second half of every affordance in this panel is that it is operable by
 *    keyboard alone. That assertion is what catches a `div` with an `onClick`.
 */
export type Factor = {
  key: string;
  direction: "up" | "down";
  headline: string;
  text: string;
};

const TRIANGLE = (up: boolean) => (
  <svg
    data-direction={up ? "up" : "down"}
    aria-hidden="true"
    viewBox="0 0 24 24"
    width="14"
    height="14"
    className="shrink-0 translate-y-px"
    fill="currentColor"
  >
    <path d={up ? "M12 6.5 21 19H3z" : "M12 17.5 3 5h18z"} />
  </svg>
);

const WORDS: Record<Factor["direction"], string> = {
  up: "for the pick",
  down: "against it",
};

export function FactorList({
  factors,
  onSelect,
  expanded,
  onToggle,
  expandable = false,
}: {
  factors: Factor[];
  onSelect?: (key: string) => void;
  /** Which factor's sentence is fully shown. Owned by the panel, so a factor can
   *  be highlighted and expanded independently. */
  expanded?: number | null;
  onToggle?: (index: number) => void;
  /** Narrow viewports clamp the sentence to its first line. Off by default, so a
   *  wide layout never hides text behind a control that adds nothing. */
  expandable?: boolean;
}) {
  const [own, setOwn] = useState<number | null>(null);
  const isOpen = (i: number) => (expanded !== undefined ? expanded === i : own === i);
  const toggle = (i: number) => {
    if (onToggle) onToggle(i);
    else setOwn((was) => (was === i ? null : i));
  };

  return (
    <ul className="flex flex-col gap-3">
      {factors.map((factor, i) => {
        const up = factor.direction === "up";
        const open = isOpen(i);
        return (
          <li key={`${factor.key}-${i}`} className="flex min-w-0 flex-col gap-1">
            <div className="flex items-start gap-2">
              <span className={up ? "mt-0.5 text-pr-win" : "mt-0.5 text-pr-loss"}>{TRIANGLE(up)}</span>
              {onSelect ? (
                <button
                  type="button"
                  data-testid={`factor-${factor.key}`}
                  data-highlighted={open ? "true" : "false"}
                  onClick={() => {
                    onSelect(factor.key);
                    if (expandable) toggle(i);
                  }}
                  aria-pressed={open}
                  className="min-w-0 flex-1 text-left text-sm font-semibold text-pr-text underline-offset-4 hover:underline"
                >
                  <span>{factor.headline}</span>{" "}
                  {/* The words carry the direction, so the meaning is never
                      carried by the colour of the triangle beside it. */}
                  <span className="text-xs font-normal uppercase tracking-wide text-pr-text-faint">
                    {WORDS[factor.direction]}
                  </span>
                </button>
              ) : (
                <h4 className="min-w-0 flex-1 text-sm font-semibold text-pr-text">
                  {factor.headline}{" "}
                  <span className="text-xs font-normal uppercase tracking-wide text-pr-text-faint">
                    {WORDS[factor.direction]}
                  </span>
                </h4>
              )}
            </div>
            <div className="flex items-start gap-2">
              <span aria-hidden="true" className="w-[14px] shrink-0" />
              <div className="min-w-0 flex-1">
                <p
                  data-expanded={open ? "true" : "false"}
                  className={`text-sm leading-relaxed text-pr-text-dim ${
                    expandable && !open ? "line-clamp-1" : ""
                  }`}
                >
                  {factor.text}
                </p>
                {expandable && (
                  <button
                    type="button"
                    onClick={() => toggle(i)}
                    aria-expanded={open}
                    className="mt-0.5 text-xs font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 hover:underline"
                  >
                    {open ? "Less" : "More"}
                  </button>
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
