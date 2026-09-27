// Synced from predictor-ui@d167633be70a. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";

/**
 * One market's answer, figure-led: the model's number large, and what the market
 * itself says directly beneath it.
 *
 * Deliberately not `StatTile`, which leads with the label. The label here IS the
 * answer, so it has to be the largest thing on the surface — and the tile's job
 * is the *comparison* between the model and the market, which is why `sub` sits
 * under the figure rather than above it. A tile that showed only the model's
 * number would be a vanity metric.
 *
 * A tile whose market the facts do not carry renders **nothing at all**: no
 * dashed box, no zero, no placeholder. Spec §6 says an absent market renders
 * nothing, and a dash would read as "we looked and there is nothing here", which
 * is a different claim from "there is no such market".
 */
export type MarketTile = {
  /** The facts' own `market` key. A factor's `key` refers to this, which is what
   *  makes the §13c highlight a lookup rather than a second naming scheme. */
  market: string;
  label: string;
  value: string;
  sub?: ReactNode;
  /** A colour for the market's own identity, where one exists. */
  tint?: string;
};

export function KeyNumberTile({ tile, highlighted }: { tile: MarketTile; highlighted?: boolean }) {
  if (!tile.value) return null;
  const { market, value, sub, tint } = tile;
  return (
    <dl
      data-testid={`tile-${market}`}
      data-highlighted={highlighted ? "true" : "false"}
      className={`flex min-w-0 flex-col gap-0.5 rounded-pr border px-3 py-2.5 transition-colors duration-150 ${
        highlighted ? "border-pr-accent bg-pr-panel-2" : "border-pr-rule bg-pr-panel"
      }`}
    >
      <dd
        className="font-pr-display text-2xl font-semibold leading-none text-pr-text"
        style={tint ? { color: tint } : undefined}
      >
        {value}
      </dd>
      <dt className="text-xs leading-snug text-pr-text-dim">{sub ?? market}</dt>
    </dl>
  );
}
