// Synced from predictor-ui@130ae20c8e7e. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";
import { GlobalNav, type NavEntry } from "./GlobalNav";
import { SportTabs } from "./SportTabs";
import type { HubTab } from "../trackRecord/HubTabs";

export type Sport = "pl" | "f1" | "nfl" | "cfb" | "nba" | "hub";
export type SiteLink = { sport: Sport; label: string; href: string };
export type Tab = { id: string; label: string };

type Props = {
  sport: Sport;
  sportName: string;
  sites: SiteLink[];
  tabs: Tab[];
  activeTab: string;
  onTab: (id: string) => void;
  children: ReactNode;
  homeHref?: string;
  paperTradingHref?: string;
  activeEntry?: string | null;
};

/** The shared frame: family wordmark, global nav, page tabs. */
export function AppFrame({
  sport,
  sportName,
  sites,
  tabs,
  activeTab,
  onTab,
  children,
  homeHref,
  paperTradingHref,
  activeEntry,
}: Props) {
  const resolvedHome = homeHref ?? sites.find((s) => s.sport === "hub")?.href ?? "/";
  const activeNav = activeEntry === undefined ? (sport === "hub" ? null : sport) : activeEntry;

  const entries: NavEntry[] = sites
    .filter((s) => s.sport !== "hub")
    .map((s) => ({ id: s.sport, label: s.label, href: s.href, sport: s.sport }));
  if (paperTradingHref) {
    entries.push({ id: "paper", label: "Paper Trading", href: paperTradingHref, sport: "hub" });
  }

  const hubTabs: HubTab[] = tabs.map((t) => ({ id: t.id, label: t.label }));

  return (
    <div data-sport={sport} className="min-h-screen">
      <header className="border-b border-pr-rule">
        <div className="mx-auto flex max-w-screen-xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 pt-3 sm:px-6">
          <a
            href={resolvedHome}
            aria-current={activeNav === null ? "page" : undefined}
            className="font-pr-display text-sm font-bold uppercase tracking-[0.2em] text-pr-text-dim"
          >
            Predictor
          </a>
          <GlobalNav entries={entries} activeId={activeNav} />
        </div>
        <div className="mx-auto flex max-w-screen-xl flex-wrap items-end justify-between gap-x-6 gap-y-3 px-4 pb-0 pt-3 sm:px-6">
          <h1 className="pb-3 font-pr-display text-3xl font-bold uppercase tracking-wide text-pr-text">{sportName} Predictor</h1>
          <SportTabs tabs={hubTabs} active={activeTab} onChange={onTab} label="Pages" />
        </div>
      </header>
      <main className="mx-auto max-w-screen-xl px-4 py-6 sm:px-6">{children}</main>
    </div>
  );
}