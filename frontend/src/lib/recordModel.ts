import type {
  CalibrationSeries,
  MarketTally,
  RecordModel,
  WeekPoint,
} from "../predictor-ui";
import { marketLabel } from "../predictor-ui";
import type { TrackRecordResponse } from "../types";

type PlMarket = "exact_score" | "match_result" | "over_under_2_5" | "btts";

const MARKETS: PlMarket[] = ["match_result", "over_under_2_5", "exact_score", "btts"];

/** PL's own words for its four markets. Unknown keys fall back to themselves
 *  via `marketLabel`, so a raw key never reaches the page. */
const marketLabelFor = (key: PlMarket): string => {
  switch (key) {
    case "exact_score":
      return "Exact score";
    case "match_result":
      return "Match result";
    case "over_under_2_5":
      return "Goals O/U 2.5";
    case "btts":
      return "BTTS";
  }
  return marketLabel(key);
};

function tally(pct: number | null, n: number): { correct: number; n: number } {
  return { correct: Math.round((pct ?? 0) * n), n };
}

function marketTally(summary: { by_market?: Record<string, { pct_correct: number | null; n_resolved: number }>; pre_kickoff?: { by_market?: Record<string, { pct_correct: number | null; n_resolved: number }> } }, key: PlMarket): MarketTally {
  const byMarket = summary.by_market?.[key];
  const counted = tally(byMarket?.pct_correct, byMarket?.n_resolved ?? 0);
  const pre = summary.pre_kickoff?.by_market?.[key];
  const preKickoff = pre && pre.n_resolved > 0 ? tally(pre.pct_correct, pre.n_resolved) : undefined;
  return {
    key,
    label: marketLabelFor(key),
    counted,
    ...(preKickoff ? { preKickoff } : {}),
  };
}

function weekPoint(gw: { gameweek: number | null; n_fixtures: number; pct_correct: number | null; n_made_before_kickoff?: number }): WeekPoint {
  const correct = gw.pct_correct != null && gw.n_fixtures > 0 ? Math.round(gw.pct_correct * gw.n_fixtures) : 0;
  return { weekStart: gw.gameweek != null ? `GW${gw.gameweek}` : "No gameweek", n: gw.n_fixtures, correct, tracked: gw.pct_correct != null };
}

function calibrationSeries(by_market: { exact_score?: { pct_correct: number | null; n_resolved: number }; match_result?: { pct_correct: number | null; n_resolved: number }; over_under_2_5?: { pct_correct: number | null; n_resolved: number }; btts?: { pct_correct: number | null; n_resolved: number } } | undefined): CalibrationSeries[] {
  const entries = Object.entries(by_market ?? {}).filter(([, stat]) => stat.n_resolved > 0 && stat.pct_correct != null);
  if (entries.length === 0) return [];
  return [
    {
      key: "by_market",
      label: "By market",
      buckets: entries.map(([key, stat]) => ({
        bucket: key,
        n: stat.n_resolved,
        correct: Math.round((stat.pct_correct ?? 0) * stat.n_resolved),
      })),
    },
  ];
}

function recentPicks(per_pick: TrackRecordResponse["summary"]["per_pick"] | undefined): { id: string; label: string; pick: string; actual: string; hit: boolean | null; when: string; madeBeforeKickoff: boolean }[] {
  const rows = (per_pick ?? [])
    .filter((p) => p.counted !== false && p.made_before_kickoff)
    .sort((a, b) => (b.snapshotted_at ?? "").localeCompare(a.snapshotted_at ?? ""))
    .slice(0, 10);
  return rows.map((p) => ({
    id: `${p.event_id ?? "unknown"}-${p.snapshotted_at ?? "unknown"}`,
    label: "Match result",
    pick: p.team_home ?? "—",
    actual: p.team_away ?? "—",
    hit: p.hit ?? null,
    when: p.snapshotted_at ?? "—",
    madeBeforeKickoff: p.made_before_kickoff ?? false,
  }));
}

/**
 * PL adapter: its `/hub/track-record` payload onto the shared `RecordModel`.
 *
 * PL's headline is every counted pick (`counted` mode per PRODUCT.md), with the
 * pre-kickoff subset shown beside it — which is exactly what PL's shipped page
 * already does, now expressed through the shared kit.
 */
export function toRecordModel(response: TrackRecordResponse): RecordModel {
  const { summary } = response;
  const markets = MARKETS.map((key) => marketTally(summary, key));
  const lead = markets[0]?.key ?? "match_result";
  return {
    markets,
    lead,
    headline: "counted",
    weeks: (summary.gameweek_trend ?? []).map(weekPoint),
    calibration: calibrationSeries(summary.by_market),
    recent: recentPicks(summary.per_pick),
  };
}
