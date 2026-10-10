// Synced from predictor-ui@1b32485a6a33. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { MatchupRow } from "../components/MatchupBrief";

const isText = (v: unknown): v is string => typeof v === "string" && v.trim() !== "";
const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

const drawable = (m: MatchupRow) =>
  !!m &&
  isText(m.id) &&
  isText(m.attacker) &&
  isText(m.defender) &&
  isText(m.stat) &&
  isText(m.foil) &&
  isInt(m.attacker_rank) &&
  isInt(m.defender_rank) &&
  isInt(m.n_teams) &&
  m.n_teams >= 2 &&
  m.attacker_rank >= 1 &&
  m.attacker_rank <= m.n_teams &&
  m.defender_rank >= 1 &&
  m.defender_rank <= m.n_teams;

function stripSuffix(id: string): string {
  return id.replace(/:home$|:away$/, "");
}

// Label map for known stat keys (sentence case, matching the stat field)
const LABEL_MAP: Record<string, string> = {
  goals_scored_per_match: "Goals scored per match",
  goals_conceded_per_match: "Goals conceded per match",
  shots_on_target_per_match: "Shots on target per match",
  shots_on_target_faced_per_match: "Shots on target faced per match",
  possession_pct: "Possession %",
  pass_off_vs_pass_def: "Pass off vs pass def",
  rush_off_vs_rush_def: "Rush off vs rush def",
  turnover_diff: "Turnover diff",
};

function labelFromKey(key: string): string {
  if (LABEL_MAP[key]) return LABEL_MAP[key];
  // Fallback: sentence case from snake_case
  return key
    .split("_")
    .map((w, i) => (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/**
 * Pivots matchup rows into a table structure with home on the left, away on the right.
 * Returns null when nothing drawable.
 * 
 * For each base key (id without :home/:away suffix):
 * - homeRank comes from the :home row's attacker_rank (home team's rank for that stat)
 * - awayRank comes from the :away row's attacker_rank (away team's rank for that stat)
 * - If a side's row is missing, that rank is null
 */
export function pivotMatchups(rows: MatchupRow[]): {
  home: string;
  away: string;
  rows: { key: string; label: string; homeRank: number | null; awayRank: number | null; n: number }[];
} | null {
  const valid = (Array.isArray(rows) ? rows : []).filter(drawable);
  if (valid.length === 0) return null;

  // Determine home/away from :home rows (attacker is home team in :home rows)
  const homeRow = valid.find(r => r.id.endsWith(":home"));
  const awayRow = valid.find(r => r.id.endsWith(":away"));
  
  let home: string;
  let away: string;
  
  if (homeRow) {
    home = homeRow.attacker;
    away = homeRow.defender;
  } else if (awayRow) {
    // Fallback: if only :away rows, attacker is away team
    away = awayRow.attacker;
    home = awayRow.defender;
  } else {
    // No suffix rows - use first row
    home = valid[0].attacker;
    away = valid[0].defender;
  }

  // Group by base key, track home/away ranks separately
  const byKey = new Map<string, { homeRank: number | null; awayRank: number | null; n: number }>();

  for (const row of valid) {
    const key = stripSuffix(row.id);
    const existing = byKey.get(key) ?? { homeRank: null, awayRank: null, n: row.n_teams };
    
    // :home row: attacker=home, defender=away
    // :away row: attacker=away, defender=home
    if (row.id.endsWith(":home")) {
      if (row.attacker === home) existing.homeRank = row.attacker_rank;
      if (row.defender === away) existing.awayRank = row.defender_rank;
    } else if (row.id.endsWith(":away")) {
      if (row.attacker === away) existing.awayRank = row.attacker_rank;
      if (row.defender === home) existing.homeRank = row.defender_rank;
    } else {
      // No suffix - try to match by team
      if (row.attacker === home) existing.homeRank = row.attacker_rank;
      else if (row.defender === home) existing.homeRank = row.defender_rank;
      if (row.attacker === away) existing.awayRank = row.attacker_rank;
      else if (row.defender === away) existing.awayRank = row.defender_rank;
    }
    
    byKey.set(key, existing);
  }

  // Sort by key for deterministic output (alphabetical)
  const sortedEntries = [...byKey.entries()].sort((a, b) => a[0].localeCompare(b[0]));

  return {
    home,
    away,
    rows: sortedEntries.map(([key, data]) => ({
      key,
      label: labelFromKey(key),
      homeRank: data.homeRank,
      awayRank: data.awayRank,
      n: data.n,
    })),
  };
}