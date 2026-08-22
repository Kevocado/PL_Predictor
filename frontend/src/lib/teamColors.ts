// Deterministic color + initials per team name, so badges are stable across
// reloads without needing a crest-image dependency.

const PALETTE = [
  "#E90052", // pink
  "#05F0D1", // cyan
  "#6B32A0", // violet
  "#F97316", // orange
  "#22C55E", // green
  "#3B82F6", // blue
  "#EAB308", // gold
  "#EC4899", // fuchsia
  "#14B8A6", // teal
  "#F43F5E", // rose
];

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

export function teamColor(team: string): string {
  return PALETTE[hashString(team) % PALETTE.length];
}

const OVERRIDES: Record<string, string> = {
  "Man City": "MCI",
  "Man United": "MUN",
  "Nott'm Forest": "NFO",
  Tottenham: "TOT",
  "Aston Villa": "AVL",
  "Crystal Palace": "CRY",
  "West Brom": "WBA",
  "Sheffield United": "SHU",
  "West Ham": "WHU",
};

export function teamInitials(team: string): string {
  if (OVERRIDES[team]) return OVERRIDES[team];
  const words = team.replace(/'/g, "").split(/\s+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return words
    .map((w) => w[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
}
