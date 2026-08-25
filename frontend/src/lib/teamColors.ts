// Fallback colour + initials for a local crest asset that is unavailable.

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

const CREST_FILES: Record<string, string> = {
  Arsenal: "arsenal-logo-footylogos.svg",
  "Aston Villa": "aston-villa-logo-footylogos.svg",
  Bournemouth: "afc-bournemouth-logo-footylogos.svg",
  Brentford: "brentford-logo-footylogos.svg",
  Brighton: "brighton-and-hove-albion-logo-footylogos.svg",
  Chelsea: "chelsea-logo-footylogos.svg",
  Coventry: "coventry-city-logo-footylogos.svg",
  "Crystal Palace": "crystal-palace-logo-footylogos.svg",
  Everton: "everton-logo-footylogos.svg",
  Fulham: "fulham-logo-footylogos.svg",
  Hull: "hull-city-logo-footylogos.svg",
  Ipswich: "ipswich-town-logo-footylogos.svg",
  Leeds: "leeds-united-logo-footylogos.svg",
  Liverpool: "liverpool-fc-logo-footylogos.svg",
  "Man City": "manchester-city-logo-footylogos.svg",
  "Man United": "manchester-united-logo-footylogos.svg",
  Newcastle: "newcastle-united-logo-footylogos.svg",
  "Nott'm Forest": "nottingham-forest-logo-footylogos.svg",
  Sunderland: "sunderland-logo-footylogos.svg",
  Tottenham: "tottenham-hotspur-logo-footylogos.svg",
};

export function teamCrestUrl(team: string): string | null {
  const filename = CREST_FILES[team];
  return filename ? `/badges/${filename}` : null;
}

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
