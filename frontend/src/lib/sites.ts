import type { SiteLink } from "../predictor-ui";

// Every Predictor site, for the family switcher. Change the hosts here when
// the real domain lands.
const HOST = "40-160-91-131.sslip.io";
export const SITES: SiteLink[] = [
  { sport: "pl", label: "PL", href: "/" },
  { sport: "f1", label: "F1", href: `https://f1.${HOST}` },
  { sport: "nfl", label: "NFL", href: `https://sports.${HOST}/?sport=nfl` },
  { sport: "cfb", label: "CFB", href: `https://sports.${HOST}/?sport=cfb` },
  { sport: "nba", label: "NBA", href: `https://nba.${HOST}` },
];
