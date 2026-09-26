import { kickoff, parseKickoff } from "../predictor-ui";

// Kickoff times in the viewer's zone, via the family formatter. Cards show
// day and time; the zone is stated once per gameweek (kickoffZones).

export function kickoffParts(iso: string, timeZone?: string): { day: string; time: string } {
  const [day, rest = ""] = kickoff(iso, timeZone).split(" · ");
  // "7:30 PM CDT" -> "7:30 PM"; the zone is said once for the page.
  return { day, time: rest.split(" ").slice(0, 2).join(" ") };
}

/** "CDT", or "BST/GMT" across a clock change. */
export function kickoffZones(isos: string[], timeZone?: string): string {
  const zones = isos
    .filter((iso) => !Number.isNaN(parseKickoff(iso).getTime()))
    .map((iso) => kickoff(iso, timeZone).split(" ").pop() ?? "");
  return [...new Set(zones)].filter(Boolean).join("/");
}
