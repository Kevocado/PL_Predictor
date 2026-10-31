// Plain words for the model's own enums, so no snake_case reaches the screen.

const RESULT: Record<string, (home: string, away: string) => string> = {
  home_win: (home) => `${home} win`,
  away_win: (_home, away) => `${away} win`,
  draw: () => "Draw",
};

/** "away_win" -> "Forest win"; "over" with label "Cards O/U 3.5" -> "Over 3.5 cards".
 *  `actual` values that are bare numbers read as "2 cards" / "9 corners". */
export function verdictWords(label: string, value: string, home: string, away: string): string {
  const result = RESULT[value];
  if (result) return result(home, away);
  const ou = /^(Corners|Cards)\s+O\/U\s+([\d.]+)$/i.exec(label);
  const noun = ou ? ou[1].toLowerCase() : null;
  if (ou && (value === "over" || value === "under")) {
    return `${value === "over" ? "Over" : "Under"} ${ou[2]} ${noun}`;
  }
  if (noun && /^\d+(\.\d+)?$/.test(value)) return `${value} ${noun}`;
  return value.replace(/_/g, " ");
}

/** "Called it" / "Missed" sentence for a finished match's headline pick. */
export function verdictSentence(hit: boolean, pickLabel: string, finalScore: string): string {
  const score = finalScore.replace("-", "–");
  return hit ? `Called it: ${pickLabel} was the pick; it finished ${score}.` : `Missed: ${pickLabel} was the pick; it finished ${score}.`;
}

/** One date style for sub-lists: "23 Oct 2026". */
export function dateShort(iso: string): string {
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(iso) ? `${iso}T12:00:00Z` : iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

/** "About 11 corners expected · 63% chance of 10 or more" */
export function overUnderWords(noun: string, expected: number, line: number, over: number): string {
  const threshold = Number.isInteger(line) ? `more than ${line}` : `${Math.ceil(line)} or more`;
  return `About ${Math.round(expected)} ${noun} expected · ${Math.round(over * 100)}% chance of ${threshold}`;
}
