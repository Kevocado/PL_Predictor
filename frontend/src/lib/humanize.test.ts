import { describe, expect, it } from "vitest";
import { dateShort, overUnderWords, verdictSentence, verdictWords } from "./humanize";

describe("humanize", () => {
  it("turns result enums into team words", () => {
    expect(verdictWords("Match result", "away_win", "Ipswich", "Forest")).toBe("Forest win");
    expect(verdictWords("Match result", "draw", "Ipswich", "Forest")).toBe("Draw");
  });
  it("turns over/under enums into the line it was judged on", () => {
    expect(verdictWords("Cards O/U 3.5", "over", "A", "B")).toBe("Over 3.5 cards");
    expect(verdictWords("Cards O/U 3.5", "2", "A", "B")).toBe("2 cards");
  });
  it("never leaves an underscore on screen", () => {
    expect(verdictWords("Something", "both_score", "A", "B")).toBe("both score");
  });
  it("writes the verdict like a person", () => {
    expect(verdictSentence(false, "Forest", "1-1")).toBe("Missed: Forest was the pick; it finished 1–1.");
    expect(verdictSentence(true, "Arsenal", "2-0")).toBe("Called it: Arsenal was the pick; it finished 2–0.");
  });
  it("says corners in plain words", () => {
    expect(overUnderWords("corners", 10.7, 9.5, 0.63)).toBe("About 11 corners expected · 63% chance of 10 or more");
    expect(overUnderWords("cards", 3.2, 3, 0.4)).toBe("About 3 cards expected · 40% chance of more than 3");
  });
  it("formats dates once", () => {
    expect(dateShort("2026-10-23")).toBe("23 Oct 2026");
  });
});
