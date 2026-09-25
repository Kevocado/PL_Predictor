import { describe, expect, it } from "vitest";
import { matchPick } from "./pick";

describe("matchPick", () => {
  it("picks the most likely side and names it", () => {
    expect(matchPick(0.36, 0.26, 0.38, "Tottenham", "Aston Villa")).toEqual({ side: "away_win", label: "Aston Villa win", prob: 0.38 });
    expect(matchPick(0.57, 0.21, 0.22, "Brentford", "Chelsea").label).toBe("Brentford win");
    expect(matchPick(0.3, 0.4, 0.3, "A", "B").label).toBe("Draw");
  });
  it("breaks ties home, then draw, then away, like the API", () => {
    expect(matchPick(0.4, 0.4, 0.2, "A", "B").side).toBe("home_win");
    expect(matchPick(0.2, 0.4, 0.4, "A", "B").side).toBe("draw");
  });
});
