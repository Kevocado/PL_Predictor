import { describe, expect, it } from "vitest";
import { matchPick, pickCall } from "./pick";

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


describe("pickCall", () => {
  it("names the favourite with its percentage", () => {
    const c = pickCall(0.6, 0.22, 0.18, "Man City", "Fulham");
    expect(c.text).toBe("Man City · 60%");
    expect(c.tossUp).toBe(false);
  });
  it("calls it a toss-up when the top two are within 2 points", () => {
    const c = pickCall(0.38, 0.24, 0.38, "Liverpool", "Brighton");
    expect(c.tossUp).toBe(true);
    expect(c.text).toBe("Toss-up · Liverpool 38%, Brighton 38%");
  });
  it("keeps matchPick's side so a finished verdict never changes", () => {
    expect(pickCall(0.38, 0.24, 0.37, "Palace", "Newcastle").side).toBe("home_win");
  });
  it("names the draw when it leads", () => {
    expect(pickCall(0.3, 0.4, 0.3, "A", "B").text).toBe("Draw · 40%");
  });
});
