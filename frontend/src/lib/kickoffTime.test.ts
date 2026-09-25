import { describe, expect, it } from "vitest";
import { kickoffParts, kickoffZones } from "./kickoffTime";

describe("kickoff times", () => {
  it("reads the API's zoneless times as UTC (an 8 PM BST kickoff is 19:00 UTC)", () => {
    expect(kickoffParts("2026-08-21T19:00:00")).toEqual({ day: "Fri 21 Aug", time: "2:00 PM" });
    expect(kickoffParts("2026-08-21T19:00:00+00:00")).toEqual({ day: "Fri 21 Aug", time: "2:00 PM" });
    expect(kickoffParts("2026-08-21T19:00:00", "Europe/London").time).toBe("8:00 PM");
  });
  it("names both zones once when a gameweek spans a clock change", () => {
    expect(kickoffZones(["2026-10-24T14:00:00", "2026-10-26T20:00:00"], "Europe/London")).toBe("GMT+1/GMT");
  });
});
