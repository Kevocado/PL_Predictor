import { describe, expect, it } from "vitest";
import { prekickoffTally } from "./CurrentGameweekSection";

const f = (finished: boolean, hit: boolean | null, backfilled: boolean) => ({ finished, hit, backfilled });

describe("prekickoffTally", () => {
  it("counts only finished picks made before kickoff", () => {
    expect(prekickoffTally([f(true, true, true), f(true, false, false), f(true, true, false), f(false, null, false)]))
      .toEqual({ hits: 1, settled: 2, rebuilt: 1 });
  });
  it("reports zero settled when every finished pick was rebuilt", () => {
    expect(prekickoffTally([f(true, true, true), f(true, false, true)])).toEqual({ hits: 0, settled: 0, rebuilt: 2 });
  });
});
