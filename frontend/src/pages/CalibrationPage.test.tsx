import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { CalibrationPage } from "./CalibrationPage";

vi.mock("../api/client", () => {
  const failing = () => Promise.reject(new Error("not needed for this test"));
  // A group that HAS a real call count but no `calibration` array. This is the
  // shape the existing `calls` guard does not catch, and the one the deployed
  // endpoint can produce: `calibration` is a separate field from `calls`, so a
  // partial group passes the guard and then `.reduce`s on undefined.
  const partial = () => Promise.resolve({
    snapshot: { calls: 12, call_hits: 7, call_hit_rate: 7 / 12, goal_brier: 0.21 },
    reconstructed: {},
  });
  const target: Record<string, unknown> = {
    // The exact shape PUBLIC_MODE serves: background tracking is skipped
    // on the public host, so both groups arrive as empty objects.
    scorerTrackRecord: () => Promise.resolve({ snapshot: {}, reconstructed: {} }),
  };
  return {
    api: new Proxy(target, {
      get: (t, prop) => {
        if (prop in t) return t[prop as keyof typeof t];
        if (prop === "scorerTrackRecordPartial") return partial;
        return failing;
      },
    }),
  };
});

describe("CalibrationPage scorer track record", () => {
  it("renders a not-published note instead of crashing on the empty PUBLIC_MODE shape", async () => {
    render(<CalibrationPage />);
    expect(await screen.findAllByText("Not published for this deployment.")).toHaveLength(2);
    expect(document.body.textContent).not.toMatch(/undefined/);
  });

  it("survives a group that has calls but no calibration buckets", async () => {
    // The review's case. `stats.calibration.reduce(...)` was unguarded, and the
    // `typeof stats.calls === "number"` guard in front of it passes for this
    // payload — so the crash was one expression later than the fix that was
    // believed to cover it.
    const { api } = await import("../api/client");
    const spy = vi.spyOn(api, "scorerTrackRecord" as never).mockResolvedValue({
      snapshot: { calls: 12, call_hits: 7, call_hit_rate: 7 / 12, goal_brier: 0.21 },
      reconstructed: {},
    } as never);

    render(<CalibrationPage />);

    // The real call count is still shown — the group is real, only the buckets
    // are missing, and hiding the whole group would be over-correcting.
    expect(await screen.findByText(/7\/12/)).toBeInTheDocument();
    expect(document.body.textContent).toContain("across 0 confirmed starters");
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
    spy.mockRestore();
  });
});
