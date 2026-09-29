import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { CalibrationPage } from "./CalibrationPage";

vi.mock("../api/client", () => {
  const failing = () => Promise.reject(new Error("not needed for this test"));
  return {
    api: new Proxy(
      {
        // The exact shape PUBLIC_MODE serves: background tracking is skipped
        // on the public host, so both groups arrive as empty objects.
        scorerTrackRecord: () => Promise.resolve({ snapshot: {}, reconstructed: {} }),
      },
      { get: (target, prop) => (prop in target ? target[prop as keyof typeof target] : failing) },
    ),
  };
});

describe("CalibrationPage scorer track record", () => {
  it("renders a not-published note instead of crashing on the empty PUBLIC_MODE shape", async () => {
    render(<CalibrationPage />);
    expect(await screen.findAllByText("Not published for this deployment.")).toHaveLength(2);
    expect(document.body.textContent).not.toMatch(/undefined/);
  });
});
