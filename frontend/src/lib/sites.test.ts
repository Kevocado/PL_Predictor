import { describe, expect, it } from "vitest";
import { SITES } from "./sites";

describe("SITES", () => {
  it("has a hub entry as the first item with a non-empty label and an absolute https:// href", () => {
    const hub = SITES[0];
    expect(hub.sport).toBe("hub");
    expect(hub.label).toBeTruthy();
    expect(hub.label.length).toBeGreaterThan(0);
    expect(hub.href).toMatch(/^https:\/\//);
  });
});
