import { afterEach, expect, it, vi } from "vitest";
import { api } from "./client";

afterEach(() => vi.unstubAllGlobals());

it("loadContext is built on the same base as explain(\"pl\", id)", async () => {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ matchups: [] }) });
  vi.stubGlobal("fetch", fetchMock);
  await api.loadContext("12345");
  expect(fetchMock.mock.calls[0][0]).toBe("/api/explain/pl/12345/context");
});
