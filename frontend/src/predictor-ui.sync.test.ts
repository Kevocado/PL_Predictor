import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { describe, expect, it } from "vitest";

// src/predictor-ui is vendored from predictor-hub/packages/predictor-ui by
// scripts/sync-ui.mjs. Edits belong in the hub; this fails if the copy here
// drifts from the manifest the sync wrote (same rules as `sync-ui --check`).
const dir = resolve(__dirname, "predictor-ui");
const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");
const walk = (d: string): string[] =>
  readdirSync(d).flatMap((n) => (statSync(join(d, n)).isDirectory() ? walk(join(d, n)) : [join(d, n)]));

describe("vendored predictor-ui", () => {
  it("matches its sync manifest exactly", () => {
    expect(existsSync(join(dir, "SYNC.json")), "run: node ../../predictor-hub/scripts/sync-ui.mjs src").toBe(true);
    const { files } = JSON.parse(readFileSync(join(dir, "SYNC.json"), "utf8")) as { files: Record<string, string> };
    for (const [rel, hash] of Object.entries(files)) {
      expect(sha256(readFileSync(join(dir, rel), "utf8").replace(/\r\n/g, "\n")), rel).toBe(hash);
    }
    const extra = walk(dir).map((f) => relative(dir, f).split("\\").join("/")).filter((r) => r !== "SYNC.json" && !(r in files));
    expect(extra).toEqual([]);
  });
});
