import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const src = resolve(__dirname);
const walk = (d: string): string[] =>
  readdirSync(d).flatMap((n) => (statSync(join(d, n)).isDirectory() ? (n === "predictor-ui" ? [] : walk(join(d, n))) : [join(d, n)]));

describe("type scale", () => {
  it("has no text under 12px anywhere on the site", () => {
    const offenders = walk(src)
      .filter((f) => f.endsWith(".tsx") && !f.endsWith(".test.tsx"))
      .filter((f) => /text-\[(?:[0-9]|1[01])px\]/.test(readFileSync(f, "utf8")));
    expect(offenders).toEqual([]);
  });
});
