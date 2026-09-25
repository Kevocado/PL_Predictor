import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(__dirname, "index.css"), "utf8");

describe("index.css", () => {
  it("loads the family fonts first and the tokens last", () => {
    const order = ["./predictor-ui/fonts.css", "tailwindcss", "./predictor-ui/tokens.css"].map((s) => css.indexOf(`@import "${s}"`));
    expect(order.every((i) => i >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });
  it("drops Inter, Oswald and the radial glow", () => {
    expect(css).not.toMatch(/Inter/);
    expect(css).not.toMatch(/Oswald/);
    expect(css).not.toMatch(/radial-gradient/);
  });
  it("declares the legacy aliases inline so they follow each sport's accent", () => {
    // A plain @theme resolves var(--color-pr-accent) once at :root (the Hub's
    // white); inline keeps the var() in each utility, where data-sport applies.
    expect(css).toMatch(/@theme inline\s*\{[^}]*--color-pl-pink:\s*var\(--color-pr-accent\)/);
  });
  it("maps every legacy PL colour onto a family token", () => {
    const legacy = [...css.matchAll(/--color-(?:pl-[\w-]+|win|draw|loss):\s*([^;]+);/g)].map((m) => m[1].trim());
    expect(legacy.length).toBeGreaterThan(0);
    for (const value of legacy) expect(value).toMatch(/^var\(--color-pr-/);
  });
});
