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

describe("accent in inline styles and portals", () => {
  it("sets the PL sport on <html>, so :root aliases and body portals resolve to PL magenta", async () => {
    const { readFileSync } = await import("node:fs");
    const html = readFileSync(resolve(__dirname, "../index.html"), "utf8");
    expect(html).toMatch(/<html[^>]*data-sport="pl"/);
  });
  it("keeps the second chart colour distinct from text white", () => {
    expect(css).toMatch(/--color-pl-cyan:\s*var\(--color-pr-lean\)/);
    for (const name of ["pl-blue", "pl-accent"]) expect(css).toMatch(new RegExp(`--color-${name}:\\s*var\\(--color-pr-`));
  });
});
