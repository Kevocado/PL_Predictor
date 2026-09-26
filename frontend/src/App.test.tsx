import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import App from "./App";

vi.mock("./api/client", () => {
  const pending = () => new Promise(() => {});
  return { api: new Proxy({}, { get: () => pending }) };
});

afterEach(() => vi.restoreAllMocks());

describe("App family frame", () => {
  it("carries the family wordmark, the PL accent and a switcher to every sport", () => {
    const { container } = render(<App />);
    expect(screen.getByRole("heading", { name: "PL Predictor" })).toBeInTheDocument();
    expect(container.querySelector("[data-sport='pl']")).not.toBeNull();
    const sports = screen.getByRole("navigation", { name: "Sports" });
    expect(within(sports).getByRole("link", { name: "PL" })).toHaveAttribute("aria-current", "page");
    expect(within(sports).getByRole("link", { name: "F1" }).getAttribute("href")).toMatch(/^https:\/\/f1\./);
  });

  it("switches pages with the page tabs, mounting each page on first visit", () => {
    render(<App />);
    const pages = screen.getByRole("navigation", { name: "Pages" });
    const labels = within(pages).getAllByRole("button").map((b) => b.textContent);
    expect(labels.slice(0, 3)).toEqual(["Fixtures", "Data Hub", "FPL"]);
    expect(within(pages).getByRole("button", { name: "Fixtures" })).toHaveAttribute("aria-current", "page");
    expect(screen.queryByTestId("page-fpl")).not.toBeInTheDocument();
    fireEvent.click(within(pages).getByRole("button", { name: "FPL" }));
    expect(within(pages).getByRole("button", { name: "FPL" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByTestId("page-fpl")).toBeVisible();
    expect(screen.getByTestId("page-fixtures")).not.toBeVisible();
  });
});
