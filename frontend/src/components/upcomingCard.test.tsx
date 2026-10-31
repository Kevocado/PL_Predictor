import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { CurrentGameweekCard } from "./CurrentGameweekCard";
import { ScorelineHeatmap } from "./ScorelineHeatmap";
import { SectionBar } from "./SectionBar";
import type { CurrentGameweekFixture } from "../types";

const base: CurrentGameweekFixture = {
  event_id: "e1", team_home: "Man City", team_away: "Fulham", commence_time: "2026-10-31T15:00:00Z",
  finished: false, actual_goals_home: null, actual_goals_away: null,
  predicted_home_win: 0.6, predicted_draw: 0.22, predicted_away_win: 0.18, predicted_scoreline: "2-1",
  draw_signal: false, hit: null, backfilled: false, has_live_odds: true, value_bet_flags: [],
};

describe("upcoming fixture card", () => {
  it("answers who the model likes, with the likely score beside it", () => {
    render(<CurrentGameweekCard fixture={base} onClick={() => {}} />);
    expect(screen.getByText(/Pick: Man City · 60%/)).toBeInTheDocument();
    expect(screen.getByText("Most likely score 2–1")).toBeInTheDocument();
  });
  it("says toss-up when the top two are within two points", () => {
    render(<CurrentGameweekCard fixture={{ ...base, team_home: "Liverpool", team_away: "Brighton", predicted_home_win: 0.38, predicted_draw: 0.24, predicted_away_win: 0.38 }} onClick={() => {}} />);
    expect(screen.getByText("Toss-up · Liverpool 38%, Brighton 38%")).toBeInTheDocument();
  });
  it("shows at most one badge: a value mark on the pick line, never Leans draw", () => {
    render(<CurrentGameweekCard fixture={{ ...base, draw_signal: true, value_bet_flags: ["home_win"] }} onClick={() => {}} />);
    expect(screen.getByText("Value")).toBeInTheDocument();
    expect(screen.queryByText("Leans draw")).not.toBeInTheDocument();
    expect(screen.queryByText("Value bet")).not.toBeInTheDocument();
  });
  it("paints no win/draw/loss colour on the pre-match bar", () => {
    const { container } = render(<CurrentGameweekCard fixture={base} onClick={() => {}} />);
    expect(container.innerHTML).not.toMatch(/bg-win|bg-draw|bg-loss/);
  });
});

describe("scoreline grid", () => {
  const grid = [
    [0.1, 0.08, 0.03],
    [0.2, 0.12, 0.05],
    [0.15, 0.1, 0.17],
  ];
  const top = [{ home: 2, away: 2, prob: 0.17 }, { home: 1, away: 0, prob: 0.2 }];
  it("lists the most likely scores first and shows each region's share summing to 100%", () => {
    render(<ScorelineHeatmap grid={grid} homeTeam="Home" awayTeam="Away" topScorelines={top} />);
    expect(screen.getByText("Most likely scores")).toBeInTheDocument();
    const labels = ["Home win", "Draw", "Away win"].map((l) => screen.getByRole("button", { name: new RegExp(`^${l}`) }).textContent ?? "");
    const sum = labels.map((t) => parseInt(/(\d+)%$/.exec(t)![1], 10)).reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 100)).toBeLessThanOrEqual(1);
  });
  it("reads a cell's chance when it is tapped", () => {
    render(<ScorelineHeatmap grid={grid} homeTeam="Home" awayTeam="Away" topScorelines={top} />);
    fireEvent.click(screen.getByRole("button", { name: "Home 1, Away 0: 20.0%" }));
    expect(screen.getByText("Home 1–0 Away: 20.0% chance")).toBeInTheDocument();
  });
  it("marks the chosen region pressed", () => {
    render(<ScorelineHeatmap grid={grid} homeTeam="Home" awayTeam="Away" topScorelines={top} />);
    const draw = screen.getByRole("button", { name: /^Draw/ });
    fireEvent.click(draw);
    expect(draw).toHaveAttribute("aria-pressed", "true");
  });
});

describe("section bar", () => {
  it("lists every section and marks the one clicked", () => {
    const root = { current: document.createElement("div") };
    render(<SectionBar links={[{ id: "a", label: "Pick" }, { id: "b", label: "Scores" }]} root={root} />);
    fireEvent.click(screen.getByRole("button", { name: "Scores" }));
    expect(screen.getByRole("button", { name: "Scores" })).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: "Pick" })).not.toHaveAttribute("aria-current");
  });
});
