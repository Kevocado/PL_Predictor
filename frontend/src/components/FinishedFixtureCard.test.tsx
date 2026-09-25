import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FinishedFixtureCard } from "./FinishedFixtureCard";

const base = {
  commence_time: "2026-09-19T10:30:00Z", team_home: "Tottenham", team_away: "Aston Villa",
  actual_goals_home: 2, actual_goals_away: 3, predicted_scoreline: "1-1",
  predicted_home_win: 0.36, predicted_draw: 0.26, predicted_away_win: 0.38,
  draw_signal: true, hit: true, backfilled: false, onClick: () => {},
};

describe("FinishedFixtureCard", () => {
  it("states one pick and judges the verdict against it", () => {
    render(<FinishedFixtureCard {...base} />);
    expect(screen.getByText("Pick: Aston Villa win · 38%")).toBeInTheDocument();
    expect(screen.getByText("Called it ✓")).toBeInTheDocument();
    expect(screen.getByText("Most likely score 1–1")).toBeInTheDocument();
    expect(screen.queryByText(/We predicted/)).not.toBeInTheDocument();
    expect(screen.queryByText("Leaned draw")).not.toBeInTheDocument();
  });
  it("labels the home/draw/away split", () => {
    render(<FinishedFixtureCard {...base} />);
    expect(screen.getByText("Home 36% · Draw 26% · Away 38%")).toBeInTheDocument();
  });
  it("marks a rebuilt pick and does not show the stuck scorers line", () => {
    render(<FinishedFixtureCard {...base} backfilled player_events_pending />);
    expect(screen.getByText("Rebuilt after kickoff")).toBeInTheDocument();
    expect(screen.queryByText(/BACKFILLED/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Refreshing official scorers/)).not.toBeInTheDocument();
  });
});
