import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FinishedFixtureCard } from "./FinishedFixtureCard";

// `made_before_kickoff` is the DERIVED flag the card reads. `backfilled` stays
// on the props — the payload still carries it — and is deliberately given the
// value a regression back to reading it would produce.
const base = {
  commence_time: "2026-09-19T10:30:00Z", team_home: "Tottenham", team_away: "Aston Villa",
  actual_goals_home: 2, actual_goals_away: 3, predicted_scoreline: "1-1",
  predicted_home_win: 0.36, predicted_draw: 0.26, predicted_away_win: 0.38,
  draw_signal: true, hit: true, backfilled: true, made_before_kickoff: true, onClick: () => {},
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
  it("marks a pick made after kickoff and does not show the stuck scorers line", () => {
    // This card's badge is SITE-LOCAL (a hand-written span, not the shared
    // `StatusBadge`), so the wording is ours and this repo changes it in step
    // with predictor-hub#67. It names the MOMENT, which is what the state now is.
    render(<FinishedFixtureCard {...base} made_before_kickoff={false} backfilled={false} player_events_pending />);
    expect(screen.getByText("Made after kickoff")).toBeInTheDocument();
    expect(screen.queryByText(/BACKFILLED/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Refreshing official scorers/)).not.toBeInTheDocument();
  });
  it("never judges a pick made after kickoff: one badge, no Called it or Missed", () => {
    // `backfilled={false}` here on purpose: the tick wrote this row late, so the
    // flag says live-captured while the timestamps say otherwise. The card must
    // follow the timestamps.
    render(<FinishedFixtureCard {...base} made_before_kickoff={false} />);
    expect(screen.getByText("Made after kickoff")).toBeInTheDocument();
    expect(screen.queryByText("Called it ✓")).not.toBeInTheDocument();
    expect(screen.queryByText("Missed ✗")).not.toBeInTheDocument();
  });
});
