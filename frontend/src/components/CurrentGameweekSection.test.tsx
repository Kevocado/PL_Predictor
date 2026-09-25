import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { prekickoffTally } from "./CurrentGameweekSection";

// The tests below import Testing Library lazily, so unmount explicitly.
afterEach(cleanup);

const f = (finished: boolean, hit: boolean | null, backfilled: boolean) => ({ finished, hit, backfilled });

describe("prekickoffTally", () => {
  it("counts only finished picks made before kickoff", () => {
    expect(prekickoffTally([f(true, true, true), f(true, false, false), f(true, true, false), f(false, null, false)]))
      .toEqual({ hits: 1, settled: 2, rebuilt: 1 });
  });
  it("reports zero settled when every finished pick was rebuilt", () => {
    expect(prekickoffTally([f(true, true, true), f(true, false, true)])).toEqual({ hits: 0, settled: 0, rebuilt: 2 });
  });
});

describe("gameweek headline copy", () => {
  it("does not promise that rebuilt-only gameweeks will settle later", async () => {
    const { render, screen } = await import("@testing-library/react");
    const { CurrentGameweekSection } = await import("./CurrentGameweekSection");
    const fixture = {
      event_id: "e1", team_home: "Tottenham", team_away: "Aston Villa", commence_time: "2026-09-19T10:30:00Z",
      finished: true, actual_goals_home: 2, actual_goals_away: 3, predicted_home_win: 0.36, predicted_draw: 0.26,
      predicted_away_win: 0.38, predicted_scoreline: "1-1", draw_signal: false, hit: true, backfilled: true,
      has_live_odds: false, value_bet_flags: [], home_player_events: [], away_player_events: [], player_events_pending: false,
    };
    const data = { gameweek: 5, is_current: true, min_gameweek: 1, max_gameweek: 38, fixtures: [fixture] };
    render(<CurrentGameweekSection data={data as never} onSelect={() => {}} onNavigate={() => {}} />);
    expect(screen.getByText("No picks made before kickoff this gameweek")).toBeInTheDocument();
  });
});

describe("gameweek navigator", () => {
  const fixture = (id: string, commence_time: string, over: Record<string, unknown> = {}) => ({
    event_id: id, team_home: "Tottenham", team_away: "Aston Villa", commence_time,
    finished: false, actual_goals_home: null, actual_goals_away: null, predicted_home_win: 0.36, predicted_draw: 0.26,
    predicted_away_win: 0.38, predicted_scoreline: "1-1", draw_signal: false, hit: null, backfilled: false,
    has_live_odds: true, value_bet_flags: [], home_player_events: [], away_player_events: [], player_events_pending: false, ...over,
  });

  it("heads the gameweek with its record and says which zone kickoff times are in", async () => {
    const { render, screen } = await import("@testing-library/react");
    const { CurrentGameweekSection } = await import("./CurrentGameweekSection");
    const data = {
      gameweek: 6, is_current: true, min_gameweek: 1, max_gameweek: 38,
      fixtures: [
        fixture("a", "2026-09-26T11:30:00Z", { finished: true, actual_goals_home: 2, actual_goals_away: 0, hit: true }),
        fixture("b", "2026-09-27T14:00:00Z"),
      ],
    };
    render(<CurrentGameweekSection data={data as never} onSelect={() => {}} onNavigate={() => {}} />);
    expect(screen.getByRole("heading", { name: "Gameweek 6" })).toBeInTheDocument();
    expect(screen.getByText("1/1 picks made before kickoff correct")).toBeInTheDocument();
    expect(screen.getByText("Kickoff times in CDT")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Jump to current gameweek" })).not.toBeInTheDocument();
  });

  it("offers a jump back to the current gameweek instead of a 'not current' pill", async () => {
    const { render, screen, fireEvent } = await import("@testing-library/react");
    const { CurrentGameweekSection } = await import("./CurrentGameweekSection");
    const onNavigate = vi.fn();
    const data = { gameweek: 3, is_current: false, min_gameweek: 1, max_gameweek: 38, fixtures: [fixture("a", "2026-08-30T14:00:00Z")] };
    render(<CurrentGameweekSection data={data as never} onSelect={() => {}} onNavigate={onNavigate} />);
    expect(screen.queryByText(/not current/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Jump to current gameweek" }));
    expect(onNavigate).toHaveBeenCalledWith(undefined);
    fireEvent.click(screen.getByRole("button", { name: "Next gameweek" }));
    expect(onNavigate).toHaveBeenLastCalledWith(4);
  });
});
