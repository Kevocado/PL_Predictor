import { describe, expect, it } from "vitest";
import { prekickoffTally } from "./CurrentGameweekSection";

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
    expect(screen.getByText(/No pre-kickoff picks this gameweek/)).toBeInTheDocument();
  });
});
