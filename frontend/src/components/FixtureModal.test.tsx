import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import type { FixtureDetail } from "../types";

vi.mock("../api/client", () => ({
  api: {
    fixtureDetail: vi.fn(),
    fixturePlayers: vi.fn(),
    fixturePlayerReview: vi.fn(),
  },
}));

const edge = (p: number) => ({ prob: p, implied: null, edge: null });
const ou = (over: number) => ({ lambda_: 2.7, line: 2.5, over, under: 1 - over });

const detail = {
  event_id: "e1", commence_time: "2026-09-19T10:30:00Z",
  team_home: "Tottenham", team_away: "Aston Villa",
  home_win: edge(0.44), draw: edge(0.28), away_win: edge(0.28),
  over_2_5: ou(0.52), under_2_5: ou(0.48), btts_yes_prob: 0.48,
  value_bet_flags: [], value_bet: null, has_live_odds: false,
  corners: ou(0.5), cards: ou(0.5),
  top_scoreline: "1-1", predicted_result: "home_win", draw_signal: false,
  is_fallback_prediction: false, data_confidence: "established",
  home_context: { rest_days: 3, xg_for_last_5: 1.6, xg_against_last_5: 1.1, corners_last_5: 5, cards_last_5: 2, set_piece_xg_share_last_5: 0.2 },
  away_context: { rest_days: 3, xg_for_last_5: 1.4, xg_against_last_5: 1.2, corners_last_5: 4, cards_last_5: 2, set_piece_xg_share_last_5: 0.18 },
  // ScorelineHeatmap indexes grid[0] unconditionally, so an empty grid is not a
  // valid FixtureDetail -- the real endpoint always returns rows.
  score_grid: [[0.1, 0.16, 0.12], [0.12, 0.14, 0.08], [0.09, 0.11, 0.08]],
  top_scorelines: [{ home: 1, away: 1, prob: 0.14 }],
  home_shots: null, away_shots: null, home_shots_on_target: null, away_shots_on_target: null,
  head_to_head: [], home_recent_form: [], away_recent_form: [],
  predicted_total_goals: 2.7, predicted_margin: 0.2,
  home_2plus_prob: 0.55, away_2plus_prob: 0.5,
  odds_fetched_at: null, odds_is_stale: false, recommended_bet: null,
  post_match: null, actual_stats: null, pre_match_value_bets: [],
} as unknown as FixtureDetail;

const explanation = {
  headline: "Tottenham are the narrow favourites, but this is close to a coin flip.",
  sections: [{ market: "result", title: "Why Tottenham", text: "The model has them at 44% at home." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "pl",
  pick_timing: "pre_kickoff" as const,
};

function mockApi(over: Record<string, unknown> = {}) {
  vi.mocked(api.fixtureDetail).mockResolvedValue(detail);
  vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
  vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
  Object.assign(api, over);
}

describe("FixtureModal and the plain-English panel", () => {
  it("fetches the summary for this fixture and shows its headline", async () => {
    mockApi();
    const explain = vi.fn().mockResolvedValue(explanation);
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(explain).toHaveBeenCalledWith("pl", "e1");
  });

  it("shows the fixture while the summary is still being written", async () => {
    mockApi();
    const explain = vi.fn().mockReturnValue(new Promise<typeof explanation>(() => {}));
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    expect(screen.getByText("Writing the summary…")).toBeInTheDocument();
    // The fixture itself must not wait on the summary. The team name appears in
    // several places in this modal, so assert the fixture heading reached the
    // screen rather than a first match on a common string.
    expect(await screen.findByRole("heading", { name: /Scoreline/i })).toBeInTheDocument();
  });

  it("offers a retry that asks the explainer again", async () => {
    mockApi();
    const explain = vi.fn().mockRejectedValue(new Error("down"));
    render(<FixtureModal eventId="e1" onClose={() => {}} explain={explain} />);
    const retry = await screen.findByRole("button", { name: "Try again" });
    explain.mockResolvedValue(explanation);
    fireEvent.click(retry);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(explain).toHaveBeenCalledTimes(2);
  });

  it("leaves the modal usable when there is no explainer", async () => {
    mockApi();
    render(<FixtureModal eventId="e1" onClose={() => {}} />);
    expect(screen.queryByText("Writing the summary…")).not.toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /Scoreline/i })).toBeInTheDocument();
  });
});
