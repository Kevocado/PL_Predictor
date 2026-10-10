/** Per-player Shots / SoT are hidden until the model passes its evaluation gate (see SHOW_PLAYER_SHOTS). */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PlayerScorerList, SHOW_PLAYER_SHOTS } from "./PlayerScorerList";
import type { PlayerPrediction } from "../types";

const p = (over: Partial<PlayerPrediction>): PlayerPrediction => ({
  player_id: 7, name: "Martin Odegaard", position: "MID", team: "Arsenal",
  anytime_goal_prob: 0.28, anytime_assist_prob: 0.32, anytime_goal_contribution_prob: 0.5,
  status: "a", news: "", confidence: "medium", predicted_starter: true, confirmed_starter: false,
  expected_minutes: 90, is_penalty_taker: false, is_set_piece_taker: false,
  expected_shots: 1.2, expected_shots_on_target: 0.4, anytime_shot_on_target_prob: 0.3, expected_saves: 0,
  ...over,
} as PlayerPrediction);

describe("PlayerScorerList shots", () => {
  it("does not render Shots / SoT even when the API sends them", () => {
    expect(SHOW_PLAYER_SHOTS).toBe(false);
    render(<PlayerScorerList homeTeam="Arsenal" awayTeam="Leeds" homePlayers={[p({})]} awayPlayers={[]} />);
    expect(screen.getByText("Martin Odegaard")).toBeInTheDocument();
    expect(screen.getByText(/G\+A/)).toBeInTheDocument(); // the rest of the row is untouched
    expect(screen.queryByText(/Shots/)).toBeNull();
    expect(screen.queryByText(/SoT/)).toBeNull();
  });
});
