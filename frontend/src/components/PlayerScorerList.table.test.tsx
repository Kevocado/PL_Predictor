/** The fixture modal's player table on a phone: one team at a time, full names pinned, stats scroll. */
import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { PlayerScorerList } from "./PlayerScorerList";
import type { PlayerPrediction } from "../types";

const p = (over: Partial<PlayerPrediction>): PlayerPrediction => ({
  player_id: 1, name: "Martin Odegaard", position: "MID", team: "Arsenal",
  anytime_goal_prob: 0.28, anytime_assist_prob: 0.32, anytime_goal_contribution_prob: 0.5,
  status: "a", news: "", confidence: "medium", predicted_starter: true, confirmed_starter: false,
  expected_minutes: 90, is_penalty_taker: false, is_set_piece_taker: false,
  expected_shots: 1.2, expected_shots_on_target: 0.4, anytime_shot_on_target_prob: 0.3, expected_saves: 0,
  ...over,
} as PlayerPrediction);

const home = [
  p({ player_id: 1, name: "Trent Alexander-Arnold-Longname", position: "DEF", anytime_goal_contribution_prob: 0.2 }),
  p({ player_id: 2, name: "David Raya Martin", position: "GK", expected_saves: 3.2, anytime_goal_contribution_prob: 0 }),
  p({ player_id: 3, name: "Bukayo Saka", position: "MID", is_penalty_taker: true }),
];
const away = [p({ player_id: 9, name: "Dominic Calvert-Lewin", position: "FWD" })];

const ui = (h = home, a = away) => render(<PlayerScorerList homeTeam="Arsenal" awayTeam="Leeds" homePlayers={h} awayPlayers={a} />);

describe("PlayerScorerList table", () => {
  it("shows the home team first, one team at a time", () => {
    ui();
    expect(screen.getAllByRole("table")).toHaveLength(1);
    expect(screen.getByText("Bukayo Saka")).toBeInTheDocument();
    expect(screen.queryByText("Dominic Calvert-Lewin")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Leeds" }));
    expect(screen.getByText("Dominic Calvert-Lewin")).toBeInTheDocument();
    expect(screen.queryByText("Bukayo Saka")).toBeNull();
  });

  it("keeps long names whole in the pinned first column, inside the scroller", () => {
    ui();
    const name = screen.getByText("Trent Alexander-Arnold-Longname");
    expect(name.textContent).toBe("Trent Alexander-Arnold-Longname");
    const th = name.closest("th")!;
    expect(th.className).toMatch(/sticky/);
    expect(screen.getByRole("region", { name: /scroll sideways/i }).contains(th)).toBe(true);
  });

  it("is sorted by G+A and keeps the G+A / Goal / Assist columns, as percentages", () => {
    ui();
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("Bukayo Saka");
    expect(within(rows[0]).getAllByRole("cell").map((c) => c.textContent)).toEqual(["50%", "28%", "32%", "—"]);
    expect(screen.getByRole("columnheader", { name: "Saves" })).toBeInTheDocument();
  });

  it("shows keeper saves in the Saves column and dashes for the probabilities", () => {
    ui();
    const row = screen.getByText("David Raya Martin").closest("tr")!;
    expect(within(row).getAllByRole("cell").map((c) => c.textContent)).toEqual(["—", "—", "—", "3.2"]);
  });

  it("adds no Saves column when the team has no goalkeeper listed", () => {
    ui();
    fireEvent.click(screen.getByRole("button", { name: "Leeds" }));
    expect(screen.queryByRole("columnheader", { name: "Saves" })).toBeNull();
  });

  it("still hides Shots / SoT, and says so plainly for an empty team", () => {
    ui(home, []);
    expect(screen.queryByText(/Shots|SoT/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Leeds" }));
    expect(screen.getByText("No box score for Leeds yet")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
  });
});
