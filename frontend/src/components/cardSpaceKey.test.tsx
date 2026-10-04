/**
 * Space activates these cards, the way Enter already does.
 *
 * Each of `CurrentGameweekCard`, `FinishedFixtureCard` and `NextFixtureHero` is a
 * `div` with `role="button"` and `tabIndex={0}`, so it is reachable by keyboard
 * and announced as a button — and then ignores the key a button is *most* used
 * with. The handler was `e.key === "Enter" && onClick()`: Enter worked, Space
 * did not, and Space then scrolled the page instead. A keyboard user tabbing to a
 * fixture card could open it with one key and not the other, with nothing on
 * screen to say so.
 *
 * `2026-09-25-predictor-frontend-action-plan.md` §15 asked for real controls
 * responding to Enter **and** Space and was never done; these are the four
 * places it meant.
 *
 * ## Why Space is prevented, not merely handled
 *
 * Space's default action is to scroll. Firing `onClick` and letting the scroll
 * happen would move the page under the reader on the very keypress that opened
 * the thing, so `preventDefault()` is part of the fix and not a detail.
 *
 * ## Why these stay `div`s
 *
 * A real `<button>` is the better end state and would delete the `role`,
 * the `tabIndex` and this handler outright. It is not done here because these
 * cards carry nested block layout and their own Tailwind surfaces, and button
 * default styling has to be reset deliberately — and this environment cannot
 * produce a screenshot of a rendered state, so a change to how these paint would
 * be going out unverified. Worth doing as its own piece, with eyes on it.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { FinishedFixtureCard } from "./FinishedFixtureCard";
import { CurrentGameweekCard } from "./CurrentGameweekCard";
import { NextFixtureHero } from "./NextFixtureHero";

// FinishedFixtureCard takes its fields flat as props; CurrentGameweekCard takes
// a `CurrentGameweekFixture`. Shapes read from each component's own signature
// rather than guessed — the first version of this file invented an `event_id`/
// `kickoff_time` fixture and every case died on `undefined.length`.
const finished = {
  commence_time: "2026-08-15T14:00:00Z",
  team_home: "Arsenal",
  team_away: "Chelsea",
  actual_goals_home: 2,
  actual_goals_away: 1,
  predicted_scoreline: "2-1",
  predicted_home_win: 0.5,
  predicted_draw: 0.25,
  predicted_away_win: 0.25,
  draw_signal: false,
  hit: true,
  backfilled: false,
  made_before_kickoff: true,
};

const upcoming = {
  event_id: "2",
  team_home: "Arsenal",
  team_away: "Chelsea",
  commence_time: "2099-08-15T14:00:00Z",
  finished: false,
  actual_goals_home: null,
  actual_goals_away: null,
  predicted_home_win: 0.5,
  predicted_draw: 0.25,
  predicted_away_win: 0.25,
  predicted_scoreline: null,
  draw_signal: false,
  hit: null,
  backfilled: false,
  made_before_kickoff: true,
  has_live_odds: false,
  value_bet_flags: [],
};

describe("a card that behaves like a button", () => {
  it("FinishedFixtureCard opens on Enter", () => {
    const onClick = vi.fn();
    render(<FinishedFixtureCard {...finished} onClick={onClick} />);
    fireEvent.keyDown(screen.getByRole("button"), { key: "Enter" });
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("FinishedFixtureCard opens on Space", () => {
    const onClick = vi.fn();
    render(<FinishedFixtureCard {...finished} onClick={onClick} />);
    fireEvent.keyDown(screen.getByRole("button"), { key: " " });
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("CurrentGameweekCard opens on Space", () => {
    const onClick = vi.fn();
    render(<CurrentGameweekCard fixture={upcoming} onClick={onClick} />);
    fireEvent.keyDown(screen.getByRole("button"), { key: " " });
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("Space does not also scroll the page", () => {
    // The default action for Space is scrolling. Without preventDefault the
    // reader's keypress opens the fixture AND jumps the page.
    const onClick = vi.fn();
    render(<FinishedFixtureCard {...finished} onClick={onClick} />);
    const event = new KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true });
    screen.getByRole("button").dispatchEvent(event);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(event.defaultPrevented).toBe(true);
  });

  it("still opens on a plain click", () => {
    // The fix is additive: a mouse user must be unaffected.
    const onClick = vi.fn();
    render(<FinishedFixtureCard {...finished} onClick={onClick} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("ignores the keys a button ignores", () => {
    const onClick = vi.fn();
    render(<FinishedFixtureCard {...finished} onClick={onClick} />);
    const el = screen.getByRole("button");
    fireEvent.keyDown(el, { key: "a" });
    fireEvent.keyDown(el, { key: "Tab" });
    fireEvent.keyDown(el, { key: "Escape" });
    expect(onClick).not.toHaveBeenCalled();
  });

  it("NextFixtureHero opens on Space as well as Enter", () => {
    // The third site with the same handler. It had no test file at all, so this
    // is the only thing standing between it and the same regression.
    const onClick = vi.fn();
    const { unmount } = render(<NextFixtureHero fixture={upcoming} onClick={onClick} />);
    fireEvent.keyDown(screen.getByRole("button"), { key: "Enter" });
    expect(onClick).toHaveBeenCalledTimes(1);
    unmount();

    const second = vi.fn();
    render(<NextFixtureHero fixture={upcoming} onClick={second} />);
    fireEvent.keyDown(screen.getByRole("button"), { key: " " });
    expect(second).toHaveBeenCalledTimes(1);
  });
});