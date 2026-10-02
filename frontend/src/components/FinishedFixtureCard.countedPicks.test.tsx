// Site-local copy on the finished-fixture card, after the 2026-10-01 reversal.
//
// The badge on this card is SITE-LOCAL: it is a hand-written `<span>`, not the
// shared `StatusBadge`, so predictor-hub#67 (open, unmerged) does not reach it.
// Its tooltip already said the pick counts; its visible words still described the
// pick's standing rather than the moment it was made, which after the reversal is
// the wrong thing to say — the same wording change the shared badge makes.
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FinishedFixtureCard } from "./FinishedFixtureCard";

// `ComponentProps<typeof FinishedFixtureCard>` rather than a hand-rolled object,
// so `tsc -b` checks these fixtures against the component's real prop types. A
// `Record<string, unknown>` cast would compile no matter what the component
// asks for, which is how a test ends up rendering with props the component no
// longer reads.
type CardProps = Parameters<typeof FinishedFixtureCard>[0];

const props = (over: Partial<CardProps> = {}): CardProps => ({
  team_home: "Tottenham", team_away: "Aston Villa",
  commence_time: "2026-09-19T10:30:00Z",
  actual_goals_home: 2, actual_goals_away: 3,
  predicted_scoreline: "1-1",
  predicted_home_win: 0.36, predicted_draw: 0.26, predicted_away_win: 0.38,
  // `backfilled` is deliberately NOT what this card reads. It is set to the
  // value that would be wrong, so a regression back to the flag shows up here.
  hit: true, backfilled: false, made_before_kickoff: false, onClick: () => {}, ...over,
});

describe("FinishedFixtureCard timing copy", () => {
  it("names the moment, and says the pick is counted", () => {
    render(<FinishedFixtureCard {...props()} />);
    expect(screen.getByText("Made after kickoff")).toBeInTheDocument();
    // "Counted in the track record like any other pick" is the badge's TOOLTIP,
    // not visible text, so it is read off the attribute. That is where the claim
    // lives, and under the 2026-10-01 reversal it is the true one.
    const badge = screen.getByText("Made after kickoff");
    expect(badge.getAttribute("title")).toMatch(/counted in the track record/i);
    expect(document.body.textContent).not.toMatch(/not\s+counted/i);
    expect(document.body.textContent).not.toMatch(/Rebuilt after/i);
  });

  it("still judges a pick made before kickoff, even when the flag says rebuilt", () => {
    // `backfilled: true` is the backfill job's mark. This pick's own timestamps
    // put it before kickoff, which is the only thing that decides the badge.
    render(<FinishedFixtureCard {...props({ backfilled: true, made_before_kickoff: true })} />);
    expect(screen.getByText(/Called it|Missed/)).toBeInTheDocument();
    expect(screen.queryByText("Made after kickoff")).toBeNull();
  });

  it("fails closed on a payload with no derived flag", () => {
    // A snapshot baked before `made_before_kickoff` existed. Showing the badge
    // is right: the alternative is claiming a pick was in time on no evidence.
    render(<FinishedFixtureCard {...props({ made_before_kickoff: undefined })} />);
    expect(screen.getByText("Made after kickoff")).toBeInTheDocument();
    expect(screen.queryByText(/Called it|Missed/)).toBeNull();
  });
});
