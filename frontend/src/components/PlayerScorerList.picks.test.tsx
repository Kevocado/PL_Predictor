/**
 * Phase 2 task 3 — PL's "Model's top calls", with per-arm honesty.
 *
 * Seven corrections were measured on the hubs' `origin/main` and each one is
 * pinned here rather than left to the wording to be right about:
 *
 *  1. The goal and assist arms are bare `1 - exp(-lambda)` Poisson. Only
 *     `anytime_goal_contribution_prob` has a calibrator, and serving blends it
 *     50/50 with the uncalibrated union (`NEUTRAL_BLEND_WEIGHT = 0.5`), so no
 *     row may claim calibration. G+A is not shipped as a category at all.
 *  2. `get_scorer_accuracy()` returns `{snapshot, reconstructed}` — aggregate
 *     only, never per player. No per-player ledger exists, so a row says so.
 *  3. `expected_saves` is not a category (GK only, two rows per fixture, and the
 *     cap is a ceiling). The shot-on-target probability needs an Understat merge
 *     per player, so its row count is data-dependent and is measured, not
 *     assumed.
 *  4. `availability_multiplier` returns 0.0 for status in {i, s, u}, so such a
 *     player leaves the ranking and is shown once, attributed and dated.
 *  5. Three rows per category is a ceiling, never a quota.
 *  6. One category per list.
 *  7. No "lock" / "guaranteed" / "best bet" / "edge" / "value" — there is no
 *     odds feed in any repo. The title is "Model's top calls".
 */
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { ModelTopCalls } from "./PlayerScorerList";
import type { PlayerPrediction } from "../types";

function player(over: Partial<PlayerPrediction> & { player_id: number; name: string }): PlayerPrediction {
  return {
    position: "MID",
    anytime_goal_prob: 0.2,
    anytime_assist_prob: 0.1,
    anytime_goal_contribution_prob: 0.27,
    status: "a",
    news: "",
    confidence: "medium",
    predicted_starter: true,
    confirmed_starter: false,
    expected_minutes: 90,
    is_penalty_taker: false,
    is_set_piece_taker: false,
    expected_shots: 2.1,
    expected_shots_on_target: 0.8,
    anytime_shot_on_target_prob: 0.55,
    expected_saves: 0,
    ...over,
  };
}

/** Two keepers, because `expected_saves` exists only for a GK and there are
 *  exactly two per fixture. Neither may become a picks category. */
const HOME_GK = player({ player_id: 1, name: "Home Keeper", position: "GK", expected_saves: 3.4, anytime_goal_prob: 0, anytime_assist_prob: 0 });
const AWAY_GK = player({ player_id: 2, name: "Away Keeper", position: "GK", expected_saves: 2.1, anytime_goal_prob: 0, anytime_assist_prob: 0 });

/** Six outfielders, probabilities well separated so the duplicate audit has a
 *  clean slate to be non-vacuously clean against. */
const HOME = [
  HOME_GK,
  player({ player_id: 3, name: "Home Alpha", anytime_goal_prob: 0.44, anytime_assist_prob: 0.19 }),
  player({ player_id: 4, name: "Home Bravo", anytime_goal_prob: 0.38, anytime_assist_prob: 0.41 }),
  player({ player_id: 5, name: "Home Charlie", anytime_goal_prob: 0.31, anytime_assist_prob: 0.28 }),
  player({ player_id: 6, name: "Home Delta", anytime_goal_prob: 0.22, anytime_assist_prob: 0.14 }),
  player({ player_id: 7, name: "Home Echo", anytime_goal_prob: 0.17, anytime_assist_prob: 0.11 }),
];
const AWAY = [
  AWAY_GK,
  player({ player_id: 8, name: "Away Foxtrot", anytime_goal_prob: 0.29, anytime_assist_prob: 0.33 }),
  player({ player_id: 9, name: "Away Golf", anytime_goal_prob: 0.13, anytime_assist_prob: 0.09 }),
];

const KICKOFF = "2026-10-03T14:00:00Z";

/** `STATUS_LABEL` in PlayerScorerList.tsx, restated so the out line's wording is
 *  asserted against the words rather than the raw FPL status code. */
const STATUS_WORD: Record<string, string> = {
  i: "Injured",
  s: "Suspended",
  u: "Unavailable",
  d: "Doubtful",
};

function renderList(home = HOME, away = AWAY) {
  return render(
    <ModelTopCalls homeTeam="Arsenal" awayTeam="Chelsea" homePlayers={home} awayPlayers={away} kickoff={KICKOFF} />,
  );
}

const rows = () => screen.getAllByTestId("picks-row");
const queryRows = () => screen.queryAllByTestId("picks-row");
const rowNames = () => rows().map((r) => r.querySelector("p")!.textContent!.split("·")[0].trim());
const categories = () => screen.getAllByTestId("picks-category-heading").map((h) => h.textContent);
/** The bar's label for the model's own side of the row — i.e. the row's number.
 *  `PicksList` passes no `highlightKey`, so it is selected by its segment
 *  label, which is the category the figure belongs to. */
const barLabel = (row: HTMLElement, category: string) =>
  row.querySelector<HTMLElement>(`[data-testid="pbar-label"][data-seg="${category}"]`)!;
const outLines = () => screen.getByTestId("picks-out").querySelectorAll("p");

describe("PL Model's top calls — categories", () => {
  it("is titled \"Model's top calls\" and ships exactly goals and assists", () => {
    renderList();
    expect(screen.getByTestId("picks-title")).toHaveTextContent("Model's top calls");
    expect(categories()).toEqual(["Anytime goal", "Anytime assist"]);
  });

  it("correction 3: expected_saves is not a category, even with two keepers present", () => {
    renderList();
    // The plan's categories are goals and assists. A keeper's expected_saves
    // (GK-only, two rows per fixture, a ceiling not a quota) must not become a
    // third list, and neither keeper's saves figure may appear in the list.
    expect(categories()).toHaveLength(2);
    const text = screen.getByTestId("picks-list").textContent!;
    expect(text).not.toMatch(/saves/i);
    // 3.4 and 2.1 are the two keepers' expected_saves in this fixture.
    expect(text).not.toContain("3.4");
    expect(text).not.toContain("2.1");
  });

  it("correction 3: the data-dependent shot probability does not add a category", () => {
    // Every player here carries `anytime_shot_on_target_prob`, so a shots list
    // would fill three. It is measured and reported, not shipped, because the
    // row count depends on an Understat merge being present per player.
    renderList();
    expect(categories()).not.toContain("Shot on target");
    expect(categories()).toHaveLength(2);
  });

  it("correction 5: three rows per category is a ceiling, not a quota", () => {
    renderList();
    const goalRows = rows().filter((r) => r.getAttribute("data-category") === "Anytime goal");
    const assistRows = rows().filter((r) => r.getAttribute("data-category") === "Anytime assist");
    expect(goalRows).toHaveLength(3);
    expect(assistRows).toHaveLength(3);
    // Ranked by the category's own number, descending.
    expect(rowNames().slice(0, 3)).toEqual(["Home Alpha", "Home Bravo", "Home Charlie"]);
  });

  it("correction 5: one candidate renders one row and is never padded", () => {
    renderList(
      [player({ player_id: 3, name: "Only One", anytime_goal_prob: 0.4, anytime_assist_prob: 0.3 })],
      [],
    );
    expect(rows().filter((r) => r.getAttribute("data-category") === "Anytime goal")).toHaveLength(1);
    expect(rows().filter((r) => r.getAttribute("data-category") === "Anytime assist")).toHaveLength(1);
  });

  it("shows the honest empty state, not a fabricated zero, when a category has no candidates", () => {
    renderList([], []);
    expect(queryRows()).toHaveLength(0);
    expect(screen.getAllByText("No pick yet")).toHaveLength(2);
  });

  it("never renders a 0% probability as if it were a real zero", () => {
    // A player who cannot score is still a ranked row; the figure reads "<1%",
    // which is a floor, and not the "0%" that would claim a measured zero.
    renderList([player({ player_id: 3, name: "Goalless", anytime_goal_prob: 0, anytime_assist_prob: 0 })], []);
    expect(barLabel(rows()[0], "Anytime goal").textContent).toBe("Anytime goal <1%");
    expect(screen.getByTestId("picks-list").textContent).not.toContain("0%");
  });

  it("correction 6: one category per list — a goal row never sits under an assists heading", () => {
    renderList();
    for (const row of rows()) {
      const category = row.getAttribute("data-category")!;
      // The bar's own segment label is the category the number belongs to.
      expect(barLabel(row, category)).toBeTruthy();
    }
    // And the goal rows carry the goal number, not the assist number.
    const goalRows = rows().filter((r) => r.getAttribute("data-category") === "Anytime goal");
    expect(goalRows.map((r) => barLabel(r, "Anytime goal").textContent)).toEqual([
      "Anytime goal 44%",
      "Anytime goal 38%",
      "Anytime goal 31%",
    ]);
  });
});

describe("PL Model's top calls — per-arm honesty", () => {
  it("correction 1: every row names its arm and claims no calibration", () => {
    renderList();
    for (const row of rows()) {
      const provenance = row.querySelectorAll("p")[1].textContent!;
      expect(provenance).toMatch(/Poisson/);
      expect(provenance).toMatch(/uncalibrated/);
      // "calibrated" on its own would be a claim. Only "uncalibrated" may match.
      expect(provenance).not.toMatch(/(?<!un)calibrated/);
    }
  });

  it("correction 1: the 50/50-blended G+A arm is not shipped as a category", () => {
    // `anytime_goal_contribution_prob` is the only arm with a calibrator, and
    // serving blends it 50/50 with the uncalibrated union, so it cannot be
    // labelled either way. It is left out rather than mislabelled.
    renderList();
    expect(categories()).toHaveLength(2);
    const barLabels = screen.getAllByTestId("pbar-label").map((l) => l.getAttribute("data-seg"));
    expect(barLabels).not.toContain("Goal or assist");
  });

  it("correction 2: no row presents a per-player graded record", () => {
    renderList();
    for (const row of rows()) {
      const provenance = row.querySelectorAll("p")[1].textContent!;
      expect(provenance).toMatch(/no graded record per player/i);
      // The ledger that does exist is aggregate, and must be labelled so.
      expect(provenance).toMatch(/aggregate/i);
    }
  });

  it("correction 2: no row shows a hit rate that is not this player's own", () => {
    renderList();
    const list = within(screen.getByTestId("picks-list"));
    expect(list.queryByText(/hit rate/i)).not.toBeInTheDocument();
    expect(list.queryByText(/\d+\s*\/\s*\d+\s*hit/i)).not.toBeInTheDocument();
  });

  it("says what the form behind each number is, and never silently drops a cold-start row", () => {
    // `confidence` is what `player_form.blended_current_form` named its blend:
    // "current" (a full window of this season), "prior_season" (blended back
    // toward last season), "position_avg" (no rate of this player's own) and
    // "none". The list this replaces hid the cold-start players; a ranked row
    // whose number rests on a prior is fine as long as the row says so.
    renderList(
      [
        player({ player_id: 3, name: "New Signing", anytime_goal_prob: 0.4, anytime_assist_prob: 0.2, confidence: "position_avg" }),
        player({ player_id: 4, name: "Steady Hand", anytime_goal_prob: 0.3, anytime_assist_prob: 0.1, confidence: "current" }),
        player({ player_id: 5, name: "Still Settling", anytime_goal_prob: 0.2, anytime_assist_prob: 0.05, confidence: "prior_season" }),
      ],
      [],
    );
    expect(rowNames().slice(0, 3)).toEqual(["New Signing", "Steady Hand", "Still Settling"]);
    const provenanceFor = (name: string) =>
      rows()
        .find((r) => r.textContent!.includes(name))!
        .querySelectorAll("p")[1].textContent!;
    expect(provenanceFor("New Signing")).toMatch(/position average/i);
    expect(provenanceFor("Steady Hand")).toMatch(/this season's games/i);
    expect(provenanceFor("Still Settling")).toMatch(/last season/i);
    // No row falls back to an unstated basis.
    for (const row of rows()) {
      expect(row.querySelectorAll("p")[1].textContent).not.toMatch(/unstated/i);
    }
  });

  it("names every confidence value the backend can serve, and says so for one it does not know", () => {
    // All four values in `blended_current_form`'s documented vocabulary, plus
    // an unknown one, which must be named as unknown rather than guessed at.
    for (const [confidence, expected] of [
      ["current", /this season's games/i],
      ["prior_season", /last season/i],
      ["position_avg", /position average/i],
      ["none", /no prior rate on file/i],
      ["something_new", /unstated basis/i],
    ] as const) {
      const view = renderList([player({ player_id: 3, name: "Solo", confidence })], []);
      const provenance = rows()[0].querySelectorAll("p")[1].textContent!;
      expect(provenance, `confidence=${confidence}`).toMatch(expected);
      view.unmount();
    }
  });

  it("says what the lineup basis is on every row", () => {
    renderList(
      [
        player({ player_id: 3, name: "In The Team", confirmed_starter: true, anytime_goal_prob: 0.4, anytime_assist_prob: 0.2 }),
        player({ player_id: 4, name: "Not In It", predicted_starter: false, anytime_goal_prob: 0.3, anytime_assist_prob: 0.1 }),
      ],
      [],
    );
    const provenanceFor = (name: string) =>
      rows()
        .find((r) => r.textContent!.includes(name))!
        .querySelectorAll("p")[1].textContent!;
    expect(provenanceFor("In The Team")).toMatch(/Confirmed XI/);
    expect(provenanceFor("Not In It")).toMatch(/not in the predicted XI/i);
  });

  it("correction 7: no row claims a lock, a guarantee, a bet, an edge or a value", () => {
    renderList();
    const text = screen.getByTestId("picks-list").textContent!.toLowerCase();
    for (const banned of ["lock", "guaranteed", "guarantee", "best bet", "edge", "value", "price", "odds"]) {
      expect(text, `banned word: ${banned}`).not.toMatch(new RegExp(`\\b${banned}\\b`));
    }
  });
});

describe("PL Model's top calls — an out player leaves the ranking", () => {
  /**
   * The ruled-out player is given the HIGHEST goal and assist probability in the
   * slate on purpose. With today's backend his own Poisson arms are 0, so a
   * "filter where the number is above zero" implementation would pass the test
   * below by coincidence. Giving him the top number makes the coincidence
   * impossible: if he is still absent, the removal is by name and not by value.
   *
   * It is not hypothetical that a ruled-out player carries a non-zero number —
   * `predict_goal_contribution` takes no availability argument, so the served
   * G+A is `blend_contribution(direct, 0.0, 0.5)`, which is 0.31 for a direct
   * arm of 0.62 rather than 0.
   */
  const OUT = player({
    player_id: 99,
    name: "Ruled Out Star",
    status: "i",
    news: "Knock",
    anytime_goal_prob: 0.91,
    anytime_assist_prob: 0.88,
    anytime_goal_contribution_prob: 0.31,
  });

  it("correction 4: is absent from every list and appears exactly once below them", () => {
    renderList([OUT, ...HOME], AWAY);
    for (const row of rows()) {
      expect(row.textContent).not.toContain("Ruled Out Star");
    }
    const out = screen.getByTestId("picks-out");
    // One line, not a row and not a bar: exactly one <p> mentions him.
    const mentioning = [...out.querySelectorAll("p")].filter((p) => p.textContent!.includes("Ruled Out Star"));
    expect(mentioning).toHaveLength(1);
    expect(within(out).queryAllByTestId("picks-row")).toHaveLength(0);
    expect(out.textContent!.match(/Ruled Out Star/g)).toHaveLength(1);
  });

  it("correction 4: leaves the list attributed and dated, and says it is not ranked", () => {
    renderList([OUT, ...HOME], AWAY);
    const line = outLines()[0].textContent!;
    expect(line).toMatch(/not ranked/i);
    expect(line).toMatch(/FPL squad status/);
    expect(line).toMatch(/Injured/);
    expect(line).toMatch(/gameweek/i);
    expect(line).not.toMatch(/\b\d{10,}\b/); // no epoch
  });

  it("correction 4: backfills the freed slot from the available pool, never with an out player", () => {
    renderList([OUT, ...HOME], AWAY);
    // Five available outfield goals candidates plus two keepers; the top three
    // available ones fill both lists, and the out player occupies no position.
    expect(rowNames().slice(0, 3)).toEqual(["Home Alpha", "Home Bravo", "Home Charlie"]);
    expect(rows()).toHaveLength(6);
  });

  it("correction 4: every unavailable status is removed, not just injured", () => {
    for (const status of ["i", "s", "u"]) {
      const out = player({ player_id: 98, name: `Unavailable ${status}`, status, anytime_goal_prob: 0.9, anytime_assist_prob: 0.9 });
      const view = renderList([out, ...HOME], AWAY);
      // Not ranked anywhere...
      for (const row of rows()) expect(row.textContent).not.toContain(`Unavailable ${status}`);
      // ...and named exactly once below the lists.
      const mentioning = [...screen.getByTestId("picks-out").querySelectorAll("p")].filter((p) =>
        p.textContent!.includes(`Unavailable ${status}`),
      );
      expect(mentioning).toHaveLength(1);
      expect(mentioning[0].textContent).toMatch(new RegExp(`\\(${STATUS_WORD[status]}\\)`));
      view.unmount();
    }
  });

  it("correction 4: a doubtful player is NOT removed — the backend scales him, it does not zero him", () => {
    // availability_multiplier("d", ...) returns chance/100 (0.5 when unknown),
    // never 0.0, so a doubtful player keeps a real probability and a real rank.
    const doubtful = player({ player_id: 97, name: "Doubtful Fella", status: "d", anytime_goal_prob: 0.47, anytime_assist_prob: 0.2 });
    renderList([doubtful, ...HOME], AWAY);
    expect(rowNames().slice(0, 3)).toEqual(["Doubtful Fella", "Home Alpha", "Home Bravo"]);
    expect(screen.queryByTestId("picks-out")).not.toBeInTheDocument();
  });

  it("shows no out line at all when nobody is out", () => {
    renderList();
    expect(screen.queryByTestId("picks-out")).not.toBeInTheDocument();
  });
});

/**
 * Duplicate-figure audit.
 *
 * A previous audit in this repo reported three phantom duplicates because it
 * searched the rendered text for the SUBSTRING "7%", which is contained in
 * "17%". So this one parses the whole number out of each row's own bar label,
 * anchored at the end of the string, and groups by category. The control test
 * below proves the audit can actually fail, so "no duplicates" is a result and
 * not an accident of a matcher that never matches.
 */
function duplicateRenderedFigures(): Array<{ category: string; percent: string; rows: string[] }> {
  const byCategory = new Map<string, Map<string, string[]>>();
  for (const row of rows()) {
    const category = row.getAttribute("data-category")!;
    const name = row.querySelector("p")!.textContent!.split("·")[0].trim();
    const shown = barLabel(row, category).textContent ?? "";
    // Anchored at the end, so "17%" can only ever parse as 17 and never as 7.
    const parsed = /(\d+(?:\.\d+)?)%\s*$/.exec(shown);
    if (!parsed) continue;
    const percent = parsed[1];
    if (!byCategory.has(category)) byCategory.set(category, new Map());
    const group = byCategory.get(category)!;
    if (!group.has(percent)) group.set(percent, []);
    group.get(percent)!.push(name);
  }
  return [...byCategory].flatMap(([category, group]) =>
    [...group]
      .filter(([, names]) => names.length > 1)
      .map(([percent, names]) => ({ category, percent, rows: names })),
  );
}

describe("duplicate-figure audit", () => {
  it("is non-vacuous: it detects a genuine duplicate, so a clean result means something", () => {
    // Two players with the SAME goal probability. If the audit could not find
    // this, it would also "pass" a list that really does repeat a figure.
    // Two players with the SAME goal probability, and distinct assist figures
    // so the finding below is attributable to the goal list and nothing else.
    const twins = [
      player({ player_id: 3, name: "Twin One", anytime_goal_prob: 0.31, anytime_assist_prob: 0.11 }),
      player({ player_id: 4, name: "Twin Two", anytime_goal_prob: 0.31, anytime_assist_prob: 0.29 }),
    ];
    renderList(twins, []);
    const found = duplicateRenderedFigures();
    expect(found).toHaveLength(1);
    expect(found[0].category).toBe("Anytime goal");
    expect(found[0].percent).toBe("31");
    expect(found[0].rows.sort()).toEqual(["Twin One", "Twin Two"]);
  });

  it("does not invent a duplicate out of a digit that is a substring of another", () => {
    // 17% and 7% are different figures. A substring search for "7%" would call
    // these duplicates; a parsed comparison must not.
    const sevenVsSeventeen = [
      player({ player_id: 3, name: "Seven", anytime_goal_prob: 0.07, anytime_assist_prob: 0.09 }),
      player({ player_id: 4, name: "Seventeen", anytime_goal_prob: 0.17, anytime_assist_prob: 0.11 }),
    ];
    renderList(sevenVsSeventeen, []);
    const goalRows = rows().filter((r) => r.getAttribute("data-category") === "Anytime goal");
    expect(goalRows.map((r) => barLabel(r, "Anytime goal").textContent)).toEqual([
      "Anytime goal 17%",
      "Anytime goal 7%",
    ]);
    // The substring trap this repo fell into before: a search for "7%" here
    // would report a duplicate. Prove the two figures are distinct as numbers.
    const parsed = goalRows.map((r) => parseFloat(/(\d+(?:\.\d+)?)%\s*$/.exec(barLabel(r, "Anytime goal").textContent!)![1]));
    expect(parsed).toEqual([17, 7]);
    expect(new Set(parsed).size).toBe(2);
    expect(duplicateRenderedFigures()).toEqual([]);
  });

  it("finds no duplicated figure in the standard slate", () => {
    renderList();
    expect(duplicateRenderedFigures()).toEqual([]);
  });
});
