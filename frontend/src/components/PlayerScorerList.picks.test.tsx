/**
 * PL's "Model's top calls".
 *
 * The rules below are pinned here rather than left to the wording to be right
 * about. Seven corrections were measured on the hubs' `origin/main`:
 *
 *  1. The goal and assist arms are bare `1 - exp(-lambda)` Poisson. Only
 *     `anytime_goal_contribution_prob` has a calibrator, and serving blends it
 *     50/50 with the uncalibrated union (`NEUTRAL_BLEND_WEIGHT = 0.5`), so G+A
 *     is not shipped as a category at all.
 *  2. `get_scorer_accuracy()` returns `{snapshot, reconstructed}` — aggregate
 *     only, never per player, so no row may carry a hit rate.
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
 *
 * Kevin, 2026-10-01: a row is the player, the team and the prediction. The
 * per-row sentences for the arm, the form basis, the lineup basis and the absent
 * per-player ledger are gone, and the tests that asserted them are gone with
 * them; the decisions those sentences described are still asserted above, on the
 * ranking rather than on the prose.
 */
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { ModelTopCalls, rankedRows } from "./PlayerScorerList";
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
/** The figure a row shows. Since 2026-10-01 that is the row's whole number: the
 *  shared component draws a probability as a bare percentage beside a bar, and
 *  the category it belongs to is the list heading rather than text on the row. */
const figure = (row: HTMLElement) => row.querySelector('[data-testid="picks-value"]')!.textContent!;
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
    expect(figure(rows()[0])).toBe("<1%");
    expect(screen.getByTestId("picks-list").textContent).not.toContain("0%");
  });

  it("correction 6: one category per list — a goal row never sits under an assists heading", () => {
    renderList();
    for (const row of rows()) {
      // The row names the category it belongs to, and it is the list's heading.
      expect(row.getAttribute("data-category")).toBeTruthy();
      expect(categories()).toContain(row.getAttribute("data-category"));
    }
    // And the goal rows carry the goal number, not the assist number.
    const goalRows = rows().filter((r) => r.getAttribute("data-category") === "Anytime goal");
    expect(goalRows.map(figure)).toEqual(["44%", "38%", "31%"]);
    const assistRows = rows().filter((r) => r.getAttribute("data-category") === "Anytime assist");
    expect(assistRows.map(figure)).toEqual(["41%", "33%", "28%"]);
  });
});

/**
 * Kevin, 2026-10-01: a top call is the player, the team and the prediction.
 * Nothing else. So no row carries a provenance sentence, a ± margin, a "no
 * graded record" line, a calibration note, a form-basis note or a lineup-basis
 * note -- and the row shows its figure rather than disclaiming what backs it.
 */
describe("PL Model's top calls — a row is the player, the team and the prediction", () => {

  it("carries no provenance, no ± and no availability text on any row", () => {
    // On the BUILT row, not only on the rendered page: the shared component
    // stopped drawing `provenance`, but the site must also stop building it, so
    // the text cannot come back the day a caller starts printing it again.
    const built = rankedRows(
      [
        { ...player({ player_id: 3, name: "New Signing", anytime_goal_prob: 0.4 }), team: "Arsenal", confidence: "position_avg", confirmed_starter: true } as never,
      ],
      "Anytime goal",
      (p) => p.anytime_goal_prob,
    );
    for (const row of built) {
      const text = Object.entries(row)
        .filter(([k]) => !["key", "name", "team", "value", "kind"].includes(k))
        .map(([, v]) => String(v))
        .join(" ")
        .toLowerCase();
      for (const banned of [
        "poisson",
        "uncalibrated",
        "graded record",
        "aggregate",
        "ledger",
        "hit rate",
        "±",
        "position average",
        "confirmed xi",
      ]) {
        expect(text, `stripped text on a built row: ${banned}`).not.toContain(banned);
      }
      // No provenance or margin field at all, rather than an empty one.
      expect(row).not.toHaveProperty("provenance");
      expect(row).not.toHaveProperty("margin");
    }
  });

  it("says no form basis, no lineup basis and no record on the page at all", () => {
    renderList(
      [
        player({ player_id: 3, name: "New Signing", anytime_goal_prob: 0.4, anytime_assist_prob: 0.2, confidence: "position_avg" }),
        player({ player_id: 4, name: "In The Team", anytime_goal_prob: 0.3, anytime_assist_prob: 0.1, confirmed_starter: true }),
        player({ player_id: 5, name: "Still Settling", anytime_goal_prob: 0.2, anytime_assist_prob: 0.05, confidence: "prior_season" }),
      ],
      [],
    );
    const text = screen.getByTestId("picks-list").textContent!.toLowerCase();
    for (const banned of [
      "poisson",
      "uncalibrated",
      "graded record",
      "position average",
      "last season",
      "this season",
      "confirmed xi",
      "not in the predicted xi",
      "prior rate",
      "±",
    ]) {
      expect(text, `stripped text on the page: ${banned}`).not.toContain(banned);
    }
  });

  it("still shows the player, the team and the figure", () => {
    renderList();
    const goalRow = rows().find((r) => r.getAttribute("data-category") === "Anytime goal")!;
    // Name and team, side by side...
    expect(goalRow.querySelector("p")!.textContent).toContain("Home Alpha");
    expect(goalRow.querySelector("p")!.textContent).toContain("Arsenal");
    // ...and the model's own number for that category, as a share.
    expect(goalRow.querySelector('[data-testid="picks-value"]')!.textContent).toBe("44%");
    expect(goalRow).toHaveAttribute("data-kind", "probability");
  });
});

/**
 * The rules here that are NOT text. What changed on 2026-10-01 is only the
 * per-row prose; the decisions it used to describe are still the decisions.
 */
describe("PL Model's top calls — the rules that are not text", () => {
  it("correction 1: the 50/50-blended G+A arm is not shipped as a category", () => {
    // `anytime_goal_contribution_prob` is the only arm with a calibrator, and
    // serving blends it 50/50 with the uncalibrated union, so it cannot be
    // labelled either way. It is left out rather than mislabelled.
    renderList();
    expect(categories()).toHaveLength(2);
    expect(categories()).not.toContain("Goal or assist");
  });

  it("correction 2: no row shows a hit rate that is not this player's own", () => {
    // There is no per-player ledger to draw a hit rate from in the first place
    // (`get_scorer_accuracy()` is aggregate), so no row may carry one.
    renderList();
    const list = within(screen.getByTestId("picks-list"));
    expect(list.queryByText(/hit rate/i)).not.toBeInTheDocument();
    expect(list.queryByText(/\d+\s*\/\s*\d+\s*hit/i)).not.toBeInTheDocument();
  });

  it("never silently drops a cold-start row", () => {
    // `confidence` used to be narrated per row ("position average", "last
    // season"). The narration is gone; the RANKING of a player whose rate rests
    // on a prior is not, and this is the test that holds that.
    renderList(
      [
        player({ player_id: 3, name: "New Signing", anytime_goal_prob: 0.4, anytime_assist_prob: 0.2, confidence: "position_avg" }),
        player({ player_id: 4, name: "Steady Hand", anytime_goal_prob: 0.3, anytime_assist_prob: 0.1, confidence: "current" }),
        player({ player_id: 5, name: "Still Settling", anytime_goal_prob: 0.2, anytime_assist_prob: 0.05, confidence: "prior_season" }),
      ],
      [],
    );
    expect(rowNames().slice(0, 3)).toEqual(["New Signing", "Steady Hand", "Still Settling"]);
  });

  it("ranks a player the backend serves a `none` confidence for", () => {
    // Every value in `blended_current_form`'s vocabulary, plus an unknown one,
    // ranks normally. Nothing about the basis is guessed at, and nothing about
    // the basis is printed.
    for (const confidence of ["current", "prior_season", "position_avg", "none", "something_new"]) {
      const view = renderList([player({ player_id: 3, name: "Solo", confidence })], []);
      expect(rowNames(), `confidence=${confidence}`).toContain("Solo");
      view.unmount();
    }
  });

  it("a confirmed and an unconfirmed starter are both ranked, on their own number", () => {
    renderList(
      [
        player({ player_id: 3, name: "In The Team", confirmed_starter: true, anytime_goal_prob: 0.4, anytime_assist_prob: 0.2 }),
        player({ player_id: 4, name: "Not In It", predicted_starter: false, anytime_goal_prob: 0.3, anytime_assist_prob: 0.1 }),
      ],
      [],
    );
    expect(rowNames().slice(0, 2)).toEqual(["In The Team", "Not In It"]);
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
    // The row's whole figure is now the percentage ("<1%", "44%"), so the
    // number is parsed out of it directly rather than off a bar segment label.
    const parsed = /(\d+(?:\.\d+)?)%\s*$/.exec(figure(row));
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
    expect(goalRows.map(figure)).toEqual(["17%", "7%"]);
    // The substring trap this repo fell into before: a search for "7%" here
    // would report a duplicate. Prove the two figures are distinct as numbers.
    const parsed = goalRows.map((r) => parseFloat(/(\d+(?:\.\d+)?)%\s*$/.exec(figure(r))![1]));
    expect(parsed).toEqual([17, 7]);
    expect(new Set(parsed).size).toBe(2);
    expect(duplicateRenderedFigures()).toEqual([]);
  });

  it("finds no duplicated figure in the standard slate", () => {
    renderList();
    expect(duplicateRenderedFigures()).toEqual([]);
  });
});
