/**
 * PL's fixture modal fetches and renders its `absence` row — the last hop.
 *
 * The backend shipped in cbb289c2, the component can draw the visual (hub #83),
 * and nothing between them existed: no client method, no fetch, no mount. This is
 * that hop, and deliberately the same shape as Sports' `GameDetailModal`, so
 * "a fixture page shows its signals" is one implementation and not one per sport.
 *
 * ## EVERY failure is silence, and the failures are indistinguishable
 *
 * A missing route, a 404, a network error and an honest empty list all leave `[]`.
 * Not laziness — spec §2's "no data, no row" forbids a placeholder, and a signal is
 * an *enhancement* on this page: it must never become the page's error state.
 *
 * ## Why a finished gameweek is not asked about
 *
 * `/api/signals/{event_id}` answers `[]` for one, because `facts.game_context`
 * empties the player pool once a fixture is live or final — the squad list it holds
 * is the one known BEFORE kick-off. `post_match` is `null` on an upcoming fixture
 * and an object on a finished one, so `post_match` is the flag, not a truthiness
 * test on some other field.
 *
 * The `FixtureDetail` fixture is the real shape from `FixtureModal.instant.test.tsx`
 * rather than a minimal object, because `panelFacts` indexes `home_context` and
 * `ScorelineHeatmap` indexes `grid[0]` unconditionally: a thin fixture throws
 * before the row is ever reached, and a test that dies in setup tests nothing.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";

import { FixtureModal } from "./FixtureModal";
import { api, fetchSignals } from "../api/client";
import type { FixtureDetail, SignalsResponse, TrackRecordResponse } from "../types";

vi.mock("../api/client", () => ({
  api: {
    fixtureDetail: vi.fn(),
    fixturePlayers: vi.fn(),
    fixturePlayerReview: vi.fn(),
    trackRecord: vi.fn(),
  },
  fetchSignals: vi.fn(),
}));

const edge = (p: number, implied: number | null = null) => ({ prob: p, implied, edge: null });
const ou = (over: number) => ({ lambda_: 2.7, line: 2.5, over, under: 1 - over });

/** The site's own three-way figures, in the real shape. */
const detail = {
  event_id: "e1", commence_time: "2026-10-10T14:00:00Z",
  team_home: "Tottenham", team_away: "Aston Villa",
  home_win: edge(0.44, 0.47), draw: edge(0.28, 0.27), away_win: edge(0.28, 0.26),
  over_2_5: ou(0.52), under_2_5: ou(0.48), btts_yes_prob: 0.48,
  value_bet_flags: [], value_bet: null, has_live_odds: false,
  corners: ou(0.5), cards: ou(0.5),
  top_scoreline: "2-1", predicted_result: "home_win", draw_signal: false,
  is_fallback_prediction: false, data_confidence: "established",
  home_context: { rest_days: 3, xg_for_last_5: 1.6, xg_against_last_5: 1.1, corners_last_5: 5, cards_last_5: 2, set_piece_xg_share_last_5: 0.2 },
  away_context: { rest_days: 3, xg_for_last_5: 1.4, xg_against_last_5: 1.2, corners_last_5: 4, cards_last_5: 2, set_piece_xg_share_last_5: 0.18 },
  score_grid: [[0.1, 0.16, 0.12], [0.12, 0.14, 0.08], [0.09, 0.11, 0.08]],
  top_scorelines: [{ home: 2, away: 1, prob: 0.14 }],
  home_shots: 13.1, away_shots: 9.4, home_shots_on_target: 4.8, away_shots_on_target: 3.2,
  head_to_head: [], home_recent_form: [], away_recent_form: [],
  predicted_total_goals: 2.7, predicted_margin: 0.2,
  home_2plus_prob: 0.55, away_2plus_prob: 0.5,
  odds_fetched_at: null, odds_is_stale: false, recommended_bet: null,
  post_match: null, actual_stats: null, pre_match_value_bets: [],
} as unknown as FixtureDetail;

const finished = {
  ...detail,
  post_match: { final_score: "1-0", provenance: "snapshot", verdicts: [], player_calls: [] },
} as unknown as FixtureDetail;

const trackRecord = {
  summary: {
    n_resolved_fixtures: 71, pct_correct_overall: 0.535, n_rebuilt_fixtures: 0,
    pre_kickoff: { n_resolved_fixtures: 71, pct_correct_overall: 0.535 },
  },
} as unknown as TrackRecordResponse;

/** The one absence row PL's adapter can produce, in the shape it sends. */
const ABSENCE = {
  kind: "absence",
  sport: "pl",
  game_id: "e1",
  headline: { text: "Out: Haaland, our #1 scorer, 71% to score", figures: { projection: 71 } },
  n: 2,
  source: "FPL status · 2 players out",
  as_of: "2026-10-04T16:00:00Z",
  strength: 0.71,
  pre_kickoff_only: true,
  visual: "absence_strip",
};

const envelope = (signals: unknown[]) =>
  ({ sport: "pl", id: "e1", signals }) as unknown as SignalsResponse;

function mockApi(fixture: FixtureDetail = detail) {
  vi.mocked(api.fixtureDetail).mockResolvedValue(fixture);
  vi.mocked(api.fixturePlayers).mockResolvedValue({ home_players: [], away_players: [] });
  vi.mocked(api.fixturePlayerReview).mockResolvedValue(null);
  vi.mocked(api.trackRecord).mockResolvedValue(trackRecord);
}

async function openModal(fixture: FixtureDetail = detail) {
  mockApi(fixture);
  render(
    <FixtureModal
      eventId="e1"
      onClose={() => {}}
      explain={vi.fn().mockReturnValue(new Promise(() => {}))}
    />,
  );
  await screen.findByTestId("fixture-flow");
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("PL's absence row on the fixture modal", () => {
  it("asks for the signals when an upcoming fixture opens", async () => {
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    await openModal();
    expect(fetchSignals).toHaveBeenCalledWith("e1");
  });

  it("renders the row the endpoint sent", async () => {
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    await openModal();
    // The words, read back out of the DOM: a screenshot proves a thing was drawn,
    // only the text says whether it was the RIGHT thing.
    //
    // The `source` ("2 players out") is deliberately NOT asserted here. It lives
    // inside `SignalRows`' COLLAPSED evidence line, so it is not in the DOM until
    // a reader expands it — and whether it is there, and what it says, is the
    // shared component's contract, already asserted in its own tests. This file is
    // about the last hop: that the row is fetched and mounted at all.
    // `findBy`, not `getBy`: the signals fetch resolves after the fixture detail
    // does, so `getBy` here would be racing it and the test would pass or fail on
    // timing rather than on the mount.
    expect(await screen.findByText(/Out: Haaland/)).toBeInTheDocument();
    expect(screen.getByTestId("signal-row")).toBeInTheDocument();
  });

  it("draws the figure beside the headline", async () => {
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    await openModal();
    // The whole point of `absence_strip`: the player's own projection, as a
    // plain unsigned number in the sport's units. `SignalRows` REFUSES a row
    // whose words do not state this, so its presence proves the two agree.
    const marker = await screen.findByTestId("signal-absence");
    expect(marker).toHaveTextContent("71");
  });

  it("renders NO row and no empty state when the endpoint has no signal", async () => {
    vi.mocked(fetchSignals).mockResolvedValue(envelope([]));
    await openModal();
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    expect(screen.queryByText(/Out:/)).not.toBeInTheDocument();
  });

  it("renders nothing at all when the fetch rejects", async () => {
    // A signal is an enhancement. A reader must never see a failed fetch reported
    // as a missing fact, and the 404 an older deployment returns must be as quiet
    // as an honest empty list.
    //
    // The clear comes from the HEAD of the effect, not from the `.catch` — a
    // mutation that deleted the catch's `setSignals([])` left all 8 tests green,
    // which is how that redundancy was found rather than assumed.
    vi.mocked(fetchSignals).mockRejectedValue(new Error("404"));
    await openModal();
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    expect(screen.queryByText(/Out:/)).not.toBeInTheDocument();
    // And the page itself survived it.
    expect(screen.getByTestId("fixture-flow")).toBeInTheDocument();
  });

  it("does not ask about a finished fixture", async () => {
    // The endpoint answers [] for those, so asking is a request for an answer
    // already known -- once per finished gameweek, for nothing.
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    await openModal(finished);
    expect(fetchSignals).not.toHaveBeenCalled();
    // Even when the endpoint WOULD have sent a row. A finished fixture's squad
    // list is the one known before kick-off; rendering it afterwards is hindsight.
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
  });

  it("does not read one fixture's status to decide about the next", async () => {
    // Caught by CodeRabbit on #55, and the fix is `detail.event_id === eventId`
    // rather than `!detail`. Switching fixtures leaves the PREVIOUS detail in state
    // until the new one lands, and an effect reads the value captured by its OWN
    // render -- so a `!detail` gate reads the OLD fixture's `post_match`.
    //
    // **What that costs is a redundant request, not a wrong row.** The fetch is
    // keyed on `eventId` and every write is guarded by `cancelled`, so no data
    // from the previous fixture can reach the page. It is asserted as a CALL COUNT
    // because that is the whole of the defect: with a stale UPCOMING detail, a
    // `!detail` gate fires for the new id immediately and again once its detail
    // lands; the id comparison waits and fires once.
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    // `detail.event_id` is "e1" and UPCOMING (`post_match: null`), so it does not
    // block on status -- only on being the wrong fixture.
    mockApi(detail);
    const { rerender } = render(
      <FixtureModal eventId="e1" onClose={() => {}} explain={vi.fn().mockReturnValue(new Promise(() => {}))} />,
    );
    await screen.findByTestId("fixture-flow");
    await waitFor(() => expect(fetchSignals).toHaveBeenCalledTimes(1));

    // Switch to a different fixture WITHOUT letting the new detail land.
    vi.mocked(api.fixtureDetail).mockReturnValue(new Promise(() => {}) as never);
    rerender(
      <FixtureModal eventId="e2" onClose={() => {}} explain={vi.fn().mockReturnValue(new Promise(() => {}))} />,
    );
    await act(async () => {
      await Promise.resolve();
      await new Promise((r) => setTimeout(r, 0));
    });
    // Still exactly one: the stale "e1" detail did not authorise a fetch, and the
    // new one has not arrived.
    expect(fetchSignals).toHaveBeenCalledTimes(1);
    expect(fetchSignals).toHaveBeenCalledWith("e1");
  });

  it("does not ask while the fixture detail is still in flight", async () => {
    // The bug this effect had on its first run: `detail?.post_match` is
    // `undefined` on the first render, which is falsy, so the fetch fired for
    // EVERY fixture and only the re-run after the detail landed suppressed it —
    // one wasted request per finished gameweek, and a row that could arrive before
    // the page knew whether it was allowed to show one.
    //
    // Asserted by NEVER RESOLVING the detail rather than by checking a call count
    // synchronously after `render`. A synchronous count is a race: it passes or
    // fails on whether a microtask happened to flush, which is exactly the kind of
    // test that reports a property it is not holding. A detail that never lands
    // makes "no signals without a fixture" a fact rather than a timing window.
    // Every other endpoint still answers -- the modal fetches the track record and
    // the players independently of the detail, and an unstubbed one returns
    // `undefined`, which the modal calls `.then` on.
    mockApi();
    // ...and only the DETAIL never lands.
    vi.mocked(api.fixtureDetail).mockReturnValue(new Promise(() => {}) as never);
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    render(
      <FixtureModal eventId="e1" onClose={() => {}} explain={vi.fn().mockReturnValue(new Promise(() => {}))} />,
    );
    // Let every microtask and timer the render scheduled run.
    await act(async () => {
      await Promise.resolve();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(fetchSignals).not.toHaveBeenCalled();
  });

  it("renders inside the modal's own scrolling body", async () => {
    // A row mounted outside the panel's `overflow-y-auto` body would not scroll
    // with it, and on a long fixture page would sit pinned over the content behind
    // the modal. `fixture-flow` is NOT that container -- it is `FixtureFlow`, the
    // flow-steps component, a sibling of this section -- so the assertion is on the
    // nearest element that actually scrolls.
    vi.mocked(fetchSignals).mockResolvedValue(envelope([ABSENCE]));
    await openModal();
    const marker = await screen.findByTestId("signal-absence");
    const body = marker.closest(".overflow-y-auto");
    expect(body).not.toBeNull();
    expect(within(body as HTMLElement).getByTestId("signal-row")).toBeInTheDocument();
  });
});
