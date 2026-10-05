/**
 * PL's vendored `SignalRows` with one ILLUSTRATIVE absence row.
 *
 * See `frontend/signals-harness.html` for why this exists and why the row is
 * labelled. Nothing on this page is a real PL observation.
 *
 * The two rows are the sport's OWN units side by side, because that is the
 * design this component landed with: `figures.projection` is a magnitude in the
 * sport's units and the marker cannot know which, so the headline carries it. PL's
 * is a share ("62% to score") and the unit differs from a points total — which is
 * why the component draws a bare number and `denotes` waives the unit for this
 * visual alone.
 *
 * Both rows carry an `n` far below the `n >= 30` rate floor, because for an
 * absence `n` counts INJURED PLAYERS rather than graded games.
 */
import { createRoot } from "react-dom/client";

import { SignalRows, SPEC_MIN_N, type Signal } from "../src/predictor-ui";
import "../src/index.css";

const INJURY_SOURCE = "FPL status · 2 players out";

/** A share, PL's own unit. `n: 2` — two injured players, not two games. */
const PL_ROW: Signal = {
  kind: "absence",
  sport: "pl",
  game_id: "harness",
  headline: {
    text: "Out: A. Player, our #1 scorer, 62% to score",
    figures: { projection: 62 },
  },
  n: 2,
  source: INJURY_SOURCE,
  as_of: "2026-10-04T16:00:00Z",
  strength: 0.62,
  pre_kickoff_only: true,
  visual: "absence_strip",
};

/** A magnitude in points, so the same visual reads correctly in another unit. */
const POINTS_ROW: Signal = {
  ...PL_ROW,
  game_id: "harness-2",
  headline: {
    text: "Out: B. Player, our #3 scorer, 30 pts",
    figures: { projection: 30 },
  },
  n: 1,
  source: "FPL status · 1 player out",
  strength: 1 / 3,
};

function Harness() {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-sm font-semibold text-pl-text-dim">
        PL&apos;s absence row — the phase-2 `absence_strip` visual
      </h1>
      <SignalRows signals={[PL_ROW, POINTS_ROW]} />
      <p className="max-w-[70ch] text-xs text-pl-text-faint">
        The marker is a plain unsigned number because this component cannot know
        the unit: PL&apos;s projection is a share and a points projection is not,
        so the headline carries it. It is deliberately not a bar — a projection is
        not a hit rate — and deliberately not coloured, because a player being out
        is not an outcome the model got right or wrong.
      </p>
      <p className="max-w-[70ch] text-xs text-pl-text-faint">
        Both rows draw at n=2 and n=1, far below the{" "}
        <code>n ≥ {SPEC_MIN_N}</code> rate floor: for an absence{" "}
        <code>n</code> counts injured <em>players</em>, not graded games, so that
        floor does not apply. <code>30</code> prints as a whole number rather than{" "}
        <code>30.0</code>, because a tenth of a point is precision the model never
        had.
      </p>
    </div>
  );
}

const el = document.getElementById("root");
if (!el) throw new Error("no #root to mount into");
createRoot(el).render(<Harness />);
