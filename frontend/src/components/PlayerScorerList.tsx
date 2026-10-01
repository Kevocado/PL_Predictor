import { MAX_ROWS_PER_CATEGORY, PicksList, type OutPlayer, type PickRow } from "../predictor-ui";
import type { PlayerPrediction } from "../types";
import { InfoTooltip } from "./InfoTooltip";

const STATUS_COLOR: Record<string, string> = {
  a: "bg-win",
  d: "bg-draw",
  i: "bg-loss",
  s: "bg-loss",
  u: "bg-pl-text-faint",
};

const STATUS_LABEL: Record<string, string> = {
  a: "Available",
  d: "Doubtful",
  i: "Injured",
  s: "Suspended",
  u: "Unavailable",
};

function PlayerRow({ player }: { player: PlayerPrediction }) {
  const isGoalkeeper = player.position === "GK";
  // Goal/assist probability is essentially always ~0 for a goalkeeper --
  // dimming on that basis would fade out every keeper regardless of how
  // likely they are to start, which isn't a useful signal for this position.
  const dimmed = !isGoalkeeper && player.anytime_goal_prob < 0.01 && player.anytime_assist_prob < 0.01;
  const strongestProbability = Math.max(
    player.anytime_goal_contribution_prob,
    player.anytime_goal_prob,
    player.anytime_assist_prob,
  );
  const probabilityClass = (probability: number) =>
    probability === strongestProbability ? "font-bold text-pl-text" : "font-semibold text-pl-text";
  return (
    <div className={`flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm ${dimmed ? "opacity-50" : ""}`}>
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${STATUS_COLOR[player.status] ?? "bg-pl-text-faint"}`}
          title={`${STATUS_LABEL[player.status] ?? player.status}${player.news ? ` — ${player.news}` : ""}`}
        />
        <span className="truncate text-pl-text">{player.name}</span>
        <span className="shrink-0 text-xs uppercase text-pl-text-faint">{player.position}</span>
        {player.confirmed_starter ? (
          <span className="rounded bg-win/20 px-1.5 py-0.5 text-xs font-semibold text-win">Confirmed XI</span>
        ) : player.predicted_starter ? (
          <span className="rounded bg-win/20 px-1.5 py-0.5 text-xs font-semibold text-win">XI</span>
        ) : null}
        {player.is_penalty_taker && <span className="rounded bg-pl-accent/20 px-1.5 py-0.5 text-xs font-semibold text-pl-accent">PK</span>}
        {!player.is_penalty_taker && player.is_set_piece_taker && <span className="rounded bg-pl-700 px-1.5 py-0.5 text-xs font-semibold text-pl-text">SP</span>}
      </div>
      <div className="flex shrink-0 items-center gap-2 text-xs sm:gap-3 sm:text-xs">
        {isGoalkeeper ? (
          <span className="text-pl-text-faint">
            Saves <span className="font-semibold text-pl-text">{player.expected_saves.toFixed(1)}</span>
          </span>
        ) : (
          <>
            <span className="text-pl-text-faint">
              G+A <span className={probabilityClass(player.anytime_goal_contribution_prob)}>{(player.anytime_goal_contribution_prob * 100).toFixed(0)}%</span>
            </span>
            <span className="text-pl-text-faint">
              Goal <span className={probabilityClass(player.anytime_goal_prob)}>{(player.anytime_goal_prob * 100).toFixed(0)}%</span>
            </span>
            <span className="text-pl-text-faint">
              Assist <span className={probabilityClass(player.anytime_assist_prob)}>{(player.anytime_assist_prob * 100).toFixed(0)}%</span>
            </span>
            {player.expected_shots != null && (
              <span
                className="text-pl-text-faint"
                title={`Chance of at least one shot on target: ${player.anytime_shot_on_target_prob != null ? (player.anytime_shot_on_target_prob * 100).toFixed(0) + "%" : "—"}`}
              >
                Shots <span className="font-semibold text-pl-text">{player.expected_shots.toFixed(1)}</span>
                <span className="ml-1">SoT {player.expected_shots_on_target?.toFixed(1) ?? "—"}</span>
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function TeamPlayerPredictions({ team, players }: { team: string; players: PlayerPrediction[] }) {
  const byContribution = [...players].sort(
    (left, right) => right.anytime_goal_contribution_prob - left.anytime_goal_contribution_prob,
  );
  const confirmed = players.some((player) => player.confirmed_starter);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-pl-text-faint">{team}</span>
        {confirmed && <span className="text-xs font-semibold uppercase text-win">Official lineup</span>}
        <InfoTooltip
          align="left"
          text="Goal is the chance of scoring at least once. Assist is the chance of registering at least one assist. G+A is the chance of doing either (or both), so it is not the two percentages added together."
        />
      </div>
      <div className="flex flex-col gap-1.5">
        {byContribution.map((player) => <PlayerRow key={player.player_id} player={player} />)}
      </div>
    </div>
  );
}

interface Props {
  homeTeam: string;
  awayTeam: string;
  homePlayers: PlayerPrediction[];
  awayPlayers: PlayerPrediction[];
  /** The fixture's kickoff, used only to date the out-player line. */
  kickoff?: string;
}

export function PlayerScorerList({ homeTeam, awayTeam, homePlayers, awayPlayers }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <div className="flex flex-col gap-1.5">
        <TeamPlayerPredictions team={homeTeam} players={homePlayers} />
      </div>
      <div className="flex flex-col gap-1.5">
        <TeamPlayerPredictions team={awayTeam} players={awayPlayers} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// "Model's top calls" — the ranked list, on the vendored PicksList.
//
// This used to be `PlayerHighlights`, a hand-rolled ranking of G+A with no
// provenance and no out-player rule. It is replaced rather than added to: two
// ranked call lists on one screen is one too many, and the old one claimed a
// number from the arm that cannot be labelled (see below).
//
// Everything this list needs is already on `PlayerPrediction`, so no new
// endpoint is involved. `PicksList` itself enforces three of the rules — the
// three-row ceiling, the refusal of an `out` row, and the kind/range guard —
// and the rest are enforced by the ranking below.
// ---------------------------------------------------------------------------

/**
 * The statuses the backend scales to zero, mirroring
 * `pl_predictor.data.fpl_api.UNAVAILABLE_STATUSES`. Kept as a mirror because
 * `PlayerPrediction` does not carry `availability`: `routes.py` builds each row
 * as `PlayerPrediction(**{k: p[k] for k in PlayerPrediction.model_fields})`,
 * which drops it. If a status is added to that set, this must be updated with it.
 *
 * "d" (doubtful) is deliberately absent: `availability_multiplier` returns
 * `chance / 100`, or 0.5 when the chance is unknown, so a doubtful player is
 * scaled down and not ruled out. Treating him as out would delete a ranked call
 * the model actually stands behind.
 */
const UNAVAILABLE_STATUSES = new Set(["i", "s", "u"]);

type Candidate = PlayerPrediction & { team: string };

/** What the number on the row is, in the words the arm deserves.
 *
 *  The goal and assist arms are `anytime_probability`, i.e. bare
 *  `1 - exp(-lambda)` Poisson (`player_goals.py`). Nothing calibrates them, so
 *  the row says "uncalibrated" and the word is load-bearing: the only arm with a
 *  calibrator is `anytime_goal_contribution_prob`, and serving hands that out as
 *  a 50/50 blend of the calibrated direct model and this uncalibrated union
 *  (`NEUTRAL_BLEND_WEIGHT = 0.5`), which is why that arm is not a category here
 *  at all -- a blend of one calibrated and one uncalibrated estimator is neither.
 */
const ARM_PROVENANCE = "Poisson 1 − e^−λ, uncalibrated";

/**
 * There is no per-player ledger, so there is no per-player record to quote.
 *
 * `tracking.store.get_scorer_accuracy()` returns `{snapshot, reconstructed}` --
 * aggregate groups over every resolved call, with no player key anywhere. A
 * hit rate read off that aggregate and attached to a named player would be a
 * number borrowed from a different unit of analysis, so the row says what is
 * actually true instead: the arm, and the absence.
 */
const LEDGER_PROVENANCE = "no graded record per player — the scorer ledger is aggregate only";

/** How much of this player's form is real data rather than a prior.
 *
 *  `features/cold_start.py` sets `"none"` when a player has played zero games
 *  this season, so their rate is entirely a league-average prior. The list this
 *  replaces dropped those rows silently; ranking them and saying so is the
 *  honest version, and a number the reader cannot see the basis of is the thing
 *  this phase exists to stop.
 */
const FORM_BASIS: Record<string, string> = {
  current: "form from this season's games",
  blended: "form blended with a league average",
  none: "no games this season — the rate is a league-average prior",
};

const LINEUP_BASIS = (player: PlayerPrediction): string =>
  player.confirmed_starter
    ? "Confirmed XI"
    : player.predicted_starter
      ? "Predicted XI"
      : "not in the predicted XI";

const gameweekOf = (kickoff?: string): string => {
  if (!kickoff) return "this gameweek";
  const date = new Date(kickoff);
  if (Number.isNaN(date.getTime())) return "this gameweek";
  return `${date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })} gameweek`;
};

/** Rank one category. The out filter runs BEFORE the sort and the slice, so an
 *  out player can neither occupy a rank nor backfill a freed slot. */
function rankedRows(
  candidates: Candidate[],
  category: string,
  probability: (player: PlayerPrediction) => number,
): PickRow[] {
  return candidates
    .filter((player) => !UNAVAILABLE_STATUSES.has(player.status))
    .sort((left, right) => probability(right) - probability(left))
    .slice(0, MAX_ROWS_PER_CATEGORY)
    .map((player) => ({
      key: `${player.player_id}-${category}`,
      name: player.name,
      team: player.team,
      detail: category,
      // The model's own number for this category, read straight off the row.
      value: probability(player),
      kind: "probability" as const,
      provenance: [LINEUP_BASIS(player), FORM_BASIS[player.confidence] ?? "form basis unstated", ARM_PROVENANCE, LEDGER_PROVENANCE].join(" · "),
    }));
}

export function ModelTopCalls({ homeTeam, awayTeam, homePlayers, awayPlayers, kickoff }: Props) {
  const candidates: Candidate[] = [
    ...homePlayers.map((player) => ({ ...player, team: homeTeam })),
    ...awayPlayers.map((player) => ({ ...player, team: awayTeam })),
  ];

  // The two categories, and only these two. `expected_saves` is a GK-only
  // figure -- two rows per fixture -- and is not a category; the shot-on-target
  // probability needs an Understat merge per player, so its row count is
  // data-dependent and is measured rather than assumed to be three.
  const categories = [
    { category: "Anytime goal", rows: rankedRows(candidates, "Anytime goal", (p) => p.anytime_goal_prob) },
    { category: "Anytime assist", rows: rankedRows(candidates, "Anytime assist", (p) => p.anytime_assist_prob) },
  ];

  // §D: removed from the ranking entirely, then shown once here, attributed and
  // dated. `PickRow.out` is never set -- PicksList refuses such a row by name,
  // which is the point of the field.
  const out: OutPlayer[] = candidates
    .filter((player) => UNAVAILABLE_STATUSES.has(player.status))
    .map((player) => ({
      name: player.name,
      team: player.team,
      source: `FPL squad status (${STATUS_LABEL[player.status] ?? player.status})`,
      dated: `as read for the ${gameweekOf(kickoff)}`,
    }));

  return <PicksList categories={categories} out={out} />;
}

