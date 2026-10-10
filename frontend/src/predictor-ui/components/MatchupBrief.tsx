// Synced from predictor-ui@0b2120579475. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The summary grouped into Edge, Risk and Price — the grouping a reader scans.
 *
 * `FactorList` already marks each row "for the pick" / "against it" / "context"
 * and refuses colour-only meaning. This adds the three labelled groups the AI
 * plan's design puts on one screen, and the `RankDuel` picture on the rows that
 * are about an offence-versus-defence matchup.
 *
 * ## The rule this component exists to enforce
 *
 * **A figure may only reach the page if the row's own words state it.** `RankDuel`
 * draws two ranks; the headline beside it has to say those same two ranks, written
 * as ranks. `assertStates` refuses the row otherwise — the same rule, and the same
 * decision, as `SignalRows.assertFigureIsStated`, for the same reason: a bar at
 * 94% under words that never said 94% is a number a reader cannot check.
 *
 * "Written as a rank" is the part that is easy to get wrong, so it is a regex
 * rather than a number scan: `#28` states rank 28, and `"28 points allowed"`
 * does not, even though both contain the digits. A row that happened to state the
 * two ranks as some other figure would otherwise pass by accident, and the two
 * kinds of row — a duel row and a scoring row — can share a panel.
 *
 * ## What it does not do
 *
 * It does not rank, cap or reorder. The sport code sorts duels by strength, the
 * service orders the factors, and a cap here would silently drop a row for a
 * reason this component cannot see — the argument `SignalRows` makes about
 * ranking-and-capping in its own header.
 *
 * An empty group is omitted rather than padded: "fewer rows beats a weak row"
 * is the design's rule, and a heading over nothing is a claim the panel cannot
 * support.
 *
 * **The grouped rows do not take `expandable`/`expanded`.** `FactorList` clamps
 * its sentence to one line on a narrow viewport because the sentence is long and
 * the panel is already tall; a duel row is a rank and a rank, and clamping it at
 * 390px would hide the very words `assertStates` just checked — so these rows
 * wrap, and the `context` group below still goes through `FactorList` with the
 * panel's own clamp. The `context` group therefore keeps the old behaviour and the
 * new rows do not, which is the honest split: a clamp is a decision about a long
 * sentence, and these are not long.
 */
import { FactorList, type Factor } from "./FactorList";
import { RankDuel } from "./RankDuel";

export type Slot = "edge" | "risk" | "price" | "context";

/**
 * A factor, which already carries an optional `slot`. An alias rather than a
 * second type, because two definitions of the same field is how a caller ends up
 * passing a `Factor` where a `SlottedFactor` is wanted and widening at the call
 * site instead of reading what the response already has.
 */
export type SlottedFactor = Factor;

/** One code-computed duel, in the shape `signals/matchups.to_context` emits. */
export type MatchupRow = {
  id: string;
  attacker: string;
  defender: string;
  stat: string;
  foil: string;
  attacker_rank: number;
  defender_rank: number;
  n_teams: number;
  /** True when the duel favours the pick, false when the other side, null when
   *  the code has not checked whether it explains anything. The service has
   *  already turned this into a `slot`; it rides along so the caller can see it. */
  toward_pick: boolean | null;
};

/** The three named groups. `context` is not a group: it renders as the old list. */
const TITLES: Record<Exclude<Slot, "context">, string> = {
  edge: "Edge",
  risk: "Risk",
  price: "Price",
};

/** The four groups, LOOKED UP rather than indexed, and that is the whole point.
 *
 *  The `slot` arrives on the wire, so a renamed value — or one from an explainer
 *  newer than this panel — is a shape this build does not know. Reading
 *  `out[f.slot]` straight off the object gave `undefined.push`, so one
 *  unexpected value took the whole panel down instead of downgrading one row.
 *  This is `FactorList.MARK`'s shape for the same reason: the value is checked
 *  against the vocabulary that exists here, and everything else is the
 *  fail-closed answer.
 */
const GROUP: Record<string, Slot> = {
  edge: "edge", risk: "risk", price: "price", context: "context",
};

/** The factors by the group they are drawn in, keeping order and dropping nothing. */
export function groupBySlot(factors: SlottedFactor[]): Record<Slot, SlottedFactor[]> {
  const out: Record<Slot, SlottedFactor[]> = { edge: [], risk: [], price: [], context: [] };
  for (const f of factors) out[GROUP[f.slot as string] ?? "context"].push(f);
  return out;
}

/** The row's words do not state the two ranks its duel is about to draw. */
export class DuelHeadlineMismatchError extends Error {
  constructor(key: string) {
    super(
      `MatchupBrief: the headline of ${key} does not state both ranks its duel draws, ` +
        `written as ranks ("#3", "#28"). A figure on the page that the row's own words ` +
        `do not carry is the one thing this component exists to prevent, so the row is ` +
        `refused rather than drawn beside words that disagree with it.`,
    );
    this.name = "DuelHeadlineMismatchError";
  }
}

/** A rank is written with a `#`, and only a `#` counts. See the file header. */
const STATED_RANK = /#\s?(\d+)/g;

function assertStates(f: SlottedFactor, m: MatchupRow) {
  const stated = [...f.headline.matchAll(STATED_RANK)].map((x) => Number(x[1]));
  if (!stated.includes(m.attacker_rank) || !stated.includes(m.defender_rank)) {
    throw new DuelHeadlineMismatchError(f.key);
  }
}

function Row({ factor, matchup, onSelect, highlighted }: {
  factor: SlottedFactor;
  matchup?: MatchupRow;
  /** §13c's other end: pressing a row lights the figure it names. Passed through
   *  from the panel rather than reimplemented here, so a brief row and a
   *  `FactorList` row are the same control. */
  onSelect?: (key: string) => void;
  highlighted?: string | null;
}) {
  // Checked before anything renders, so a caller that paired a duel with the
  // wrong words gets a named error rather than a bar under a sentence.
  if (matchup) assertStates(factor, matchup);
  const lit = !!highlighted && highlighted === factor.key;
  return (
    <li className="py-2.5">
      <div className="flex items-baseline gap-2">
        {onSelect ? (
          <button
            type="button"
            data-testid={`factor-${factor.key}`}
            data-highlighted={lit ? "true" : "false"}
            aria-pressed={lit}
            className="group min-w-0 flex-1 text-left"
            onClick={() => onSelect(factor.key)}
          >
            <span className="font-semibold text-pr-text underline-offset-4 group-hover:underline">
              {factor.headline}
            </span>
          </button>
        ) : (
          <p className="min-w-0 flex-1 font-semibold text-pr-text">{factor.headline}</p>
        )}
      </div>
      {matchup && (
        <div className="mt-2">
          <RankDuel
            attacker={matchup.attacker} attackerStat={matchup.stat} attackerRank={matchup.attacker_rank}
            defender={matchup.defender} defenderStat={matchup.foil} defenderRank={matchup.defender_rank}
            nTeams={matchup.n_teams}
          />
        </div>
      )}
      {factor.text && (
        <p className="mt-1.5 max-w-[70ch] font-pr-body text-sm leading-relaxed text-pr-text-dim">{factor.text}</p>
      )}
    </li>
  );
}

export function MatchupBrief({
  factors,
  matchups = [],
  onSelect,
  highlighted,
}: {
  factors: SlottedFactor[];
  matchups?: MatchupRow[];
  onSelect?: (key: string) => void;
  highlighted?: string | null;
}) {
  const groups = groupBySlot(factors);
  const byKey = new Map(matchups.map((m) => [`matchup:${m.id}`, m]));
  return (
    <div className="flex flex-col gap-4">
      {(Object.keys(TITLES) as (keyof typeof TITLES)[]).map((slot) =>
        groups[slot].length === 0 ? null : (
          <section key={slot} aria-labelledby={`brief-${slot}`}>
            <h3
              id={`brief-${slot}`}
              className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
            >
              {TITLES[slot]}
            </h3>
            <ul className="divide-y divide-pr-rule border-t border-pr-rule">
              {groups[slot].map((f) => (
                <Row
                  key={f.key}
                  factor={f}
                  matchup={byKey.get(f.key)}
                  onSelect={onSelect}
                  highlighted={highlighted}
                />
              ))}
            </ul>
          </section>
        ),
      )}
      {groups.context.length > 0 && (
        <FactorList factors={groups.context} onSelect={onSelect} highlighted={highlighted} />
      )}
    </div>
  );
}