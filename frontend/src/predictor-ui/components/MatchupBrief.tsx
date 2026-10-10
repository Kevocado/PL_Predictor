// Synced from predictor-ui@eb2ea90eee19. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The Matchup section: how the two sides' units meet, as data, with no AI.
 *
 * Rendered above the AI panel and visible before anything is asked for. It draws
 * what the fixture's `context` carries: code-computed rank duels (`matchups`)
 * through `RankDuel`, and F1's labelled form rows (`form_rows`). Either, both or
 * neither may be present; with neither the component renders NOTHING, not a
 * heading over an empty list and not a placeholder.
 *
 * Duels stay neutral. `toward_pick` rides along on the row but is not drawn: no
 * "advantage", no "edge", until a lift-gate run proves a duel type predicts.
 * NFL player props are a different block and are not touched here.
 */
import { RankDuel } from "./RankDuel";

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
  toward_pick: boolean | null;
};

/** One labelled form fact (F1: driver/constructor form, quali pace, track history). */
export type FormRow = {
  id: string;
  subject: string;
  label: string;
  value: string;
  rank: number | null;
  n: number | null;
};

/** What `/explain/{sport}/{id}/context` returns; every key optional. */
export type MatchupContext = { matchups?: MatchupRow[]; form_rows?: FormRow[]; player_context?: unknown[] };

/**
 * A loader for the context route, next to the site's own explain request.
 * `baseUrl` is the site's explain base (no trailing slash needed); the id is
 * encoded per path segment so a slash-bearing id still reaches the route.
 * Rejects on a non-2xx, a timeout or bad JSON; callers treat any rejection as
 * "no section".
 */
export function createContextLoader(baseUrl: string, timeoutMs = 10_000) {
  const base = baseUrl.replace(/\/+$/, "");
  return async (id: string): Promise<MatchupContext> => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetch(`${base}/${id.split("/").map(encodeURIComponent).join("/")}/context`, { signal: ctl.signal });
      if (!res.ok) throw new Error(`context ${res.status}`);
      return (await res.json()) as MatchupContext;
    } finally {
      clearTimeout(timer);
    }
  };
}

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

/** A duel the component can draw: both ranks and the league size are integers.
 *  Anything else is dropped rather than thrown on: this is context shown on every
 *  fixture, and one malformed row must not take the page down. */
const drawable = (m: MatchupRow) =>
  !!m && isInt(m.attacker_rank) && isInt(m.defender_rank) && isInt(m.n_teams) && m.n_teams >= 2 &&
  m.attacker_rank >= 1 && m.attacker_rank <= m.n_teams && m.defender_rank >= 1 && m.defender_rank <= m.n_teams;

const usableForm = (r: FormRow) => !!r && !!r.subject && !!r.label && !!r.value;

export function MatchupBrief({ matchups = [], formRows = [] }: { matchups?: MatchupRow[]; formRows?: FormRow[] }) {
  // `bundle.context` is untyped site data: anything that is not a list is nothing.
  const duels = (Array.isArray(matchups) ? matchups : []).filter(drawable);
  const forms = (Array.isArray(formRows) ? formRows : []).filter(usableForm);
  if (duels.length === 0 && forms.length === 0) return null;
  return (
    <section aria-labelledby="matchup-heading" data-testid="matchup-section">
      <h3
        id="matchup-heading"
        className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        Matchup
      </h3>
      <ul className="divide-y divide-pr-rule border-t border-pr-rule">
        {duels.map((m) => (
          <li key={m.id} className="py-2.5" data-testid={`matchup-${m.id}`}>
            <RankDuel
              attacker={m.attacker} attackerStat={m.stat} attackerRank={m.attacker_rank}
              defender={m.defender} defenderStat={m.foil} defenderRank={m.defender_rank}
              nTeams={m.n_teams}
            />
          </li>
        ))}
        {forms.map((r) => (
          <li key={r.id} className="flex items-baseline justify-between gap-3 py-2 text-sm" data-testid={`form-${r.id}`}>
            <span className="min-w-0 truncate text-pr-text-dim">{r.subject} · {r.label}</span>
            <span className="shrink-0 tabular-nums font-semibold text-pr-text">
              {r.value}
              {isInt(r.rank) && (
                <span className="font-normal text-pr-text-faint"> #{r.rank}{isInt(r.n) ? ` of ${r.n}` : ""}</span>
              )}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
