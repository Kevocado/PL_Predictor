import { TeamBadge } from "./TeamBadge";
import { kickoffParts } from "../lib/kickoffTime";
import type { FixturePlayerEvent } from "../types";
import { matchPick } from "../lib/pick";

function formatKickoff(iso: string): { date: string; time: string } {
  const { day, time } = kickoffParts(iso);
  return { date: day, time };
}

interface Props {
  commence_time: string;
  team_home: string;
  team_away: string;
  actual_goals_home: number | null;
  actual_goals_away: number | null;
  predicted_scoreline: string | null;
  predicted_home_win: number;
  predicted_draw: number;
  predicted_away_win: number;
  draw_signal?: boolean;
  hit: boolean | null;
  backfilled: boolean;
  // Derived per read by the backend from this pick's own `snapshotted_at` vs
  // its kickoff, as UTC instants. `backfilled` is provenance and cannot answer
  // that. Absent from a payload baked before the field existed, which reads
  // falsy and so shows the badge — the claim-hiding direction, which is right
  // for a label that says "made after kickoff".
  made_before_kickoff?: boolean;
  home_player_events?: FixturePlayerEvent[];
  away_player_events?: FixturePlayerEvent[];
  player_events_pending?: boolean;
  onClick: () => void;
}

function playerSummary(events: FixturePlayerEvent[], key: "goals" | "assists", label: string) {
  const contributors = events.filter((event) => event[key] > 0);
  if (!contributors.length) return null;
  return `${label}: ${contributors.map((event) => `${event.name}${event[key] > 1 ? ` (${event[key]})` : ""}`).join(", ")}`;
}

// Shared "already-played fixture" card — same visual language wherever a
// finished match with an honest pre-match prediction is shown (the
// Fixtures page's current-gameweek view, the Track Record tab's
// per-gameweek results). Clicking opens the same full fixture-detail
// modal both places already use.
export function FinishedFixtureCard({
  commence_time,
  team_home,
  team_away,
  actual_goals_home,
  actual_goals_away,
  predicted_scoreline,
  predicted_home_win,
  predicted_draw,
  predicted_away_win,
  draw_signal = false,
  hit,
  made_before_kickoff,
  home_player_events = [],
  away_player_events = [],
  onClick,
}: Props) {
  const { date, time } = formatKickoff(commence_time);
  const pick = matchPick(predicted_home_win, predicted_draw, predicted_away_win, team_home, team_away);
  const pct = (p: number) => `${Math.round(p * 100)}%`;

  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
          // Enter and Space, because this is announced as a button and must
          // behave like one. Space's default action is to scroll the page, so
          // preventDefault is part of the fix rather than a detail: without it
          // the keypress that opens the fixture also jumps the reader down the
          // page. `role="button"` on a div is the reason this is worth stating
          // at all -- a real <button> would give both keys for free.
          if (e.key !== "Enter" && e.key !== " ") return;
          e.preventDefault();
          onClick();
        }}
      title={draw_signal ? "The scoreline model also leaned towards a draw." : undefined}
      className={`clip-corner flex cursor-pointer flex-col gap-3 rounded-xl border bg-pl-850/70 ${!made_before_kickoff ? "border-pl-border" : hit ? "border-win/30" : "border-loss/30"} p-4 transition hover:border-pl-pink/40`}
    >
      <div className="flex flex-wrap items-center justify-between gap-1.5 text-xs font-medium uppercase tracking-wide text-pl-text-faint">
        <span>
          {date} &middot; {time}
        </span>
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          {/* Site-local badge, NOT the shared `StatusBadge`: `predictor-hub#67` (open,
              unmerged) renames the shared badge's words but cannot reach this
              hand-written span. The words match it anyway — "Made after
              kickoff" describes WHEN the pick was made, which is what this
              state now is. The old wording described the pick's standing in the
              record, and under the 2026-10-01 reversal a pick made after the
              start is counted like any other, so that standing is no longer the
              thing worth saying. The `rebuilt` PROP is kept: it is a published
              field name, and renaming it would break every call site to change
              nothing a reader sees. */}
          {/* The DERIVED label, not `backfilled`. The badge says when the pick
              was made, and `backfilled` only says which job wrote the row — the
              five-minute tracking tick writes `false` whenever it runs, so a
              pick written long after kickoff used to render as "Called it ✓"
              with no badge at all. Fails closed: an absent flag shows the badge,
              so the card never claims a pick was in time. */}
          {!made_before_kickoff && (
            <span
              title="Made from the model after this match finished. Counted in the track record like any other pick."
              className="rounded border border-pl-border px-1.5 py-0.5 text-xs font-semibold normal-case tracking-normal text-pl-text-dim"
            >
              Made after kickoff
            </span>
          )}
          {/* A pick not made in time is shown, never judged: one badge, not two. */}
          {made_before_kickoff && (
            <span className={`text-xs font-semibold ${hit ? "text-win" : "text-loss"}`}>{hit ? "Called it ✓" : "Missed ✗"}</span>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={team_home} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{team_home}</span>
        </div>
        <div className="flex flex-col items-center gap-0.5 px-1">
          <span className="text-xs font-medium uppercase text-pl-text-faint">Final</span>
          <span className="rounded-lg bg-pl-700/50 px-2.5 py-1 font-display text-xl font-semibold tracking-wide text-pl-text">
            {actual_goals_home}–{actual_goals_away}
          </span>
        </div>
        <div className="flex flex-1 flex-col items-center gap-1.5 text-center">
          <TeamBadge team={team_away} />
          <span className="text-xs font-semibold leading-tight text-pl-text">{team_away}</span>
        </div>
      </div>
      {(home_player_events.length > 0 || away_player_events.length > 0) && (
        <div className="grid grid-cols-2 gap-3 border-t border-pl-border/70 pt-2 text-xs leading-relaxed text-pl-text-dim">
          <div>{playerSummary(home_player_events, "goals", "Goals") && <p>{playerSummary(home_player_events, "goals", "Goals")}</p>}{playerSummary(home_player_events, "assists", "Assists") && <p>{playerSummary(home_player_events, "assists", "Assists")}</p>}</div>
          <div className="text-right">{playerSummary(away_player_events, "goals", "Goals") && <p>{playerSummary(away_player_events, "goals", "Goals")}</p>}{playerSummary(away_player_events, "assists", "Assists") && <p>{playerSummary(away_player_events, "assists", "Assists")}</p>}</div>
        </div>
      )}
      <div className="flex flex-col gap-1 border-t border-pl-border/70 pt-2.5 text-xs text-pl-text-dim">
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold text-pl-text">Pick: {pick.label} · {pct(pick.prob)}</span>
          {predicted_scoreline && <span>Most likely score {predicted_scoreline.replace("-", "–")}</span>}
        </div>
        <span>Home {pct(predicted_home_win)} · Draw {pct(predicted_draw)} · Away {pct(predicted_away_win)}</span>
      </div>
    </div>
  );
}
