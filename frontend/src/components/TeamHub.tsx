import { useMemo, useState } from "react";
import type { TeamHubResponse, TeamHubTeam } from "../types";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";
import { GLOSSARY } from "../lib/glossary";

function value(number: number | null, suffix = "") {
  return number === null ? "—" : `${number.toFixed(2)}${suffix}`;
}

function trend(number: number | null) {
  if (number === null) return "text-pl-text-faint";
  return number > 0 ? "text-win" : number < 0 ? "text-loss" : "text-pl-text-dim";
}

const FORM_TREND = {
  up: { arrow: "↑", label: "Improving form", className: "text-win" },
  down: { arrow: "↓", label: "Declining form", className: "text-loss" },
  steady: { arrow: "→", label: "Steady form", className: "text-pl-text-dim" },
  new: { arrow: "•", label: "Not enough current-season matches for a trend", className: "text-pl-text-faint" },
} as const;

const COMPARE_ROW_GRID = "grid-cols-[1rem_1.5rem_minmax(7rem,1fr)_repeat(7,minmax(4rem,auto))]";

function StatChip({ label, value: chipValue }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-[9px] font-semibold uppercase tracking-wide text-pl-text-faint">{label}</p>
      <p className="text-sm font-semibold tabular-nums text-pl-text">{chipValue}</p>
    </div>
  );
}

function TeamDetailPanel({ selected }: { selected: TeamHubTeam }) {
  const metrics = [
    ["Points / match", value(selected.points_per_match)],
    ["Goals", `${selected.goals_for}-${selected.goals_against}`],
    ["Shots / target", `${value(selected.shots_per_match)} / ${value(selected.shots_on_target_per_match)}`],
    ["Corners / cards", `${value(selected.corners_per_match)} / ${value(selected.cards_per_match)}`],
  ];

  return (
    <section className="rounded-xl border border-pl-border bg-pl-850/50 p-4">
      <div className="mb-4 flex items-center gap-3">
        <TeamBadge team={selected.team} size="lg" />
        <div>
          <h3 className="text-lg font-semibold text-pl-text">{selected.team}</h3>
          <p className="text-xs text-pl-text-faint">{selected.wins}W · {selected.draws}D · {selected.losses}L · Form {value(selected.form_points_per_match)} PPG <span title={FORM_TREND[selected.form_trend].label} className={`font-bold ${FORM_TREND[selected.form_trend].className}`}>{FORM_TREND[selected.form_trend].arrow}</span> · streak {selected.streak > 0 ? `+${selected.streak}` : selected.streak}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {metrics.map(([label, metric]) => (
          <div key={label} className="rounded-lg bg-pl-900/60 px-3 py-2">
            <p className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-pl-text-faint">{label}{label === "Points / match" && <InfoTooltip text={GLOSSARY.pointsPerMatch} align="left" />}</p>
            <p className="mt-1 text-sm font-semibold text-pl-text">{metric}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Underlying performance</h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="flex items-center gap-1 text-xs text-pl-text-faint">xG for <InfoTooltip text={GLOSSARY.expectedGoals} align="left" /></span><p className="font-semibold text-pl-text">{value(selected.xg_for)}</p></div>
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="flex items-center gap-1 text-xs text-pl-text-faint">xG against <InfoTooltip text={GLOSSARY.expectedGoals} align="left" /></span><p className="font-semibold text-pl-text">{value(selected.xg_against)}</p></div>
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="flex items-center gap-1 text-xs text-pl-text-faint">Goals − xG <InfoTooltip text={GLOSSARY.goalsMinusXg} align="left" /></span><p className={`font-semibold ${trend(selected.goals_minus_xg)}`}>{selected.goals_minus_xg === null ? "—" : `${selected.goals_minus_xg > 0 ? "+" : ""}${value(selected.goals_minus_xg)}`}</p></div>
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="flex items-center gap-1 text-xs text-pl-text-faint">Conceded − xGA <InfoTooltip text={GLOSSARY.concededMinusXga} align="left" /></span><p className={`font-semibold ${trend(selected.goals_conceded_minus_xg)}`}>{selected.goals_conceded_minus_xg === null ? "—" : `${selected.goals_conceded_minus_xg > 0 ? "+" : ""}${value(selected.goals_conceded_minus_xg)}`}</p></div>
          </div>
        </div>
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Style</h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="flex items-center gap-1 text-xs text-pl-text-faint">Set-piece xG share <InfoTooltip text={GLOSSARY.setPieceShare} align="left" /></span><p className="font-semibold text-pl-text">{value(selected.set_piece_xg_share === null ? null : selected.set_piece_xg_share * 100, "%")}</p></div>
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="text-xs text-pl-text-faint">Fouls / match</span><p className="font-semibold text-pl-text">{value(selected.fouls_per_match)}</p></div>
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="text-xs text-pl-text-faint">Corners / match</span><p className="font-semibold text-pl-text">{value(selected.corners_per_match)}</p></div>
            <div className="rounded-lg bg-pl-900/60 px-3 py-2"><span className="text-xs text-pl-text-faint">Cards / match</span><p className="font-semibold text-pl-text">{value(selected.cards_per_match)}</p></div>
          </div>
        </div>
      </div>

      <div className="mt-4">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">Recent matches</h4>
        <div className="grid gap-1 sm:grid-cols-2">
          {selected.recent_matches.map((match) => (
            <div key={`${match.date}-${match.opponent}-${match.venue}`} className="flex items-center justify-between rounded-lg bg-pl-900/60 px-3 py-2 text-xs">
              <span className="text-pl-text-faint">{match.date} · {match.venue === "Home" ? "vs" : "@"} {match.opponent}</span>
              <span className="flex items-center gap-2"><span className={`font-semibold ${match.result === "W" ? "text-win" : match.result === "L" ? "text-loss" : "text-pl-text-dim"}`}>{match.result} {match.score}</span><span className="text-pl-text-faint">xG {value(match.xg_for)}-{value(match.xg_against)}</span></span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function TeamHub({ data }: { data: TeamHubResponse }) {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const teams = useMemo(
    () => {
      const current = [...data.teams].sort((left, right) => right.points - left.points || (right.goals_for - right.goals_against) - (left.goals_for - left.goals_against) || right.goals_for - left.goals_for);
      const form = [...data.teams].sort((left, right) => (right.form_points_per_match ?? -1) - (left.form_points_per_match ?? -1) || (right.points_per_match ?? 0) - (left.points_per_match ?? 0));
      const formPosition = new Map(form.map((team, index) => [team.team, index + 1]));
      return current.map((team, index) => ({ ...team, currentPosition: index + 1, formPosition: formPosition.get(team.team) ?? null }));
    },
    [data.teams],
  );
  const selected = teams.find((team) => team.team === selectedTeam);

  if (teams.length === 0) return <p className="text-sm text-pl-text-faint">No current-season team data is available yet.</p>;

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm text-pl-text-dim">Ordered by current league position. Form rank shows where recent PPG suggests each club belongs; select a club for detail and select it again to collapse.</p>

      <div className="overflow-x-auto rounded-xl border border-pl-border bg-pl-850/50">
        <div className={`grid ${COMPARE_ROW_GRID} gap-2 border-b border-pl-border px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-pl-text-faint`}>
          <span />
          <span />
          <span>Team</span>
          <span className="text-center">Pos</span>
          <span className="text-center">Form rank</span>
          <span className="text-center">Goals</span>
          <span className="text-center">Assists</span>
          <span className="text-center">Corners/m</span>
          <span className="flex items-center justify-center gap-1">
            xG
            <InfoTooltip
              text={`${GLOSSARY.expectedGoals} A dash means the underlying data provider (Understat) hasn't published that team's match(es) yet this season — not that the team hasn't played.`}
              align="left"
            />
          </span>
          <span className="text-center">xA</span>
        </div>
        <div className="flex flex-col">
          {teams.map((team, i) => (
            <div key={team.team}>
              <button
                onClick={() => setSelectedTeam((current) => current === team.team ? null : team.team)}
                className={`grid w-full ${COMPARE_ROW_GRID} items-center gap-2 px-3 py-2 text-left transition ${
                  team.team === selectedTeam ? "bg-pl-pink/10" : "hover:bg-pl-900/40"
                } ${team.team === selectedTeam || i === teams.length - 1 ? "" : "border-b border-pl-border/60"}`}
              >
                <span aria-hidden className={`text-sm font-bold text-pl-text-faint transition-transform ${team.team === selectedTeam ? "rotate-90" : ""}`}>›</span>
                <TeamBadge team={team.team} size="sm" />
                <span className="min-w-0">
                  <span className="block truncate text-xs font-semibold text-pl-text">{team.team}</span>
                  <span className="flex items-center gap-1 text-[11px] text-pl-text-faint">
                    Form {value(team.form_points_per_match)} PPG
                    <span title={FORM_TREND[team.form_trend].label} className={`font-bold ${FORM_TREND[team.form_trend].className}`}>
                      {FORM_TREND[team.form_trend].arrow}
                    </span>
                  </span>
                </span>
                <StatChip label="Table" value={String(team.currentPosition)} />
                <StatChip label="Recent" value={team.formPosition === null ? "—" : String(team.formPosition)} />
                <StatChip label="For-Ag" value={`${team.goals_for}-${team.goals_against}`} />
                <StatChip label="Total" value={String(team.assists)} />
                <StatChip label="Per match" value={value(team.corners_per_match)} />
                <StatChip label="Total" value={value(team.xg_for)} />
                <StatChip label="Total" value={value(team.xa)} />
              </button>
              {team.team === selectedTeam && selected && (
                <div className={`border-b border-pl-border bg-pl-900/30 p-4 ${i === teams.length - 1 ? "border-0" : ""}`}>
                  <TeamDetailPanel selected={selected} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
