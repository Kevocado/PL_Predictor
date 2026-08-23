import type { BiggestUpset } from "../types";

function outcomeLabel(u: BiggestUpset): string {
  if (u.actual_outcome === "home_win") return `${u.team_home} win`;
  if (u.actual_outcome === "away_win") return `${u.team_away} win`;
  return "Draw";
}

export function MissesTable({ misses }: { misses: BiggestUpset[] }) {
  if (misses.length === 0) {
    return <p className="text-xs text-pl-text-faint">No resolved predictions to review yet.</p>;
  }

  return (
    <ul className="flex flex-col gap-1.5">
      {misses.map((u, i) => (
        <li key={i} className="flex items-baseline justify-between gap-3 text-xs">
          <span className="text-pl-text-dim">
            <span className="font-mono font-semibold text-pl-text">
              {u.team_home} {u.actual_goals_home}-{u.actual_goals_away} {u.team_away}
            </span>{" "}
            — model gave {outcomeLabel(u)} a {(u.predicted_prob * 100).toFixed(0)}% chance
          </span>
        </li>
      ))}
    </ul>
  );
}
