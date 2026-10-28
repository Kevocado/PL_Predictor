interface Props {
  home: number;
  draw: number;
  away: number;
  homeLabel?: string;
  awayLabel?: string;
  /** The side the model picked. Only the pick wears the accent; a bar with no
   *  pick (a toss-up) wears none. Never win/draw/loss colours: those claim an
   *  outcome, and a pre-match bar claims only a lean. */
  pick?: "home_win" | "draw" | "away_win" | null;
}

const NEUTRAL: Record<"home_win" | "draw" | "away_win", string> = {
  home_win: "var(--color-pr-text-dim)",
  draw: "var(--color-pr-fill-mute)",
  away_win: "var(--color-pr-text-faint)",
};

export function ProbabilityBar({ home, draw, away, homeLabel = "H", awayLabel = "A", pick = null }: Props) {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const fill = (side: "home_win" | "draw" | "away_win") => (pick === side ? "var(--color-pr-accent)" : NEUTRAL[side]);
  const strong = (side: "home_win" | "draw" | "away_win") => (pick === side ? "text-pr-text font-semibold" : "text-pr-text-dim");

  return (
    <div className="w-full">
      <div className="flex h-2.5 w-full gap-px overflow-hidden rounded-pr bg-pr-panel">
        <div style={{ width: pct(home), background: fill("home_win") }} />
        <div style={{ width: pct(draw), background: fill("draw") }} />
        <div style={{ width: pct(away), background: fill("away_win") }} />
      </div>
      <div className="mt-1.5 flex justify-between text-xs font-medium text-pr-text-dim">
        <span>
          {homeLabel} <span className={`tabular-nums ${strong("home_win")}`}>{pct(home)}</span>
        </span>
        <span>
          D <span className={`tabular-nums ${strong("draw")}`}>{pct(draw)}</span>
        </span>
        <span>
          {awayLabel} <span className={`tabular-nums ${strong("away_win")}`}>{pct(away)}</span>
        </span>
      </div>
    </div>
  );
}
