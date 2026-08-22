interface Props {
  home: number;
  draw: number;
  away: number;
  homeLabel?: string;
  awayLabel?: string;
}

export function ProbabilityBar({ home, draw, away, homeLabel = "H", awayLabel = "A" }: Props) {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

  return (
    <div className="w-full">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-pl-850">
        <div className="bg-win" style={{ width: pct(home) }} />
        <div className="bg-draw" style={{ width: pct(draw) }} />
        <div className="bg-loss" style={{ width: pct(away) }} />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] font-medium text-pl-text-dim">
        <span>
          {homeLabel} <span className="text-pl-text">{pct(home)}</span>
        </span>
        <span>
          D <span className="text-pl-text">{pct(draw)}</span>
        </span>
        <span>
          {awayLabel} <span className="text-pl-text">{pct(away)}</span>
        </span>
      </div>
    </div>
  );
}
