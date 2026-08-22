import type { PlayerPrediction } from "../types";

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
  const dimmed = player.anytime_goal_prob < 0.01 && player.anytime_assist_prob < 0.01;
  return (
    <div className={`flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm ${dimmed ? "opacity-50" : ""}`}>
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${STATUS_COLOR[player.status] ?? "bg-pl-text-faint"}`}
          title={`${STATUS_LABEL[player.status] ?? player.status}${player.news ? ` — ${player.news}` : ""}`}
        />
        <span className="truncate text-pl-text">{player.name}</span>
        <span className="shrink-0 text-[10px] uppercase text-pl-text-faint">{player.position}</span>
      </div>
      <div className="flex shrink-0 items-center gap-3 text-xs">
        <span className="text-pl-text-faint">
          Goal <span className="font-semibold text-pl-text">{(player.anytime_goal_prob * 100).toFixed(0)}%</span>
        </span>
        <span className="text-pl-text-faint">
          Assist <span className="font-semibold text-pl-text">{(player.anytime_assist_prob * 100).toFixed(0)}%</span>
        </span>
      </div>
    </div>
  );
}

interface Props {
  homeTeam: string;
  awayTeam: string;
  homePlayers: PlayerPrediction[];
  awayPlayers: PlayerPrediction[];
}

export function PlayerScorerList({ homeTeam, awayTeam, homePlayers, awayPlayers }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <div className="flex flex-col gap-1.5">
        <span className="text-[11px] font-semibold text-pl-text-faint">{homeTeam}</span>
        {homePlayers.map((p) => (
          <PlayerRow key={p.player_id} player={p} />
        ))}
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-[11px] font-semibold text-pl-text-faint">{awayTeam}</span>
        {awayPlayers.map((p) => (
          <PlayerRow key={p.player_id} player={p} />
        ))}
      </div>
    </div>
  );
}
