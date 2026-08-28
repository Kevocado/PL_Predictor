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
        {player.confirmed_starter ? (
          <span className="rounded bg-win/20 px-1.5 py-0.5 text-[10px] font-semibold text-win">Confirmed XI</span>
        ) : player.predicted_starter ? (
          <span className="rounded bg-win/20 px-1.5 py-0.5 text-[10px] font-semibold text-win">XI</span>
        ) : null}
        {player.is_penalty_taker && <span className="rounded bg-pl-accent/20 px-1.5 py-0.5 text-[10px] font-semibold text-pl-accent">PK</span>}
        {!player.is_penalty_taker && player.is_set_piece_taker && <span className="rounded bg-pl-700 px-1.5 py-0.5 text-[10px] font-semibold text-pl-text">SP</span>}
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

function TeamPlayerPredictions({ team, players }: { team: string; players: PlayerPrediction[] }) {
  const byContribution = [...players].sort(
    (left, right) => right.anytime_goal_contribution_prob - left.anytime_goal_contribution_prob,
  );
  const confirmed = players.some((player) => player.confirmed_starter);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold text-pl-text-faint">{team}</span>
        {confirmed && <span className="text-[10px] font-semibold uppercase text-win">Official lineup</span>}
      </div>
      <div className="flex flex-col gap-1.5">
        {players.map((player) => <PlayerRow key={player.player_id} player={player} />)}
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-[10px] font-semibold uppercase text-pl-text-faint">Goal + assist chance</span>
        {byContribution.map((player) => (
          <div key={`contribution-${player.player_id}`} className="flex items-center justify-between rounded-lg bg-pl-850/40 px-3 py-1.5 text-xs">
            <span className="truncate text-pl-text">{player.name}</span>
            <span className="font-semibold text-pl-text">{(player.anytime_goal_contribution_prob * 100).toFixed(0)}%</span>
          </div>
        ))}
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
        <TeamPlayerPredictions team={homeTeam} players={homePlayers} />
      </div>
      <div className="flex flex-col gap-1.5">
        <TeamPlayerPredictions team={awayTeam} players={awayPlayers} />
      </div>
    </div>
  );
}

type HighlightTier = "confirmed" | "predicted" | "model_pick";

const TIER_LABEL: Record<HighlightTier, string> = {
  confirmed: "Confirmed XI",
  predicted: "Predicted XI",
  model_pick: "Model pick",
};

export function PlayerHighlights({ homePlayers, awayPlayers }: Pick<Props, "homePlayers" | "awayPlayers">) {
  const allPlayers = [...homePlayers, ...awayPlayers].filter(
    (player) => player.status === "a" && player.confidence !== "none",
  );
  // Three tiers, falling through to whichever is non-empty: confirmed
  // lineups (close to kickoff) > the lineup model's own >50% start
  // probability (player_goals.py's predicted_starter) > any available
  // player at all, ranked by goal-contribution chance. That last tier
  // matters early in a season specifically: with only a game or two of
  // current-season data, the calibrated start-probability model can
  // genuinely clear 50% for nobody on a given team, which isn't the same
  // as "no recommendation exists" — the underlying goal/assist model still
  // has a real, ranked opinion on every available player regardless.
  let tier: HighlightTier = "confirmed";
  let pool = allPlayers.filter((player) => player.confirmed_starter);
  if (pool.length === 0) {
    tier = "predicted";
    pool = allPlayers.filter((player) => player.predicted_starter);
  }
  if (pool.length === 0) {
    tier = "model_pick";
    pool = allPlayers;
  }
  const picks = [...pool].sort((left, right) => right.anytime_goal_contribution_prob - left.anytime_goal_contribution_prob).slice(0, 3);

  return (
    <section>
      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
        Top {tier === "confirmed" ? "confirmed" : "predicted"} player calls
      </h3>
      {picks.length === 0 ? (
        <p className="rounded-lg bg-pl-850/60 px-3 py-2 text-xs text-pl-text-faint">No available players to call for this fixture.</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {picks.map((player) => (
            <div key={`highlight-${player.player_id}`} className="flex items-center justify-between rounded-lg bg-pl-850/60 px-3 py-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="font-medium text-pl-text">{player.name}</span>
                <span className="rounded bg-win/20 px-1.5 py-0.5 text-[10px] font-semibold text-win">
                  {TIER_LABEL[tier]}
                </span>
              </div>
              <span className="text-xs text-pl-text-faint">
                G+A <span className="font-semibold text-pl-text">{(player.anytime_goal_contribution_prob * 100).toFixed(0)}%</span>
                <span className="ml-2">G {(player.anytime_goal_prob * 100).toFixed(0)}% · A {(player.anytime_assist_prob * 100).toFixed(0)}%</span>
              </span>
            </div>
          ))}
        </div>
      )}
      <p className="mt-1 text-[11px] text-pl-text-faint">Player calls are model projections, not live odds recommendations.</p>
    </section>
  );
}
