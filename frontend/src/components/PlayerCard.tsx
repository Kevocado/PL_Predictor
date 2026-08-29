import type { FPLPlayerProjection } from "../types";
import { TeamBadge } from "./TeamBadge";

interface PlayerCardProps {
  player: FPLPlayerProjection;
  marker?: string;
  size?: "sm" | "md";
}

export function PlayerCard({ player, marker, size = "md" }: PlayerCardProps) {
  const isSmall = size === "sm";
  return (
    <div
      className={`relative min-w-0 rounded-xl border border-pl-border bg-pl-900/90 text-center shadow-sm ${
        isSmall ? "px-1.5 py-1.5" : "px-3 py-3"
      }`}
    >
      {marker && (
        <span className="absolute -right-1 -top-1 rounded-full bg-pl-pink px-1.5 py-0.5 text-[9px] font-bold text-white">
          {marker}
        </span>
      )}
      <div className="flex items-center justify-center gap-1">
        <TeamBadge team={player.team} size="sm" />
        <span
          className={`rounded bg-pl-850 px-1 py-0.5 font-semibold text-pl-text-dim ${
            isSmall ? "text-[8px]" : "text-[9px]"
          }`}
        >
          {player.position}
        </span>
        {player.status !== "a" && <span className="text-loss">●</span>}
      </div>
      <p className={`mt-1 font-bold text-pl-pink ${isSmall ? "text-sm" : "text-lg"}`}>
        {player.projected_points.toFixed(1)}
      </p>
      <p className={`truncate font-semibold text-white ${isSmall ? "text-[10px]" : "text-xs"}`}>
        {player.web_name || player.name}
      </p>
      {!isSmall && <p className="text-[10px] text-pl-text-faint">£{player.price.toFixed(1)}</p>}
    </div>
  );
}
