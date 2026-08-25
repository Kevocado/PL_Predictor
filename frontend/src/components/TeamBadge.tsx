import { useState } from "react";
import { teamColor, teamCrestUrl, teamInitials } from "../lib/teamColors";

interface Props {
  team: string;
  size?: "sm" | "md" | "lg";
}

const SIZES = {
  sm: "h-7 w-7 text-[10px]",
  md: "h-10 w-10 text-xs",
  lg: "h-14 w-14 text-sm",
};

export function TeamBadge({ team, size = "md" }: Props) {
  const color = teamColor(team);
  const crestUrl = teamCrestUrl(team);
  const [crestFailed, setCrestFailed] = useState(false);
  const showCrest = crestUrl !== null && !crestFailed;
  return (
    <div
      className={`flex shrink-0 items-center justify-center overflow-hidden rounded-full font-bold text-white ${SIZES[size]}`}
      style={{
        background: showCrest ? "transparent" : `linear-gradient(135deg, ${color}, ${color}cc)`,
        boxShadow: showCrest ? "none" : `0 0 0 1px rgba(255,255,255,0.08), 0 2px 8px -2px ${color}88`,
      }}
      title={team}
    >
      {showCrest ? (
        <img src={crestUrl} alt={`${team} crest`} className="h-full w-full object-contain p-0.5" onError={() => setCrestFailed(true)} />
      ) : teamInitials(team)}
    </div>
  );
}
