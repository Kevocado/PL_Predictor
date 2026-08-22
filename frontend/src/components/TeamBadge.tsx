import { teamColor, teamInitials } from "../lib/teamColors";

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
  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-full font-bold text-white shadow-sm ${SIZES[size]}`}
      style={{
        background: `linear-gradient(135deg, ${color}, ${color}cc)`,
        boxShadow: `0 0 0 1px rgba(255,255,255,0.08), 0 2px 8px -2px ${color}88`,
      }}
      title={team}
    >
      {teamInitials(team)}
    </div>
  );
}
