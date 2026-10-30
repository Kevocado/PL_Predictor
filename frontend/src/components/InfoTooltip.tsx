import { useRef, useState } from "react";
import { createPortal } from "react-dom";

interface Props {
  text: string;
  align?: "left" | "center" | "right";
}

const WIDTH = 208; // px, matches w-52

export function InfoTooltip({ text, align = "center" }: Props) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const iconRef = useRef<HTMLSpanElement>(null);

  const show = () => {
    const rect = iconRef.current?.getBoundingClientRect();
    if (!rect) return;

    let left = rect.left + rect.width / 2 - WIDTH / 2;
    if (align === "left") left = rect.left;
    if (align === "right") left = rect.right - WIDTH;
    left = Math.min(Math.max(left, 8), window.innerWidth - WIDTH - 8);

    setPos({ top: rect.top - 8, left });
    setOpen(true);
  };

  return (
    <span
      ref={iconRef}
      className="relative inline-flex cursor-help align-middle"
      onMouseEnter={show}
      onMouseLeave={() => setOpen(false)}
      onFocus={show}
      onBlur={() => setOpen(false)}
      tabIndex={0}
    >
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full bg-pr-panel-2 text-xs font-bold leading-none text-pr-text-dim ring-1 ring-pr-rule transition hover:bg-pr-accent hover:text-pr-accent-ink">
        ?
      </span>
      {open &&
        pos &&
        createPortal(
          <span
            className="pointer-events-none fixed z-[100] rounded-lg border border-pr-rule bg-pr-stage p-2.5 text-xs font-normal normal-case leading-snug tracking-normal text-pr-text-dim shadow-xl"
            style={{ top: pos.top, left: pos.left, width: WIDTH, transform: "translateY(-100%)" }}
          >
            {text}
          </span>,
          document.body
        )}
    </span>
  );
}
